"""Centralized file watcher for all active conversations.

Scans every active conversation with a running or recently running (2s) agent
process, plus conversations whose backlog is being streamed after a reopen
(``start_reload`` / ``finish_reload``). Reloading conversations are
force-included regardless of process activity and share a per-tick wall-clock
wallet (§7.3) so multiple parallel reloads never starve the GUI thread.

A conversation is also active while an unattached (orphaned) agent holds its
io dir's agent lock (process_lock.py) — a GUI restart/crash can leave the
agent alive and owning the streams while the reopened GUI has no Popen for
it. The held-lock arm keeps scanning those conversations (late records —
e.g. a missed cancel_turn_ack — parse and route normally), a lock drop is
recorded like a process exit (the 2s post-exit tail scans the holder's
final bytes), and either verdict flip fires the conversation's
on_liveness_changed so buttons/status re-render without a user action.
Probes are TTL-cached (see _probe_lock_held / _conditional_lock_held).

``wake()`` interrupts the poll sleep so a send / permission response /
subprocess launch renders on an immediately woken tick (§5.10) instead of
waiting up to a full tick interval.
"""

import threading
import time
import logging
from typing import Optional, Dict, List, Set

from fluximp_protocol.user_agent.file_io.defs import MessageType
from fluximp_agent.process_lock import probe as _probe_agent_lock


class FileWatcher:
    """Centralized file watcher that monitors all active conversations.

    Replaces per-conversation watcher threads with a single thread that
    scans all active conversations based on the current mode.
    """

    def __init__(self):
        self._conversations: Dict[str, object] = {}
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        # Conversations whose backlog is being streamed after a reopen.
        self._reloading: Set[str] = set()
        # Conversations whose reload scan raised (disables the conversation
        # until it is re-registered on a later reopen).
        self._failed_conversations: Set[str] = set()
        # Wakes the poll immediately (idempotent set) on sends, permission
        # responses, and subprocess launches.
        self._wake_event = threading.Event()

        # Track process exit times for "recently running" detection
        # conversation_id -> time.time() when process exit was detected
        self._process_exit_times: Dict[str, float] = {}

        # Track which conversations had running processes last check
        # conversation_id -> bool
        self._was_process_running: Dict[str, bool] = {}

        # Liveness probe verdict caches (process_lock.py arm): TTL-bounded
        # so the 50ms tick's per-conversation lock checks cost at most one
        # syscall pair per interval per conversation. conversation_id ->
        # (verdict, time.time()) for the primary io-dir lock; str(run_dir)
        # -> (verdict, time.time()) for conditional run dirs.
        self._lock_verdicts: Dict[str, tuple] = {}
        self._run_lock_verdicts: Dict[str, tuple] = {}
        self._lock_probe_interval = 0.5
        self.logger = logging.getLogger("file_watcher")

    def start(self):
        """Start the file watcher thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._watch_loop, daemon=True, name="FileWatcher"
        )
        self._thread.start()
        self.logger.info("File watcher started")

    def stop(self):
        """Stop the file watcher thread."""
        self._stop_event.set()
        self._wake_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        self.logger.info("File watcher stopped")

    def register_conversation(self, conversation):
        """Register a conversation to be watched.

        The watcher will scan this conversation's output files when it has
        a running or recently running agent process, or while it is reloading.
        A conversation that previously failed to reload is re-armed on
        re-registration (a later reopen can retry).
        """
        with self._lock:
            self._conversations[conversation.conversation_id] = conversation
            self._was_process_running[conversation.conversation_id] = False
            self._failed_conversations.discard(conversation.conversation_id)
        self.logger.debug(f"Registered conversation {conversation.conversation_id}")

    def unregister_conversation(self, conversation_id: str):
        """Unregister a conversation from being watched."""
        with self._lock:
            self._conversations.pop(conversation_id, None)
            self._process_exit_times.pop(conversation_id, None)
            self._was_process_running.pop(conversation_id, None)
            # Drop this conversation's liveness probe-cache entry (the
            # run-dir cache is path-keyed and self-refreshes via TTL).
            self._lock_verdicts.pop(conversation_id, None)
            # A mid-load close must never fire finish_reload afterwards.
            self._reloading.discard(conversation_id)
            self._failed_conversations.discard(conversation_id)
        self.logger.debug(f"Unregistered conversation {conversation_id}")

    def wake(self):
        """Wake the watcher loop immediately (safe from any thread).

        Called after a send / permission response / subprocess launch so the
        newly written input record renders on an immediately woken tick
        instead of waiting up to a full 50 ms interval (§5.10). Idempotent.
        """
        self._wake_event.set()

    def start_reload(self, conversation_id: str):
        """Begin reload streaming for a registered conversation.

        Valid only for an already-registered conversation (the watcher must
        know it to force-include it regardless of process activity). Raises
        ValueError for an unknown id. The conversation's streams must already
        have their size snapshot taken (``_refresh_stream_sizes``) so the
        finalize-once predicate is well-defined.
        """
        with self._lock:
            if conversation_id not in self._conversations:
                raise ValueError(
                    f"cannot start reload for unknown conversation {conversation_id}"
                )
            self._reloading.add(conversation_id)
            self._failed_conversations.discard(conversation_id)
        self.wake()
        self.logger.info(f"Started reload for conversation {conversation_id}")

    def finish_reload(self, conversation_id: str):
        """End reload streaming without finalizing (mid-load close).

        Called by the GUI when a conversation is closed during reload so the
        watcher never fires on_reload_complete for it afterwards.
        """
        with self._lock:
            self._reloading.discard(conversation_id)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _watch_loop(self):
        """Main file watching loop."""
        tick_interval = 0.05
        while not self._stop_event.is_set():
            now = time.time()

            with self._lock:
                # Compute the reload wallet for this tick: 0.85 x tick_interval
                # shared among the reloading conversations scanned this tick
                # (§7.3). Live (non-reloading) scans are never throttled.
                reloading_now = [
                    c for c in self._conversations.values()
                    if c.conversation_id in self._reloading
                ]
                if reloading_now:
                    time_budget = (tick_interval * 0.85) / max(1, len(reloading_now))
                else:
                    time_budget = None

                conversations_to_scan = self._get_conversations_to_scan(now)

            # Scan conversations (outside the lock to avoid blocking)
            for conversation in conversations_to_scan:
                try:
                    # Skip if the conversation was unregistered after the scan
                    # list was built (defense-in-depth against stale references)
                    with self._lock:
                        if conversation.conversation_id not in self._conversations:
                            continue
                        is_reloading = conversation.conversation_id in self._reloading

                    # Reloading conversations stream under batch brackets (§6):
                    # the tab flushes once per batch instead of per message.
                    if is_reloading and conversation.on_scan_batch_start:
                        conversation.on_scan_batch_start()
                    try:
                        if is_reloading:
                            conversation._scan_io_files(time_budget=time_budget)
                        else:
                            conversation._scan_io_files()
                    finally:
                        if is_reloading and conversation.on_scan_batch_end:
                            conversation.on_scan_batch_end()

                    # Finalize once every stream is caught up to its snapshot
                    # size (§7.3): drop from _reloading and fire
                    # on_reload_complete exactly once.
                    if is_reloading:
                        self._maybe_finalize_reload(conversation)
                except Exception as e:
                    self.logger.error(
                        f"Error scanning conversation "
                        f"{conversation.conversation_id}: {e}"
                    )
                    with self._lock:
                        if conversation.conversation_id in self._reloading:
                            self._reloading.discard(conversation.conversation_id)
                            self._failed_conversations.add(conversation.conversation_id)
                            self.logger.error(
                                f"Reload failed for conversation "
                                f"{conversation.conversation_id}: {e}"
                            )

            # Check for process status changes
            self._check_process_status(now)

            # Sleep for 50ms, but wake immediately if stop is requested or
            # wake() was called (a send / permission response / subprocess
            # launch). wake() sets the event; stop() sets both events.
            self._wake_event.wait(tick_interval)
            self._wake_event.clear()

    def _maybe_finalize_reload(self, conversation):
        """Finalize a reloading conversation once every stream is caught up.

        The caught-up predicate (§7.3) holds when each stream in the reload
        snapshot (``conversation._stream_sizes``) has been committed to its
        snapshot size or beyond. On caught-up: copy-assign the timestamp
        sort, drop the id from ``_reloading``, and fire ``on_reload_complete``
        exactly once. A mid-load close (unregister) removes the id from
        ``_reloading`` first, so a stale predicate can never finalize a
        closed conversation.
        """
        if not conversation.reload_caught_up():
            return
        with self._lock:
            if conversation.conversation_id not in self._reloading:
                return
            self._reloading.discard(conversation.conversation_id)
        # Copy-assign (never sort in place): the finalization reads on the GUI
        # thread must see a stable list (§7.3).
        conversation.messages = sorted(
            conversation.messages, key=lambda m: m.timestamp)
        self.logger.info(
            f"Reload finalized for conversation {conversation.conversation_id}"
        )
        if conversation.on_reload_complete:
            conversation.on_reload_complete()


    def _get_conversations_to_scan(self, now: float) -> List:
        """Get the list of conversations to scan this tick.

        Reloading conversations are force-included regardless of process
        activity so the backlog streams in after the process has exited.
        Every other conversation is scanned only while active (running or
        recently running). Reload failures disable the conversation until it
        is re-registered on a later reopen.
        """
        result = []
        for conv in self._conversations.values():
            if conv.conversation_id in self._reloading:
                result.append(conv)
            elif conv.conversation_id not in self._failed_conversations:
                if self._is_conversation_active(conv, now):
                    result.append(conv)
        return result

    def _probe_lock_held(self, conv, now: float) -> bool:
        """True when the conversation's io-dir agent lock is held.

        A held lock with no attached Popen means an unattached agent
        process (orphaned by a GUI exit/crash) owns the io directory —
        alive by lock, absorbed by the tab rather than respawned. Verdicts
        are cached per conversation (a probe is a flock attempt or mutex
        syscall); 'unknown' (probe failure) is never alive — callers
        degrade to the attached-Popen-only behavior of the pre-lock era.
        """
        conv_id = conv.conversation_id
        cached = self._lock_verdicts.get(conv_id)
        if cached is not None and now - cached[1] < self._lock_probe_interval:
            return cached[0] == 'held'
        try:
            verdict = _probe_agent_lock(conv.io_dir, 'agent')
        except Exception:
            verdict = 'unknown'
        self._lock_verdicts[conv_id] = (verdict, now)
        return verdict == 'held'

    def _conditional_lock_held(self, conv, now: float) -> bool:
        """True when any discovered conditional run dir's lock is held.

        A conditional run orphaned by a GUI exit still owns its run dir
        (each run dir is its own single-owner scope, process_lock.py)
        while the primary agent may be dead; the held arm keeps the
        conversation scanning so the run's report streams live during
        this GUI session. Probe cost is TTL-bounded per run dir.
        """
        targets = getattr(conv, '_conditional_scan_targets', None)
        if not targets:
            return False
        for run_dir in targets:
            key = str(run_dir)
            cached = self._run_lock_verdicts.get(key)
            if (cached is not None
                    and now - cached[1] < self._lock_probe_interval):
                if cached[0] == 'held':
                    return True
                continue
            try:
                verdict = _probe_agent_lock(run_dir, 'agent')
            except Exception:
                verdict = 'unknown'
            self._run_lock_verdicts[key] = (verdict, now)
            if verdict == 'held':
                return True
        return False

    def _is_conversation_active(self, conv, now: float) -> bool:
        """Check if a conversation has a running or recently running agent.

        A conversation is considered active if:
        - Its process is currently running (poll() returns None), OR
        - Its process exited within the last 2 seconds (the post-exit tail
          scan catches the final records), OR
        - Its Popen object exists with an exited process whose exit has not
          yet been recorded (ensures a scan for last output), OR
        - Its io dir's agent lock is held (process_lock.py): no Popen
          exists, but an unattached (orphaned) agent process owns the
          conversation — alive by lock. The reopened GUI keeps streaming
          its records through the ordinary scan path, OR
        - Any discovered conditional run dir's lock is held (a conditional
          orphan survived the GUI exit while the primary may be dead).
        """
        if conv.process is not None:
            if conv.process.poll() is None:
                return True  # Process is running
            else:
                # Process has exited but Popen object still exists
                exit_time = self._process_exit_times.get(conv.conversation_id)
                if exit_time is None:
                    # Exit not yet recorded by _check_process_status
                    # Treat as active so we can scan for any last output
                    return True
                elif now - exit_time < 2.0:
                    return True
        else:
            # Process was set to None (terminated) - check exit time
            exit_time = self._process_exit_times.get(conv.conversation_id)
            if exit_time and now - exit_time < 2.0:
                return True

        # Alive-by-lock arms (process_lock.py): an unattached agent holds
        # the primary io dir's lock (absorbed orphan — a missed
        # cancel_turn_ack gets parsed the moment it lands), or an orphaned
        # conditional run owns its run dir. Verdicts are TTL-cached.
        if self._probe_lock_held(conv, now):
            return True
        if self._conditional_lock_held(conv, now):
            return True
        return False

    def _check_process_status(self, now: float):
        """Check for liveness changes (attached process + lock verdicts).

        A conversation's alive verdict folds in its io dir's agent-lock
        probe (process_lock.py): with no attached Popen, ``held`` means an
        unattached (orphaned) agent owns the conversation's streams — alive.
        Verdict transitions are recorded exactly like attached-process
        transitions:
        - alive -> not stamps ``_process_exit_times`` so the 2-second
          post-exit tail scans the holder's final bytes (an orphan's late
          cancel_turn_ack lands in this window), and
        - either flip fires the conversation's on_liveness_changed callback
          (watcher thread; the owning tab marshals) so Stop/Resume/status
          re-render without a user action.
        """
        with self._lock:
            for conv_id, conv in self._conversations.items():
                attached_alive = (
                    conv.process is not None and conv.process.poll() is None
                )
                # Only conversations without a live attached Popen are probed:
                # the attached window owns the verdict, so a just-spawned
                # GUI agent is never raced by the lock's acquire latency.
                lock_alive = ((not attached_alive)
                              and self._probe_lock_held(conv, now))
                is_running = attached_alive or lock_alive
                was_running = self._was_process_running.get(conv_id, False)

                transition_cb = None
                if is_running and not was_running:
                    # Started: a GUI attach, or an orphan discovered alive
                    # by lock on the first check after registration.
                    self._was_process_running[conv_id] = True
                    self._process_exit_times.pop(conv_id, None)
                    transition_cb = getattr(conv, "on_liveness_changed", None)
                elif not is_running and was_running:
                    # Exited: the attached Popen, or a held lock dropped (a
                    # live orphan wrote its last record and quit). The exit
                    # stamp drives the 2s post-exit tail for both shapes.
                    self._was_process_running[conv_id] = False
                    self._process_exit_times[conv_id] = now
                    transition_cb = getattr(conv, "on_liveness_changed", None)
                elif not is_running and not was_running:
                    # Not running and never seen running: record an
                    # attached-but-exited Popen exactly once so the 2s tail
                    # still scans for last output.
                    if (conv.process is not None
                            and conv.process.poll() is not None
                            and conv_id not in self._process_exit_times):
                        self._process_exit_times[conv_id] = now

                if transition_cb is not None:
                    try:
                        transition_cb()
                    except Exception:
                        self.logger.exception(
                            f"on_liveness_changed callback failed for {conv_id}"
                        )
