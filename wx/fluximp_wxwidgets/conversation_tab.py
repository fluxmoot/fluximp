"""Conversation tab - UI for a single workflow."""

import wx
import logging
import subprocess
import threading
import time
from typing import Optional, List, Dict, Tuple, Callable
from datetime import datetime
from pathlib import Path

from fluximp_wxwidgets.workflow_history_controller import WorkflowHistoryController
from fluximp_wxwidgets.workflow_history_view import WorkflowHistoryView
from fluximp_wxwidgets.conversation_manager import Conversation, ConversationMessage
from fluximp_wxwidgets.conditional_sequence import (
    SequenceCallbacks, ConditionalSequenceState,
    on_primary_agent_work_completion_summary, on_conditional_agent_complete,
    on_resume_button_pressed, load_state, save_state,
    build_conversation_context_messages, apply_offline_terminal_transition,
)
from fluximp_agent.conditional_agent import ConditionalAgentConfigInfo
from fluximp_agent.process_lock import probe as _probe_agent_lock
from fluximp_protocol.user_agent.fork_turn_state import IOFactKind, IOTurnStatus
from fluximp_wxwidgets.tab_status_icons import (
    STATUS_IDLE, STATUS_RUNNING, STATUS_WAITING, STATUS_PAUSED,
    STATUS_STOPPED, STATUS_ERROR, STATUS_COMPLETED, STATUS_APPROVAL,
)




class ConversationTab(wx.Panel):
    """Tab panel for a single conversation."""

    # Replay chunk size (§8.5): one watcher batch is replayed in chunks of
    # this many events, yielding to the event loop between chunks so paint
    # events are not starved while the backlog streams.
    RELOAD_REPLAY_CHUNK = 12

    # Input-box grey hints, paired with the box's enabled state (§2.2): the
    # enabled hint invites a message; the disabled one explains the screen
    # lock and how to interject. Wording is user-specified verbatim.
    _INPUT_HINT_ENABLED = "Type your message here..."
    _INPUT_HINT_DISABLED = (
        "The agent can not receive a message while its working.  "
        "Press the Stop button if you want to interrupt its work and interject.")

    def __init__(self, parent, conversation: Conversation,
                 file_watcher=None, config_mgr=None, right_panel=None,
                 app_settings=None):
        super().__init__(parent)

        self.conversation = conversation
        self.file_watcher = file_watcher  # Centralized file watcher for mode switching
        self.right_panel = right_panel  # Shared singleton; all forwards are display-gated (see _panel_displays)
        # App-wide settings, forwarded to the history view so request cards
        # can honour the View > API Request Card Metadata visibility toggles.
        self.app_settings = app_settings

        # Track whether the agent is actively working (API request in progress)
        self._agent_active: bool = False
        self._stop_pressed: bool = False  # True after Stop pressed, reset on new process
        self._closed: bool = False  # Set to True when tab is closed/destroyed

        # Spawn window (Option 2e): True between the Send-driven spawn and
        # the process attach, so the header reads 'running' immediately.
        self._process_starting: bool = False
        # Tab-header status icon (Option F). on_status_changed is wired by
        # the main frame to its notebook-bitmap updater; _last_reported_status
        # dedupes those updates; _last_terminal carries the most recent turn
        # outcome so a dead agent's icon can read error/completed.
        self.on_status_changed: Optional[Callable[[], None]] = None
        self._last_reported_status: Optional[str] = None
        self._last_terminal: Optional[str] = None

        # Liveness probe verdict cache (tab-local, GUI thread):
        # (verdict, monotonic timestamp). Render recomputes reuse the
        # cache; click handlers re-probe with force=True so no action
        # ever acts on a stale verdict (process_lock.py).
        self._alive_probe: Tuple[str, Optional[float]] = ('free', None)
        # The view fork is now derived from fork_turn_state.latest_fork at each
        # use site; the next turn is computed from fork_turn_state.next_turn().
        # (No stored _view_fork / _next_turn state.)

        # Fork mode state tracking
        self._fork_mode: bool = False
        self._fork_message: Optional[ConversationMessage] = None
        # History tracking now lives in the WorkflowHistoryController/view.

        # Main sizer
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        # Header with conversation info
        self.header = wx.StaticText(self, label=self._get_header_text())
        font = self.header.GetFont()
        font.SetWeight(wx.FONTWEIGHT_BOLD)
        self.header.SetFont(font)
        main_sizer.Add(self.header, 0, wx.EXPAND | wx.ALL, 5)

        main_sizer.Add(wx.StaticLine(self), 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 5)

        # Message history (scrolled) - owned by the WorkflowHistoryView
        self.history_view = WorkflowHistoryView(
            self, logger=self.conversation.logger,
            app_settings=self.app_settings)
        main_sizer.Add(self.history_view, 1, wx.EXPAND | wx.ALL, 5)

        # Footer container: input area, hidden during reload and replaced by
        # the position status line (§5.10.1).
        self.footer_panel = wx.Panel(self)
        footer_sizer = wx.BoxSizer(wx.VERTICAL)

        # Input area
        input_sizer = wx.BoxSizer(wx.HORIZONTAL)

        self.input_ctrl = wx.TextCtrl(self.footer_panel, style=wx.TE_MULTILINE, size=(-1, 85))
        self.input_ctrl.SetHint(self._INPUT_HINT_ENABLED)
        self.input_ctrl.Bind(wx.EVT_KEY_DOWN, self._on_input_key)
        input_sizer.Add(self.input_ctrl, 1, wx.EXPAND | wx.RIGHT, 5)

        # Button sizer for Send, Stop, and Cancel
        btn_sizer = wx.BoxSizer(wx.VERTICAL)

        self.send_btn = wx.Button(self.footer_panel, label="Send")
        self.send_btn.Bind(wx.EVT_BUTTON, self._on_send)
        btn_sizer.Add(self.send_btn, 0, wx.EXPAND)

        self.stop_btn = wx.Button(self.footer_panel, label="Stop")
        self.stop_btn.Bind(wx.EVT_BUTTON, self._on_stop)
        self.stop_btn.Hide()
        btn_sizer.Add(self.stop_btn, 0, wx.EXPAND | wx.TOP, 5)

        self.resume_btn = wx.Button(self.footer_panel, label="Resume")
        self.resume_btn.Bind(wx.EVT_BUTTON, self._on_resume)
        self.resume_btn.Hide()
        btn_sizer.Add(self.resume_btn, 0, wx.EXPAND | wx.TOP, 5)

        self.cancel_btn = wx.Button(self.footer_panel, label="Cancel")
        self.cancel_btn.Bind(wx.EVT_BUTTON, self._on_cancel)
        self.cancel_btn.Hide()
        btn_sizer.Add(self.cancel_btn, 0, wx.EXPAND | wx.TOP, 5)

        self.resume_sequence_btn = wx.Button(self.footer_panel, label="Resume Sequence")
        self.resume_sequence_btn.Bind(wx.EVT_BUTTON, self._on_resume_sequence)
        self.resume_sequence_btn.Hide()
        btn_sizer.Add(self.resume_sequence_btn, 0, wx.EXPAND | wx.TOP, 5)

        input_sizer.Add(btn_sizer, 0, wx.EXPAND)

        footer_sizer.Add(input_sizer, 0, wx.EXPAND | wx.ALL, 5)

        self.footer_panel.SetSizer(footer_sizer)
        main_sizer.Add(self.footer_panel, 0, wx.EXPAND)

        # Position status line (shown only while a reload streams, §5.10.1):
        # replaces the footer's input/button UI during the load and is hidden
        # again once the footer is restored.
        self.position_label = wx.StaticText(self, label="Viewing Fork 0")
        self.position_label.Hide()
        main_sizer.Add(self.position_label, 0, wx.EXPAND | wx.ALL, 5)

        self.SetSizer(main_sizer)

        # Conditional agent sequence tracking
        self._config_mgr = config_mgr
        # §7.7: the conditional run dicts are keyed by run_id (fresh per
        # launch); run→name is resolved at the completion call site.
        self._conditional_subprocesses: Dict[str, subprocess.Popen] = {}
        self._conditional_event_key: Optional[str] = None
        self._conditional_agent_list: List[ConditionalAgentConfigInfo] = []
        self._conditional_io_dirs: Dict[str, Path] = {}  # run_id -> io_dir
        self._conditional_run_names: Dict[str, str] = {}  # run_id -> agent name
        self._run_terminal_consumed: Dict[str, bool] = {}  # run_id -> terminal routed
        self._deferred_ticks: Dict[str, int] = {}  # run_id -> deferred poll-exit ticks
        self._stopped_sequence_event_key: Optional[str] = None  # §7.7 park-on-stop

        # Timer for polling conditional agent subprocess completion
        self._conditional_timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self._on_conditional_timer, self._conditional_timer)

        # Bind to conversation events
        self.conversation.on_message = self._on_message
        self.conversation.on_message_updated = self._on_message_updated
        self.conversation.on_turn_complete = self._on_turn_complete
        self.conversation.on_fork_turn_update = self._on_fork_turn_update
        self.conversation.on_request_started = self._on_request_started
        self.conversation.on_cancel_ack = self._on_cancel_ack
        self.conversation.on_error = self._on_error
        self.conversation.on_shutdown_starting = self._on_shutdown_starting
        self.conversation.on_shutdown_complete = self._on_shutdown_complete

        # Liveness-transition callback (process_lock.py arm): the watcher
        # thread fires it when the alive verdict flips (attached Popen
        # exit, orphan discovered alive, or a held lock dropped). The
        # handler marshals the recompute to the GUI thread.
        self.conversation.on_liveness_changed = self._on_liveness_changed

        # Displayed-tab-only right-panel forwards. The shared panel used to
        # bind these per-conversation slots itself: the callbacks carried no
        # tab identity, so a background tab's agent/process transitions
        # drove the panel showing whichever tab was active, and switching
        # away/back (the panel's set_conversation) was the only resync.
        # The tab owns its conversation's slots now and gates every
        # forward on the panel's displayed identity (_panel_displays /
        # _notify_panel_config_state).
        self.conversation.on_process_starting = self._on_process_starting
        self.conversation.on_process_attached = self._on_process_attached

        # History controller drives routing, ordering, batching, and policies.
        self.history_controller = WorkflowHistoryController(
            self.conversation,
            view=self.history_view,
            file_watcher=file_watcher,
            permission_response_handler=self._handle_permission_response,
            logger=self.conversation.logger,
        )
        self.history_view.set_controller(self.history_controller)
        self.history_view.set_tab(self)
        self.history_view.set_edit_handler(self._enter_fork_mode)

        # Phase 2: reload streaming. The history controller owns load_existing
        # (deferred by the main frame until after the page is added); the
        # watcher streams the backlog and finalization runs via
        # on_reload_complete. Batch brackets make the view flush once per
        # watcher batch instead of per message (§6).
        self.history_controller.set_tab(self)
        self.conversation.on_reload_complete = self._on_reload_complete
        self.conversation.on_scan_batch_start = self._on_scan_batch_start
        self.conversation.on_scan_batch_end = self._on_scan_batch_end

        # Constructed with the footer suppressed (§5.10.1): brand-new
        # conversations settle via the empty short-circuit, reopened ones at
        # finalization (§7.4 step 9).
        self.footer_panel.Hide()
        self.position_label.Show()

        # Reload batch buffer (§8.5): set on the watcher thread by
        # _on_scan_batch_start; watcher callbacks buffer their events into it
        # so each watcher tick marshals one wx.CallAfter replay instead of one
        # per backlog record.
        self._reload_batch_buffer = None
        # Chunks of the current batch replay still queued on the GUI thread
        # (finalize_reload defers itself until this drains, §8.5).
        self._reload_replay_pending = 0

        # Show Resume button for conversations with existing messages (not brand new)
        self._update_button_states()

    def _ensure_process_running(self) -> bool:
        """Ensure the conversation has a running agent process.

        Returns True if process is running (or was successfully started),
        False if resume failed.
        """
        # Check if process exists and is running
        if self.conversation.process is not None:
            if self.conversation.process.poll() is None:
                return True
            # Process has exited, clear it
            self.conversation.process = None

        # Absorbed-orphan guard (process_lock.py): an unattached agent (a
        # previous GUI session's process) holds the io dir's agent lock.
        # A second process on one io dir would race input files and
        # duplicate expensive turns (the agent-side lock would fail the
        # spawn too). Absorb: no spawn, no notify; the watcher's
        # held-lock arm keeps scanning the orphan's records and the input
        # written here is read by the holder's existing input handles.
        if self._probe_orphan_alive(force=True):
            self.conversation.logger.info(
                "ensure_process_running: unattached agent holds the io dir "
                "lock; absorbing (no new process started)")
            return True
        # Notify that process is starting (disables config controls in UI)
        # This runs on main thread since we're called from button click handler
        if self.conversation.on_process_starting:
            self.conversation.on_process_starting()

        # Spawn window (Option 2e): the status derivation reads this as
        # alive-ish so the first Send shows 'running' immediately rather
        # than a transient idle before the process attaches.  Cleared on
        # attach (or spawn failure) in _start_process_in_background.
        self._process_starting = True
        # Start process in background thread to keep UI responsive
        # This handles shutdown wait, clearing end signals, and process startup
        import threading
        threading.Thread(
            target=self._start_process_in_background,
            daemon=True
        ).start()

        return True

    def _start_process_in_background(self):
        """Start the agent process in a background thread.

        Handles shutdown wait, clearing end signals, and process startup.
        """
        try:
            # Wait for any pending shutdown to complete
            if self.conversation._is_shutting_down:
                self.conversation.logger.debug("Shutdown in progress, waiting...")
                if not self.conversation.wait_for_shutdown(timeout=10.0):
                    self.conversation.logger.error("Timeout waiting for shutdown")
                    return

            # Clear any end signal from previous termination before starting new process
            self.conversation.clear_end_signal()

            # Start new process with current configuration
            process = self.conversation.start_agent_process(
                self.conversation.agent_type,
                self.conversation.current_provider,
                self.conversation.current_model,
                100000,
                persist_checkpoints=True
            )
            self.conversation.attach_process(process)

            # Process is now attached. The agent is idle until
            # _on_request_started fires (mid-turn recovery case).
            # Reset _agent_active so Send is enabled and Stop appears.
            self._agent_active = False
            self._notify_agent_active_state()
        except Exception as e:
            self.conversation.logger.error(f"Failed to start process: {e}")
            self._agent_active = False
            self._stop_pressed = False
            self._notify_agent_active_state()
        finally:
            # Spawn window (Option 2e): close it on every exit — attach
            # success, spawn failure, or the early return above waiting on a
            # shutdown — so the header status can never stick on the
            # alive-ish 'running'/'waiting' icon after a spawn that never
            # produced a process.
            self._process_starting = False
            wx.CallAfter(self._update_button_states)

    def _is_any_agent_active(self) -> bool:
        """Check if any agent (workflow or conditional) is currently active."""
        return self._agent_active or len(self._conditional_subprocesses) > 0

    def _notify_agent_active_state(self):
        """Sync the conversation's persisted agent-active flag.

        The shared right panel no longer binds on_agent_active_changed (the
        callback is deleted — it carried no tab identity, so a background
        tab's transition disabled the combos of whichever tab you were
        looking at): the displayed tab now forwards its input/Send state
        from _update_button_states. Only the flag update remains here —
        the panel reads it as set_conversation's fallback when no tab was
        passed.
        """
        self.conversation.is_agent_active = self._is_any_agent_active()

    def get_config_enabled(self) -> bool:
        """Read this tab's Agent/Provider/Model combo enable state.

        Sole consumer: the right panel's tab-switch sync. The combos follow
        the config gate (_config_gate_enabled) — Send's idle gate, OR
        unlocked whenever no process of any kind is running — so the panel
        asks the tab rather than the conversation: is_agent_active omits
        those gates and is stale during live conditional runs
        (launch/completion never re-notified it).
        """
        if not self._is_window_valid():
            return False
        try:
            return self._config_gate_enabled(self.send_btn.IsEnabled())
        except RuntimeError:
            return False

    def get_restart_available(self) -> bool:
        """True when some control actually restarts the agent process.

        The right panel's stopped-status phrasing only promises "will
        restart on next send" when this is True: Send enabled (a send
        respawns through _ensure_process_running) or Resume visible (the
        resume restarts). A stopped non-resumable workflow has neither
        (§7.8) — for that shape the panel shows the plain stopped form
        and promises nothing.
        """
        if not self._is_window_valid():
            return False
        try:
            return self.send_btn.IsEnabled() or self.resume_btn.IsShown()
        except RuntimeError:
            return False

    def get_status(self, alive: Optional[bool] = None) -> str:
        """Derive this tab's header-status icon key (Option 2e).

        One ordered derivation over an explicit signal set, shape-first
        (colour-blind safe).  Precedence, highest first:

          1. invalid window                     -> idle
          2. alive/starting + pending approval  -> approval
          3. alive/starting + turn in flight    -> running
          4. alive/starting                     -> waiting (idle, your move)
          5. non-resumable                      -> stopped
          6. terminal error                     -> error
          7. terminal completed (with messages) -> completed
          8. messages                           -> paused
          9. otherwise                          -> idle

        *alive* is an optional pre-computed verdict so the central recompute
        does not probe twice.  "Alive-ish" folds in the spawn window so the
        first Send reads running immediately instead of a transient idle.
        """
        if not self._is_window_valid():
            return STATUS_IDLE
        if alive is None:
            alive = self._is_agent_alive()
        alive_ish = alive or self._process_starting
        if alive_ish:
            fork, turn = self.conversation.fork_turn_state.active_position
            if self.history_controller.has_pending_permission(fork, turn):
                return STATUS_APPROVAL
            if self._turn_in_flight():
                return STATUS_RUNNING
            return STATUS_WAITING
        if self.history_controller.non_resumable:
            return STATUS_STOPPED
        if self._last_terminal == STATUS_ERROR:
            return STATUS_ERROR
        if self._last_terminal == STATUS_COMPLETED and self.conversation.messages:
            return STATUS_COMPLETED
        if self.conversation.messages:
            return STATUS_PAUSED
        return STATUS_IDLE

    def _notify_status_changed(self, alive: Optional[bool] = None):
        """Fire on_status_changed when the derived status actually changes.

        Called from _update_button_states (the central recompute) and from
        the non-resumable settle, so the frame's tab-header icon tracks
        every state transition.  The dedupe keeps the notebook repaint to
        real changes only.
        """
        cb = self.on_status_changed
        if cb is None:
            return
        try:
            status = self.get_status(alive)
        except Exception:
            return
        if status == self._last_reported_status:
            return
        self._last_reported_status = status
        cb()

    def _panel_displays(self) -> bool:
        """True when the shared right panel currently displays this tab.

        A plain-attribute identity read (right_panel is set once at
        construction; current_conversation is swapped on the GUI thread;
        identity reads are atomic), so the gate is safe to call from any
        thread, including the watcher.
        """
        panel = self.right_panel
        return (panel is not None
                and self.conversation is panel.current_conversation)

    def _notify_panel_config_state(self, config_enabled: bool):
        """Forward this tab's config lock state to the shared panel.

        The forwarded state is the Agent/Provider/Model combos' gate
        (Send's idle gate OR no process of any kind running — see
        _config_gate_enabled), no longer the input/Send lock itself.

        Runs on the GUI thread (the recompute sites are button handlers,
        timer ticks, or CallAfter-marshalled paths). The identity gate
        drops every other tab's forward: the panel is a singleton over one
        displayed conversation and holds no per-conversation bindings, so
        a background tab's transitions can never reach it again.
        """
        if not self._panel_displays():
            return
        self.right_panel.apply_displayed_tab_state(config_enabled)

    def _on_process_starting(self):
        """A process start is beginning on this tab (GUI thread, fired from
        _ensure_process_running): stamp the shared panel when displayed."""
        if not self._is_window_valid() or not self._panel_displays():
            return
        self.right_panel.note_process_starting()

    def _on_process_attached(self):
        """A process was attached (the manager marshals this to the GUI
        thread): stamp the shared panel's running status when displayed."""
        if not self._is_window_valid() or not self._panel_displays():
            return
        self.right_panel.note_process_attached()

    def _forward_panel_grid_refresh(self):
        """GUI-thread: refresh the shared fork/turn grid when displayed.

        The schedule-time gate in _on_fork_turn_update kills background
        churn; this re-gate covers a tab switch between schedule and apply
        (a switch rebuilds the grid at set_conversation, so either outcome
        is correct).
        """
        if not self._is_window_valid() or not self._panel_displays():
            return
        self.right_panel.refresh_fork_grid()

    def _is_process_running(self) -> bool:
        """Check if the workflow agent process is actually running."""
        if self.conversation.process is None:
            return False
        return self.conversation.process.poll() is None

    def _probe_orphan_alive(self, force: bool = False) -> bool:
        """Lock-probe the conversation's io dir: True when an unattached
        agent (an orphan from a previous GUI session) holds its liveness
        lock (process_lock.py).

        Verdicts are cached briefly (a probe is a flock attempt / mutex
        syscall) for render-time recomputes; click handlers pass
        force=True so no action ever acts on a stale verdict. 'unknown'
        (probe failure) is never alive — behavior degrades to the
        attached-Popen-only semantics of the pre-lock era.
        """
        cached_ts = self._alive_probe[1]
        if (not force and cached_ts is not None
                and time.monotonic() - cached_ts < 0.5):
            return self._alive_probe[0] == 'held'
        try:
            verdict = _probe_agent_lock(self.conversation.io_dir, 'agent')
        except Exception:
            verdict = 'unknown'
        self._alive_probe = (verdict, time.monotonic())
        return verdict == 'held'

    def _is_agent_alive(self) -> bool:
        """True when the workflow agent is alive: the attached Popen (this
        GUI spawned it) OR an unattached lock holder (an orphaned agent
        absorbed at reopen). The single aliveness truth Stop/Resume key on.
        """
        if self.conversation.process is not None:
            if self.conversation.process.poll() is None:
                return True
        return self._probe_orphan_alive()

    def _config_gate_enabled(self, send_enabled: bool,
                             alive: Optional[bool] = None) -> bool:
        """Shared Agent/Provider/Model combos gate (both forward paths).

        The combos keep Send's idle gate (§2.2) as the primary rule, and
        additionally unlock whenever no process of any kind is running:
        no primary agent process (no attached Popen, no orphan lock
        holder), no live conditional run (§7.7), and not in the spawn
        starting window. A mid-turn process death (e.g. an API error) can
        leave the turn's io facts non-terminal forever — the one process
        that would have written the terminal record is gone, and Send
        stays protocol-locked — which must never deadlock the
        Provider/Model switch that is the recovery for that state.
        A non-resumable workflow keeps the combos locked (§7.8).
        """
        if self.history_controller.non_resumable:
            return False
        if alive is None:
            alive = self._is_agent_alive()
        return (send_enabled
                or (not alive
                    and not self._process_starting
                    and len(self._conditional_subprocesses) == 0))

    def _orphan_turn_in_flight(self) -> bool:
        """Re-derive mid-turn state from io facts for an unattached
        (orphan) agent's Stop routing — facts, not the replayed
        _agent_active flag, because the orphan's streams may hold records
        this GUI never parsed and the flag can be stale in either
        direction.

        The active position's turn is in flight when it has input and its
        output stream is non-terminal (parsed but no terminal record yet —
        e.g. a REQUEST whose turn was never completed or acked).
        """
        state = self.conversation.fork_turn_state
        fork, turn = state.active_position
        facts = state.facts
        has_input = any(
            f.kind is IOFactKind.IN and f.fork == fork and f.turn == turn
            for f in facts)
        if not has_input:
            # No input at the active position: the (live) holder is idle
            # at a turn boundary — the .end route reaches it at its poll
            # position; cancel_turn would have nowhere ack-able to land.
            return False
        out_status = next(
            (f.status for f in facts
             if f.kind is IOFactKind.OUT
             and f.fork == fork and f.turn == turn),
            None)
        if out_status is None:
            # Input posted but no output stream yet (the holder has not
            # begun writing the turn): in flight — cancel_turn is the
            # route that can still reach the agent at that position.
            return True
        return out_status not in (IOTurnStatus.COMPLETE_NORMAL,
                                  IOTurnStatus.COMPLETE_ERROR,
                                  IOTurnStatus.CANCELLED,
                                  IOTurnStatus.ABANDONED)

    def _turn_in_flight(self) -> bool:
        """True when a turn is in flight for the header status (Option 2e).

        Facts-based, not the _agent_active busy flag: the active position has
        an input whose output stream is not yet terminal (reusing the orphan
        Stop-routing predicate), or a conditional run is live.  Because
        Conversation.send_message records the IN fact synchronously, this is
        true the instant Send is pressed — no watcher-scan lag, and no
        dependence on the spawn thread's _agent_active reset.
        """
        return (self._orphan_turn_in_flight()
                or len(self._conditional_subprocesses) > 0)

    def _on_liveness_changed(self):
        """Watcher-thread callback: the conversation's alive verdict flipped
        (an attached Popen exited, an unattached agent was discovered
        alive, or a held lock dropped — the orphan exited). Marshal the
        button/status recompute to the GUI thread.
        """
        wx.CallAfter(self._safe_ui, self._on_liveness_changed_ui)

    def _on_liveness_changed_ui(self):
        """GUI thread: settle buttons to the new aliveness truth.

        _update_button_states forwards the displayed shared panel's
        status/config state, so the label refresh rides along.
        """
        self._update_button_states()

    def _update_button_states(self):
        """Centralized method to update Stop, Resume, and Send button states.

        Should be called (via wx.CallAfter from background threads, or directly
        from UI thread handlers) at every state transition.

        Stop:    Visible if the agent process is alive (attached Popen or an
                 unattached lock-held orphan) and Stop hasn't been pressed,
                 or a conditional run is live (§7.7 — Stop routes to the run).
        Resume:  Visible if the agent process is not alive and there are
                 messages — or a Stop-parked conditional run awaits Resume
                 (§7.7). A non-resumable workflow never shows it (§7.8).
        Send:    Enabled if no agent is active and no §7.7 sequence lock holds
                 the surface; disabled entirely on a non-resumable workflow.
        Input:   The input box mirrors Send exactly — disabled whenever Send
                 is disabled (agent active, §7.7 sequence lock, or
                 non-resumable) and enabled otherwise (§2.2: this method is
                 the sole owner of the pair's enabled state); the grey hint
                 follows the state via _apply_input_enable (enabled invite
                 vs the disabled working/Stop advisory).
        Combos:  The shared Agent/Provider/Model combos follow the config
                 gate (_config_gate_enabled) — Send's idle gate, OR unlocked
                 whenever no process of any kind is running — forwarded only
                 from the displayed tab. §2.2's exact Send mirror is relaxed:
                 a mid-turn process death leaves the turn's io facts
                 non-terminal forever, which must never deadlock the
                 Provider/Model switch that is the recovery for that state.
        """
        if not self._is_window_valid():
            return

        # Aliveness truth (process_lock.py): the attached Popen (this
        # GUI's spawn) or an unattached agent holding the io dir's lock
        # (an orphaned agent from a previous GUI session, absorbed at
        # reopen). Stop/Resume visibility keys on this — NOT the
        # _agent_active busy flag, which now only gates the Send/Input
        # pair and Stop's cancel-vs-idle routing (a turn-protocol
        # concern: one user input per active turn).
        alive = self._is_agent_alive()

        # Stop button: visible while the agent is alive (unless already
        # pressed), or while a conditional run is live (§7.7 — Stop routes
        # to the run).
        show_stop = (alive and not self._stop_pressed
                     or len(self._conditional_subprocesses) > 0)
        self.stop_btn.Show(show_stop)

        # Resume button: visible when the agent is not alive and there
        # are messages — plus a Stop-parked conditional run awaiting
        # Resume (§7.7). A non-resumable workflow never shows Resume
        # (§7.8: _on_resume refuses the click for the same reason, so the
        # rendered-but-refused quirk is gone).
        show_resume = (not alive
                       and bool(self.conversation.messages)
                       and len(self._conditional_subprocesses) == 0)
        show_resume = show_resume or self._stopped_sequence_event_key is not None
        if self.history_controller.non_resumable:
            show_resume = False
        self.resume_btn.Show(show_resume)

        # Send button: enabled when no agent is active and no §7.7 sequence
        # lock (live run, parked orphan, or Stop-parked chain) holds the
        # surface; a non-resumable workflow never enables Send (§7.8).
        # The input box is bound to `send_enabled` itself — disabled
        # whenever Send is disabled — so the lock rule covers both
        # controls from this one owner (§2.2); no other call site may
        # toggle the box's enabled state independently.
        send_enabled = (not self._is_any_agent_active()
                        and not self._has_sequence_lock()
                        and not self.history_controller.non_resumable)
        self.send_btn.Enable(send_enabled)
        self._apply_input_enable(send_enabled)
        # The shared panel's Agent/Provider/Model combos no longer mirror the
        # pair's `send_enabled` exactly (§2.2 relaxed): they follow the config
        # gate — Send's idle gate, OR unlocked whenever no process of any kind
        # is running. A mid-turn process death (e.g. an API error) leaves the
        # turn's io facts non-terminal forever — no one lives to write the
        # terminal record — which used to deadlock the combos on the very
        # Provider/Model switch that is the recovery for that state.
        config_enabled = self._config_gate_enabled(send_enabled, alive)
        # Only the displayed tab reaches the panel: the forward's identity
        # gate drops every other tab's recompute, so background transitions
        # can never reach the panel again (the pre-refactor per-conversation
        # callback's corruption mode is gone).
        self._notify_panel_config_state(config_enabled)

        self.Layout()
        # Header status icon (Option F): the central recompute is the single
        # site that fires the frame's tab-bitmap update on a real change.
        self._notify_status_changed(alive)

    def _apply_input_enable(self, enabled: bool):
        """Set the input box's enabled state and its matching grey hint.

        The hint is part of the box's screen-lock pairing (§2.2): the
        enabled box invites a message; the disabled box explains the lock
        (the agent is working — Stop lets the user interject). Called from
        the central recompute and from the two intentional direct-disable
        windows (_on_shutdown_starting, _on_stop), so the hint always
        matches the enabled state it pairs with.
        """
        self.input_ctrl.Enable(enabled)
        self.input_ctrl.SetHint(
            self._INPUT_HINT_ENABLED if enabled else self._INPUT_HINT_DISABLED)

    def _get_header_text(self) -> str:
        """Get header text for the conversation."""
        timestamp = self.conversation.start_time.strftime("%Y-%m-%d %H:%M:%S")
        return f"{self.conversation.workflow_name}: {self.conversation.conversation_id} | Started: {timestamp}"


    def _is_window_valid(self) -> bool:
        """Check if this tab's window is still valid for UI operations.
        
        Returns False if the tab has been closed or its C++ window object
        has been destroyed (e.g., during app shutdown or tab close).
        """
        if self._closed:
            return False
        try:
            if self.IsBeingDeleted():
                return False
        except RuntimeError:
            return False
        return True

    def _safe_ui(self, func, *args, **kwargs):
        """Call a UI function only if this tab's window is still valid.
        
        Use this for wx.CallAfter targets to guard against operations on
        destroyed windows.
        """
        if self._is_window_valid():
            try:
                func(*args, **kwargs)
            except RuntimeError:
                pass  # Window was destroyed between check and call

    def _reload_streaming(self) -> bool:
        """True while a backlog streams through the watcher (§7.1).

        Per-event tab-level refreshes (button states, position label,
        whole-tab Layout, right-panel notifications) are pure churn during a
        reload: the footer is hidden (§5.10.1) and finalization settles all
        of them once (§7.4 step 9). Marshalling hundreds of them per backlog
        saturates the GUI thread and starves paint events — the flashing and
        "tab not assembled until finalization" symptom.
        """
        return self.history_controller.load_in_progress




    # ------------------------------------------------------------------
    # Reload status mode + finalization (§5.10.1, §7.4)
    # ------------------------------------------------------------------

    def enter_reload_status_mode(self):
        """Suppress the interactive footer during a reload (§5.10.1)."""
        if not self._is_window_valid():
            return
        self.footer_panel.Hide()
        self.position_label.Show()
        self.Layout()

    def exit_reload_status_mode(self):
        """Restore the footer after a reload settles (§7.4 step 9)."""
        if not self._is_window_valid():
            return
        self.footer_panel.Show()
        self.position_label.Hide()
        self.Layout()

    def _on_scan_batch_start(self):
        """Watcher thread callback: open the reload batch buffer.

        During a reload every watcher callback for this tick (on_message,
        on_request_started, ...) appends to the buffer instead of marshalling
        one wx.CallAfter per backlog record — hundreds of individually
        marshalled events per tick flood the GUI event queue and starve paint
        (§8.5). One CallAfter per batch replays the whole tick on the GUI
        thread inside one begin_batch/end_batch bracket (§6), preserving the
        FIFO ordering the marshalled brackets relied on.
        """
        self._reload_batch_buffer = []

    def _on_scan_batch_end(self):
        """Watcher thread callback: marshal the buffered batch to the GUI."""
        buffer = self._reload_batch_buffer
        self._reload_batch_buffer = None
        if buffer is None:
            # Defensive: brackets are paired in the watcher loop; if the open
            # was missed there is nothing buffered to replay.
            return
        # Drain the tick's buffered fact transitions (§8.5): one CallAfter
        # applies them all on the GUI thread instead of one wx.CallAfter per
        # fact (which also fired a right-panel grid rebuild per fact).
        conversation = self.conversation
        facts = None
        if conversation is not None:
            facts = conversation._reload_fact_buffer
            conversation._reload_fact_buffer = []
        # Logging plan (2026-10-10 resume-desync diagnosis): one DEBUG line per
        # reload tick — the buffered backlog counts are now answerable from
        # the log alone.
        if conversation is not None:
            conversation.logger.debug(
                "reload scan batch end: buffered_events=%s buffered_facts=%s",
                len(buffer), len(facts) if facts else 0)
        # Chunk the replay so the GUI thread yields to paints between chunks
        # (§8.5); finalize_reload defers itself until _reload_replay_pending
        # drains.
        if buffer:
            n_chunks = ((len(buffer) + self.RELOAD_REPLAY_CHUNK - 1)
                        // self.RELOAD_REPLAY_CHUNK)
            self._reload_replay_pending += n_chunks
            if facts:
                wx.CallAfter(self._safe_ui, self._apply_reload_facts, facts)
            wx.CallAfter(self._safe_ui, self._replay_reload_batch, buffer, 0)
        elif facts:
            wx.CallAfter(self._safe_ui, self._apply_reload_facts, facts)

    def _replay_reload_batch(self, events, start=0):
        """GUI thread: replay one chunk of a watcher batch in a bracket.

        A tick's batch can hold hundreds of records; replaying all of them
        in one blocking chunk starves the event loop's paints (cards only
        appear when the block happens to end — the flashing). Chunk the
        replay and yield to the event loop between chunks so the view
        paints incrementally. Each chunk runs inside one begin_batch /
        end_batch bracket, so a flush happens at most once per chunk.
        """
        controller = self.history_controller
        controller.begin_batch()
        try:
            for event in events[start:start + self.RELOAD_REPLAY_CHUNK]:
                kind = event[0]
                if kind == 'message':
                    message = event[1]
                    controller.on_message(message)
                    if (message.metadata.get('run_terminal')
                            and message.metadata.get('run_id')):
                        self._set_run_terminal_consumed(
                            message.metadata['run_id'])
                elif kind == 'message_updated':
                    controller.on_message_updated(event[1])
                elif kind == 'turn_complete':
                    controller.on_turn_complete(event[1], event[2])
                elif kind == 'error':
                    controller.on_error(event[1], event[2], event[3])
                elif kind == 'request_started':
                    controller.on_request_started(
                        event[1], event[2], event[3])
                elif kind == 'cancel_ack':
                    controller.on_cancel_ack(event[1], event[2])
        finally:
            controller.end_batch()
            self._reload_replay_pending -= 1
        # Logging plan (2026-10-10 resume-desync diagnosis): per-chunk replay
        # counts (GUI thread) — what the backlog replay actually processed,
        # answerable from the log alone.
        if self.conversation is not None:
            self.conversation.logger.debug(
                "reload replay chunk: replayed=%s start=%s total=%s "
                "chunks_pending=%s",
                min(self.RELOAD_REPLAY_CHUNK, len(events) - start),
                start, len(events), self._reload_replay_pending)
        end = start + self.RELOAD_REPLAY_CHUNK
        if end < len(events):
            wx.CallAfter(self._safe_ui, self._replay_reload_batch, events, end)

    def _apply_reload_facts(self, facts):
        """GUI thread: apply one tick's buffered fact transitions (§8.5).

        During a reload the watcher's fact posts are buffered on the
        conversation (one wx.CallAfter + one right-panel grid rebuild per
        fact flooded the GUI queue and starved paint events); the batch-end
        replay applies the tick's facts here in order. fork_turn_state
        converges exactly as it would per-fact; the settled right-panel
        notification fires once at finalization (§7.4 step 9).
        """
        conversation = self.conversation
        if conversation is None:
            return
        for fact in facts:
            conversation._apply_fact_transition(fact)

    def _on_reload_complete(self):
        """Watcher thread callback: marshal finalization to the GUI thread."""
        wx.CallAfter(self._safe_ui, self.finalize_reload)

    def finalize_reload(self):
        """Finalize a reload (§7.4 steps 2-9).

        Runs on the GUI thread when the watcher reports the backlog is caught
        up. The backlog is already timestamp-sorted (copy-assigned by the
        watcher's finalize-once predicate). Step 5 (measure-all, §7.4) is a
        chunked controller pass that populates every item's exact height before
        `_end_load()` makes the end-of-load scroll choice; the continuation
        (steps 6-9) runs in `_finish_reload_after_measure`. Step 8
        (sequence-state resolution) runs after the header refresh and before
        the footer reveal so the parked-run lock, `suspended` re-surfacing, and
        Stop/Resume routing are settled before any control appears.
        Non-resumable workflows (§7.8) skip step 8 and step 9 is a no-op.
        """
        if not self._is_window_valid():
            return
        # Chunked batch replays may still be pending (the watcher can
        # finalize in the same tick it queued them): defer finalization
        # until every replay chunk has run so the whole backlog is routed
        # before the measure-all and the final flush (§7.4).
        if self._reload_replay_pending > 0:
            wx.CallAfter(self._safe_ui, self.finalize_reload)
            return
        conversation = self.conversation
        messages = conversation.messages

        # Step 2: fill unset agent_type/provider/model from the sorted backlog.
        if not conversation.current_provider or not conversation.current_model:
            for m in reversed(messages):
                md = m.metadata
                if not conversation.current_provider and md.get('provider'):
                    conversation.current_provider = md['provider']
                if not conversation.current_model and md.get('requested_model'):
                    conversation.current_model = md['requested_model']
                if conversation.current_provider and conversation.current_model:
                    break

        # Step 3: pre-population re-check over the fully sorted list (§7.5).
        # The live `_on_message` check is the sole owner; this re-check over
        # the fully-sorted list is the single arbiter at caught-up and
        # pre-populates a pre-close report exactly once. Non-resumable
        # workflows receive no pre-population (§7.8).
        if messages and not self.history_controller.non_resumable:
            last_msg = messages[-1]
            if last_msg.metadata.get('is_conditional_report'):
                self._prefill_conditional_report(
                    last_msg.content,
                    last_msg.metadata.get('conditional_agent_name', ''))

        # Step 3.5: reload fork-visibility convergence (§5.8 re-check, the
        # §12 oracle). The two-phase backlog streams records from EVERY
        # fork, so the item model currently holds the fork-shadowed items
        # the live Send-Fork destroy removed; re-check each item against
        # is_visible(view_fork=latest_fork) and destroy that set from both
        # structures BEFORE the reorder / measure-all / end-of-load flush,
        # so shadowed items are never sorted, measured, or painted - "a
        # reopen after the send rebuilds the identical visible set". Runs
        # on the GUI thread inside the bracket (load_in_progress is still
        # set); the re-check's own flush is a no-op here (the one-shot
        # head paint is done and load flushes do no view work, §8.5).
        self.history_controller.on_fork_turn_update()
        # Reorder the item model to match the watcher's timestamp-sorted
        # backlog before the measure-all and the final flush (§7.4). The
        # watcher's two-phase scan (inputs -> outputs -> conditional) would
        # otherwise cluster roles on reopen.
        self.history_controller.reorder_items_chronologically()

        # Step 4: settle pending cards. With no live agent they render
        # static (no timer runs against a card that will never resolve);
        # an unattached-but-alive orphan (lock held, process_lock.py)
        # keeps the live timer — its response may still land through the
        # watcher's held-lock scan arm.
        live_agent = self._is_agent_alive()
        for item in self.history_controller._items:
            if item.item_id.kind != 'card' or not item.pending:
                continue
            card = self.history_controller._widget(item)
            if card is None:
                continue
            try:
                if live_agent and hasattr(card, 'set_pending'):
                    card.set_pending()
                elif hasattr(card, 'set_static_pending'):
                    card.set_static_pending()
            except RuntimeError:
                pass

        # Logging plan (2026-10-10 resume-desync diagnosis): one summary line at
        # caught-up. A non-empty unanswered-request FIFO here with no live
        # process is exactly the crash+resume state that desynced the
        # 2026-10-10 resumed turn (reported "no tool expanders") — it is
        # visible on one line at reload time now. Diagnostics must never
        # break finalization, hence the guard.
        try:
            _pending_cards = sum(
                1 for it in self.history_controller._items
                if it.item_id.kind == 'card' and it.pending)
            _fifo = {}
            for k, seqs in conversation._pending_request_seqs.items():
                if seqs:
                    _fifo[f"{k.fork}.{k.turn}"] = list(seqs)
            for target in conversation._conditional_scan_targets.values():
                for k, seqs in target.pending_request_seqs.items():
                    if seqs:
                        _fifo[f"{target.agent_name}:{k.fork}.{k.turn}"] = list(seqs)
            conversation.logger.info(
                "reload finalize summary: items=%s pending_cards=%s "
                "unanswered_request_fifos=%s live_agent=%s",
                len(self.history_controller._items), _pending_cards,
                _fifo, live_agent)
        except Exception:
            pass
        # Step 5: chunked measure-all (§7.4). Every item's exact height is
        # populated before `_end_load()` makes the end-of-load scroll choice;
        # the continuation (steps 6-9) runs once the pass completes.
        self.history_controller.measure_all(on_complete=self._finish_reload_after_measure)

    def _finish_reload_after_measure(self):
        """§7.4 steps 6-9, run after the step-5 measure-all completes.

        Step 6 clears the reload bracket and applies the end-of-load scroll
        decision (via the §8.1 force-bottom flag); step 7 refreshes the
        header; step 8 resolves sequence state; step 9 settles the footer and
        position in one same-turn pass.
        """
        if not self._is_window_valid():
            return

        # Step 6: end load (clear flag, apply scroll decision, restore footer).
        self.history_controller._end_load()

        # Step 7: refresh the header.
        self.header.SetLabel(self._get_header_text())

        # Step 8: sequence-state resolution (§7.7). Classify the backlog's
        # tagged summaries through the consumed/ignored/unconsumed rule and
        # fire each unconsumed one into the engine in timestamp order (each
        # fired exactly once, the window is quiescent by construction); then
        # apply the §7.7 restart rule to every sequence state the §7.1 step-0
        # glob loaded. Non-resumable workflows skip this step (§7.8).
        if not self.history_controller.non_resumable:
            self._drain_backlog_summaries()
            self._apply_restart_rule()

        # Step 9: settle the footer and position in one same-turn pass.
        # §5.5 restore: re-arm genuinely live approve/deny buttons for
        # non-terminal permissions that streamed resolved during the reload
        # (load_in_progress is now False, so is_turn_terminal reads the real
        # fork_turns and only genuinely-terminal turns stay deactivated).
        # For a non-resumable conversation this step is a no-op — the
        # permanent label was installed at step 0 and the controls are never
        # restored (§7.8).
        # The reload buffered the fork/turn fact -> right-panel grid
        # notifications (§8.5); fire the settled grid update once here
        # (§7.4 step 9).
        if self.conversation.on_fork_turn_update:
            self.conversation.on_fork_turn_update()
        if self.history_controller.non_resumable:
            return
        # Reload un-wedge self-heal (process_lock.py): a replayed REQUEST
        # can leave _agent_active stuck True when the turn's terminal
        # record was written after the old GUI died and never parsed by
        # this one. With no attached process and nothing alive-by-lock,
        # re-derive from disk: a provably terminal active turn releases
        # the busy flag so Send unlocks without a Resume click (a
        # non-terminal turn keeps the busy gate — Resume as mid-turn
        # recovery is that path, and the aliveness-keyed button rule
        # already shows it).
        if (self._agent_active
                and self.conversation.process is None
                and not self._probe_orphan_alive(force=True)):
            fork, turn = self.conversation.fork_turn_state.active_position
            if self.history_controller.is_turn_terminal(fork, turn):
                self._agent_active = False
        self.history_controller.restore_live_permissions()
        self._update_position()
        self._update_button_states()
        # The reload suppressed per-event right-panel notifications; the
        # settled agent-active state reaches the panel once here (§7.4 step 9).
        self._notify_agent_active_state()
        self.Layout()


    def _on_message(self, message: ConversationMessage):
        """Called when a new message arrives - route through the history controller."""
        self.conversation.logger.debug(
            f"_on_message: role={message.role}, fork={message.fork}, "
            f"turn={message.turn}")
        # Reload batching (§8.5): buffer the routing for the batch replay
        # instead of one wx.CallAfter per backlog record. The replay owns the
        # run-terminal consumption marking; pre-population and the sequence
        # engine are finalize/live-only concerns.
        if self._reload_batch_buffer is not None:
            self._reload_batch_buffer.append(('message', message))
            return
        # The watcher fires these callbacks on its background thread, so all
        # widget-touching controller calls are marshalled to the GUI thread.
        wx.CallAfter(self._safe_ui, self.history_controller.on_message, message)

        # A permission request means the agent is blocked on the user: nudge
        # the header status to 'approval' once the routing above has
        # registered the pending marker (queued after it, Option 2e).
        if message.role == 'permission_request':
            wx.CallAfter(self._safe_ui, self._notify_status_changed)
        # §7.7: mark the run's terminal record consumed on the GUI thread in
        # the same branch that routes a run_terminal=True message for the run
        # (the timer's O(1) deferral gate).
        if message.metadata.get('run_terminal') and message.metadata.get('run_id'):
            wx.CallAfter(self._safe_ui, self._set_run_terminal_consumed,
                         message.metadata['run_id'])

        # §7.5 sole-owner pre-population: a live conditional report writes the
        # input box once at message-scan time (the engine callback is gone).
        # Non-resumable workflows receive no pre-population at all (§7.8).
        if (message.role == 'assistant'
                and message.metadata.get('is_conditional_report')
                and not self.history_controller.non_resumable):
            wx.CallAfter(self._safe_ui, self._prefill_conditional_report,
                         message.content,
                         message.metadata.get('conditional_agent_name', ''))

        # Detect work_completion_summary tag on assistant messages. Live-path
        # guard (§9): never fire the sequence engine from a backlog scan (the
        # flag is set while a reload streams), from a conditional run's
        # summary, or from a non-resumable workflow (§7.8).
        if (message.role == 'assistant'
                and not self.history_controller.load_in_progress
                and not self.history_controller.non_resumable
                and not message.metadata.get('is_conditional', False)):
            tags = message.metadata.get('tags', [])
            if 'work_completion_summary' in tags:
                wx.CallAfter(self._on_work_completion_summary, message)






    def _on_message_updated(self, message: ConversationMessage):
        """Called when an existing message is updated (e.g., tool result arrives)."""
        if self._reload_batch_buffer is not None:
            self._reload_batch_buffer.append(('message_updated', message))
            return
        wx.CallAfter(self._safe_ui, self.history_controller.on_message_updated, message)


    def _on_turn_complete(self, fork: int, turn: int):
        """Called when a turn completes."""
        self._agent_active = False
        self._last_terminal = STATUS_COMPLETED
        self._notify_agent_active_state()
        if not self._reload_streaming():
            wx.CallAfter(self._update_button_states)
        if self._reload_batch_buffer is not None:
            self._reload_batch_buffer.append(('turn_complete', fork, turn))
            return
        wx.CallAfter(self._safe_ui, self.history_controller.on_turn_complete, fork, turn)
        if not self._reload_streaming():
            wx.CallAfter(self._update_position)
            wx.CallAfter(self._safe_ui, self.Layout)
        

    def _handle_permission_response(self, permission_message, approved, explanation):
        """Handle permission response from a RequestCard."""
        if self.history_controller.non_resumable:
            # §7.8: no event reaches permission responses on a non-resumable
            # workflow.
            return
        # Terminal-turns deactivate the live send/approve surface; refuse to
        # send a permission response for a turn that has already completed.
        if self.history_controller.is_turn_terminal(
                permission_message.fork, permission_message.turn):
            return
        self.conversation.send_permission_response(
            permission_message.fork,
            permission_message.turn,
            approved,
            explanation
        )
        # The agent resumes on the same turn, so the turn's terminal events
        # will not clear the pending-permission marker; clear it here so the
        # header icon leaves 'approval' the moment the user answers
        # (Option 2e).
        self.history_controller.mark_permission_answered(
            permission_message.fork, permission_message.turn)
        self._notify_status_changed()
        # Wake the file watcher so the permission response renders on an
        # immediately woken tick (§5.10).
        if self.file_watcher:
            self.file_watcher.wake()
        # Restart the agent process so it can read the permission response
        # and continue. When the process is already running, this is a no-op.
        # When reloading a historical conversation, the process has exited
        # and needs to be restarted to process the response.
        self._ensure_process_running()

    def _on_error(self, fork: int, turn: int, error_message: str):
        """Called when a turn errors (e.g., max requests reached)."""
        self._agent_active = False
        self._last_terminal = STATUS_ERROR
        self._notify_agent_active_state()
        if not self._reload_streaming():
            wx.CallAfter(self._update_button_states)
        if self._reload_batch_buffer is not None:
            self._reload_batch_buffer.append(
                ('error', fork, turn, error_message))
            return
        wx.CallAfter(self._safe_ui, self.history_controller.on_error, fork, turn, error_message)
        if not self._reload_streaming():
            wx.CallAfter(self._safe_ui, self.Layout)


    def _on_fork_turn_update(self):
        """Called when fork/turn data updates."""
        # The position status line is shown only while a reload streams;
        # finalization recomputes it once (§7.4 step 9).
        if self._reload_batch_buffer is not None:
            return
        if not self._reload_streaming():
            # §5.8/§9 (plan line 2440): the controller owns the visibility
            # re-check; forward the notification into it, marshaled like
            # every other watcher-thread arrival (the scan-path fires come
            # from _parse_output_file on the watcher thread; the live
            # send-path fire is already GUI-side and lands after
            # _exit_fork_mode either way, so the sweep is an idempotent
            # belt there). Reload ticks stay silent: the buffered gate
            # above returns first, and finalize_reload's step 3.5 owns the
            # reload convergence.
            wx.CallAfter(self._safe_ui,
                         self.history_controller.on_fork_turn_update)
            wx.CallAfter(self._update_position)
            # Shared grid forward, gated to the displayed tab (thread-safe
            # plain-attribute read) so background tabs' facts stop
            # scheduling no-op panel rebuilds. The re-gate at apply time
            # covers a tab switch between schedule and apply; a switch
            # rebuilds the grid at set_conversation either way.
            if self._panel_displays():
                wx.CallAfter(self._safe_ui, self._forward_panel_grid_refresh)

    def _on_request_started(self, fork: int, turn: int, metadata: dict):
        """Called when an API request starts."""
        self.conversation.logger.debug(f"_on_request_started: fork={fork}, turn={turn}")
        is_conditional = metadata.get('is_conditional', False)
        if not is_conditional:
            self._agent_active = True
            self._stop_pressed = False
            self._last_terminal = None
            self._notify_agent_active_state()
            if not self._reload_streaming():
                wx.CallAfter(self._update_button_states)
        if self._reload_batch_buffer is not None:
            self._reload_batch_buffer.append(
                ('request_started', fork, turn, metadata))
            return
        wx.CallAfter(self._safe_ui, self.history_controller.on_request_started, fork, turn, metadata)
        if not self._reload_streaming():
            wx.CallAfter(self._safe_ui, self.Layout)


    def _on_shutdown_starting(self):
        """Called when shutdown begins - disable the input/Send pair."""
        if not self._is_window_valid():
            return
        # Direct disables, no recompute: shutdown starts before the
        # agent-active flags settle, and the input box must mirror the
        # Send disable through the whole shutdown window.
        self.send_btn.Enable(False)
        self._apply_input_enable(False)
        # The shared panel's combos mirror the pair through the shutdown
        # window too (this tab's display gate applies inside the forward).
        self._notify_panel_config_state(False)
        if self._fork_mode:
            self.cancel_btn.Enable(False)
        self.send_btn.Update()
        self.cancel_btn.Update()

    def _on_shutdown_complete(self):
        """Called when shutdown completes - update button states."""
        if not self._is_window_valid():
            return
        self._agent_active = False
        self._notify_agent_active_state()
        self._update_button_states()
        if self._fork_mode:
            self.cancel_btn.Enable(True)
            self.cancel_btn.Update()


    def _on_cancel_ack(self, fork: int, turn: int):
        """Called when the agent acknowledges the cancel request."""
        self._agent_active = False
        self._last_terminal = None
        self._notify_agent_active_state()
        if not self._reload_streaming():
            wx.CallAfter(self._update_button_states)
        if self._reload_batch_buffer is not None:
            self._reload_batch_buffer.append(('cancel_ack', fork, turn))
            return
        wx.CallAfter(self._safe_ui, self.history_controller.on_cancel_ack, fork, turn)
        if self._reload_streaming():
            # A backlog's historical cancel ack has no live process to wait
            # on (the dead reopen process is already gone), and the terminate
            # spawn / layout would churn the UI for nothing.
            return
        wx.CallAfter(self._safe_ui, self.Layout)

        # The agent writes cancel_turn_ack before it exits. The process may
        # still be alive when this callback fires. Call terminate() in a
        # background thread to wait for the process to actually exit (with
        # escalation if needed), clear the process reference, and trigger
        # on_shutdown_complete which updates button states and status label.
        import threading
        threading.Thread(target=self.conversation.terminate, daemon=True).start()

    def _on_stop(self, event: wx.Event):
        """Handle stop button click - stop the agent process."""
        if self.history_controller.non_resumable:
            # §7.8: no event reaches the engine on a non-resumable workflow.
            return
        if self._conditional_subprocesses:
            # §7.7 park-on-stop: while a conditional run is live, Stop routes
            # to the conditional run — terminate it, touch its
            # <run_dir>/0.0.end tombstone at stop time, mute its cards, and
            # park the chain in memory without completion-processing (a Resume
            # relaunch mints a fresh run dir).
            self._stopped_sequence_event_key = self._conditional_event_key
            self._stop_conditional_runs()
            self._update_button_states()
            return
        self._stop_pressed = True
        was_active = self._agent_active
        self._agent_active = True  # Prevent Send during termination
        self._notify_agent_active_state()

        # Hide Stop button immediately for user feedback, and disable the
        # input/Send pair together so the box mirrors Send through the
        # termination window until the recompute after cancel-ack /
        # shutdown-complete settles the pair centrally.
        self.stop_btn.Hide()
        self.send_btn.Enable(False)
        self._apply_input_enable(False)
        # The shared panel's combos mirror the pair through the termination
        # window too (displayed tab only — the gate lives in the forward).
        self._notify_panel_config_state(False)
        self.Layout()

        if self.conversation.process is not None:
            if was_active:
                # Agent is actively processing - send cancel_turn message.
                # _on_cancel_ack will fire when the agent acknowledges.
                self.conversation.cancel_current_turn()
            else:
                # Agent is idle between turns - terminate process directly.
                # _on_shutdown_complete will fire when the process has exited.
                self.conversation.terminate()
        else:
            # Unattached but alive (Stop is aliveness-keyed, process_lock.py):
            # an orphaned agent from a previous GUI session owns the io dir,
            # and its records stream through the watcher's held-lock arm. No
            # Popen exists, so there is no escalation ladder — the graceful
            # input-stream routes are the Stop semantics for an orphan (an
            # expensive in-flight API call is never killed on the way out),
            # and the ack / lock-drop settles the buttons through the
            # liveness-transition callback.
            if self._orphan_turn_in_flight():
                # Mid-turn: write cancel_turn — the agent acks when the
                # in-flight call returns and exits. A stale second cancel
                # is drained by mid-turn recovery on the next start.
                self.conversation.cancel_current_turn()
            else:
                # Idle at a turn boundary: write the .end at the agent's
                # poll position so it exits quietly within one ~50ms poll
                # (clear_end_signal removes the leftover on next launch).
                self.conversation.end_conversation()

    def _on_resume(self, event: wx.Event):
        """Handle resume button click - restart the agent process."""
        if self.history_controller.non_resumable:
            # §7.8: no event reaches the engine on a non-resumable workflow.
            return
        # Logging plan (2026-10-10 resume-desync diagnosis): Resume was
        # completely unlogged while the resumed turn's pairing desync streamed
        # ~2 cards per request cycle — the press and the branch it takes must
        # be answerable from the conversation log.
        if self._stopped_sequence_event_key is not None:
            # §7.7 park-on-stop: Resume relaunches the Stop-parked conditional
            # run's current step with a fresh run_id/run dir.
            self.conversation.logger.info(
                "Resume pressed: relaunching Stop-parked conditional "
                "sequence (event_key=%s)", self._stopped_sequence_event_key)
            self._resume_stopped_conditional()
            return
        if self._probe_orphan_alive(force=True):
            # Action-time re-probe (process_lock.py): an unattached agent
            # came to own the io dir since the button rendered (the
            # liveness transition may not have re-rendered yet). Nothing
            # to start — the watcher's held-lock arm is already streaming
            # its records; settle the controls to the live truth. A second
            # process on one io dir would race inputs and duplicate
            # expensive turns (the agent-side lock would fail it too).
            self.conversation.logger.info(
                "Resume pressed: orphaned agent alive by lock — settling "
                "controls to the live agent (no relaunch)")
            self._update_button_states()
            return
        # Restart the agent process. _ensure_process_running() handles dead
        # Popen cleanup, end signal clearing, and process startup.
        self.conversation.logger.info(
            "Resume pressed: relaunching agent process (conversation_id=%s "
            "io_dir=%s)", self.conversation.conversation_id,
            self.conversation.io_dir)
        if not self._ensure_process_running():
            self.conversation.logger.warning(
                "Resume failed: could not start the agent process")
            wx.MessageBox(
                "Failed to resume workflow. Please try again.",
                "Resume Error",
                wx.OK | wx.ICON_ERROR
            )
            return
        # Mark agent as active, reset stop state for new process
        self._agent_active = True
        self._stop_pressed = False
        self._update_button_states()
        self.Layout()


    def _update_position(self):
        """Update the position indicator."""
        if not self._is_window_valid():
            return
        text = f"Viewing Fork {self.conversation.fork_turn_state.latest_fork}"
        if self._fork_mode:
            text += " (Fork Mode)"
        self.position_label.SetLabel(text)

    def _on_input_key(self, event: wx.KeyEvent):
        """Handle key events in input control."""
        if event.GetKeyCode() == wx.WXK_RETURN and (event.ControlDown() or event.ShiftDown()):
            self._on_send(event)
        else:
            event.Skip()

    def _on_send(self, event: wx.Event):
        """Handle send button click."""
        if self.history_controller.non_resumable or self._has_sequence_lock():
            # §7.8: no event reaches the send path on a non-resumable workflow;
            # §7.7: the input/Send lock covers live runs, parked orphans, and
            # Stop-parked chains.
            return
        if self._is_any_agent_active():
            return
        content = self.input_ctrl.GetValue().strip()
        if not content:
            return
        
        # Ensure process is running before sending
        if not self._ensure_process_running():
            wx.MessageBox(
                "Failed to resume workflow. Please try again.",
                "Resume Error",
                wx.OK | wx.ICON_ERROR
            )
            return
        
        if self._fork_mode and self._fork_message:
            # Sending in fork mode - creates a new fork
            new_fork = self.conversation.fork_turn_state.latest_fork + 1
            self.conversation.logger.info(
                f"_on_send: FORK MODE - creating new fork {new_fork} "
                f"from fork={self._fork_message.fork}, turn={self._fork_message.turn}")

            # Send message to the NEW fork - turn numbering continues from fork point
            fork, turn = self.conversation.send_message(
                content,
                fork=new_fork,
                turn=self._fork_message.turn
            )

            # Exit fork mode
            self._exit_fork_mode()
            self.input_ctrl.SetValue("")
        else:
            # Normal send - use latest fork and its next turn
            view_fork = self.conversation.fork_turn_state.latest_fork
            next_turn = self.conversation.fork_turn_state.next_turn(view_fork)
            self.conversation.logger.info(
                f"_on_send: NORMAL send on fork {view_fork}, turn {next_turn}")
            fork, turn = self.conversation.send_message(
                content, fork=view_fork, turn=next_turn)
            self.input_ctrl.SetValue("")
        
        # Wake the file watcher so the newly written input record renders on an
        # immediately woken tick (§5.10) instead of waiting up to a full tick.
        if self.file_watcher:
            self.file_watcher.wake()

        # Update position
        self._update_position()

        # Mark agent as active, reset stop state for new process
        self._agent_active = True
        self._stop_pressed = False
        self._update_button_states()

    def _on_cancel(self, event: wx.Event):
        """Handle cancel button click - exit fork mode."""
        self._exit_fork_mode(cancelled=True)

    # ---------------------------------------------------------------
    # Conditional agent sequence methods
    # ---------------------------------------------------------------

    def _on_work_completion_summary(self, message: ConversationMessage):
        """Handle a work_completion_summary event from the primary agent."""
        if self.history_controller.non_resumable:
            # §7.8: no event reaches the engine on a non-resumable workflow.
            return
        if self._config_mgr is None:
            return

        workflow_name = self.conversation.workflow_name
        event_key = "work_completion_summary"
        agent_names = self._config_mgr.get_event_sequence_for_agent_type(
            workflow_name, event_key, self.conversation.agent_type)
        if not agent_names:
            return

        # Resolve agent names to ConditionalAgentConfigInfo objects
        agent_list = []
        for name in agent_names:
            config = self._config_mgr.get_conditional_agent(name)
            if config is None:
                self.conversation.logger.warning(
                    f"Conditional agent '{name}' not found, skipping")
                continue
            agent_list.append(config)

        if not agent_list:
            return

        conversation_messages = build_conversation_context_messages(
            self.conversation.messages,
            self.conversation.fork_turn_state)

        # Store event key and agent list for later use
        self._conditional_event_key = event_key
        self._conditional_agent_list = agent_list

        callbacks = self._make_sequence_callbacks()
        on_primary_agent_work_completion_summary(
            self.conversation.conversation_dir,
            self.conversation.working_dir,
            event_key,
            agent_list,
            conversation_messages,
            provider=self.conversation.current_provider,
            model=self.conversation.current_model,
            callbacks=callbacks,
        )

    def _make_sequence_callbacks(self) -> SequenceCallbacks:
        """Create SequenceCallbacks bound to this ConversationTab.

        §7.5: the engine no longer owns the input box — the ``_on_message``
        message path is the sole writer — so no ``pre_populate_user_input``
        field exists here.
        """
        return SequenceCallbacks(
            show_resume_button=self._on_show_resume_button,
            on_sequence_completed=self._on_sequence_completed,
            subprocess_launched=self._on_subprocess_launched,
        )

    def _on_subprocess_launched(self, proc: subprocess.Popen, agent_name: str, io_dir: Path, run_id: str):
        """Callback: a conditional agent subprocess was launched (§7.7).

        Keyed by the minted ``run_id`` — a relaunch's fresh run dir never
        collides with a stopped/closed predecessor's. run→name is resolved at
        the completion call site; the engine's ``agent_outputs[name]``
        semantics unchanged.
        """
        self._conditional_subprocesses[run_id] = proc
        self._conditional_io_dirs[run_id] = io_dir
        self._conditional_run_names[run_id] = agent_name
        self._run_terminal_consumed[run_id] = False
        self._deferred_ticks[run_id] = 0

        # Start the polling timer if not already running
        if not self._conditional_timer.IsRunning():
            self._conditional_timer.Start(500)

        # Lock the interactive surface while a conditional run is live: the
        # central recompute derives both Send and the input box from the
        # tracked-run dicts, so the pair can never diverge (§2.2). The
        # callback always runs on the GUI thread (engine calls arrive via
        # button handlers or wx.CallAfter-marshalled message paths), so
        # the synchronous recompute also closes the sub-tick window a
        # posted recompute would leave.
        self._update_button_states()

        # Wake the file watcher so the conditional run's writes render on an
        # immediately woken tick (§5.10).
        if self.file_watcher:
            self.file_watcher.wake()

    def _prefill_conditional_report(self, text: str, agent_name: str):
        """§7.5 sole-owner pre-population of the input box.

        Called from the live ``_on_message`` check (message-scan time) and from
        the §7.4 finalization re-check over the fully-sorted list. The engine's
        ``pre_populate_user_input`` callback is gone — this is the only writer,
        forever.
        """
        if not self._is_window_valid():
            return
        # An empty (or whitespace-only) report is "nothing to report" — the
        # ``[From: ...]`` header labels a report body and must never stand
        # alone (mirrors the engine's ``if output:`` header guard in
        # _get_prior_agent_message and the ``if not output.strip():``
        # advance rule in on_conditional_agent_complete). Both call sites
        # (the live _on_message check and the §7.4 step-3 re-check) funnel
        # through this single writer, so one gate covers live and reopen.
        if not (text or "").strip():
            return
        self.input_ctrl.SetValue(f"[From: {agent_name}]\n{text}")
        self.input_ctrl.SetFocus()

    def _on_show_resume_button(self, event_key: str):
        """Callback: show the Resume Sequence button."""
        if not self._is_window_valid():
            return
        # Get current agent name and retrigger count from sequence state
        state = load_state(self.conversation.conversation_dir, event_key)
        if state and self._conditional_agent_list:
            idx = min(state.current_step_index, len(self._conditional_agent_list) - 1)
            agent_name = self._conditional_agent_list[idx].name
            count = state.retrigger_counts.get(agent_name, 0)
            self.resume_sequence_btn.SetLabel(
                f"Resume: {agent_name} ({count} cycles)")
        self.resume_sequence_btn.Show()
        self.Layout()

    def _on_sequence_completed(self, event_key: str):
        """Callback: the conditional agent sequence has completed."""
        if not self._is_window_valid():
            return
        self.resume_sequence_btn.Hide()
        if self._conditional_timer.IsRunning():
            self._conditional_timer.Stop()
        self._conditional_subprocesses.clear()
        self._conditional_io_dirs.clear()
        self._conditional_run_names.clear()
        self._run_terminal_consumed.clear()
        self._deferred_ticks.clear()
        self._stopped_sequence_event_key = None
        self._conditional_event_key = None
        self._conditional_agent_list = []
        # Unlock the interactive surface once the conditional run is done —
        # the central recompute owns the pair, so the box follows Send
        # here instead of enabling independently (§2.2).
        self.Layout()
        self._update_button_states()

    def _on_conditional_timer(self, event: wx.TimerEvent):
        """Poll conditional agent subprocesses for completion (§7.7).

        Keyed by ``run_id``. A ``poll()`` exit observed before the run's
        ``run_terminal=True`` record has streamed defers (never completes
        empty); the deferred-crash escape — N consecutive deferred ticks
        (20 ≈ 10 s at 500 ms) with ``poll()`` exited and the run's out-stream
        at EOF — applies the §7.4 step-8 offline error-marker transition
        directly (no launch, cards muted via ``pending_muted``, input/Send
        released, run removed from the dicts so the timer stops).
        """
        completed_runs = []
        escaped_runs = []
        for run_id, proc in list(self._conditional_subprocesses.items()):
            if proc.poll() is None:
                continue
            if not self._run_terminal_consumed.get(run_id):
                # Terminal record not yet streamed. Defer; bounded crash escape.
                self._deferred_ticks[run_id] = self._deferred_ticks.get(run_id, 0) + 1
                io_dir = self._conditional_io_dirs.get(run_id)
                out_file = io_dir / "0.0.out.txt" if io_dir is not None else None
                if (self._deferred_ticks[run_id] >= 20
                        and out_file is not None
                        and self.conversation.output_stream_at_eof(out_file)):
                    escaped_runs.append(run_id)
                continue
            output = self._derive_run_report(run_id)
            completed_runs.append((run_id, output))

        # Deferred-crash escape: markerless live death. Apply the §7.4 step-8
        # offline error-marker transition directly (advance once, no bump, no
        # launch), mute the run's cards, release the lock, and drop the run
        # from the dicts so the timer stops.
        for run_id in escaped_runs:
            agent_name = self._conditional_run_names.get(run_id, "")
            try:
                apply_offline_terminal_transition(
                    self.conversation.conversation_dir,
                    self._conditional_event_key,
                    agent_name,
                    "",
                    self._conditional_agent_list,
                )
            except Exception as e:
                self.conversation.logger.error(
                    f"offline error transition failed for run {run_id}: {e}")
            if agent_name:
                self.history_controller.mute_pending_for_conditional(agent_name)
            self._drop_conditional_run(run_id)

        # Process completions (terminal record already streamed). run→name is
        # resolved here; the engine's agent_outputs[name] semantics unchanged.
        for run_id, output in completed_runs:
            agent_name = self._conditional_run_names.get(run_id, "")
            self._drop_conditional_run(run_id)

            # Finalize any pending conditional cards for this agent.  The
            # controller mutes (pending_muted) the run's still-pending card so
            # its pending timer stops and the card is marked resolved.
            if agent_name:
                self.history_controller.mute_pending_for_conditional(agent_name)

            # Notify the sequence engine
            callbacks = self._make_sequence_callbacks()
            conversation_messages = build_conversation_context_messages(
                self.conversation.messages,
                self.conversation.fork_turn_state)
            on_conditional_agent_complete(
                self.conversation.conversation_dir,
                self.conversation.working_dir,
                self._conditional_event_key,
                agent_name,
                output,
                self._conditional_agent_list,
                conversation_messages,
                provider=self.conversation.current_provider,
                model=self.conversation.current_model,
                callbacks=callbacks,
            )

        # Stop timer if no more subprocesses
        if not self._conditional_subprocesses:
            self._conditional_timer.Stop()
            # Safety: settle the pair if the sequence engine did not run
            # _on_sequence_completed (abnormal exit path). The central
            # recompute owns the input/Send pair, so the box unlocks
            # exactly when Send unlocks; a subsequent launch re-locks both
            # via _on_subprocess_launched. (The Stop path stops the timer
            # inside _drop_conditional_run, so a Stop-parked lock never
            # reaches this settle.)
            self._update_button_states()

        # Update button states if any agents completed or escaped
        if completed_runs or escaped_runs:
            wx.CallAfter(self._update_button_states)

    def _on_resume_sequence(self, event: wx.Event):
        """Handle Resume Sequence button press."""
        if self.history_controller.non_resumable:
            # §7.8: no event reaches the engine on a non-resumable workflow.
            return
        if not self._conditional_event_key:
            return

        # Logging plan (2026-10-10 resume-desync diagnosis): the Resume Sequence
        # press was previously completely unlogged.
        self.conversation.logger.info(
            "Resume Sequence pressed: re-firing event_key=%s provider=%s "
            "model=%s", self._conditional_event_key,
            self.conversation.current_provider, self.conversation.current_model)
        conversation_messages = build_conversation_context_messages(
            self.conversation.messages,
            self.conversation.fork_turn_state)

        on_resume_button_pressed(
            self.conversation.conversation_dir,
            self.conversation.working_dir,
            self._conditional_event_key,
            self._conditional_agent_list,
            conversation_messages,
            provider=self.conversation.current_provider,
            model=self.conversation.current_model,
            callbacks=self._make_sequence_callbacks(),
        )
        self.resume_sequence_btn.Hide()
        self.Layout()


    def _enter_fork_mode(self, message: ConversationMessage):
        """Enter fork mode, previewing the new fork's item set."""
        if self.history_controller.non_resumable:
            # §7.8: forking is withheld on a non-resumable workflow — no
            # event reaches the fork path.
            return
        self._fork_mode = True
        self._fork_message = message

        # Preview the new fork's item set through the controller's item model
        # (is_visible_with_extra_fork is a pure function - no mutation of
        # fork_turn_state).  The controller hides items not visible in the
        # prospective fork and restores them on exit.
        self.history_controller.enter_fork_preview(message.turn)

        self.conversation.logger.info(
            f"_enter_fork_mode: previewing fork "
            f"{self.conversation.fork_turn_state.latest_fork + 1} "
            f"from fork={message.fork}, turn={message.turn}")

        # Pre-fill input with fork message content
        self.input_ctrl.SetValue(message.content)
        self.input_ctrl.SetFocus()

        # Update buttons
        self.send_btn.SetLabel("Send Fork")
        self.cancel_btn.Show()
        self.Layout()

    def _exit_fork_mode(self, cancelled: bool = False):
        """Exit fork mode, restoring the item set through the controller."""
        self.conversation.logger.info(
            f"_exit_fork_mode: cancelled={cancelled}, view_fork="
            f"{self.conversation.fork_turn_state.latest_fork}")
        if cancelled:
            self.input_ctrl.SetValue("")
        self.history_controller.exit_fork_preview(cancelled)
        self._fork_mode = False
        self._fork_message = None

        # Restore buttons
        self.send_btn.SetLabel("Send")
        self.cancel_btn.Hide()

        self.Layout()




    def _touch_end_tombstone(self, run_dir: Path):
        """Touch the run's <run_dir>/0.0.end tombstone (§7.7).

        GUI-thread write recording killed-at-close evidence for a tracked run
        (close hook and Stop). A plain file touch — no watcher participation.
        """
        try:
            (run_dir / "0.0.end").touch(exist_ok=True)
        except OSError:
            self.conversation.logger.warning(
                f"failed to touch END tombstone in {run_dir}")

    def _install_non_resumable_label(self):
        """Install the permanent non-resumable label (§7.8).

        Renders through the Phase 2 suppression frame: the footer stays
        hidden, the status line shows a read-only message, and no control is
        ever restored.
        """
        if not self._is_window_valid():
            return
        self.footer_panel.Hide()
        self.position_label.SetLabel(
            "Read-only: sequence state could not be loaded")
        self.position_label.Show()
        self.Layout()
        self._notify_status_changed()

    def _has_sequence_lock(self) -> bool:
        """True when the input/Send surface must stay locked.

        A live conditional run holds the lock via ``_conditional_subprocesses``;
        an untracked orphan run parked by the §7.7 restart rule holds it via a
        loaded ``running`` state; a Stop-parked run holds it until Resume.
        """
        return (len(self._conditional_subprocesses) > 0
                or self.history_controller.has_running_loaded_state()
                or self._stopped_sequence_event_key is not None)

    def _derive_run_report(self, run_id: str) -> str:
        """Derive a run's completion report from its own streamed messages (§7.7).

        The run's last ``CONDITIONAL_REPORT`` (its content, possibly empty),
        falling back to its last ``MODEL_MESSAGE_TO_USER``, else empty. The
        watcher is the sole reader of the run's files; completion never
        re-parses the closed files.
        """
        for m in reversed(self.conversation.messages):
            md = m.metadata
            if md.get('run_id') != run_id:
                continue
            if md.get('is_conditional_report'):
                return m.content or ""
            if (m.role == 'assistant'
                    and md.get('is_conditional', False)
                    and not md.get('is_api_response', False)
                    and not md.get('is_conditional_report', False)):
                return m.content or ""
        return ""

    def _drop_conditional_run(self, run_id: str):
        """Remove a run from the tracked dicts; stop the timer when empty (§7.7)."""
        self._conditional_subprocesses.pop(run_id, None)
        self._conditional_io_dirs.pop(run_id, None)
        self._conditional_run_names.pop(run_id, None)
        self._run_terminal_consumed.pop(run_id, None)
        self._deferred_ticks.pop(run_id, None)
        if not self._conditional_subprocesses and self._conditional_timer.IsRunning():
            self._conditional_timer.Stop()

    def _set_run_terminal_consumed(self, run_id: str):
        """GUI-thread marker: the run's terminal record was routed (§7.7)."""
        if run_id:
            self._run_terminal_consumed[run_id] = True

    def _stop_conditional_runs(self):
        """Park every live conditional run without completion-processing (§7.7).

        Terminates each subprocess, touches the run's END tombstone at stop
        time, mutes the run's pending cards, and drops the run from the dicts
        so the timer never routes it into completion. The persisted state
        stays ``running`` — Resume relaunches it with a fresh run dir.
        """
        for run_id, proc in list(self._conditional_subprocesses.items()):
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            io_dir = self._conditional_io_dirs.get(run_id)
            if io_dir is not None:
                try:
                    self._touch_end_tombstone(io_dir)
                except Exception:
                    pass
            agent_name = self._conditional_run_names.get(run_id, "")
            if agent_name:
                self.history_controller.mute_pending_for_conditional(agent_name)
            self._drop_conditional_run(run_id)

    def _resume_stopped_conditional(self):
        """Relaunch a Stop-parked sequence (§7.7, park-on-stop).

        Re-resolves ``_conditional_agent_list`` if needed, builds the context
        messages, and drives ``on_resume_button_pressed`` (broadened to accept
        a ``running`` state) so the current step relaunches with a fresh
        run_id/run dir.
        """
        event_key = self._stopped_sequence_event_key
        if event_key is None:
            return
        # Logging plan (2026-10-10 resume-desync diagnosis): this resume branch
        # relaunches a Stop-parked conditional step — the action must be
        # answerable from the conversation log.
        self.conversation.logger.info(
            "Resume (stopped conditional): relaunching Stop-parked sequence "
            "step event_key=%s", event_key)
        if self._config_mgr is not None and not self._conditional_agent_list:
            workflow_name = self.conversation.workflow_name
            agent_names = self._config_mgr.get_event_sequence_for_agent_type(
                workflow_name, event_key, self.conversation.agent_type)
            agent_list = []
            for name in agent_names or []:
                config = self._config_mgr.get_conditional_agent(name)
                if config is not None:
                    agent_list.append(config)
            self._conditional_agent_list = agent_list
        if not self._conditional_agent_list:
            self._stopped_sequence_event_key = None
            return
        self._conditional_event_key = event_key
        conversation_messages = build_conversation_context_messages(
            self.conversation.messages,
            self.conversation.fork_turn_state)
        on_resume_button_pressed(
            self.conversation.conversation_dir,
            self.conversation.working_dir,
            event_key,
            self._conditional_agent_list,
            conversation_messages,
            provider=self.conversation.current_provider,
            model=self.conversation.current_model,
            callbacks=self._make_sequence_callbacks(),
        )
        self._stopped_sequence_event_key = None
        self._update_button_states()

    def _drain_backlog_summaries(self):
        """§7.4 step 8: fire each unconsumed backlog summary exactly once (§7.7).

        Classifies the fully-sorted backlog's tagged ``work_completion_summary``
        assistant messages through the consumed/ignored/unconsumed rule
        (consumed ⇔ a run dir launched for it — GUI-written ``0.0.in.txt``
        launch ts >= summary ts, tie ⇒ consumed; ignored ⇔ no such launch but a
        then-live run's [launch ts, terminal ts] window covered it; unconsumed
        ⇔ neither) and fires each unconsumed one into the engine in timestamp
        order. The window is quiescent by construction (§7.7), so nothing is
        queued or deferred.
        """
        summaries = [
            m for m in self.conversation.messages
            if (m.role == 'assistant'
                and not m.metadata.get('is_conditional', False)
                and 'work_completion_summary' in m.metadata.get('tags', []))
        ]
        if not summaries:
            return
        # Run evidence: per non-empty run_id, the launch ts (from the scan
        # target, GUI-written at the run's first positioned input read) and the
        # run's terminal ts (from its run_terminal=True message if any).
        launch_by_run = {}
        terminal_by_run = {}
        for target in self.conversation._conditional_scan_targets.values():
            if target.run_id and target.launch_ts is not None:
                launch_by_run[target.run_id] = target.launch_ts
        for m in self.conversation.messages:
            md = m.metadata
            if md.get('run_id') and md.get('run_terminal'):
                terminal_by_run[md['run_id']] = m.timestamp
        unconsumed = []
        for m in summaries:
            ts = m.timestamp
            consumed = any(launch >= ts for launch in launch_by_run.values())
            if consumed:
                continue
            ignored = any(
                launch <= ts and (term is None or ts <= term)
                for launch, term in (
                    (launch_by_run.get(run_id), terminal_by_run.get(run_id))
                    for run_id in launch_by_run
                )
            )
            if ignored:
                continue
            unconsumed.append(m)
        for m in sorted(unconsumed, key=lambda x: x.timestamp):
            self._on_work_completion_summary(m)

    def _apply_restart_rule(self):
        """§7.4 step 8: apply the §7.7 restart rule to every loaded state.

        Per ``state.event_key`` from the §7.1 step-0 glob: ``running`` parks
        (resolved by stream terminal markers — the §7.7 offline transition,
        never launching — or a ``<run_dir>/0.0.end`` tombstone naming its run
        dir, which relabels ``waiting_for_user``; each disk-side resolution
        also mirrors its post-resolution status into the in-memory step-0
        entries so the lock releases exactly when the disk resolves — a
        ``suspended`` landing additionally re-arms Resume Sequence; the
        crash/no-marker case stays parked with the lock engaged);
        ``suspended`` re-shows the Resume Sequence button re-armed from the
        loaded state's own event_key; ``idle``/``completed`` take no work.
        """
        states = self.history_controller._loaded_sequence_states
        for event_key, state in states.items():
            status = getattr(state, 'status', None)
            if status == "suspended":
                self._rearm_suspended_state(event_key)
            elif status == "running":
                self._park_running_state(event_key, state)
            # idle / completed: no step-8 work.

    def _rearm_suspended_state(self, event_key: str):
        """Re-arm the Resume Sequence button from a loaded suspended state (§7.7)."""
        if not self._is_window_valid():
            return
        if self._config_mgr is None:
            return
        workflow_name = self.conversation.workflow_name
        agent_names = self._config_mgr.get_event_sequence_for_agent_type(
            workflow_name, event_key, self.conversation.agent_type)
        if not agent_names:
            return
        agent_list = []
        for name in agent_names:
            config = self._config_mgr.get_conditional_agent(name)
            if config is not None:
                agent_list.append(config)
        if not agent_list:
            return
        self._conditional_event_key = event_key
        self._conditional_agent_list = agent_list
        state = load_state(self.conversation.conversation_dir, event_key)
        if state and self._conditional_agent_list:
            idx = min(state.current_step_index, len(self._conditional_agent_list) - 1)
            agent_name = self._conditional_agent_list[idx].name
            count = state.retrigger_counts.get(agent_name, 0)
            self.resume_sequence_btn.SetLabel(
                f"Resume: {agent_name} ({count} cycles)")
        self.resume_sequence_btn.Show()
        self.Layout()

    def _mirror_loaded_state_resolution(self, event_key: str):
        """Mirror a §7.7 disk-side park resolution into the in-memory entries.

        Branches 1/2 of ``_park_running_state`` resolve a parked ``running``
        chain on disk only: each loads a fresh state, relabels/transitions
        it, and saves — the in-memory §7.1 step-0 entry read by
        ``has_running_loaded_state()`` still says ``running`` and would hold
        the input/Send lock for the rest of the session. Re-read the
        resolved state here and replace the entry so the lock releases
        exactly when the disk says so; a ``suspended`` landing also re-arms
        Resume Sequence (the restart-rule pass dispatched this entry on the
        stale ``running`` status, so its suspended branch never ran).
        """
        resolved = load_state(self.conversation.conversation_dir, event_key)
        if resolved is None:
            # Unreadable state file: the entry stays ``running`` — kept
            # parked, matching the sibling silent-exception handling of
            # branch 2.
            return
        if getattr(resolved, "status", None) == "running":
            # The disk transition failed (logged by branch 1): stay parked
            # by design with the lock engaged.
            return
        self.history_controller.resolve_loaded_sequence_state(event_key, resolved)
        if getattr(resolved, "status", None) == "suspended":
            self._rearm_suspended_state(event_key)

    def _park_running_state(self, event_key: str, state):
        """§7.7 restart rule for a persisted ``running`` state (never launching).

        The offline transition needs the real ``agent_list`` (for
        ``max_retriggers`` and the current agent); on a reopen no live trigger
        has run, so re-resolve it through the same config lookup the live
        trigger performs.
        """
        if not self._conditional_agent_list and self._config_mgr is not None:
            workflow_name = self.conversation.workflow_name
            agent_names = self._config_mgr.get_event_sequence_for_agent_type(
                workflow_name, event_key, self.conversation.agent_type)
            agent_list = []
            for name in agent_names or []:
                config = self._config_mgr.get_conditional_agent(name)
                if config is not None:
                    agent_list.append(config)
            self._conditional_agent_list = agent_list
            self._conditional_event_key = event_key
        run_id = getattr(state, 'active_run_id', '') or ''
        run_dir = None
        if run_id:
            io_dir_str = getattr(state, 'active_io_dir', None)
            if io_dir_str:
                run_dir = Path(io_dir_str)
            else:
                # Fall back to io-tree discovery: run dirs are minted as
                # conditional_{agent}_{run_id} (conditional_sequence.py's
                # _launch_agent_subprocess), so glob on the run_id suffix —
                # the plain conditional_{run_id} name never matches.
                matches = sorted(
                    self.conversation.io_dir.glob(f"conditional_*_{run_id}"))
                if matches and matches[0].is_dir():
                    run_dir = matches[0]
        # 1. Stream terminal marker in the run's own scan (content wins).
        if run_id:
            terminal = None
            for m in self.conversation.messages:
                if m.metadata.get('run_id') == run_id and m.metadata.get('run_terminal'):
                    terminal = m
                    break
            if terminal is not None:
                agent_name = self._conditional_run_names.get(run_id, "")
                if not agent_name:
                    agent_name = self._agent_name_for_run_id(run_id)
                report = self._derive_run_report(run_id)
                try:
                    apply_offline_terminal_transition(
                        self.conversation.conversation_dir,
                        event_key,
                        agent_name,
                        report,
                        self._conditional_agent_list,
                    )
                except Exception as e:
                    self.conversation.logger.error(
                        f"offline terminal transition failed for {event_key}: {e}")
                # The transition is disk-only: the in-memory §7.1 step-0
                # entry still reads ``running``. Mirror the resolved status
                # so has_running_loaded_state() releases the input/Send lock
                # exactly when the disk resolves (a ``suspended`` landing
                # also re-arms Resume Sequence; a still-``running`` disk
                # keeps the park — see the helper's internal guards).
                self._mirror_loaded_state_resolution(event_key)
                return
        # 2. Tombstone naming the run dir → killed-at-close relabel.
        if run_dir is not None and (run_dir / "0.0.end").exists():
            try:
                loaded = load_state(self.conversation.conversation_dir, event_key)
                if loaded is not None:
                    loaded.status = "waiting_for_user"
                    save_state(self.conversation.conversation_dir, loaded)
                    # The relabel is disk-only: mirror the resolved state
                    # into the in-memory §7.1 step-0 entry so the lock
                    # releases (``loaded`` is already the post-resolution
                    # disk state — no fresh re-read needed, unlike
                    # branch 1).
                    self.history_controller.resolve_loaded_sequence_state(
                        event_key, loaded)
            except Exception:
                pass
            return
        # 3. No terminal marker, no tombstone: probe the run dir's agent
        # lock (process_lock.py) before parking the chain.
        if run_dir is not None:
            verdict = _probe_agent_lock(run_dir, 'agent')
            if verdict == 'held':
                # Still alive: the run's agent survived the GUI exit and
                # owns its run dir (a fresh launch here would also fail the
                # agent-side single-owner lock). Keep the park exactly as
                # the pre-lock rule did — the watcher's held-lock arm
                # streams the run's records, and the terminal marker
                # resolves the chain on a later reopen (branch 1).
                self.conversation.logger.info(
                    f"conditional run {run_id} for {event_key} is still "
                    f"alive (run-dir lock held); keeping parked lock")
                return
            if verdict == 'free':
                # Dead with the GUI (crash or kill): the run will never
                # write again. Resolve like the killed-at-close relabel
                # (branch 2) so the §7.7 input/Send lock cannot hold
                # forever on a process that no longer exists.
                try:
                    loaded = load_state(
                        self.conversation.conversation_dir, event_key)
                    if loaded is not None:
                        loaded.status = "waiting_for_user"
                        save_state(self.conversation.conversation_dir, loaded)
                        self.history_controller.resolve_loaded_sequence_state(
                            event_key, loaded)
                except Exception as e:
                    self.conversation.logger.error(
                        f"liveness relabel failed for {event_key}: {e}")
                return
        # 'unknown' (probe failure) or an unresolvable run dir: stay
        # parked with the lock engaged (conservative — the §7.7 restart
        # rule keeps the input/Send lock while ANY loaded chain is
        # running); a later reopen re-probes.
        return

    def _agent_name_for_run_id(self, run_id: str) -> str:
        """Recover the agent name for a run id from the scan targets (§7.7)."""
        for target in self.conversation._conditional_scan_targets.values():
            if target.run_id == run_id:
                return target.agent_name
        return ""

    def get_title(self) -> str:
        """Get the tab title."""
        return self.conversation.start_time.strftime("%H:%M:%S")

    def close(self):
        """Close the conversation tab."""
        self._closed = True

        # Nil the reload streaming callbacks so the watcher never finalizes
        # (or brackets) a closing conversation.
        self.conversation.on_scan_batch_start = None
        self.conversation.on_scan_batch_end = None
        self.conversation.on_reload_complete = None
        if self.file_watcher is not None:
            try:
                self.file_watcher.finish_reload(self.conversation.conversation_id)
            except Exception:
                pass

        # Clean up conditional agent resources
        if hasattr(self, '_conditional_timer') and self._conditional_timer.IsRunning():
            self._conditional_timer.Stop()
        if hasattr(self, '_conditional_subprocesses'):
            # §7.7: keyed by run_id; each still-tracked run is terminated and
            # its <run_dir>/0.0.end tombstone touched after terminate() so a
            # reopen resolves the killed-at-close run (killed-at-close relabel).
            for run_id, proc in list(self._conditional_subprocesses.items()):
                try:
                    proc.terminate()
                    proc.wait(timeout=2)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                io_dir = self._conditional_io_dirs.get(run_id)
                if io_dir is not None:
                    try:
                        self._touch_end_tombstone(io_dir)
                    except Exception:
                        pass
            self._conditional_subprocesses.clear()
            self._conditional_io_dirs.clear()
            if hasattr(self, '_conditional_run_names'):
                self._conditional_run_names.clear()
            if hasattr(self, '_run_terminal_consumed'):
                self._run_terminal_consumed.clear()
            if hasattr(self, '_deferred_ticks'):
                self._deferred_ticks.clear()

        # Stop all request-card pending timers owned by the history view.
        # A running 1-second timer could otherwise fire against these
        # (soon-to-be-destroyed) cards after the tab's native window tree
        # is torn down.
        self.history_view.stop_all_pending_timers()

        # Clear all callbacks FIRST to prevent the file_watcher thread from
        # invoking stale callbacks that reference this (soon-to-be-destroyed)
        # tab via wx.CallAfter.  Without this, a race between the file_watcher
        # scan loop and tab destruction causes a crash when the main thread
        # processes a wx.CallAfter targeting a destroyed C++ window.
        self.conversation.on_message = None
        self.conversation.on_message_updated = None
        self.conversation.on_turn_complete = None
        self.conversation.on_fork_turn_update = None
        self.conversation.on_request_started = None
        self.conversation.on_cancel_ack = None
        self.conversation.on_error = None
        self.conversation.on_shutdown_starting = None
        self.conversation.on_shutdown_complete = None
        self.conversation.on_process_attached = None
        self.conversation.on_process_starting = None

        self.conversation.on_liveness_changed = None
        # Terminate the agent process in a background thread. terminate()
        # blocks for up to ~9s (graceful wait -> SIGTERM -> SIGKILL). Running
        # it synchronously here freezes the wx event loop, during which the
        # file-watcher thread keeps flooding the queue with wx.CallAfter
        # events for the other open tabs. When the main thread finally
        # resumes and destroys this tab's native windows, those deferred
        # events race window destruction and the process halts with no
        # traceback. Launching termination on a daemon thread keeps the GUI
        # responsive so the watcher can finish scanning before teardown.
        # (This mirrors _on_cancel_ack, which already uses this pattern.)
        threading.Thread(target=self.conversation.terminate, daemon=True).start()
