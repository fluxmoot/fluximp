"""GUI-agnostic conversation management for Fluximp agents."""

import os
import sys
import json
import time
import logging
import subprocess
import threading
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, List, Tuple, Any
from datetime import datetime, timezone
from enum import Enum
from collections import deque

from fluximp_protocol.user_agent.file_io.user import FileIOUserProtocol
from fluximp_protocol.user_agent.file_io.defs import (
    extract_fork_turn_from_filename_strict,
    ParsedHeader,
    MessageType,
    parse_utc_timestamp,
)
from fluximp_agent.process_utils import build_agent_command, launch_agent_process, terminate_process
from fluximp_protocol.user_agent.file_io.wire_format import (
    parse_response_payload,
    _is_coordinate_shaped,
)
from fluximp_protocol.user_agent.fork_turn_state import (
    ForkTurn,
    IOFactKind,
    IOTurnFact,
    IOTurnStatus,
    ForkTurnState,
    ForkTurnStateError,
)

class ConversationStatus(Enum):
    """Status of a conversation turn."""
    EMPTY = "empty"
    COMPLETED = "completed"
    ACTIVE = "active"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class ForkTurnInfo:
    """Information about a specific fork/turn."""
    fork: int
    turn: int
    status: ConversationStatus
    has_tool_use: bool = False
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    # Sum of the turn's per-response API durations (the "Time:" value shown
    # on request cards): response end_time minus request start_time.
    time_seconds: float = 0.0


@dataclass
class PermissionRequest:
    """A tool permission request from the agent."""
    tool_name: str
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class ToolMessageKey:
    """Identity of a pending tool message, discriminated by stream."""
    stream: str  # 'primary' or 'conditional'
    agent_name: Optional[str]  # None for primary stream
    fork: int
    turn: int
    request_seq: int

@dataclass
class ConversationMessage:
    """A message in the conversation."""
    fork: int
    turn: int
    role: str  # 'user', 'assistant', 'tool', 'tool_result', 'permission_request'
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _ConditionalScanTarget:
    """Per-agent scanning state for a conditional agent's I/O directory.

    Holds the FileIOUserProtocol and per-agent tracking dicts so that
    multiple conditional agents can be scanned independently without key
    collisions (all conditional agents use fork=0, turn=0).
    """
    agent_name: str
    protocol: FileIOUserProtocol
    run_id: str = ""
    anchor_fork: Optional[int] = None
    anchor_turn: Optional[int] = None
    launch_ts: Optional[datetime] = None
    request_seq: Dict[ForkTurn, int] = field(default_factory=dict)
    pending_request_seqs: Dict[ForkTurn, deque] = field(default_factory=dict)
    request_start_times: Dict[Tuple[int, int, int], str] = field(default_factory=dict)
    request_metadata: Dict[Tuple[int, int, int], Dict] = field(default_factory=dict)
    turn_metadata: Dict[ForkTurn, Dict] = field(default_factory=dict)


class Conversation:
    """Represents a single agent conversation.

    This class manages the file I/O protocol with main.py and tracks
    the state of forks and turns.
    """

    def __init__(self, conversation_dir: Path, working_dir: Path, process: Optional[subprocess.Popen],
                 conversation_id: str, start_time: datetime, read_root_dir: Optional[Path] = None,
                 write_file: Optional[Path] = None, agent_type: str = 'developer',
                 workspace: Optional['Workspace'] = None):
        # Setup logging
        self.log_dir = conversation_dir / "log"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_file = self.log_dir / f"session_{start_time.strftime('%Y%m%d_%H%M%S')}.log"
        self.logger = logging.getLogger(f"conversation_{conversation_id}")
        self.logger.setLevel(logging.DEBUG)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        formatter.converter = time.gmtime
        file_handler.setFormatter(formatter)
        # F-B5 (conversation log doubling, 2026-10-07 11:59 log): the
        # logger name is stable across opens, so an in-process close ->
        # reopen accumulated a SECOND FileHandler on the same logger —
        # every post-reopen line was logged twice (the dev_workflow_
        # 20261007_113915 session log's 11:58:43+ lines are doubled
        # pairs; the 11:59:37 fresh process's lines are single). Remove
        # this conversation logger's existing FileHandlers (and close
        # them — a Windows handle release) before attaching the new
        # one; the app-wide root logger is untouched.
        for old_handler in list(self.logger.handlers):
            if isinstance(old_handler, logging.FileHandler):
                self.logger.removeHandler(old_handler)
                try:
                    old_handler.close()
                except Exception:
                    pass
        self.logger.addHandler(file_handler)
        # Prevent conversation logs from bubbling up to the app-wide log
        self.logger.propagate = False

        self.conversation_dir = conversation_dir
        self.io_dir = conversation_dir / "io"
        self.working_dir = working_dir
        self.read_root_dir = read_root_dir
        self.write_file = write_file
        
        # Back-reference to the owning Workspace. The test-runner plugin
        # directory is a workspace-level (not per-conversation) setting, so
        # it is NOT stored on the Conversation; start_agent_process reads
        # it from the workspace at launch time so Settings changes apply to
        # every conversation's next agent launch/restart.
        self.workspace = workspace

        self.process = process
        self.conversation_id = conversation_id
        self.start_time = start_time
        self.agent_type = agent_type
        self.agent_configs: List[str] = []  # list of agent config file paths for --agent-config
        self.workflow_name: str = 'Dev Workflow'  # which workflow this conversation belongs to

        # Create subdirectories
        self.io_dir.mkdir(parents=True, exist_ok=True)

        # Client-side file I/O protocol handler
        self._protocol = FileIOUserProtocol(self.io_dir)

        # Conditional agent scan targets: discovered automatically from disk.
        # Both the live file-watcher and reload-from-disk paths call
        # _scan_io_files() which calls _discover_conditional_dirs().
        self._conditional_scan_targets: Dict[Path, '_ConditionalScanTarget'] = {}

        # Reload stream-size snapshot: path -> byte size at the moment the
        # reload began (used by the watcher's finalize-once predicate).
        self._stream_sizes: Dict[str, int] = {}

        # Per-file last header timestamp (debug assert, §7.6): header
        # timestamps must be strictly increasing within each file.
        self._last_header_ts_by_file: Dict[str, datetime] = {}

        self.fork_turns: Dict[ForkTurn, ForkTurnInfo] = {}
        self.messages: List[ConversationMessage] = []
        self.fork_turn_state = ForkTurnState()

        # Provider/model configuration tracking
        self.current_provider: str = ""
        self.current_model: str = ""
        # Maps provider name to last used model for that provider
        self.provider_models: Dict[str, str] = {}

        # Track request sequence number per (fork, turn): (fork, turn) -> request_seq
        # Used to distinguish multiple API requests within a single turn
        self._current_request_seq: Dict[ForkTurn, int] = {}

        # Track pending request sequences per (fork, turn) for matching REQUEST to RESPONSE and TOOL_USE
        # Uses FIFO queue: REQUEST pushes, RESPONSE pops, TOOL_USE peeks
        from collections import deque
        self._pending_request_seqs: Dict[ForkTurn, deque] = {}

        # Track request start times per (fork, turn, request_seq): (fork, turn, request_seq) -> start_time_iso
        # Used to calculate request duration
        self._request_start_times: Dict[Tuple[int, int, int], str] = {}

        # Track per-request metadata from request and response headers: (fork, turn, request_seq) -> Dict
        # Used to store request-specific metadata
        self._request_metadata: Dict[Tuple[int, int, int], Dict] = {}

        # Track per-turn metadata from request and response headers: (fork, turn) -> Dict
        # Used to combine metadata for model_message_to_user messages (aggregates across all requests in a turn)
        self._turn_metadata: Dict[ForkTurn, Dict] = {}

        # Track pending tool messages by (fork, turn, request_seq) for updating with results
        # Each (fork, turn, request_seq) can have multiple tool calls, so we store a list
        self._pending_tool_messages: Dict[ToolMessageKey, List[ConversationMessage]] = {}

        # Track pending system agent messages by agent_type for updating with stopped status
        self._pending_system_agents: Dict[str, ConversationMessage] = {}
        # File watching is now handled by the centralized FileWatcher (gui.file_watcher)

        # Process lifecycle tracking
        self._is_shutting_down = False
        # Transitional conditional-run identity (Phase 1): run_id -> anchor
        self._conditional_anchors: Dict[str, Tuple[int, int]] = {}
        # system_agent_ord per (fork, turn, agent_type) for deterministic ordering
        self._system_agent_ord: Dict[Tuple[int, int, str], int] = {}
        self._shutdown_lock = threading.Lock()

        # Callbacks (GUI-agnostic event handling)
        self.on_message: Optional[Callable[[ConversationMessage], None]] = None
        self.on_message_updated: Optional[Callable[[ConversationMessage], None]] = None  # Called when message content/status changes
        self.on_turn_complete: Optional[Callable[[int, int], None]] = None
        self.on_error: Optional[Callable[[int, int, str], None]] = None
        self.on_fork_turn_update: Optional[Callable[[], None]] = None
        self.on_request_started: Optional[Callable[[int, int, dict], None]] = None  # fork, turn, metadata
        self.on_cancel_ack: Optional[Callable[[int, int], None]] = None  # fork, turn - called when cancel is acknowledged
        self.on_process_attached: Optional[Callable[[], None]] = None  # Called when a new process is attached
        self.on_process_starting: Optional[Callable[[], None]] = None  # Called when a new process is starting
        self.on_shutdown_starting: Optional[Callable[[], None]] = None  # Called when shutdown begins
        self.on_shutdown_complete: Optional[Callable[[], None]] = None  # Called when shutdown finishes

        # Liveness-transition callback (process_lock.py arm): fired on the
        # watcher thread when the conversation's alive verdict flips — an
        # attached Popen exits, an unattached (orphaned) agent is discovered
        # alive via its io-dir agent lock, or a held lock drops (the orphan
        # exited). The owning tab marshals the recompute to the GUI thread.
        self.on_liveness_changed: Optional[Callable[[], None]] = None
        # Reload streaming brackets + completion callback (Phase 2 §6/§7.3).
        self.on_scan_batch_start: Optional[Callable[[], None]] = None
        self.on_scan_batch_end: Optional[Callable[[], None]] = None
        self.on_reload_complete: Optional[Callable[[], None]] = None
        # Reload state (§8.5): set by the history controller's reload bracket.
        # While True, _post_fact_transition buffers facts for one batched
        # application per tick (instead of one wx.CallAfter + one right-panel
        # grid rebuild per fact — the flood that starved paint events), and
        # scan-path on_fork_turn_update fires are suppressed (the settled
        # notification fires once at finalization, §7.4 step 9).
        self.reload_in_progress: bool = False
        self._reload_fact_buffer: List[IOTurnFact] = []

        # Agent active state tracking. The enabled-state consumer is the
        # owning tab's right-panel forward: the shared panel no longer
        # binds on_agent_active_changed (deleted — the callback carried no
        # tab identity, so background tabs' transitions corrupted the
        # shared panel); the combos mirror the displayed tab's input/Send
        # pair instead, with this flag as set_conversation's fallback read.
        self.is_agent_active: bool = False


    def _discover_conditional_dirs(self):
        """Discover conditional agent I/O subdirectories on disk.

        Scans the primary I/O directory for subdirectories matching the
        naming convention 'conditional_*'.  Each subdirectory is an I/O
        directory created by _launch_agent_subprocess() in
        conditional_sequence.py.  The agent name is recovered from
        agent_config.json inside the subdirectory, or parsed from the
        directory name as a fallback.
        """
        if not self.io_dir.exists():
            return
        for subdir in self.io_dir.iterdir():
            if not subdir.is_dir():
                continue
            if not subdir.name.startswith("conditional_"):
                continue
            if subdir in self._conditional_scan_targets:
                continue  # Already tracking this target
            agent_name, run_id = self._read_conditional_agent_name(subdir)
            target = _ConditionalScanTarget(
                agent_name=agent_name,
                run_id=run_id,
                protocol=FileIOUserProtocol(subdir),
            )
            self._conditional_scan_targets[subdir] = target
            self.logger.info(f"Discovered conditional agent I/O directory: {subdir} (agent: {agent_name}, run: {run_id})")

    @staticmethod
    def _read_conditional_agent_name(io_dir: Path) -> Tuple[str, str]:
        """Recover a conditional agent's (display name, run_id) from its I/O dir.

        Reads agent_config.json (written by _launch_agent_subprocess) and
        extracts the agent name.  The run_id is the directory-name suffix
        (8 hex chars minted by the sequence engine); it is always parsed from
        the directory name because agent_config.json does not carry it.
        Falls back to parsing the whole directory name.
        """
        name_part = io_dir.name[len("conditional_"):]
        last_underscore = name_part.rfind('_')
        run_id = ""
        if last_underscore > 0:
            run_id = name_part[last_underscore + 1:]

        config_path = io_dir / "agent_config.json"
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                conditional_agents = config_data.get("conditional_agents", {})
                if conditional_agents:
                    # The dict has one key: the agent name
                    agent_name = next(iter(conditional_agents.keys()))
                    return agent_name, run_id
            except (json.JSONDecodeError, OSError):
                pass

        # Fallback: parse directory name after 'conditional_' prefix
        # Format: conditional_<Agent_Name>_<run_id>
        if last_underscore > 0:
            name_part = name_part[:last_underscore]
        return name_part.replace('_', ' '), run_id

    def _scan_io_files(self, update_facts: bool = True,
                       time_budget: Optional[float] = None) -> set:
        """Scan all io streams for new content (two-phase, inputs first).

        Phase 1: primary inputs -> primary outputs (output discovery still
        discovers conditional dirs). Phase 2: each just-discovered conditional
        dir's inputs -> outputs. Input scans are never throttled; output
        parsing is bounded by ``time_budget`` (an elapsed-time deadline shared
        across streams in the calling tick, §7.3).

        Args:
            update_facts: Whether to post fact transitions into
                fork_turn_state. False replays already-parsed files without
                re-posting facts (the IN/END seed owns the snapshot).
            time_budget: Maximum wall-clock seconds this call may spend on
                output parsing (an absolute deadline is derived per tick). None
                means unlimited.

        Returns:
            Set of MessageType values found during this scan.
        """
        types_found = set()
        deadline = None
        if time_budget is not None:
            deadline = time.monotonic() + time_budget

        def _budget_exhausted() -> bool:
            return deadline is not None and time.monotonic() >= deadline

        # --- Phase 1a: primary inputs (never throttled) ---
        for in_file, fork, turn in self._protocol.scan_input_files():
            self._parse_input_file(
                in_file, fork, turn, protocol=self._protocol,
                is_conditional=False, conditional_agent_name="")
            types_found.add(MessageType.USER_MESSAGE)

        # --- Phase 1b: primary outputs (budgeted) ---
        for out_file, fork, turn in self._protocol.scan_output_files():
            if _budget_exhausted():
                break
            # Post the UNPARSED observation for a primary output file that
            # does not yet have an OUT fact (before its contents are parsed).
            if update_facts and not any(
                f.kind is IOFactKind.OUT and f.fork == fork and f.turn == turn
                for f in self.fork_turn_state.facts
            ):
                self._post_fact_transition(
                    IOTurnFact(fork, turn, IOFactKind.OUT, IOTurnStatus.UNPARSED))
            sub_types, _ = self._parse_output_file(
                out_file, fork, turn,
                protocol=self._protocol,
                is_conditional=False,
                conditional_agent_name="",
                update_facts=update_facts,
                deadline=deadline)
            types_found.update(sub_types)

        # --- Phase 2: discover and scan conditional agent subdirectories ---
        self._discover_conditional_dirs()
        for target in self._conditional_scan_targets.values():
            if _budget_exhausted():
                break
            for in_file, fork, turn in target.protocol.scan_input_files():
                self._parse_input_file(
                    in_file, fork, turn,
                    protocol=target.protocol,
                    is_conditional=True,
                    conditional_agent_name=target.agent_name,
                    scan_target=target)
            for out_file, fork, turn in target.protocol.scan_output_files():
                if _budget_exhausted():
                    break
                sub_types, _ = self._parse_output_file(
                    out_file, fork, turn,
                    protocol=target.protocol,
                    is_conditional=True,
                    conditional_agent_name=target.agent_name,
                    scan_target=target,
                    update_facts=update_facts,
                    deadline=deadline)
                types_found.update(sub_types)

        return types_found

    def _parse_input_file(self, input_file: Path, fork: int, turn: int,
                          protocol=None, is_conditional=False,
                          conditional_agent_name="", scan_target=None) -> int:
        """Parse unread records from a primary/conditional input file.

        Primary USER_MESSAGE records are the sole io-stream source of user
        panels (routed through on_message, §5.10); cancel_turn and
        tool_permission_response records in a primary input are command
        records and create no panel. Conditional inputs (``0.0.in.txt``) are
        run context only -- never a primary panel and never post facts.

        Returns:
            Number of records consumed.
        """
        if protocol is None:
            protocol = self._protocol
        records_consumed = 0
        with protocol.new_input_records(input_file) as records:
            for record in records:
                header, payload = record.header, record.payload
                if header.is_type(MessageType.USER_MESSAGE) and not is_conditional:
                    # Primary user message: the sole io-stream event source.
                    timestamp = parse_utc_timestamp(header.timestamp)
                    message = ConversationMessage(
                        fork=fork, turn=turn, role='user',
                        content=payload, timestamp=timestamp,
                    )
                    self.messages.append(message)
                    if self.on_message:
                        self.on_message(message)
                    # Defensive IN-fact dedup: the live send path posts the IN
                    # fact eagerly, but a reload scanning inputs must still see
                    # the IN fact for visibility/upsert machinery.
                    if not any(
                        f.kind is IOFactKind.IN and f.fork == fork and f.turn == turn
                        for f in self.fork_turn_state.facts
                    ):
                        self._post_fact_transition(
                            IOTurnFact(fork, turn, IOFactKind.IN, IOTurnStatus.NOT_APPLICABLE))
                if (is_conditional and scan_target is not None
                        and header.is_type(MessageType.USER_MESSAGE)
                        and scan_target.anchor_fork is None):
                    # Conditional run context (0.0.in.txt launch header). The
                    # launch record's header timestamp is the authoritative
                    # anchor source (§7.6): the latest primary user message
                    # whose timestamp is <= the launch timestamp; ties resolve
                    # to the user message; (0, -1) if none. Idempotent
                    # thereafter and stamped onto the run's identity.
                    launch_ts = parse_utc_timestamp(header.timestamp)
                    scan_target.launch_ts = launch_ts
                    anchor = (0, -1)
                    for m in self.messages:
                        if (m.role == 'user'
                                and not m.metadata.get('is_conditional', False)
                                and m.timestamp <= launch_ts):
                            anchor = (m.fork, m.turn)
                    scan_target.anchor_fork, scan_target.anchor_turn = anchor
                    self._conditional_anchors[scan_target.run_id] = anchor
                # Conditional input records (and primary cancel/permission
                # command records) never create a panel or post facts.
                protocol.commit_input_position(input_file, record.next_pos)
                records_consumed += 1
        return records_consumed

    def seed_in_end_facts(self, io_dir: Path) -> ForkTurnState:
        """Build the IN/END fact seed for a reload from filenames only.

        Walks ``*.in.txt`` and ``*.end`` filenames (never outputs -- those are
        learned live by the watcher) and constructs the reconciled
        ForkTurnState the watcher upserts against (§7.2). IN/END facts are
        NOT_APPLICABLE; malformed coordinate-shaped names fail fast exactly
        like the snapshot scanner.
        """
        facts = set()
        for glob_pattern, kind in (
            ("*.in.txt", IOFactKind.IN),
            ("*.end", IOFactKind.END),
        ):
            for path in io_dir.glob(glob_pattern):
                if not path.is_file():
                    continue
                if not _is_coordinate_shaped(path.name, kind):
                    continue
                fork, turn = extract_fork_turn_from_filename_strict(path.name)
                facts.add(IOTurnFact(fork, turn, kind, IOTurnStatus.NOT_APPLICABLE))
        return ForkTurnState(facts).reconcile_snapshot()

    def _iter_discovered_streams(self):
        """Yield every io stream path currently discovered on disk.

        Primary ``*.in.txt`` / ``*.out.txt`` / ``*.end`` plus each discovered
        conditional dir's own three stream kinds. Discovery of conditional
        dirs is refreshed so the finalize-once predicate sees them.
        """
        self._discover_conditional_dirs()
        for pattern in ("*.in.txt", "*.out.txt", "*.end"):
            for path in sorted(self.io_dir.glob(pattern)):
                if path.is_file():
                    yield path
        for target in self._conditional_scan_targets.values():
            for pattern in ("*.in.txt", "*.out.txt", "*.end"):
                for path in sorted(target.protocol.io_path.glob(pattern)):
                    if path.is_file():
                        yield path

    def _refresh_stream_sizes(self) -> None:
        """Snapshot current byte sizes of every discovered io stream.

        Populates ``self._stream_sizes`` used by the watcher's finalize-once
        predicate (the reload is "caught up" when each stream's committed
        position reaches its snapshot size). Called by ``load_existing``
        (step 0.5) before ``start_reload`` so the predicate captures the
        pre-reload backlog; late-discovered conditional dirs (genuinely new
        runs) are appended too.
        """
        self._stream_sizes.clear()
        for stream in self._iter_discovered_streams():
            try:
                self._stream_sizes[str(stream)] = stream.stat().st_size
            except FileNotFoundError:
                self._stream_sizes[str(stream)] = 0

    def reload_caught_up(self) -> bool:
        """True when every parsed stream in the reload snapshot is committed.

        The caught-up predicate (§7.3): each ``*.in.txt`` / ``*.out.txt``
        stream recorded in ``self._stream_sizes`` has a committed read
        position that reached (or passed) its snapshot byte size, so nothing
        from the pre-reload backlog remains unparsed. ``.end`` files are
        terminal markers (never parsed by the watcher) and are always
        considered caught up. Streams that vanished after the snapshot (e.g.
        a deleted conditional dir) are also considered caught up.
        """
        for path_str, size in self._stream_sizes.items():
            path = Path(path_str)
            if path.name.endswith(".end"):
                continue
            protocol = self._protocol_for_path(path)
            if protocol is None:
                continue
            if path.name.endswith(".in.txt"):
                committed = protocol.get_input_file_position(path)
            else:
                committed = protocol.get_file_position(path)
            if committed < size:
                return False
        return True

    def _protocol_for_path(self, path: Path):
        """Resolve the protocol that owns an io stream path.

        The primary protocol owns files directly in the conversation io dir;
        each conditional scan target owns files in its own subdir.
        """
        if path.parent == self.io_dir:
            return self._protocol
        for target in self._conditional_scan_targets.values():
            if path.parent == target.protocol.io_path:
                return target.protocol
        return None

    def output_stream_at_eof(self, out_file: Path) -> bool:
        """True when an out-stream's committed position is at its EOF (§5.10).

        Read-only stat-vs-committed-position check on the protocol's per-file
        position map — the §7.7 deferred-crash escape's EOF gate. Advisory
        GUI-thread query: a single-value dict read; worst case one tick's
        staleness delays the escape. A missing file is treated as at-EOF.
        """
        if not out_file.exists():
            return True
        protocol = self._protocol_for_path(out_file)
        if protocol is None:
            return False
        committed = protocol.get_file_position(out_file)
        try:
            size = out_file.stat().st_size
        except FileNotFoundError:
            return True
        return committed >= size

    def _abandon_unanswered_requests(self, key, pending_request_seqs,
                                     request_start_times, request_seq):
        """Fix A (2026-10-10 resume pairing desync): abandon a (fork, turn)'s
        unanswered request seqs when a new REQUEST row arrives at it.

        The pairing FIFO pushes every REQUEST seq and pops on RESPONSE; a seq
        whose request never completes (a crashed session's last row, or an
        aborted in-session retry) stays queued forever. A crash+resume makes
        this concrete: the reopen replays the dead session's unanswered
        REQUEST row (the FIFO gains an orphan seq), and the resumed session's
        first RESPONSE then pops that orphan — pairing every subsequent cycle
        shifted by one, spawning duplicate contentless response cards while
        the live request's card (holding the tool expanders) was destroyed by
        the controller's retry-supersede (the 2026-10-10 02:10 "no tool
        expanders on the resumed turn" report).

        Safe by protocol (§7.6): one in-flight request per stream with
        strictly-increasing rows — a second REQUEST row at the same
        (fork, turn) strictly implies the earlier request will never receive
        its RESPONSE, so those FIFO entries are dead weight. A live
        in-flight request is never affected: no second REQUEST row
        intervenes before its RESPONSE arrives.
        """
        fifo = pending_request_seqs.get(key)
        if not fifo:
            return
        abandoned = list(fifo)
        start_times = [request_start_times.get((key.fork, key.turn, seq))
                       for seq in abandoned]
        fifo.clear()
        # WARNING, not debug: this was THE invisible event behind the
        # 2026-10-10 report — it must be findable in the conversation log.
        self.logger.warning(
            "REQUEST supersedes unanswered request(s): fork=%s turn=%s "
            "new_seq=%s — abandoned seqs=%s start_times=%s "
            "(a second REQUEST row at this (fork, turn) means the earlier "
            "request aborted: crash+resume or retry; its RESPONSE can "
            "never arrive — one in-flight request per stream, §7.6)",
            key.fork, key.turn, request_seq, abandoned, start_times)

    def _log_request_pushed(self, key, pending_request_seqs,
                            fork: int, turn: int, request_seq: int):
        """Logging plan (2026-10-10 resume-desync diagnosis): the pairing
        FIFO state after each REQUEST row parse."""
        self.logger.debug(
            "request row parsed: fork=%s turn=%s seq=%s pushed; fifo=%s",
            fork, turn, request_seq,
            list(pending_request_seqs.get(key, deque())))

    def _parse_output_file(self, output_file: Path, fork: int, turn: int,
                            protocol=None, is_conditional=False,
                            conditional_agent_name="", scan_target=None,
                            update_facts: bool = True,
                            deadline: Optional[float] = None) -> Tuple[set, int]:
        """Parse a single output file for new messages.

        Uses the FileIOUserProtocol to track position and read new output.

        Args:
            output_file: Path to the output file to parse.
            fork: Fork number for this output.
            turn: Turn number for this output.
            protocol: Protocol to use for reading (defaults to self._protocol).
            is_conditional: Whether this output is from a conditional agent.
            conditional_agent_name: Name of the conditional agent (if applicable).
            scan_target: The conditional scan target (if conditional).
            update_facts: Whether to post fact transitions.
            deadline: Absolute monotonic deadline for this call (shared across
                files by the caller). None means unlimited.

        Returns:
            Tuple of (types_found, records_consumed).
        """
        if protocol is None:
            protocol = self._protocol

        types_found = set()
        has_updates = False
        records_consumed = 0

        # Stream unread records through the protocol's positioned cursor. The
        # cursor yields fully-validated StreamedRecords and commits nothing
        # itself; we commit a record's next_pos only after it is fully
        # processed so a failure never silently skips a wire record, and we can
        # stop at a message budget without draining the remaining generator.
        with protocol.new_output_records(output_file) as records:
            for record in records:
                if deadline is not None and time.monotonic() >= deadline:
                    break

                header, payload = record.header, record.payload

                # Debug assert (§7.6): header timestamps are strictly
                # increasing — unique — within each file. Active only under
                # FLUXIMP_DEBUG_ASSERTS (the writer's monotonic-stamp
                # guarantee and this reader check agree).
                if os.getenv('FLUXIMP_DEBUG_ASSERTS') and header.timestamp is not None:
                    file_key = str(output_file)
                    ts = parse_utc_timestamp(header.timestamp)
                    last = self._last_header_ts_by_file.get(file_key)
                    if last is not None:
                        assert ts > last, (
                            f"non-monotonic header timestamp in {output_file}: "
                            f"{ts.isoformat()} <= {last.isoformat()}")
                    self._last_header_ts_by_file[file_key] = ts

                # Upgrade the observed UNPARSED OUT to INCOMPLETE once we begin
                # parsing the file's contents (primary stream only; conditional
                # sub-streams never enter the primary snapshot). Post only when
                # the file actually has new content and the tracked OUT is not
                # already INCOMPLETE or terminal, so re-scans of a partially-
                # parsed or completed file do not re-post (the live transition
                # machine rejects INCOMPLETE->INCOMPLETE and terminal downgrades).
                if update_facts and not is_conditional and records_consumed == 0:
                    existing_out = next(
                        (f.status for f in self.fork_turn_state.facts
                         if f.kind is IOFactKind.OUT and f.fork == fork and f.turn == turn),
                        None,
                    )
                    if existing_out is None or existing_out is IOTurnStatus.UNPARSED:
                        self._post_fact_transition(
                            IOTurnFact(fork, turn, IOFactKind.OUT, IOTurnStatus.INCOMPLETE))
                # Track request headers for timing calculations
                if header.is_request:
                    types_found.add(MessageType.REQUEST)

                    # Track request metadata using the appropriate namespace
                    if is_conditional and scan_target is not None:
                        # Conditional agent metadata (per-target namespace)
                        key = ForkTurn(fork, turn)
                        scan_target.request_seq[key] = scan_target.request_seq.get(key, 0) + 1
                        request_seq = scan_target.request_seq[key]

                        # Fix A (2026-10-10 resume-desync — two consecutive
                        # REQUEST rows head 0.0.out.txt after a crash+resume):
                        # a new REQUEST row at this (fork, turn) while
                        # unanswered seqs sit in the pairing FIFO means those
                        # requests can never be matched (one in-flight request
                        # per stream, §7.6) — abandon them so this request's
                        # response pops THIS seq instead of an orphan's.
                        self._abandon_unanswered_requests(
                            key, scan_target.pending_request_seqs,
                            scan_target.request_start_times, request_seq)

                        if key not in scan_target.pending_request_seqs:
                            scan_target.pending_request_seqs[key] = deque()
                        scan_target.pending_request_seqs[key].append(request_seq)
                        self._log_request_pushed(
                            key, scan_target.pending_request_seqs,
                            fork, turn, request_seq)

                        request_key = (fork, turn, request_seq)
                        scan_target.request_start_times[request_key] = header.start_time
                        scan_target.request_metadata[request_key] = header.to_metadata_dict()

                        if key not in scan_target.turn_metadata:
                            scan_target.turn_metadata[key] = {}
                        scan_target.turn_metadata[key].update(header.to_metadata_dict())
                    else:
                        # Primary agent metadata
                        key = ForkTurn(fork, turn)
                        self._current_request_seq[key] = self._current_request_seq.get(key, 0) + 1
                        request_seq = self._current_request_seq[key]

                        # Fix A (2026-10-10 resume-desync): same pairing
                        # reset as the conditional namespace above. Pre-fix,
                        # the reopened GUI replayed the dead session's
                        # unanswered REQUEST row (the FIFO gained an orphan
                        # seq), and the resumed session's first RESPONSE then
                        # popped that orphan — every subsequent cycle spawned
                        # a duplicate contentless response card while the
                        # live request's card (holding the tool expanders)
                        # was destroyed by the controller's retry-supersede.
                        self._abandon_unanswered_requests(
                            key, self._pending_request_seqs,
                            self._request_start_times, request_seq)

                        if key not in self._pending_request_seqs:
                            self._pending_request_seqs[key] = deque()
                        self._pending_request_seqs[key].append(request_seq)
                        self._log_request_pushed(
                            key, self._pending_request_seqs,
                            fork, turn, request_seq)

                        request_key = (fork, turn, request_seq)
                        self._request_start_times[request_key] = header.start_time
                        self._request_metadata[request_key] = header.to_metadata_dict()

                        if key not in self._turn_metadata:
                            self._turn_metadata[key] = {}
                        self._turn_metadata[key].update(header.to_metadata_dict())

                    # Fire callback for request started (includes conditional metadata)
                    metadata = {
                        "agent_type": header.agent_type,
                        "provider": header.provider,
                        "requested_model": header.requested_model,
                        "request_seq": scan_target.request_seq.get(ForkTurn(fork, turn), 0) if (is_conditional and scan_target)
                                       else self._current_request_seq.get(ForkTurn(fork, turn), 0),
                    }
                    if is_conditional and scan_target is not None:
                        metadata['is_conditional'] = True
                        metadata['conditional_agent_name'] = conditional_agent_name
                        metadata['run_id'] = scan_target.run_id
                        anchor = self._resolve_conditional_anchor(scan_target.run_id)
                        metadata['anchor_fork'], metadata['anchor_turn'] = anchor
                        scan_target.anchor_fork, scan_target.anchor_turn = anchor
                    if self.on_request_started:
                        self.on_request_started(fork, turn, metadata)

                if self._process_message(fork, turn, header, payload,
                                         is_conditional=is_conditional,
                                         conditional_agent_name=conditional_agent_name,
                                         scan_target=scan_target,
                                         update_facts=update_facts):
                    has_updates = True

                # Track message type (REQUEST already tracked above)
                if not header.is_request:
                    try:
                        types_found.add(MessageType(header.msg_type))
                    except ValueError:
                        pass  # Unknown message type, skip

                protocol.commit_output_position(output_file, record.next_pos)
                records_consumed += 1

        if (has_updates and not is_conditional and self.on_fork_turn_update
                and not getattr(self, 'reload_in_progress', False)):
            # During a reload the scan-path notification is suppressed (§8.5):
            # the settled notification fires once at finalization (§7.4 step 9).
            self.on_fork_turn_update()

        return types_found, records_consumed

    def _process_message(self, fork: int, turn: int,
                         header: ParsedHeader, payload: str,
                         is_conditional=False,
                         conditional_agent_name="", scan_target=None,
                         update_facts: bool = True) -> bool:
        """Process a single message from the output file.

        Returns True if this message updated fork/turn state.
        
        """
        updated = False

        # Select the appropriate metadata namespace based on is_conditional.
        # Conditional agents use separate dicts to avoid key collisions with
        # the primary agent (both use fork=0, turn=0).
        if is_conditional and scan_target is not None:
            pending_request_seqs = scan_target.pending_request_seqs
            request_start_times = scan_target.request_start_times
            current_request_seq = scan_target.request_seq
            turn_metadata = scan_target.turn_metadata
        else:
            pending_request_seqs = self._pending_request_seqs
            request_start_times = self._request_start_times
            current_request_seq = self._current_request_seq
            turn_metadata = self._turn_metadata

        # Logging plan (2026-10-10 resume-desync diagnosis): every record now
        # carries its request-pairing scope (the seq counter that
        # tools/permissions route by + the pending-request FIFO that
        # responses pop from), so a turn's routing can be traced from the
        # conversation log alone.
        ns_key = ForkTurn(fork, turn)
        self.logger.debug(
            "Processing message: type=%s, fork=%s, turn=%s, "
            "current_request_seq=%s, pending_fifo=%s",
            header.msg_type, fork, turn,
            current_request_seq.get(ns_key, 0),
            list(pending_request_seqs.get(ns_key, deque())))

        if header.timestamp is None:
            raise ValueError(
                f"corrupt stream: {header.msg_type!r} header at fork {fork}, "
                f"turn {turn} has no timestamp"
            )
        timestamp = parse_utc_timestamp(header.timestamp)

        # Fail fast on system-agent-only report types in the primary stream.
        # These are only ever written by system/conditional agents into their
        # own I/O subdirectories; seeing them in the primary output stream
        # indicates on-disk corruption (wrong directory or misrouted write).
        if not is_conditional:
            if header.is_type(MessageType.SCRUTINATOR_REPORT):
                raise ForkTurnStateError(
                    f"SCRUTINATOR_REPORT encountered in primary output stream "
                    f"at fork {fork}, turn {turn}"
                )
            if header.is_type(MessageType.SUMMARY):
                raise ForkTurnStateError(
                    f"SUMMARY encountered in primary output stream "
                    f"at fork {fork}, turn {turn}"
                )
        if header.is_response:
            # Pop request_seq from queue to match this RESPONSE with its REQUEST (FIFO)
            key = ForkTurn(fork, turn)
            request_seq = 0
            if key in pending_request_seqs and len(pending_request_seqs[key]) > 0:
                request_seq = pending_request_seqs[key].popleft()
                # Logging plan (2026-10-10 resume-desync): every pairing
                # decision is now traceable — the 02:10 resumed turn
                # streamed ~2 cards per request cycle with no diagnostic
                # trail anywhere in the logs.
                self.logger.debug(
                    "response matched request: fork=%s turn=%s seq=%s fifo_remain=%s",
                    fork, turn, request_seq, list(pending_request_seqs[key]))
            else:
                self.logger.warning(
                    "RESPONSE with empty pending-request queue (orphan response): "
                    "fork=%s turn=%s — pairing desync: request_seq falls back to 0",
                    fork, turn)
            request_key = (fork, turn, request_seq) if request_seq > 0 else None
            
            
            # Store response metadata for this turn (merges with request metadata)
            if key not in turn_metadata:
                turn_metadata[key] = {}
            turn_metadata[key].update(header.to_metadata_dict())
            
            # Request start time for this response (use request-specific
            # key). Hoisted above the ForkTurnInfo block so both the Time
            # accumulation and the message metadata below use it.
            start_time = None
            if request_key and request_key in request_start_times:
                start_time = request_start_times[request_key]

            # Accumulate token counts into ForkTurnInfo (primary agent only)
            if not is_conditional:
                if key not in self.fork_turns:
                    self.fork_turns[key] = ForkTurnInfo(fork, turn, ConversationStatus.ACTIVE)
                info = self.fork_turns[key]
                if header.prompt_tokens is not None:
                    info.prompt_tokens += header.prompt_tokens
                if header.cached_tokens is not None:
                    info.cached_tokens += header.cached_tokens
                if header.completion_tokens is not None:
                    info.completion_tokens += header.completion_tokens
                if header.reasoning_tokens is not None:
                    info.reasoning_tokens += header.reasoning_tokens
                # Sum the per-response durations (the "Time:" value shown on
                # the request cards) for this turn: request-header start_time
                # -> response-header end_time, with the same guarding.
                if header.end_time and start_time:
                    try:
                        end_dt = parse_utc_timestamp(header.end_time)
                        start_dt = parse_utc_timestamp(start_time)
                        duration = (end_dt - start_dt).total_seconds()
                        if duration >= 0:
                            info.time_seconds += duration
                    except (ValueError, TypeError):
                        pass

            # Parse subheader-encoded payload for content and reasoning
            parsed = parse_response_payload(payload) if payload else {}
            content_text = parsed.get("content", "")
            reasoning_text = parsed.get("reasoning", "")

            # Create assistant message with content, including the request_seq in metadata
            msg_metadata = {**turn_metadata.get(key, {}), 'reasoning': reasoning_text, 'is_api_response': True, 'request_seq': request_seq}
            if is_conditional:
                msg_metadata.update(self._conditional_identity(scan_target, conditional_agent_name))
            message = ConversationMessage(
                fork=fork, turn=turn, role='assistant',
                content=content_text,
                timestamp=timestamp,
                metadata=msg_metadata
            )
            self.messages.append(message)
            if self.on_message:
                self.on_message(message)

        elif header.is_type(MessageType.MODEL_MESSAGE_TO_USER):
            # model_message_to_user carries tags and format in header
            # Metadata also includes accumulated turn metadata (request + response headers)
            key = ForkTurn(fork, turn)
            request_seq = current_request_seq.get(key, 0)
            request_key = (fork, turn, request_seq) if request_seq > 0 else None
            
            metadata = dict(turn_metadata.get(key, {}))
            # Include request start time for duration calculation (use request-specific key)
            if request_key and request_key in request_start_times:
                metadata['start_time'] = request_start_times[request_key]
            
            # Include tags and format from header
            if header.tags is not None:
                metadata['tags'] = header.tags
            if header.format is not None:
                metadata['format'] = header.format
            if is_conditional:
                metadata.update(self._conditional_identity(scan_target, conditional_agent_name))
            message = ConversationMessage(
                fork=fork, turn=turn, role='assistant',
                content=payload,
                timestamp=timestamp,
                metadata=metadata
            )
            self.messages.append(message)
            if self.on_message:
                self.on_message(message)

            # Update fork/turn status (primary agent only)
            if not is_conditional:
                if key not in self.fork_turns:
                    self.fork_turns[key] = ForkTurnInfo(fork, turn, ConversationStatus.COMPLETED)
                else:
                    self.fork_turns[key].status = ConversationStatus.COMPLETED
                updated = True
                if update_facts:
                    self._post_fact_transition(
                        IOTurnFact(fork, turn, IOFactKind.OUT, IOTurnStatus.COMPLETE_NORMAL))


                if self.on_turn_complete:
                    self.on_turn_complete(fork, turn)

        elif header.is_type(MessageType.TOOL_USE):
            key = ForkTurn(fork, turn)
            
            # Track tool use in fork_turns (primary agent only)
            if not is_conditional:
                if key not in self.fork_turns:
                    self.fork_turns[key] = ForkTurnInfo(fork, turn, ConversationStatus.ACTIVE)
                self.fork_turns[key].has_tool_use = True
            updated = True

            # Parse JSON payload for tool_calls
            display_payload = payload
            parsed_tool_calls = None
            try:
                data = json.loads(payload)
                if isinstance(data, dict) and "tool_calls" in data:
                    parsed_tool_calls = data["tool_calls"]
                    display_payload = json.dumps(data, indent=2)
            except json.JSONDecodeError:
                pass

            is_send_message = "send_message(" in display_payload

            # TOOL_USE follows its RESPONSE in a contiguous block,
            # so use the current request_seq for this (fork, turn)
            request_seq = current_request_seq.get(key, 0)

            metadata = {'status': 'pending', 'result': None, 'is_send_message': is_send_message,
                        'request_seq': request_seq}
            if parsed_tool_calls is not None:
                metadata['tool_calls'] = parsed_tool_calls
            if is_conditional:
                metadata.update(self._conditional_identity(scan_target, conditional_agent_name))

            message = ConversationMessage(
                fork=fork, turn=turn, role='tool',
                content=display_payload,
                timestamp=timestamp,
                metadata=metadata
            )
            self.messages.append(message)

            request_key = self._tool_message_key(
                fork, turn, request_seq, is_conditional, conditional_agent_name)
            if request_key is not None:
                if request_key not in self._pending_tool_messages:
                    self._pending_tool_messages[request_key] = []
                self._pending_tool_messages[request_key].append(message)

            if self.on_message:
                self.on_message(message)

        elif header.is_type(MessageType.TOOL_USE_RESULT):
            key = ForkTurn(fork, turn)

            # Extract execution_successful from header
            metadata = header.to_metadata_dict()
            execution_successful = bool(metadata.get('execution_successful', 0))

            # TOOL_USE_RESULT follows TOOL_USE in a contiguous block,
            # so use the current request_seq for this (fork, turn)
            request_seq = current_request_seq.get(key, 0)
            request_key = self._tool_message_key(
                fork, turn, request_seq, is_conditional, conditional_agent_name)

            # Match the first pending tool for this request_seq (FIFO within the block)
            message = None
            if request_key and request_key in self._pending_tool_messages:
                pending_list = self._pending_tool_messages[request_key]
                if pending_list:
                    message = pending_list.pop(0)
                    if not pending_list:
                        del self._pending_tool_messages[request_key]

            if message:
                message.metadata['status'] = 'completed'
                message.metadata['result'] = payload
                message.metadata.update(metadata)
                # Tool messages stay rendered (including successful
                # send_message) — the on_message_removed plumbing is gone.
                if self.on_message_updated:
                    self.on_message_updated(message)
            else:
                self.logger.warning(
                    f"TOOL_USE_RESULT with no pending TOOL_USE at fork {fork}, turn {turn}, request_seq {request_seq}"
                )

        elif header.is_type(MessageType.TOOL_PERMISSION_REQUEST):
            # Conditional streams never produce permissions (the conditional
            # agent cannot prompt the user mid-scan). Fail fast.
            if is_conditional:
                raise ForkTurnStateError(
                    f"TOOL_PERMISSION_REQUEST encountered in conditional output stream "
                    f"at fork {fork}, turn {turn}"
                )
            # TOOL_PERMISSION_REQUEST follows TOOL_USE in a contiguous block,
            # so use the current request_seq for this (fork, turn)
            key = ForkTurn(fork, turn)
            request_seq = current_request_seq.get(key, 0)
            try:
                data = json.loads(payload)
                request = PermissionRequest(
                    tool_name=data.get('tool_name', 'unknown'),
                    reason=data.get('reason', ''),
                    details={k: v for k, v in data.items()
                            if k not in ('tool_name', 'reason')}
                )
                if key not in self.fork_turns:
                    self.fork_turns[key] = ForkTurnInfo(fork, turn, ConversationStatus.ACTIVE)

                message = ConversationMessage(
                    fork=fork, turn=turn, role='permission_request',
                    content=f"Permission request: {request.tool_name}",
                    timestamp=timestamp,
                    metadata={'request': request, 'request_seq': request_seq}
                )
                self.messages.append(message)

                if self.on_message:
                    self.on_message(message)
                updated = True
            except json.JSONDecodeError:
                pass
        elif header.is_type(MessageType.MAX_REQUESTS_REACHED):
            key = ForkTurn(fork, turn)
            if is_conditional:
                # Conditional run terminal: record run-terminal marker on the
                # run's own card; never touch the primary snapshot or callbacks.
                maxreq_metadata = self._conditional_identity(scan_target, conditional_agent_name)
                maxreq_metadata.update({'run_terminal': True, 'status': 'error'})
                maxreq_message = ConversationMessage(
                    fork=fork, turn=turn, role='error',
                    content=f"Max requests reached for conditional agent {conditional_agent_name}",
                    timestamp=timestamp,
                    metadata=maxreq_metadata,
                )
                self.messages.append(maxreq_message)
                if self.on_message:
                    self.on_message(maxreq_message)
            else:
                if key not in self.fork_turns:
                    self.fork_turns[key] = ForkTurnInfo(fork, turn, ConversationStatus.ERROR)
                else:
                    self.fork_turns[key].status = ConversationStatus.ERROR
                updated = True
                if update_facts:
                    self._post_fact_transition(
                        IOTurnFact(fork, turn, IOFactKind.OUT, IOTurnStatus.COMPLETE_ERROR))
                if self.on_error:
                    self.on_error(fork, turn, f"Max requests reached at fork {fork}, turn {turn}")

        elif header.is_type(MessageType.CANCEL_TURN_ACK):
            # Acknowledgment that the cancel was received and processed.
            # Mark the turn CANCELLED and post the terminal CANCELLED fact
            # (primary stream only; conditional (0,0) facts never enter the
            # primary snapshot).
            if is_conditional:
                # Conditional streams never cancel a primary turn: record the
                # run-terminal marker on the run's own card and fire nothing.
                cancel_metadata = self._conditional_identity(scan_target, conditional_agent_name)
                cancel_metadata.update({'run_terminal': True, 'status': 'CANCELLED'})
                cancel_message = ConversationMessage(
                    fork=fork, turn=turn, role='error',
                    content=f"Conditional agent {conditional_agent_name} cancelled",
                    timestamp=timestamp,
                    metadata=cancel_metadata,
                )
                self.messages.append(cancel_message)
                if self.on_message:
                    self.on_message(cancel_message)
            else:
                key = ForkTurn(fork, turn)
                if key not in self.fork_turns:
                    self.fork_turns[key] = ForkTurnInfo(fork, turn, ConversationStatus.CANCELLED)
                else:
                    self.fork_turns[key].status = ConversationStatus.CANCELLED
                if update_facts:
                    self._post_fact_transition(
                        IOTurnFact(fork, turn, IOFactKind.OUT, IOTurnStatus.CANCELLED))
                if self.on_cancel_ack:
                    self.on_cancel_ack(fork, turn)

        elif header.is_type(MessageType.CONDITIONAL_REPORT):
            # Conditional agent's final report (via submit_report tool).
            # A CONDITIONAL_REPORT in the primary stream is never valid and
            # indicates on-disk corruption, so fail fast.
            if not is_conditional:
                raise ForkTurnStateError(
                    f"CONDITIONAL_REPORT encountered in primary output stream "
                    f"at fork {fork}, turn {turn}"
                )
            report_metadata = self._conditional_identity(scan_target, conditional_agent_name)
            report_metadata.update({
                'is_conditional_report': True,
                'run_terminal': True,
            })
            message = ConversationMessage(
                fork=fork, turn=turn, role='assistant',
                content=payload or "",
                timestamp=timestamp,
                metadata=report_metadata,
            )
            self.messages.append(message)
            if self.on_message:
                self.on_message(message)

        elif header.is_type(MessageType.SYSTEM_AGENT_STARTED):
            # System agents run in the primary agent's context; use the
            # fork/turn from the output file (which is the primary agent's
            # fork/turn for this turn).
            agent_type = header.system_agent_type or ''
            subdir = header.system_agent_subdir or ''

            # Deterministic same-role ordering: monotonic per (fork, turn, type).
            ord_key = (fork, turn, agent_type)
            prev = self._system_agent_ord.get(ord_key, 0)
            self._system_agent_ord[ord_key] = prev + 1
            system_agent_ord = self._system_agent_ord[ord_key]
            # Debug-build invariant: strictly monotonic per key (§4). The
            # counter is monotonic by construction; the assert documents the
            # guarantee and is active only under FLUXIMP_DEBUG_ASSERTS.
            if os.getenv('FLUXIMP_DEBUG_ASSERTS'):
                assert system_agent_ord == prev + 1

            message = ConversationMessage(
                fork=fork, turn=turn, role='system_agent',
                content=f"System Agent: {agent_type}",
                timestamp=timestamp,
                metadata={
                    'is_system_agent': True,
                    'system_agent_type': agent_type,
                    'system_agent_ord': system_agent_ord,
                    'system_agent_subdir': subdir,
                    'status': 'running',
                    'system_agent_start_time': timestamp.isoformat() if timestamp else None,
                }
            )
            self.messages.append(message)
            self._pending_system_agents[agent_type] = message
            if self.on_message:
                self.on_message(message)

        elif header.is_type(MessageType.SYSTEM_AGENT_STOPPED):
            agent_type = header.system_agent_type or ''
            success = header.system_agent_success
            message = self._pending_system_agents.pop(agent_type, None)
            if message:
                message.metadata['status'] = 'completed'
                message.metadata['success'] = bool(success) if success is not None else True
                message.metadata['system_agent_stop_time'] = timestamp.isoformat() if timestamp else None
                if success is False and payload:
                    message.metadata['error'] = payload.strip()
                # Scan the system agent's output files for provider, model, tokens
                subdir = message.metadata.get('system_agent_subdir', '')
                agent_metadata = self._scan_system_agent_output(subdir)
                if agent_metadata:
                    message.metadata.update(agent_metadata)
                if self.on_message_updated:
                    self.on_message_updated(message)
            else:
                if payload:
                    self.logger.warning(
                        f"SYSTEM_AGENT_STOPPED with no pending STARTED for agent "
                        f"'{agent_type}': {payload.strip()!r}"
                    )
                else:
                    self.logger.warning(
                        f"SYSTEM_AGENT_STOPPED with no pending STARTED for agent '{agent_type}'"
                    )
        return updated


    def _resolve_conditional_anchor(self, run_id: str) -> Tuple[int, int]:
        """Resolve the anchor for a conditional run (§7.6).

        The authoritative source is the run's first positioned input read:
        ``_parse_input_file`` resolves it from the 0.0.in.txt launch header
        timestamp (the latest primary user message whose timestamp is <= the
        launch; ties resolve to the user message; (0, -1) if none) and stamps
        it into ``self._conditional_anchors``. This method is a cache lookup
        with a legacy fallback for pre-refactor runs whose input stream
        predates the positioned input scan (latest primary position with a
        user panel).
        """
        if run_id in self._conditional_anchors:
            return self._conditional_anchors[run_id]
        anchor = (0, -1)
        for m in self.messages:
            if m.role == 'user' and not m.metadata.get('is_conditional', False):
                anchor = (m.fork, m.turn)
        self._conditional_anchors[run_id] = anchor
        return anchor

    def _conditional_identity(self, scan_target, conditional_agent_name: str) -> dict:
        """Build the identity metadata stamped on a conditional message."""
        identity = {
            'is_conditional': True,
            'conditional_agent_name': conditional_agent_name,
        }
        if scan_target is not None:
            identity['run_id'] = scan_target.run_id
            if scan_target.anchor_fork is not None:
                identity['anchor_fork'] = scan_target.anchor_fork
                identity['anchor_turn'] = scan_target.anchor_turn
        return identity


    def _scan_system_agent_output(self, subdir_name: str) -> dict:
        """Scan a system agent's output files to extract REQUEST/RESPONSE metadata.

        Reads the system agent's output files in its subdirectory to extract
        provider, model, token counts, and timing information from the
        system agent's own REQUEST and RESPONSE protocol messages.

        Returns:
            Dict with metadata from the system agent's REQUEST and RESPONSE headers.
        """
        if not subdir_name:
            return {}

        subdir_path = self.io_dir / subdir_name
        if not subdir_path.exists():
            self.logger.warning(f"System agent subdirectory not found: {subdir_path}")
            return {}

        protocol = FileIOUserProtocol(subdir_path)
        metadata = {}

        try:
            for out_file, fork, turn in protocol.scan_output_files():
                for header, payload, _ in protocol.read_new_output(out_file):
                    if header.is_request:
                        metadata.update(header.to_metadata_dict())
                    elif header.is_response:
                        metadata.update(header.to_metadata_dict())
        except Exception as e:
            self.logger.warning(f"Error scanning system agent output in {subdir_path}: {e}")

        return metadata

    def send_message(self, content: str, fork: int, turn: int) -> Tuple[int, int]:
        """Send a message to the agent at the specified fork and turn.

        Raises ValueError if the turn already has input (one message per turn).

        Returns the (fork, turn) where the message was sent.
        """
        self.logger.debug(f"Sending message: fork={fork}, turn={turn}")

        # Write the input file via protocol (which checks for existing files)
        self._protocol.send_message(content, fork, turn)

        # Record the IN fact immediately after the protocol write and before
        # constructing/dispatching the user message. Fork-mode ordering
        # invariant: the IN fact must exist before visibility is evaluated.
        self.fork_turn_state = self.fork_turn_state.with_fact(
            IOTurnFact(fork, turn, IOFactKind.IN, IOTurnStatus.NOT_APPLICABLE)
        ).reconcile_snapshot()
        if self.on_fork_turn_update:
            self.on_fork_turn_update()


        return fork, turn


    def cancel_current_turn(self) -> None:
        """Cancel the current turn, causing the agent to shut down.

        Writes a cancel_turn message to the current fork/turn input file. The agent
        detects this message, writes cancel_turn_ack, and shuts down. When the user
        presses Resume, the GUI restarts the agent process which performs mid-turn
        recovery and continues processing where it left off.
        """
        fork, turn = self.fork_turn_state.active_position
        self.logger.debug(f"cancel_current_turn: fork={fork}, turn={turn}")
        self._protocol.cancel_current_turn(fork, turn)

    def send_permission_response(self, fork: int, turn: int,
                                   approved: bool, explanation: str = ""):
        """Send a permission response to the agent."""
        self._protocol.send_permission_response(fork, turn, approved, explanation)

    def end_conversation(self):
        """Signal the end of the conversation.

        The .end file is created for the NEXT turn to prevent it from starting,
        not for the current turn which has already completed.
        """
        # The end file is created for the next turn after the active position.
        end = self.fork_turn_state.end_position()
        self.logger.debug(f"Creating end file for fork {end.fork}, turn {end.turn}")
        self._protocol.create_end_file(end.fork, end.turn)

    def clear_end_signal(self):
        """Remove the .end file to allow the conversation to continue."""
        # The end file is created for the next turn after the active position.
        end = self.fork_turn_state.end_position()
        self.logger.debug(f"Removing end file for fork {end.fork}, turn {end.turn}")
        self._protocol.remove_end_file(end.fork, end.turn)

    def attach_process(self, process: subprocess.Popen):
        """Attach a running agent process to this conversation.
        
        Used when resuming a conversation to start a new agent process
        without creating a new Conversation instance.
        """
        import wx
        with self._shutdown_lock:
            self._is_shutting_down = False
        self.process = process
        # File watching is handled by the centralized FileWatcher (gui.file_watcher)
        
        # Notify listeners that process was attached (use wx.CallAfter for UI thread safety)
        if self.on_process_attached:
            wx.CallAfter(self.on_process_attached)

    def wait_for_shutdown(self, timeout: float = 10.0) -> bool:
        """Wait for shutdown to complete.
        
        Returns True if shutdown completed, False if timeout.
        """
        start = time.time()
        while time.time() - start < timeout:
            with self._shutdown_lock:
                if not self._is_shutting_down and self.process is None:
                    return True
            time.sleep(0.1)
        return False

    def terminate(self):
        """Terminate the agent process gracefully."""
        with self._shutdown_lock:
            if self._is_shutting_down or self.process is None:
                return  # Already shutting down or no process
            self._is_shutting_down = True

        # Notify listeners that shutdown is starting (disable Send buttons in UI)
        if self.on_shutdown_starting:
            import wx
            wx.CallAfter(self.on_shutdown_starting)

        self.logger.info(f"Terminating conversation {self.conversation_id}")
        # File watching is handled by the centralized FileWatcher (gui.file_watcher)
        if self.process:
            def pre_terminate():
                self.logger.debug("Creating end file for graceful shutdown")
                self.end_conversation()

            def cleanup():
                self.process = None
                with self._shutdown_lock:
                    self._is_shutting_down = False
                self.logger.info("Process terminated")
                # Notify listeners that shutdown is complete (re-enable Send buttons in UI)
                if self.on_shutdown_complete:
                    import wx
                    wx.CallAfter(self.on_shutdown_complete)

            terminate_process(
                self.process,
                timeout_graceful=5.0,
                timeout_terminate=2.0,
                timeout_kill=2.0,
                pre_terminate_hook=pre_terminate,
                cleanup_hook=cleanup,
            )

    def start_agent_process(self, agent_type: str, provider: str, model: str,
                             max_requests: int = 100000,
                             persist_checkpoints: bool = False) -> subprocess.Popen:
        """Start the main.py subprocess for this conversation."""
        # Workspace-level test-runner plugin dir, resolved at launch time
        # (not captured at construction) so Check-and-Save changes in the
        # workspace Settings tab apply to every conversation's next agent
        # launch/restart. None omits --test-runner-plugin-dir (run_tests
        # is dropped agent-side in that case).
        plugin_dir = self.workspace.resolve_plugin_dir() if self.workspace is not None else None
        self.logger.debug(f"start_agent_process: test-runner plugin dir resolved to {plugin_dir}")
        cmd = build_agent_command(
            agent_type=agent_type,
            io_dir=self.io_dir,
            working_dir=self.working_dir,
            read_root_dir=self.read_root_dir,
            write_file=self.write_file,
            provider=provider,
            model=model,
            max_requests=max_requests,
            input_timeout=86400,
            permission_timeout=86400,
            plugin_dir=plugin_dir,
            persist_checkpoints=persist_checkpoints,
            agent_configs=self.agent_configs if self.agent_configs else None,
        )

        stdout_file = self.io_dir / "agent_stdout.log"
        return launch_agent_process(
            cmd,
            stdout=open(stdout_file, 'a', encoding='utf-8'),
            stderr=subprocess.STDOUT,
        )
    def restart_with_agent_type(self, agent_type: str):
        """Terminate current process and restart with new agent type."""
        self.agent_type = agent_type
        self.terminate()
        process = self.start_agent_process(agent_type, self.current_provider, self.current_model, 100000, persist_checkpoints=True)
        self.attach_process(process)

    def _save_config(self):
        """Save configuration to disk."""
        config_path = self.conversation_dir / ".config.json"
        try:
            config_data = {
                'agent_type': self.agent_type,
                'agent_configs': self.agent_configs,
                'workflow_name': self.workflow_name,
                'current_provider': self.current_provider,
                'current_model': self.current_model,
                'provider_models': self.provider_models
            }
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save config: {e}")

    def _load_config(self):
        """Load configuration from disk."""
        config_path = self.conversation_dir / ".config.json"
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                self.agent_type = config_data.get('agent_type', self.agent_type)
                self.agent_configs = config_data.get('agent_configs', self.agent_configs)
                self.workflow_name = config_data.get('workflow_name', self.workflow_name)
                self.current_provider = config_data.get('current_provider', '')
                self.current_model = config_data.get('current_model', '')
                self.provider_models = config_data.get('provider_models', {})
            except Exception as e:
                self.logger.warning(f"Failed to load config: {e}")

    def get_fork_turns(self) -> List[ForkTurnInfo]:
        """Get all fork/turn information sorted by fork then turn."""
        return sorted(self.fork_turns.values(), key=lambda x: (x.fork, x.turn))

    def get_messages_for_turn(self, fork: int, turn: int) -> List[ConversationMessage]:
        """Get all messages for a specific fork/turn."""
        return [m for m in self.messages if m.fork == fork and m.turn == turn]
    

    def _post_fact_transition(self, fact: IOTurnFact) -> None:
        """Package and post a single fact observation to the GUI thread.

        The file-watcher performs byte parsing on its own thread, but every
        fork_turn_state transition is applied on the GUI thread (single-writer
        rule). During a reload (§8.5) facts are buffered on the conversation and
        applied once per watcher tick by the tab's batch replay — one
        wx.CallAfter + one right-panel grid rebuild per fact flooded the GUI
        queue and starved paint events. Tests call _apply_fact_transition
        directly.
        """
        if getattr(self, 'reload_in_progress', False):
            buf = getattr(self, '_reload_fact_buffer', None)
            if buf is not None:
                buf.append(fact)
                return
        import wx
        wx.CallAfter(self._apply_fact_transition, fact)

    def _apply_fact_transition(self, fact: IOTurnFact) -> None:
        """Apply a single fact transition on the GUI thread.

        This is the only place self.fork_turn_state is rebound outside of
        load. OUT facts use with_out_upsert (maintaining one OUT per position
        and the status-transition machine); other kinds use with_fact. Every
        transition is followed by reconcile_snapshot() so ABANDONED can be
        assigned after supersession.
        """
        if fact.kind is IOFactKind.OUT:
            new_state = self.fork_turn_state.with_out_upsert(
                fact.fork, fact.turn, fact.status)
        else:
            new_state = self.fork_turn_state.with_fact(fact)
        self.fork_turn_state = new_state.reconcile_snapshot()
        # During a reload the right-panel grid is not notified per fact
        # (one grid rebuild per fact flooded the GUI queue and starved
        # paint events, §8.5); the settled notification fires once at
        # finalization (§7.4 step 9).
        if self.on_fork_turn_update and not getattr(
                self, 'reload_in_progress', False):
            self.on_fork_turn_update()

    def _tool_message_key(self, fork: int, turn: int, request_seq: int,
                          is_conditional: bool, conditional_agent_name: str):
        """Build the ToolMessageKey for a pending tool message, or None.

        Primary agent messages use stream='primary' with agent_name=None;
        conditional agent messages use stream='conditional' with the agent
        name so that distinct conditional agents (all at fork=0, turn=0) do
        not collide.
        """
        if request_seq <= 0:
            return None
        if is_conditional:
            return ToolMessageKey("conditional", conditional_agent_name,
                                  fork, turn, request_seq)
        return ToolMessageKey("primary", None, fork, turn, request_seq)

