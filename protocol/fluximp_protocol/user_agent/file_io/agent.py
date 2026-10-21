#!/usr/bin/env python3
"""
File I/O implementation of the agent side of the user-agent protocol.

FileIOAgentProtocol reads input files and writes output files,
counterpart to FileIOUserProtocol on the user side.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple, List

from fluximp_protocol.user_agent.base import (
    AgentProtocol,
    RequestLifecycleHooks,
    ToolPermissionService,
    PermissionResult,
    PermissionServiceProtocolError,
    TurnCancelled,
)
from fluximp_protocol.user_agent.file_io.base import FileIOProtocolBase, FileIOInputInfo
from fluximp_protocol.user_agent.file_io._permission_scan import _OutputPermissionScan
from fluximp_protocol.user_agent.file_io.wire_format import (
    write_request_header,
    write_response,
    write_payload,
    write_model_message_to_user,
    read_exact,
    read_payload,
    read_streaming_header,
)
from fluximp_protocol.user_agent.fork_turn_state import (
    ForkTurn,
    IOFactKind,
    IOTurnStatus,
    ForkTurnState,
)
from fluximp_protocol.user_agent.file_io.defs import (
    MessageType,
    parse_bool,
    _read_bounded_header_line,
    _validate_iso_timestamp,
    expected_field_count,
    extract_fork_turn_from_filename_strict,
    IOFilenameError,
    monotonic_stamp,
    parse_utc_timestamp,
    utc_now_iso,
)
# ============================================================================
# FileIOAgentProtocol
# =============================================================================

class FileIOAgentProtocol(AgentProtocol, FileIOProtocolBase, RequestLifecycleHooks, ToolPermissionService):
    """File I/O Protocol handler for the agent side of the user-agent protocol.
    
    Owns file handles, manages the wire format, and provides check/get
    methods for input polling. Implements AgentProtocol for the IO-agnostic
    interface, FileIOProtocolBase for shared file-IO path resolution,
    RequestLifecycleHooks so it can be passed directly to create_agent()
    for request header writing, and ToolPermissionService for handling
    permission requests via the same file I/O streams.
    
    The coordinator owns all loops - this class provides non-blocking
    check methods and write methods that the coordinator calls.
    """

    def __init__(self, io_path: Path, agent_type: str, provider: str,
                 requested_model: str, input_timeout: float = 120.0,
                 permission_timeout: float = 120.0):
        """Initialize the FileIOAgentProtocol.
        
        Args:
            io_path: Directory containing the conversation files
            agent_type: The agent type (e.g., 'coder', 'scrutinator')
            provider: The model provider name
            requested_model: The model name requested
            input_timeout: Timeout in seconds for waiting on input files
            permission_timeout: Timeout in seconds for waiting on permission responses
        """
        FileIOProtocolBase.__init__(self, io_path)
        self.agent_type = agent_type
        self.provider = provider
        self.requested_model = requested_model
        self.input_timeout = input_timeout
        self.permission_timeout = permission_timeout
        self.seen_files = set()
        self._input_handle = None
        self._output_handle = None
        self._output_file = None
        # Permission state
        self._is_mid_turn_recovery = False
        self._last_request_timestamp: Optional[datetime] = None
        # Incremental permission-scan cache over the current output file
        # (see _permission_scan._OutputPermissionScan).
        self._output_scan = None

    # --- RequestLifecycleHooks ---
    
    def on_request_start(self, start_time: datetime) -> None:
        """Write request header to output file when API call starts."""
        if self._output_handle:
            write_request_header(self._output_handle, start_time.isoformat(),
                                 self.agent_type, self.provider, self.requested_model,
                                 file_id=str(self._output_file))

    # --- Input: check/get (non-blocking, no polling loop) ---

    def check_for_new_input(self, fork: int, turn: int) -> Optional[FileIOInputInfo]:
        """Non-blocking scan for the next input file.
        
        Looks for files matching {f}.{t}.in.txt where f >= current fork
        and turn >= current turn (for same fork).
        
        Args:
            fork: Current fork number
            turn: Current turn number
            
        Returns:
            FileIOInputInfo if new input found, None if nothing new.
            Raises Exception if multiple new files detected (protocol violation).
        """
        # Check for end file at current position (live processing)
        current_end = self.io_path / f"{fork}.{turn}.end"
        if current_end.is_file():
            return FileIOInputInfo(
                input_file=None, output_file=None,
                end_file=current_end, fork=fork, turn=turn, is_end_signal=True
            )

        # Scan directory for new input files
        candidates = []
        
        for path in self.io_path.glob("*.in.txt"):
            if not path.is_file():
                # A directory/special file shaped like a primary filename is not
                # a valid input stream; skip it exactly like the snapshot
                # scanner (_scan_io_facts_raw) so it can never reach open() and
                # fail there with a bare IsADirectoryError/PermissionError.
                continue
            if path in self.seen_files:
                continue

            # Parse filename: {fork}.{turn}.in.txt. Unrelated files (not
            # coordinate-shaped, e.g. "notes.in.txt") are skipped.
            # Coordinate-shaped but malformed names -- leading zeros, negatives,
            # extra components (e.g. "0.1.2.in.txt") -- are kept so the strict
            # parser rejects them and fails fast rather than silently reading
            # at the wrong (fork, turn).
            parts = path.name.split('.')
            if not (len(parts) >= 4 and parts[-2] == 'in' and parts[-1] == 'txt'):
                continue
            try:
                file_fork, file_turn = extract_fork_turn_from_filename_strict(path.name)
            except IOFilenameError as e:
                raise Exception(
                    f"ERROR: malformed input filename: {path.name}"
                ) from e

            # Only consider files with fork >= current fork
            if file_fork < fork:
                continue

            # For current fork, only consider turns >= current turn
            if file_fork == fork and file_turn < turn:
                continue

            candidates.append((file_fork, file_turn, path))
        
        if not candidates:
            return None
        
        # If multiple candidates found, check if the current turn's input
        # file is among them. If newer input files also exist, the user has
        # abandoned the current turn (e.g., stopped the agent mid-turn then
        # sent a new message). Mark the current turn's input as seen
        # (abandoned) and proceed with the newer input.
        if len(candidates) > 1:
            current_candidates = [(f, t, p) for f, t, p in candidates if f == fork and t == turn]
            newer_candidates = [(f, t, p) for f, t, p in candidates if not (f == fork and t == turn)]
            
            if current_candidates and newer_candidates:
                for f, t, p in current_candidates:
                    self.seen_files.add(p)
                    print(f"Abandoning mid-turn recovery for fork {f} turn {t} - newer input found")
                candidates = newer_candidates
            
            if len(candidates) > 1:
                candidate_strs = [f"{f}.{t}.in.txt" for f, t, _ in sorted(candidates)]
                error_msg = f"ERROR: Multiple new input files detected: {candidate_strs}"
                print(error_msg)
                raise Exception(error_msg)
        
        # Sort by (fork, turn) and return the first one
        candidates.sort(key=lambda x: (x[0], x[1]))
        file_fork, file_turn, input_file = candidates[0]
        output_file = self.io_path / f"{file_fork}.{file_turn}.out.txt"
        end_file = self.io_path / f"{file_fork}.{file_turn}.end"
        return FileIOInputInfo(
            input_file=input_file, output_file=output_file,
            end_file=end_file, fork=file_fork, turn=file_turn
        )

    def check_cancel_turn(self) -> bool:
        """Non-blocking check for cancel_turn on input handle.

        Returns True if cancel_turn found. Raises Exception if a
        non-cancel message is found on the input stream (protocol error).
        """
        pos = self._input_handle.tell()
        # Bounded header read (see read_streaming_header): a header fragment
        # exceeding MAX_HEADER_BYTES without a terminating newline is
        # corruption and fails fast in _read_bounded_header_line instead of
        # buffering the rest of the file via an unbounded readline().
        line, _header_len = _read_bounded_header_line(self._input_handle)
        if line is None:
            # Nothing appended yet (EOF), or a header without its terminating
            # newline is still being written by the peer. Seek back so we do
            # NOT consume a partial header and return (this is a non-blocking
            # check; a later poll will see the completed line once its
            # trailing newline has been written).
            self._input_handle.seek(pos)
            return False

        # Realign exactly after the header line: the bounded read may have
        # pulled bytes beyond the newline (the next record's bytes) into the
        # handle, so reset the logical position (mirrors parse_streamed_records).
        self._input_handle.seek(pos + _header_len)

        line_str = line.decode('utf-8').rstrip('\r\n')
        parts = line_str.split(',')

        msg_type = parts[1] if len(parts) > 1 else ""
        expected = expected_field_count(msg_type)

        # The header is newline-terminated, so it is complete. A field count
        # that differs from what the type requires is corruption -- fail fast.
        if len(parts) != expected:
            raise Exception(
                f"cancel_turn handler: corrupt stream, {msg_type!r} header has "
                f"{len(parts)} fields, expected {expected}: {line_str!r}"
            )

        # The timestamp field must be a valid ISO-8601 timestamp, exactly as
        # read_streaming_header and the two record parsers enforce. A corrupt
        # timestamp means the stream is corrupt and must fail fast here -- an
        # invalid cancel_turn record must never cancel the turn.
        _validate_iso_timestamp(
            parts[2], f"cancel_turn handler: corrupt stream, {msg_type!r} header"
        )

        # The size field must be a valid non-negative int; otherwise the stream
        # is corrupt -- fail fast.
        try:
            payload_size = int(parts[0])
        except ValueError:
            raise Exception(
                f"cancel_turn handler: corrupt stream, payload_size "
                f"{parts[0]!r} is not an integer: {line_str!r}"
            ) from None
        if payload_size < 0:
            raise Exception(
                f"cancel_turn handler: corrupt stream, negative payload_size "
                f"{payload_size}: {line_str!r}"
            )
        if msg_type == 'tool_permission_response':
            # A tool_permission_response at a cancel-check point is legitimate
            # only while a matching permission request is still pending: during
            # a mid-turn recovery such a response was intentionally left in the
            # input stream for request_permission() to consume on resume (the
            # recovery drain only consumes responses for *acked* requests). It
            # is not a cancel -- leave it unconsumed and report "not a cancel"
            # so request_permission() can read it later. If no permission
            # request is pending, the response is orphaned/corrupt and we fail
            # fast rather than silently dropping it.
            if self._has_any_pending_permission_request():
                self._input_handle.seek(pos)
                return False
            self._input_handle.seek(pos)
            raise Exception(
                f"cancel_turn message header expected but got {msg_type} "
                f"(no pending permission request)"
            )
        if msg_type != 'cancel_turn':
            self._input_handle.seek(pos)
            raise Exception(
                f"cancel_turn message header expected but got {msg_type}"
            )
        if payload_size != 0:
            self._input_handle.seek(pos)
            raise Exception(
                f"cancel_turn not expected to have payload, but size is {payload_size}"
            )

        print("Turn cancelled by user, shutting down...")
        return True

    def read_user_message(self) -> Tuple[str, str]:
        """Read user_message payload from input handle.

        Returns:
            Tuple of (msg_type, content) where msg_type should be 'user_message'

        Raises:
            Exception: If the next message on the input stream is not a
                user_message. The stream must fail fast on a corrupt or
                mispositioned message rather than silently treating a
                cancel_turn or permission response as user content.
        """
        msg_type, content = read_payload(self._input_handle, timeout=self.input_timeout)
        if msg_type != MessageType.USER_MESSAGE.value:
            raise Exception(
                f"user_message message expected; invalid filesystem input "
                f"stream (got {msg_type!r})"
            )
        return msg_type, content

    # --- Handle management ---

    def open_handles(self, input_file: Path, output_file: Path):
        """Open file handles for the current turn.
        
        Args:
            input_file: Path to the input file
            output_file: Path to the output file
        """
        self._input_handle = open(input_file, 'rb')
        self._output_handle = open(output_file, 'ab')
        self._output_file = output_file
        # The incremental permission-scan cache is per output file: a new
        # turn (or a recovery re-open of an already populated output file,
        # which must see the whole-file balance) starts from a fresh cache
        # at byte 0.
        self._output_scan = None

    def close_handles(self):
        """Close file handles for the current turn."""
        if self._input_handle:
            self._input_handle.close()
        if self._output_handle:
            self._output_handle.close()
        self._input_handle = None
        self._output_handle = None
        self._output_file = None
        self._last_request_timestamp = None
        self._is_mid_turn_recovery = False
        # The incremental permission-scan cache is per output file
        self._output_scan = None

    def set_mid_turn_recovery(self, is_mid_turn_recovery: bool) -> None:
        """Set whether the protocol is operating in a mid-turn recovery context.

        During mid-turn recovery, ``request_permission()`` may replay a
        prior-approval result for a tool that was already granted and
        acknowledged during the original run (see ``_has_prior_approval``).
        Outside of an actual recovery this flag must be False so a fresh,
        in-turn re-request is never granted without a new user decision.
        """
        self._is_mid_turn_recovery = bool(is_mid_turn_recovery)

    # --- ToolPermissionService implementation ---

    def has_pending_request(self, tool_name: str) -> bool:
        """Check if there is already a pending permission request for this tool.

        Delegates the output-file scan to :meth:`_update_output_permission_scan`,
        which folds each record into a cached state exactly once, so the scan
        backing every ``request_permission`` call parses only newly appended
        records instead of re-reading and re-parsing the whole output file.

        Args:
            tool_name: Name of the tool to check for pending request

        Returns:
            True if a pending request exists for this tool (request found but
            no ack after it), False otherwise

        Raises:
            PermissionServiceProtocolError: On corruption detected in any
                scanned record (see _permission_scan._OutputPermissionScan.fold).
        """
        print(f"DEBUG has_pending_request")
        if not self._output_file or not self._output_handle:
            print(f"DEBUG has_pending_request not self._output_file")
            return False

        scan = self._update_output_permission_scan()

        # Return True if there is any outstanding (un-acked) request for
        # this tool.
        if scan.pending_by_tool.get(tool_name, 0) > 0:
            # The globally last request must be for the queried tool: the
            # protocol only ever has one request in flight, so an outstanding
            # request here plus a later request for another tool is corrupt
            # interleaving. This guard is actually reachable because fold()
            # records EVERY request's tool name, not (as in the pre-cache
            # version) only the queried tool's -- which had made the check
            # dead code.
            if scan.last_request_tool_name != tool_name:
                raise PermissionServiceProtocolError(
                    f"corrupt output file: last permission request was for "
                    f"{scan.last_request_tool_name!r}, not requested tool "
                    f"{tool_name!r}"
                )
            self._last_request_timestamp = scan.last_request_timestamp
            return True

        return False

    def _update_output_permission_scan(self) -> _OutputPermissionScan:
        """Fold all complete output-file records appended since the last scan
        into the incremental permission-scan cache and return it.

        See the ``_permission_scan`` module. All agent-side output writers
        flush (or close) before these scans run, so ``advance`` only parses
        genuinely new appends; ``open_handles`` resets the cache so a
        mid-turn-recovery scan starts from byte 0 and sees the whole-file
        balance the recovery drain and prior-approval checks rely on. A torn
        trailing record (a prior process died mid-write) fails fast in
        ``advance`` instead of silently freezing the cache at the torn
        position.
        """
        if self._output_scan is None:
            self._output_scan = _OutputPermissionScan()
        self._output_scan.advance(self._output_file)
        return self._output_scan

    def _write_permission_request(self, tool_name: str, request_data: dict) -> None:
        """Write a tool_permission_request to the output stream.

        The matching tool_permission_response_ack is written by
        request_permission() itself after the response is consumed, so every
        request written here is a real request that consumed (or will consume)
        a tool_permission_response from the input stream.
        """
        # Build permission request payload. The canonical ``tool_name`` argument
        # wins over any ``tool_name`` key inside ``request_data``: the output
        # stream must agree with the tool name used by the pending/ack
        # bookkeeping, otherwise the request/ack balance would desynchronize.
        permission_request = {
            **request_data,
            "tool_name": tool_name,
        }

        # Write permission request to output handle
        # Format: <payload_bytes>,tool_permission_request,<timestamp>\n<payload>
        stamp = monotonic_stamp(str(self._output_file))
        self._last_request_timestamp = parse_utc_timestamp(stamp)
        payload = json.dumps(permission_request)
        if not payload.endswith('\n'):
            payload += '\n'
        header = f"{len(payload.encode('utf-8'))},tool_permission_request,{stamp}\n"
        content = header + payload

        # Write to output handle (already opened by caller)
        self._output_handle.write(content.encode('utf-8'))
        self._output_handle.flush()

    def request_permission(
        self,
        tool_name: str,
        request_data: dict
    ) -> PermissionResult:
        """Request permission for a tool execution.

        Writes a tool_permission_request to the output stream and waits for a
        tool_permission_response from the input stream, then writes the
        matching tool_permission_response_ack before returning. The ack is
        owned here, not by the caller.

        During mid-turn recovery, if there is no pending request and the most
        recent acknowledged decision for ``tool_name`` was an approval, this
        returns a prior-approval result without writing anything: the user has
        already granted this exact tool earlier in the turn, so replaying the
        checkpoint does not prompt again or add new request/ack records.

        Args:
            tool_name: Name of the tool requesting permission
            request_data: Dictionary containing tool-specific request data
                          (e.g., script_path, args, timeout, reason)

        Returns:
            PermissionResult with approval status and user explanation

        Raises:
            TimeoutError: If no permission response received within timeout.
                On timeout the input stream is restored to the position
                before this record (permission-response header included), so
                a caller that handles the timeout and retries re-reads the
                whole record instead of resuming mid-payload.
        """
        print("DEBUG FileIOAgentProtocol.request_permission")

        # Check if there is already a pending request for this tool (resume scenario)
        if self.has_pending_request(tool_name):
            print(f"DEBUG Found existing pending permission request for {tool_name}. Resuming wait...")
        elif self._is_mid_turn_recovery and self._has_prior_approval(tool_name):
            print(f"DEBUG Prior approval found for {tool_name}; replaying the already-granted permission")
            # The user already approved and the approval was acknowledged in
            # the original run. No new request/response was consumed, so
            # nothing is written here: the output stream already contains the
            # balanced request/ack pair and its ack already contributed to the
            # recovery drain count.
            return PermissionResult(
                approved=True,
                user_explanation=(
                    "Permission previously granted and acknowledged during "
                    "the original run (mid-turn recovery prior approval)."
                ),
            )
        else:
            self._write_permission_request(tool_name, request_data)
            print(f"DEBUG Permission request written for {tool_name}. Waiting for response...")
        
        # Wait for permission response from input handle
        # The client writes the response to the same input file
        # Format: <payload_bytes>,tool_permission_response,<timestamp>,<approved>\n<payload>
        # Use a single monotonic deadline (Option B) shared across the header
        # and payload waits, so the total wait cannot exceed permission_timeout
        # and the budget is robust against wall-clock jumps (unlike time.time()).
        wait_deadline = time.monotonic() + self.permission_timeout

        def _remaining_budget() -> float:
            return max(0.0, wait_deadline - time.monotonic())
        while True:
            # Track the stream position before this record's header so the
            # payload-timeout handlers below can restore the stream to
            # "this permission response not consumed" -- the whole-record
            # convention of read_exact's docstring and the drain helpers (a
            # retry then re-reads the entire record instead of resuming after
            # the consumed header, which would misparse payload bytes as a
            # header).
            record_start = self._input_handle.tell()
            # Block until a complete, valid header line is available on the
            # input handle. read_streaming_header seeks back on partial writes
            # (no terminating newline yet) and waits with the remaining time
            # budget, and fails fast on a corrupt stream (wrong field count, or
            # an unparseable/negative payload size).
            parts = read_streaming_header(self._input_handle, timeout=_remaining_budget())
            msg_type = parts[1]
            payload_size = int(parts[0])

            if msg_type == 'cancel_turn':
                # A cancel_turn must carry no payload; a nonzero size means the
                # stream is corrupt -- fail fast before writing the cancel ack.
                if payload_size != 0:
                    raise ValueError(
                        f"corrupt stream: cancel_turn not expected to have "
                        f"payload, but size is {payload_size}"
                    )
                # Cancel received while waiting for permission response
                self.write_cancel_ack()
                raise TurnCancelled("Turn cancelled by user during permission wait")
            if msg_type != 'tool_permission_response':
                raise Exception(
                    "tool_permission_response message expected; invalid filesystem input stream"
                )

            # Validate that the response timestamp is well-formed; a malformed
            # one means the stream is corrupt -- fail fast like the rest of the
            # Option 2 header validation.
            try:
                response_timestamp = parse_utc_timestamp(parts[2])
            except ValueError:
                raise ValueError(
                    f"corrupt stream: tool_permission_response timestamp "
                    f"{parts[2]!r} is not an ISO timestamp"
                ) from None
            # A malformed approved field means the stream is corrupt -- fail
            # fast instead of silently coercing it to a denial (False).
            approved_value = parse_bool(parts[3])
            if approved_value is None:
                raise ValueError(
                    f"corrupt stream: tool_permission_response approved field "
                    f"{parts[3]!r} is not a valid boolean"
                )
            approved = approved_value

            # Validate that response timestamp is AFTER the request timestamp
            # This prevents stale permission responses from being consumed
            # Only apply the stale-skip when a real request timestamp exists;
            # comparing a datetime against None would raise a bare TypeError
            # instead of failing fast cleanly as a corrupt-stream error.
            if (self._last_request_timestamp is not None
                    and response_timestamp < self._last_request_timestamp):
                print(f"DEBUG Skipping stale permission response (response timestamp {response_timestamp} < request timestamp {self._last_request_timestamp})")
                # Skip this stale response and continue reading. Recompute the
                # remaining budget immediately before the wait so the payload
                # read can't out-live the intended timeout.
                try:
                    if payload_size > 0:
                        read_exact(
                            self._input_handle, payload_size,
                            timeout=_remaining_budget(),
                        )
                except TimeoutError:
                    # Stale records are normally complete on disk, but if
                    # this wait does time out, do not leave the stream with
                    # the record's header consumed (read_exact seeks back
                    # only to the payload start). Seek back to the record
                    # start like the fresh-response path below and the drain
                    # helpers, so a retry re-reads the whole stale record.
                    self._input_handle.seek(record_start)
                    raise
                continue

            # Read payload (streaming: wait until the full byte count arrives).
            # Recompute the remaining budget immediately before the wait so the
            # payload read can't out-live the intended timeout.
            try:
                payload_bytes = read_exact(
                    self._input_handle, payload_size,
                    timeout=_remaining_budget(),
                )
            except TimeoutError:
                # The header of this response record was already consumed
                # above, and read_exact seeks back only to its own start (the
                # payload start). Restore the stream to the position before
                # this record's header -- the whole-record convention of
                # read_exact's docstring and the drain helpers -- so a caller
                # that handles the timeout and retries re-reads the entire
                # permission response (header included) instead of resuming
                # mid-payload, which would misparse the explanation bytes as
                # a header.
                self._input_handle.seek(record_start)
                raise
            user_explanation = payload_bytes.decode('utf-8')

            print(f"Permission response: approved={approved}, explanation={user_explanation[:50]}...")

            # A real response was consumed: write the matching ack before
            # returning so the request/ack pairing is owned here, not by the
            # caller. (The prior-approval short-circuit above returns without
            # consuming a response and therefore without an ack.)
            result = PermissionResult(approved=approved, user_explanation=user_explanation)
            self._write_permission_response_ack(tool_name, approved, user_explanation)
            return result

    def _write_permission_response_ack(self, tool_name: str, approved: bool, explanation: str):
        """Write a permission response acknowledgment to the output stream.

        Called from request_permission() immediately after a real
        tool_permission_response has been consumed from the input stream, so
        the request/ack pairing is owned entirely by request_permission().
        Each ack pairs with the request written earlier in the same
        request_permission() call.

        Format: <payload_size>,tool_permission_response_ack,<timestamp>\n<payload>

        Args:
            tool_name: Name of the tool for the permission request
            approved: Whether permission was granted
            explanation: User explanation for the decision
        """
        timestamp = monotonic_stamp(str(self._output_file))

        ack_payload = {
            "tool_name": tool_name,
            "approved": approved,
            "explanation": explanation
        }
        payload = json.dumps(ack_payload)
        if not payload.endswith('\n'):
            payload += '\n'

        header = f"{len(payload.encode('utf-8'))},tool_permission_response_ack,{timestamp}\n"
        content = header + payload

        self._output_handle.write(content.encode('utf-8'))
        self._output_handle.flush()

        print(f"DEBUG write_permission_response_ack: {tool_name}, approved={approved}")

    def drain_stale_permission_responses(self):
        """Drain stale tool_permission_response messages from the input stream.

        During mid-turn recovery, the input file is re-opened from the beginning.
        If a run* tool was executed during the original run, its
        tool_permission_response is still in the input file after read_user_message().
        This method reads and discards those stale responses so that
        check_cancel_turn() won't crash when it encounters them.

        Counts acknowledged permission requests in the output file (each ack
        corresponds to a consumed response from the original run) and drains
        that many tool_permission_response messages from the input stream.
        Unconsumed responses (for pending unacked tools) are left in place.
        """
        acked_count = self._count_acked_permission_requests()
        if acked_count == 0:
            return
        drained = 0
        while drained < acked_count:
            pos = self._input_handle.tell()
            # Bounded header read (see read_streaming_header): a header
            # fragment exceeding MAX_HEADER_BYTES without a terminating
            # newline is corruption and fails fast in
            # _read_bounded_header_line instead of buffering the rest of the
            # file via an unbounded readline().
            line, _header_len = _read_bounded_header_line(self._input_handle)
            if line is None:
                # No terminating newline within the bound -- either EOF or a
                # header still being written by the peer: seek back, leave
                # the partial header unconsumed and break.
                self._input_handle.seek(pos)
                break
            # Realign exactly after the header line: the bounded read may
            # have pulled bytes beyond the newline (the payload) into the
            # handle, so reset the logical position before the read_exact
            # payload wait below (mirrors parse_streamed_records).
            self._input_handle.seek(pos + _header_len)
            line_str = line.decode('utf-8').rstrip('\r\n')
            parts = line_str.split(',')
            msg_type = parts[1] if len(parts) > 1 else ""
            expected = expected_field_count(msg_type)
            # The header is newline-terminated, so it is complete. A field
            # count that differs from what the type requires is corruption --
            # fail fast.
            if len(parts) != expected:
                raise Exception(
                    f"drain_stale_permission_responses: corrupt stream, "
                    f"{msg_type!r} header has {len(parts)} fields, expected "
                    f"{expected}: {line_str!r}"
                )
            # The timestamp field must be a valid ISO-8601 timestamp, exactly
            # as read_streaming_header and the two record parsers enforce; a
            # corrupt one means the stream is corrupt and must fail fast rather
            # than being silently discarded here.
            _validate_iso_timestamp(
                parts[2],
                f"drain_stale_permission_responses: corrupt stream, "
                f"{msg_type!r} header",
            )
            if msg_type != 'tool_permission_response':
                # Not a permission response -- seek back and let check_cancel_turn() handle it
                self._input_handle.seek(pos)
                break
            # The size field must be a valid non-negative int; otherwise the
            # stream is corrupt -- fail fast.
            try:
                payload_size = int(parts[0])
            except ValueError:
                raise Exception(
                    f"drain_stale_permission_responses: corrupt stream, "
                    f"payload_size {parts[0]!r} is not an integer: {line_str!r}"
                ) from None
            if payload_size < 0:
                raise Exception(
                    f"drain_stale_permission_responses: corrupt stream, "
                    f"negative payload_size {payload_size}: {line_str!r}"
                )
            if payload_size > 0:
                try:
                    read_exact(self._input_handle, payload_size, timeout=self.input_timeout)
                except TimeoutError:
                    # Don't leave the stream mid-payload -- restore to the
                    # position before this header so callers see the same state.
                    self._input_handle.seek(pos)
                    break
            drained += 1
        print(f"DEBUG Drained {drained} stale permission responses (acked_count={acked_count})")
        if drained < acked_count:
            raise PermissionServiceProtocolError(
                f"corrupt input stream: expected {acked_count} "
                f"tool_permission_response records but only drained {drained}"
            )

    def drain_stale_cancel_turn(self):
        """Drain stale cancel_turn messages from the input stream.

        During mid-turn recovery after a cancel, the input file is re-opened
        from the beginning. The stale cancel_turn message is still in the file.
        This method reads and discards any cancel_turn messages so that
        check_cancel_turn() won't cancel again immediately after recovery.
        """
        while True:
            pos = self._input_handle.tell()
            # Bounded header read (see read_streaming_header): a header
            # fragment exceeding MAX_HEADER_BYTES without a terminating
            # newline is corruption and fails fast in
            # _read_bounded_header_line instead of buffering the rest of the
            # file via an unbounded readline().
            line, _header_len = _read_bounded_header_line(self._input_handle)
            if line is None:
                # No terminating newline within the bound -- either EOF or a
                # header still being written by the peer: seek back, leave
                # the partial header unconsumed and break.
                self._input_handle.seek(pos)
                break
            # Realign exactly after the header line: the bounded read may
            # have pulled bytes beyond the newline (the payload) into the
            # handle, so reset the logical position before the read_exact
            # payload wait below (mirrors parse_streamed_records).
            self._input_handle.seek(pos + _header_len)
            line_str = line.decode('utf-8').rstrip('\r\n')
            parts = line_str.split(',')
            msg_type = parts[1] if len(parts) > 1 else ""
            expected = expected_field_count(msg_type)
            # The header is newline-terminated, so it is complete. A field
            # count that differs from what the type requires is corruption --
            # fail fast.
            if len(parts) != expected:
                raise Exception(
                    f"drain_stale_cancel_turn: corrupt stream, {msg_type!r} header "
                    f"has {len(parts)} fields, expected {expected}: {line_str!r}"
                )
            # The timestamp field must be a valid ISO-8601 timestamp, exactly
            # as read_streaming_header and the two record parsers enforce; a
            # corrupt one means the stream is corrupt and must fail fast rather
            # than being silently discarded here.
            _validate_iso_timestamp(
                parts[2],
                f"drain_stale_cancel_turn: corrupt stream, {msg_type!r} header",
            )
            # Match on the parsed message type, not a substring: a header whose
            # msg_type merely *contains* 'cancel_turn' (e.g. 'cancel_turn_ack')
            # is not a cancel and must not be drained.
            if msg_type != 'cancel_turn':
                # Not a cancel_turn -- seek back
                self._input_handle.seek(pos)
                break
            # The size field must be a valid non-negative int; otherwise the
            # stream is corrupt -- fail fast.
            try:
                payload_size = int(parts[0])
            except ValueError:
                raise Exception(
                    f"drain_stale_cancel_turn: corrupt stream, payload_size "
                    f"{parts[0]!r} is not an integer: {line_str!r}"
                ) from None
            if payload_size < 0:
                raise Exception(
                    f"drain_stale_cancel_turn: corrupt stream, negative "
                    f"payload_size {payload_size}: {line_str!r}"
                )
            if payload_size != 0:
                raise Exception(
                    f"drain_stale_cancel_turn: corrupt stream, cancel_turn "
                    f"not expected to have payload, but size is {payload_size}: "
                    f"{line_str!r}"
                )
            print("DEBUG Drained stale cancel_turn message during recovery")

    def _count_acked_permission_requests(self) -> int:
        """Count request-ack pairs in the output file.

        Each tool_permission_response_ack corresponds to a
        tool_permission_response that was consumed from the input stream, so
        this count is the number of stale responses the recovery drain must
        consume.

        Uses the incremental output-file scan (see
        :meth:`_update_output_permission_scan`). ``open_handles`` resets the
        scan cache, so during mid-turn recovery this first call folds the
        entire pre-existing output file (the whole-file balance the drain
        depends on); subsequent calls only parse newly appended records.
        """
        if not self._output_file:
            return 0
        return self._update_output_permission_scan().acked_count

    def _has_any_pending_permission_request(self) -> bool:
        """True if the output stream has any outstanding (un-acked) permission
        request.

        Used by :meth:`check_cancel_turn` to distinguish a legitimate
        ``tool_permission_response`` left in the input stream for a
        still-pending request (consumed later by ``request_permission``, e.g.
        during a mid-turn recovery) from an orphaned/corrupt response.
        """
        if not self._output_file or not self._output_handle:
            return False
        scan = self._update_output_permission_scan()
        return any(count > 0 for count in scan.pending_by_tool.values())

    def _has_prior_approval(self, tool_name: str) -> bool:
        """Check if there is a prior acknowledged grant for this tool.

        Called from request_permission() when has_pending_request() returns
        False. Returns True only if the MOST RECENT acknowledged decision for
        this tool was an approval, so a later denial supersedes an earlier
        grant and a previously-denied permission is never granted on
        prior-approval replay.

        Uses the incremental output-file scan (see
        :meth:`_update_output_permission_scan`); corruption in any scanned
        record (including permission payloads) fails fast exactly as it does
        in ``has_pending_request``.
        """
        if not self._output_file:
            return False
        scan = self._update_output_permission_scan()
        return scan.last_ack_by_tool.get(tool_name) is True

    # --- Output writing ---

    def write_response(self, content: str, reasoning_content: Optional[str],
                       metadata: dict) -> None:
        """Write response message with LLM content, reasoning, and metadata.

        All token/metadata handling (including the nested ``token_usage`` and
        ``*_details`` conventions, and the top-level ``actual_model``/``model``
        key aliases) is performed by the wire-format writer so this method and
        the module-level ``write_response`` always agree on the header layout.
        """
        write_response(
            self._output_handle, content, reasoning_content, metadata,
            file_id=str(self._output_file)
        )

    def write_with_metadata(self, payload_type: str, content: str,
                            execution_successful: int = 0,
                            tool_calls: Optional[List[Tuple[str, dict]]] = None) -> None:
        """Write payload (tool_use, tool_use_result, etc.).

        Not a valid channel for ``model_message_to_user``: that message type
        requires ``tags`` and ``format`` header fields and has a dedicated
        5-field writer (:meth:`write_model_message_to_user`); routing it
        here would emit a 3-field header that every reader rejects as
        corrupt.
        """
        payload = content
        if tool_calls is not None:
            payload = json.dumps(
                {"tool_calls": tool_calls},
                default=lambda o: None if o is Ellipsis else str(o)
            )

        write_payload(
            self._output_handle, payload_type, payload,
            execution_successful=execution_successful,
            file_id=str(self._output_file)
        )

    def write_model_message_to_user(self, content: str, tags: list, format: str = "markdown") -> None:
        """Write model_message_to_user with tags and format metadata."""
        write_model_message_to_user(
            self._output_handle, content,
            tags=tags, format=format,
            file_id=str(self._output_file)
        )

    def write_conditional_report(self, content: str) -> None:
        """Write conditional_report payload to output file."""
        write_payload(self._output_handle, "conditional_report", content,
                      file_id=str(self._output_file))


    def write_cancel_ack(self):
        """Write cancel_turn_ack to output file."""
        write_payload(self._output_handle, "cancel_turn_ack", None,
                      file_id=str(self._output_file))

    def write_max_requests_reached(self, message: str):
        """Write max_requests_reached marker to output file."""
        write_payload(self._output_handle, "max_requests_reached", str(message),
                      file_id=str(self._output_file))

    # --- Seen files ---

    def pre_populate_seen_files(self, state: ForkTurnState):
        """Pre-populate seen files from the reconciled fact snapshot.

        Marks every input file seen whose ``IN`` fact has a terminal ``OUT``
        (``COMPLETE_NORMAL`` or ``COMPLETE_ERROR``) at the same
        ``(fork, turn)``. Unfinished and ``ABANDONED`` turns stay unseen so a
        genuine mid-turn ``IN`` at the startup cursor is re-opened; superseded
        turns are then excluded from later polls by ``check_for_new_input``'s
        fork/turn cursor.

        Args:
            state: The reconciled ForkTurnState snapshot.
        """
        # CANCELLED is deliberately NOT in the mark-seen set: a cancelled turn
        # may hold a matching checkpoint with interrupted work, so its input
        # must stay re-openable for mid-turn recovery. Completed cancelled
        # turns are still skipped by the cursor (file_turn < turn filter).
        terminal = {
            IOTurnStatus.COMPLETE_NORMAL,
            IOTurnStatus.COMPLETE_ERROR,
        }
        marked = 0
        for fact in state.facts:
            if fact.kind is not IOFactKind.IN:
                continue
            if fact.status is IOTurnStatus.ABANDONED:
                continue
            out = next(
                (f for f in state.facts
                 if f.kind is IOFactKind.OUT
                 and f.fork == fact.fork and f.turn == fact.turn),
                None,
            )
            if out is not None and out.status in terminal:
                self.seen_files.add(self.io_path / f"{fact.fork}.{fact.turn}.in.txt")
                marked += 1
        if marked:
            print(f"  (Pre-marked {marked} files as already processed)")

    def mark_seen(self, input_file: Path):
        """Mark an input file as seen so we don't process it again."""
        self.seen_files.add(input_file)

    # --- Validation ---

    def validate_end_files(self, state: ForkTurnState, start_fork: int,
                           start_turn: int, is_fresh_start: bool) -> bool:
        """Validate end files before entering processing loop.

        The old glob-based malformed-name handling is removed: every matched
        malformed ``.end`` has already failed the strict snapshot scan with
        ``IOFilenameError``. The remaining rules are unchanged: no end facts on
        a fresh start; at most one end fact; it must sit at the startup cursor
        and returns ``True`` when it does.

        Args:
            state: The reconciled ForkTurnState snapshot.
            start_fork: Startup cursor fork.
            start_turn: Startup cursor turn.
            is_fresh_start: True for a fresh start (no recovery).

        Returns:
            True if an end fact sits at the startup cursor (conversation complete).

        Raises:
            Exception: If end facts are in an invalid state (misplaced or multiple).
        """
        end_positions = state.end_positions
        if is_fresh_start:
            if end_positions:
                names = [f"{f.fork}.{f.turn}.end" for f in sorted(end_positions)]
                raise Exception(
                    f"ERROR: Found end file(s) but starting fresh conversation: {names}"
                )
        else:
            if len(end_positions) > 1:
                names = [f"{f.fork}.{f.turn}.end" for f in sorted(end_positions)]
                raise Exception(f"ERROR: Multiple end files detected: {names}")
            if end_positions:
                end_pos = next(iter(end_positions))
                if end_pos != ForkTurn(start_fork, start_turn):
                    raise Exception(
                        f"ERROR: End file {end_pos.fork}.{end_pos.turn}.end at "
                        f"unexpected position (expected {start_fork}.{start_turn}.end)"
                    )
                # End file at expected position - conversation complete
                print(f"End signal found at fork {start_fork}, turn {start_turn}. "
                      f"Conversation already complete.")
                return True
        return False
