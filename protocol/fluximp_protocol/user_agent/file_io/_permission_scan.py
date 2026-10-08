"""Incremental output-file permission-scan state for FileIOAgentProtocol.

Shared by ``has_pending_request``, ``_has_prior_approval`` and
``_count_acked_permission_requests`` (see agent.py): instead of re-reading and
re-parsing the entire output file on every query, each complete record is
folded into this state exactly once, keyed by ``parsed_pos`` -- the byte
offset immediately after the last record folded in.

The agent is the only output writer and only ever appends fully flushed
records, so the cache never needs a reset mid-turn: every query sees exactly
the state a whole-file scan would produce, at the cost of parsing only the
newly appended tail bytes. ``open_handles`` resets the cache so a
mid-turn-recovery scan starts from byte 0 and sees the whole-file balance
that the recovery paths (drain_count / prior approvals) rely on. A torn
trailing record (a prior process died mid-write) never completes -- a
recovered run appends new records after it -- so ``advance`` fails fast on
one instead of silently freezing the cache at the torn position.
"""

import json

from fluximp_protocol.user_agent.base import PermissionServiceProtocolError
from fluximp_protocol.user_agent.file_io.defs import parse_output_messages, parse_utc_timestamp


class _OutputPermissionScan:
    """Incrementally accumulated state of the permission-related records in
    the current output file.

    Attributes:
        parsed_pos: Byte offset immediately after the last complete record
            folded into the state.
        pending_by_tool: Outstanding (un-acked) permission requests, per tool.
            A single "acked" boolean is not enough: it would incorrectly report
            "no pending" for [request, request, ack], where the last request is
            still pending. Requests and acks are paired per tool: an ack for
            one tool must not satisfy an outstanding request for a different
            tool, and an ack whose tool has no matching outstanding request is
            corruption -- fold() fails fast rather than silently granting
            or duplicating a request on resume.
        acked_count: Total request/ack pairs completed. Each ack corresponds
            to a tool_permission_response consumed from the input stream, so
            this is the recovery drain count.
        last_request_tool_name: Tool name of the last tool_permission_request
            seen.
        last_request_timestamp: Timestamp of that last request (datetime).
        last_ack_by_tool: Most recent ack decision per tool (None if never
            acked).
    """

    __slots__ = (
        "parsed_pos",
        "pending_by_tool",
        "acked_count",
        "last_request_tool_name",
        "last_request_timestamp",
        "last_ack_by_tool",
    )

    def __init__(self):
        # Byte offset immediately after the last complete record folded in.
        self.parsed_pos = 0
        # Outstanding (un-acked) permission requests, per tool.
        self.pending_by_tool = {}
        # Total request/ack pairs completed; each ack corresponds to a
        # tool_permission_response consumed from the input stream, so this is
        # the recovery drain count.
        self.acked_count = 0
        # Tool name of the last tool_permission_request seen.
        self.last_request_tool_name = None
        self.last_request_timestamp = None
        # Most recent ack decision per tool (None if never acked).
        self.last_ack_by_tool = {}

    def fold(self, header, payload: str, record_start: int) -> None:
        """Fold one complete output-file record into this state.

        ``record_start`` is the record's own byte offset (not the offset
        after it) so corruption diagnostics point at the offending record.

        Raises:
            PermissionServiceProtocolError: On any corruption detected in the
                record (missing/unparseable timestamp, non-JSON or non-object
                permission payload, missing tool_name, missing/non-boolean
                approved, or an ack with no matching outstanding request for
                its tool).
        """
        msg_type = header.msg_type
        msg_timestamp_str = header.timestamp
        # A missing or unparseable timestamp in any output message means the
        # stream is corrupt -- fail fast rather than silently skipping the
        # message (which could hide a pending permission request on resume).
        if not msg_timestamp_str:
            raise PermissionServiceProtocolError(
                f"corrupt output file: {msg_type!r} message at byte "
                f"{record_start} has no timestamp"
            )
        try:
            msg_timestamp = parse_utc_timestamp(msg_timestamp_str)
        except ValueError:
            raise PermissionServiceProtocolError(
                f"corrupt output file: {msg_type!r} message at byte "
                f"{record_start} has unparseable timestamp "
                f"{msg_timestamp_str!r}"
            ) from None

        if msg_type == 'tool_permission_request':
            try:
                req_payload = json.loads(payload)
            except json.JSONDecodeError:
                raise PermissionServiceProtocolError(
                    f"corrupt output file: tool_permission_request payload "
                    f"at byte {record_start} is not valid JSON"
                ) from None
            # Validate the object shape BEFORE touching its fields so a
            # non-object JSON value fails fast as a PermissionServiceProtocolError
            # instead of raising a bare AttributeError from .get().
            if not isinstance(req_payload, dict):
                raise PermissionServiceProtocolError(
                    f"corrupt output file: tool_permission_request payload "
                    f"at byte {record_start} is not a JSON object"
                )
            req_tool_name = req_payload.get('tool_name')
            if req_tool_name is None:
                raise PermissionServiceProtocolError(
                    f"corrupt output file: tool_permission_request payload "
                    f"at byte {record_start} has no tool_name"
                )
            # tool_name must be a string: it doubles as a dict key in the
            # per-tool pending/ack balance and is compared against the str tool
            # names queried by has_pending_request(). An unhashable value would
            # crash the dict lookups below with a bare TypeError, and a hashable
            # non-string becomes a phantom key that can never match a real tool,
            # making the request silently invisible to has_pending_request() and
            # letting request_permission() re-issue it (consuming the parked
            # response) before a misleading ordering error surfaces later. Fail
            # fast instead.
            if not isinstance(req_tool_name, str):
                raise PermissionServiceProtocolError(
                    f"corrupt output file: tool_permission_request payload "
                    f"at byte {record_start} has non-string tool_name "
                    f"{req_tool_name!r}"
                )
            self.pending_by_tool[req_tool_name] = (
                self.pending_by_tool.get(req_tool_name, 0) + 1
            )
            # Every request sets the one-in-flight marker: the protocol only
            # ever has one request in flight, so has_pending_request() compares
            # against this and request_permission() uses the timestamp to skip
            # stale responses.
            self.last_request_tool_name = req_tool_name
            self.last_request_timestamp = msg_timestamp

        elif msg_type == 'tool_permission_response_ack':
            try:
                ack_payload = json.loads(payload)
            except json.JSONDecodeError:
                raise PermissionServiceProtocolError(
                    f"corrupt output file: tool_permission_response_ack "
                    f"payload at byte {record_start} is not valid JSON"
                ) from None
            if not isinstance(ack_payload, dict):
                raise PermissionServiceProtocolError(
                    f"corrupt output file: tool_permission_response_ack "
                    f"payload at byte {record_start} is not a JSON object"
                )
            ack_tool_name = ack_payload.get('tool_name')
            if ack_tool_name is None:
                raise PermissionServiceProtocolError(
                    f"corrupt output file: tool_permission_response_ack "
                    f"payload at byte {record_start} has no tool_name"
                )
            # tool_name must be a string: it doubles as a dict key in the
            # per-tool pending/ack balance and is compared against the str tool
            # names queried by has_pending_request(). An unhashable value would
            # crash the dict lookups below with a bare TypeError, and a hashable
            # non-string becomes a phantom key that can never match a real tool
            # -- surfacing only as a misleading ordering error. Fail fast.
            if not isinstance(ack_tool_name, str):
                raise PermissionServiceProtocolError(
                    f"corrupt output file: tool_permission_response_ack "
                    f"payload at byte {record_start} has non-string tool_name "
                    f"{ack_tool_name!r}"
                )
            # A missing or non-boolean approved field means the stream is
            # corrupt: the writer always emits a JSON boolean, and
            # _has_prior_approval treats a non-bool as "no prior grant", which
            # would silently mis-recover a previously granted permission
            # instead of failing fast.
            if 'approved' not in ack_payload or not isinstance(
                ack_payload['approved'], bool
            ):
                raise PermissionServiceProtocolError(
                    f"corrupt output file: tool_permission_response_ack "
                    f"payload at byte {record_start} has missing or "
                    f"non-boolean 'approved' field: "
                    f"{ack_payload.get('approved')!r}"
                )
            # An ack for ANY tool whose outstanding-request balance is already
            # zero is corruption (an ack before its request, or more acks than
            # requests). Validate against that ack's own tool -- not just the
            # queried tool -- so a cross-tool mismatch fails fast rather than
            # silently granting permission or duplicating a request on resume.
            if self.pending_by_tool.get(ack_tool_name, 0) <= 0:
                raise PermissionServiceProtocolError(
                    f"corrupt output file: tool_permission_response_ack for "
                    f"{ack_tool_name!r} at byte {record_start} appears before "
                    f"any un-acked tool_permission_request for it"
                )
            self.pending_by_tool[ack_tool_name] -= 1
            # Every ack corresponds to a tool_permission_response consumed from
            # the input stream, so it increments the recovery drain count.
            self.acked_count += 1
            self.last_ack_by_tool[ack_tool_name] = ack_payload['approved']

    def advance(self, output_file) -> "_OutputPermissionScan":
        """Fold all complete output-file records appended since the last
        scan into this state and return it.

        A file that shrank below ``parsed_pos`` is corruption and fails fast
        (mirroring the positioned-reader shrink detection in user.py). A
        trailing record that fails to parse to completion is corruption and
        fails fast as well: the agent is the only output writer and every
        write is flushed before any scan can run, so an incomplete trailing
        record can only exist if a prior process died mid-write. Such a torn
        tail never completes -- a recovered run reopens this same file in
        append mode and writes new records AFTER it -- so treating it as a
        partial write would silently freeze this cache (and every query
        backed by it) at the torn position instead of failing fast.

        Returns:
            self, for chaining at the call sites.
        """
        size = output_file.stat().st_size
        if size < self.parsed_pos:
            raise PermissionServiceProtocolError(
                f"corrupt output file: {output_file.name} shrank from parsed "
                f"position {self.parsed_pos} to size {size}"
            )
        if size == self.parsed_pos:
            return self
        # Read only the newly appended tail. parse_output_messages positions
        # are relative to the bytes it was given, so translate back with a
        # bias to absolute file offsets (parsed_pos is the buffer's index 0).
        with open(output_file, 'rb') as f:
            f.seek(self.parsed_pos)
            new_bytes = f.read()
        bias = self.parsed_pos
        record_start = bias
        for header, payload, next_pos in parse_output_messages(new_bytes):
            self.fold(header, payload, record_start)
            record_start = bias + next_pos
            self.parsed_pos = bias + next_pos
        tail_end = bias + len(new_bytes)
        if self.parsed_pos < tail_end:
            # parse_output_messages stopped before consuming the whole tail:
            # the file ends in an incomplete record. The agent is the only
            # writer and every write is flushed before any scan runs, so this
            # is a torn tail from a dead process (see the docstring) -- fail
            # fast instead of silently freezing the cache at the torn
            # position while a recovery run appends records past it.
            raise PermissionServiceProtocolError(
                f"corrupt output file: {output_file.name} ends in an "
                f"incomplete record at byte {self.parsed_pos} (read tail to "
                f"byte {tail_end}, file size {size}); a torn tail only "
                f"exists if a process died mid-record"
            )
        return self
