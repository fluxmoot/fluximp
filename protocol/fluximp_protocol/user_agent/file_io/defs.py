#!/usr/bin/env python3
'''

Shared definitions for the file I/O implementation of the user-agent protocol.

This module contains the wire-format definitions and primitives that are
common to both sides of the file I/O implementation -- the agent and the user.
Both sides need to agree on these values.

Contents:
- MessageType enum (canonical definition of all protocol message types)
- ParsedHeader dataclass (structured access to wire-format header fields)
- Filename helpers (get_io_filename, get_input_filename, etc.)
- extract_fork_turn_from_filename
- Generic output parser (parse_output_messages)
'''

import os
import time
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional, Tuple, Generator, List


# ============================================================================
# MessageType - Canonical definition of all protocol message types
# ============================================================================

class MessageType(Enum):
    '''Types of messages in the file I/O protocol.
    
    Used by both the agent (writing output) and clients (writing input).
    '''
    # Client -> Agent messages (written to .in.txt files)
    USER_MESSAGE = "user_message"
    TOOL_PERMISSION_RESPONSE = "tool_permission_response"
    CANCEL_TURN = "cancel_turn"
    
    # Agent -> Client messages (written to .out.txt files)
    TOOL_PERMISSION_REQUEST = "tool_permission_request"
    TOOL_PERMISSION_RESPONSE_ACK = "tool_permission_response_ack"
    CANCEL_TURN_ACK = "cancel_turn_ack"
    
    # Agent output message types
    MODEL_MESSAGE_TO_USER = "model_message_to_user"
    SCRUTINATOR_REPORT = "scrutinator_report"
    CONDITIONAL_REPORT = "conditional_report"
    SUMMARY = "summary"
    MAX_REQUESTS_REACHED = "max_requests_reached"
    INFRASTRUCTURE_ERROR = "infrastructure_error"
    TOOL_USE = "tool_use"
    TOOL_USE_RESULT = "tool_use_result"
    REQUEST = "request"
    RESPONSE = "response"
    
    # System agent lifecycle messages (written to primary output by AgentService)
    SYSTEM_AGENT_STARTED = "system_agent_started"
    SYSTEM_AGENT_STOPPED = "system_agent_stopped"
# ============================================================================

def parse_bool(value: Optional[str]) -> Optional[bool]:
    """Parse a boolean wire-format value.

    Accepts the canonical 'true'/'false' serialization and, for backwards
    compatibility with older files, the legacy '1'/'0' form.  Matching is
    case-insensitive.  Returns None when the value cannot be parsed as a
    boolean.
    """
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in ("true", "1", "yes"):
        return True
    if normalized in ("false", "0", "no"):
        return False
    return None


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 timestamp string.

    Always timezone-aware (``+00:00``), so header timestamps can be parsed and
    compared without mixing naive and aware datetimes.
    """
    return datetime.now(timezone.utc).isoformat()


def parse_utc_timestamp(value: str) -> datetime:
    """Parse an ISO-8601 timestamp, requiring a timezone-aware UTC value.

    Writers always emit timezone-aware UTC timestamps. Any other value — a
    naive timestamp or one carrying a non-UTC offset — is a mismatched
    stream and fails fast so downstream comparisons are always well-defined.
    """
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None or dt.utcoffset() != timedelta(0):
        raise ValueError(
            f"{value!r} is not a UTC ISO-8601 timestamp "
            "(expected a timezone-aware value with a +00:00 offset)"
        )
    return dt

def _validate_iso_timestamp(value: str, message: str) -> None:
    """Validate the common ISO-8601 timestamp header field.

    Every header carries a non-empty ``timestamp`` as its third field (see
    :class:`ParsedHeader`). The writers always emit timezone-aware UTC
    timestamps via :func:`utc_now_iso`, so a completed header whose timestamp
    is empty or unparseable is a corrupt stream and must fail fast rather than
    propagate a malformed timestamp to consumers that might silently fall back
    to ``datetime.now()``.
    """
    if not value:
        raise ValueError(f"{message}: timestamp is empty")
    try:
        parse_utc_timestamp(value)
    except ValueError:
        raise ValueError(
            f"{message}: timestamp {value!r} is not a UTC ISO-8601 timestamp"
        ) from None

# Per-file monotonic header stamps. The wire format's header timestamps must be
# strictly increasing per output stream so the §7.4.1 reload sort and §7.7
# trigger classification can rely on them. Writers key this by the protocol-
# known file path; the helper waits (bounded) for the wall clock to advance
# past the previous stamp and otherwise emits `last + 1us` so the guarantee is
# independent of clock resolution.
_last_stamp_by_file: Dict[str, datetime] = {}


def monotonic_stamp(file_id: str) -> str:
    """Return a strictly-increasing ISO-8601 UTC timestamp for ``file_id``.

    Each write path to a file calls this with the file's identity (its
    protocol-known path). The returned stamp is guaranteed to be strictly
    greater than the previous stamp returned for the same ``file_id``: if the
    wall clock has advanced past the last stamp that value is used, otherwise
    the helper waits for the clock to advance (bounded to ~1 ms) and finally
    falls back to ``last + 1 microsecond`` so the property holds even at
    microsecond clock resolution. Files written after this refactor therefore
    carry per-file strictly-increasing header timestamps; legacy files are
    never retroactively re-stamped.
    """
    last = _last_stamp_by_file.get(file_id)
    if last is None:
        stamp = datetime.now(timezone.utc)
        _last_stamp_by_file[file_id] = stamp
        return stamp.isoformat()
    deadline = time.monotonic() + 0.001
    stamp = datetime.now(timezone.utc)
    while stamp <= last and time.monotonic() < deadline:
        time.sleep(0.0001)
        stamp = datetime.now(timezone.utc)
    if stamp <= last:
        stamp = last + timedelta(microseconds=1)
    # Debug-build invariant: strictly increasing per file (§7.7). The helper
    # guarantees this by construction; the assert documents the guarantee and
    # is active only under FLUXIMP_DEBUG_ASSERTS.
    if os.getenv('FLUXIMP_DEBUG_ASSERTS'):
        assert stamp > last
    _last_stamp_by_file[file_id] = stamp
    return stamp.isoformat()


# ============================================================================
# ParsedHeader - Structured representation of a parsed message header
# ============================================================================

@dataclass
class ParsedHeader:
    """Structured representation of a parsed message header.
    
    Wire format headers have 3 common fields, followed by type-specific fields:
        Common (all messages):
            0  payload_size        Payload byte count
            1  msg_type            Message type string
            2  timestamp           ISO 8601 timestamp when header was written
        
        Request-specific:
            3  agent_type          Agent type (e.g., 'coder', 'scrutinator')
            4  provider            Model provider name
            5  requested_model     Model name that was requested
            6  start_time          ISO 8601 timestamp when request started
        
        Response-specific:
            3  end_time            ISO 8601 timestamp when response completed
            4  actual_model        Actual model name returned by API
            5  finish_reason       Why generation stopped
            6  prompt_tokens       Prompt token count
            7  completion_tokens   Completion token count
            8  total_tokens        Total token count
            9  cached_tokens       Cached token count
            10 reasoning_tokens    Reasoning token count
            11 system_fingerprint  System fingerprint from API response
            12 created             Unix timestamp from API response
        
        
        Tool_use_result-specific:
            3  execution_successful 'true'/'false' (legacy '1'/'0' accepted)
        
        model_message_to_user-specific:
            3  tags               '+'-separated tag values (e.g., 'findings_report+work_completion_summary')
            4  format             Message format ('plain_text' or 'markdown')
        
        tool_permission_response-specific:
            3  approved            'true'/'false' (legacy '1'/'0' accepted)
        Other message types (tool_use, etc.):
            No additional fields beyond the common 3.
    """
    # Common fields (all messages)
    msg_type: str
    payload_size: int
    timestamp: Optional[str] = None
    
    # Request-specific fields (only populated for 'request' messages)
    agent_type: Optional[str] = None
    provider: Optional[str] = None
    requested_model: Optional[str] = None
    start_time: Optional[str] = None
    
    # Response-specific fields (only populated for 'response' messages)
    end_time: Optional[str] = None
    actual_model: Optional[str] = None
    finish_reason: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    system_fingerprint: Optional[str] = None
    created: Optional[int] = None
    
    
    # Tool_use_result-specific fields
    execution_successful: Optional[bool] = None
    
    # model_message_to_user-specific fields
    tags: Optional[List[str]] = None
    format: Optional[str] = None
    
    # tool_permission_response-specific fields
    approved: Optional[bool] = None
    # System agent lifecycle fields (system_agent_started/stopped messages)
    system_agent_type: Optional[str] = None
    system_agent_subdir: Optional[str] = None
    system_agent_success: Optional[bool] = None

    @property
    def is_request(self) -> bool:
        """True if this is a request header."""
        return self.msg_type == MessageType.REQUEST.value

    @property
    def is_response(self) -> bool:
        """True if this is a response header."""
        return self.msg_type == MessageType.RESPONSE.value

    def is_type(self, mt: MessageType) -> bool:
        """Check if this header matches a MessageType enum value."""
        return self.msg_type == mt.value

    def to_metadata_dict(self) -> Dict[str, Any]:
        """Convert header fields to a metadata dict.
        
        Only includes fields relevant to the message type:
        - Request: agent_type, provider, requested_model, start_time
        - Response: end_time, model, finish_reason, token counts, etc.
        - Tool_use_result: execution_successful
        - Other: empty dict
        """
        metadata: Dict[str, Any] = {}
        
        if self.is_request:
            if self.agent_type is not None:
                metadata['agent_type'] = self.agent_type
            if self.provider is not None:
                metadata['provider'] = self.provider
            if self.requested_model is not None:
                metadata['requested_model'] = self.requested_model
            if self.start_time is not None:
                metadata['start_time'] = self.start_time
        
        elif self.is_response:
            if self.end_time is not None:
                metadata['end_time'] = self.end_time
            if self.actual_model is not None:
                metadata['model'] = self.actual_model
            if self.finish_reason is not None:
                metadata['finish_reason'] = self.finish_reason
            if self.prompt_tokens is not None:
                metadata['prompt_tokens'] = self.prompt_tokens
            if self.completion_tokens is not None:
                metadata['completion_tokens'] = self.completion_tokens
            if self.total_tokens is not None:
                metadata['total_tokens'] = self.total_tokens
            if self.cached_tokens is not None:
                metadata['cached_tokens'] = self.cached_tokens
            if self.reasoning_tokens is not None:
                metadata['reasoning_tokens'] = self.reasoning_tokens
            if self.system_fingerprint is not None:
                metadata['system_fingerprint'] = self.system_fingerprint
            if self.created is not None:
                metadata['created'] = self.created
        
        elif self.is_type(MessageType.TOOL_USE_RESULT):
            metadata['execution_successful'] = self.execution_successful if self.execution_successful is not None else False
        
        elif self.is_type(MessageType.MODEL_MESSAGE_TO_USER):
            if self.tags is not None:
                metadata['tags'] = self.tags
            if self.format is not None:
                metadata['format'] = self.format
        elif self.is_type(MessageType.TOOL_PERMISSION_RESPONSE):
            if self.approved is not None:
                metadata['approved'] = self.approved
        return metadata

    @classmethod
    def from_parts(cls, parts: List[str]) -> "ParsedHeader":
        """Construct a ParsedHeader from a split header line.
        
        Parses fields based on message type, since each type has
        a different header layout beyond the common 3 fields.
        """
        if not parts:
            raise ValueError("corrupt header: header has no fields")
        msg_type = parts[1] if len(parts) > 1 else ""
        try:
            payload_size = int(parts[0])
        except ValueError:
            # An unparseable payload_size means the stream is corrupt -- fail
            # fast rather than silently treating it as an empty payload.
            raise ValueError(
                f"corrupt header: payload_size {parts[0]!r} is not an integer"
            ) from None

        def _str(idx: int) -> Optional[str]:
            if idx >= len(parts):
                return None
            val = parts[idx]
            return val if val else None

        def _int(idx: int, name: str) -> Optional[int]:
            val = _str(idx)
            if val is None:
                return None
            try:
                return int(val)
            except ValueError:
                # A present-but-unparseable integer field means the header is
                # corrupt -- fail fast rather than silently treating it as
                # absent (matching _bool below).
                raise ValueError(
                    f"corrupt header: {msg_type!r} field {name!r} value {val!r} "
                    f"is not a valid integer"
                ) from None

        def _bool(idx: int) -> Optional[bool]:
            val = _str(idx)
            if val is None:
                return None
            parsed = parse_bool(val)
            if parsed is None:
                # A present-but-unparseable boolean field means the header is
                # corrupt -- fail fast rather than silently treating it as absent.
                raise ValueError(
                    f"corrupt header: {msg_type!r} boolean field {val!r} is "
                    f"not a valid boolean"
                ) from None
            return parsed

        # Common fields
        timestamp = _str(2)

        # Type-specific parsing
        if msg_type == MessageType.REQUEST.value:
            return cls(
                msg_type=msg_type, payload_size=payload_size,
                timestamp=timestamp,
                agent_type=_str(3), provider=_str(4),
                requested_model=_str(5), start_time=_str(6),
            )
        elif msg_type == MessageType.RESPONSE.value:
            return cls(
                msg_type=msg_type, payload_size=payload_size,
                timestamp=timestamp,
                end_time=_str(3), actual_model=_str(4),
                finish_reason=_str(5),
                prompt_tokens=_int(6, "prompt_tokens"), completion_tokens=_int(7, "completion_tokens"),
                total_tokens=_int(8, "total_tokens"), cached_tokens=_int(9, "cached_tokens"),
                reasoning_tokens=_int(10, "reasoning_tokens"),
                system_fingerprint=_str(11), created=_int(12, "created"),
            )
        elif msg_type == MessageType.TOOL_USE_RESULT.value:
            return cls(
                msg_type=msg_type, payload_size=payload_size,
                timestamp=timestamp,
                execution_successful=_bool(3),
            )
        elif msg_type == MessageType.MODEL_MESSAGE_TO_USER.value:
            tags_str = _str(3)
            tags = tags_str.split('+') if tags_str else []
            return cls(
                msg_type=msg_type, payload_size=payload_size,
                timestamp=timestamp,
                tags=tags,
                format=_str(4),
            )
        elif msg_type == MessageType.SYSTEM_AGENT_STARTED.value:
            return cls(
                msg_type=msg_type, payload_size=payload_size,
                timestamp=timestamp,
                system_agent_type=_str(3),
                system_agent_subdir=_str(4),
            )
        elif msg_type == MessageType.SYSTEM_AGENT_STOPPED.value:
            return cls(
                msg_type=msg_type, payload_size=payload_size,
                timestamp=timestamp,
                system_agent_type=_str(3),
                system_agent_success=_bool(4),
            )
        elif msg_type == MessageType.TOOL_PERMISSION_RESPONSE.value:
            return cls(
                msg_type=msg_type, payload_size=payload_size,
                timestamp=timestamp,
                approved=_bool(3),
            )
        else:
            # Well-known types that carry only the 3 common header fields.
            if _FIELD_COUNTS.get(msg_type) != 3:
                # An unrecognized msg_type is impossible on a well-formed
                # stream; treat it as corruption and fail fast rather than
                # returning a generic header that downstream dispatchers would
                # silently skip (hiding a pending permission request, etc.).
                raise ValueError(
                    f"corrupt stream: unrecognized message type {msg_type!r}"
                )
            return cls(
                msg_type=msg_type, payload_size=payload_size,
                timestamp=timestamp,
            )

# Filename helpers
# ============================================================================

def get_io_filename(fork: int, turn: int, suffix: str) -> str:
    '''Build filename for fork.turn protocol.
    
    Args:
        fork: Fork number (increments for time travel branches)
        turn: Turn number (increments within each fork)
        suffix: File suffix (e.g., '.in.txt', '.out.txt', '.end')
        
    Returns:
        Filename string in format {fork}.{turn}{suffix}
    '''
    return f"{fork}.{turn}{suffix}"


def get_input_filename(fork: int, turn: int) -> str:
    '''Build input filename for fork.turn protocol.
    
    Args:
        fork: Fork number (increments for time travel branches)
        turn: Turn number (increments within each fork)
        
    Returns:
        Filename string in format {fork}.{turn}.in.txt
    '''
    return get_io_filename(fork, turn, ".in.txt")


def get_output_filename(fork: int, turn: int) -> str:
    '''Build output filename for fork.turn protocol.
    
    Args:
        fork: Fork number (increments for time travel branches)
        turn: Turn number (increments within each fork)
        
    Returns:
        Filename string in format {fork}.{turn}.out.txt
    '''
    return get_io_filename(fork, turn, ".out.txt")


def get_end_filename(fork: int, turn: int) -> str:
    '''Build end signal filename for fork.turn protocol.
    
    Args:
        fork: Fork number (increments for time travel branches)
        turn: Turn number (increments within each fork)
        
    Returns:
        Filename string in format {fork}.{turn}.end
    '''
    return get_io_filename(fork, turn, ".end")


def extract_fork_turn_from_filename(filename: str) -> Optional[Tuple[int, int]]:
    '''Extract fork and turn numbers from an I/O filename.

    Args:
        filename: Filename like '0.5.out.txt' or '1.3.in.txt'

    Returns:
        Tuple of (fork, turn) or None if not a valid I/O filename
    '''
    # Remove suffixes to get base like '0.5.out' or '0.5.in'
    if '.out.txt' in filename:
        base = filename.replace('.out.txt', '')
    elif '.in.txt' in filename:
        base = filename.replace('.in.txt', '')
    else:
        return None

    # Split by dots - must have exactly 2 parts (fork.turn); extra components
    # (e.g. "0.1.2.out.txt") indicate a malformed filename and are rejected.
    parts = base.split('.')
    if len(parts) != 2:
        return None

    try:
        fork = int(parts[0])
        turn = int(parts[1])
        return (fork, turn)
    except ValueError:
        return None

class IOFilenameError(Exception):
    """A matched primary protocol filename violates the wire-format name rules."""


def _is_valid_coordinate(value: str) -> bool:
    """True if ``value`` is a non-negative integer with no leading zeros."""
    if not value or not value.isdecimal():
        return False
    return value == "0" or not value.startswith("0")


def extract_fork_turn_from_filename_strict(filename: str) -> Tuple[int, int]:
    """Strictly parse a primary protocol filename into ``(fork, turn)``.

    Accepts exactly ``{fork}.{turn}.in.txt``, ``{fork}.{turn}.out.txt``, or
    ``{fork}.{turn}.end`` with non-negative integer coordinates that have no
    leading zeros and no extra components. Any other matched primary filename
    raises :class:`IOFilenameError`.
    """
    parts = filename.split(".")
    if len(parts) == 4 and parts[2] in ("in", "out") and parts[3] == "txt":
        fork_s, turn_s = parts[0], parts[1]
    elif len(parts) == 3 and parts[2] == "end":
        fork_s, turn_s = parts[0], parts[1]
    else:
        raise IOFilenameError(
            f"malformed primary filename (expected {{fork}}.{{turn}}.{{kind}} "
            f"[.txt]): {filename!r}"
        )
    if not (_is_valid_coordinate(fork_s) and _is_valid_coordinate(turn_s)):
        raise IOFilenameError(
            f"malformed primary filename coordinates: {filename!r}"
        )
    return int(fork_s), int(turn_s)


# Wire-format header field counts for every known message type. All headers
# share 3 common fields (size,msg_type,timestamp); type-specific headers add
# more. Only the types listed here are recognized -- any other msg_type is
# impossible on a well-formed stream (forward-compatibility is not a concern)
# and is treated as corruption so the parser fails fast instead of silently
# accepting a generic 3-field record.
_FIELD_COUNTS = {
    MessageType.USER_MESSAGE.value: 3,
    MessageType.TOOL_PERMISSION_RESPONSE.value: 4,
    MessageType.CANCEL_TURN.value: 3,
    MessageType.TOOL_PERMISSION_REQUEST.value: 3,
    MessageType.TOOL_PERMISSION_RESPONSE_ACK.value: 3,
    MessageType.CANCEL_TURN_ACK.value: 3,
    MessageType.MODEL_MESSAGE_TO_USER.value: 5,
    MessageType.SCRUTINATOR_REPORT.value: 3,
    MessageType.CONDITIONAL_REPORT.value: 3,
    MessageType.SUMMARY.value: 3,
    MessageType.MAX_REQUESTS_REACHED.value: 3,
    MessageType.INFRASTRUCTURE_ERROR.value: 3,
    MessageType.TOOL_USE.value: 3,
    MessageType.TOOL_USE_RESULT.value: 4,
    MessageType.REQUEST.value: 7,
    MessageType.RESPONSE.value: 13,
    MessageType.SYSTEM_AGENT_STARTED.value: 5,
    MessageType.SYSTEM_AGENT_STOPPED.value: 5,
}


def expected_field_count(msg_type: str) -> int:
    """The number of comma-separated fields a valid header carries for ``msg_type``.

    All headers share 3 common fields (``size,msg_type,timestamp``); type-specific
    headers add more (see :class:`ParsedHeader`).

    This is used to validate a completed (newline-terminated) header: a field
    count that differs from expected means the stream is corrupt -- fail fast.
    (A header that is not yet newline-terminated is handled separately as a
    partial write.) An unrecognized ``msg_type`` is also corruption -- fail
    fast rather than silently assuming it carries only the 3 common fields,
    which would let a corrupt/mangled record slip past type-specific
    validation and be silently skipped by downstream dispatchers.
    """
    try:
        return _FIELD_COUNTS[msg_type]
    except KeyError:
        raise ValueError(
            f"corrupt stream: unrecognized message type {msg_type!r}"
        ) from None


# ============================================================================
# Generic output parser
# ============================================================================

def parse_output_messages(content_bytes: bytes, start_pos: int = 0) -> Generator[Tuple[ParsedHeader, str, int], None, None]:
    '''Parse output file content into individual messages.

    Generic parser that yields each message found in the output file bytes.
    This replaces the duplicated header+payload parsing loops found in
    conversation_manager, test_utils, agent_service, and run_scrutinator_tests.

    Format: <payload_size>,<msg_type>,<timestamp>
<payload>

    Args:
        content_bytes: Raw bytes of the output file
        start_pos: Byte position to start parsing from (for incremental reading)

    Yields:
        Tuples of (header, payload, next_position)
        - header: ParsedHeader with structured access to header fields
        - payload: The decoded payload string (empty string if payload_size == 0)
        - next_position: Byte position after this message (for incremental reading)
    '''
    pos = start_pos

    while pos < len(content_bytes):
        # Find the next newline (end of header). A header fragment exceeding
        # MAX_HEADER_BYTES without a terminating newline is structurally
        # impossible wire data -- corruption -- and fails fast here exactly as
        # _read_bounded_header_line does for the streaming readers (see
        # read_streaming_header's bounded-read rationale); silently treating it
        # as a partial write below would mask corruption from this parser while
        # parse_streamed_records fails, leaving callers (permission scan,
        # status classification) answering from frozen/stale state.
        header_end = content_bytes.find(b'\n', pos)
        if header_end == -1:
            if len(content_bytes) - pos > MAX_HEADER_BYTES:
                raise ValueError(
                    f"corrupt output stream at byte {pos}: header exceeds "
                    f"MAX_HEADER_BYTES ({MAX_HEADER_BYTES}) without terminating "
                    f"newline"
                )
            break
        # A completed header line longer than the same bound is corruption for
        # the same reason (the streaming readers reject it via
        # _read_bounded_header_line), so fail fast instead of accepting a
        # record here that every other reader refuses.
        if header_end - pos > MAX_HEADER_BYTES:
            raise ValueError(
                f"corrupt output stream at byte {pos}: header line exceeds "
                f"MAX_HEADER_BYTES ({MAX_HEADER_BYTES}): "
                f"{content_bytes[pos:header_end][:128]!r}..."
            )

        # Extract header line (invalid UTF-8 in a header means the stream is
        # corrupt -- fail fast with ValueError rather than a UnicodeDecodeError).
        try:
            header_line = content_bytes[pos:header_end].decode('utf-8').rstrip('\r')
        except UnicodeDecodeError:
            raise ValueError(
                f"corrupt output stream at byte {pos}: header is not valid UTF-8"
            ) from None

        # Parse header - split by comma
        parts = header_line.split(',')

        msg_type = parts[1] if len(parts) > 1 else ""
        expected = expected_field_count(msg_type)

        # The header is newline-terminated (we located the '\n' above), so it
        # is complete. A field count that differs from what the type requires
        # is corruption -- fail fast rather than silently stopping. Only a
        # header with no trailing newline (handled above by 'header_end == -1')
        # or an incomplete payload (handled below) is a legitimate partial write.
        if len(parts) != expected:
            raise ValueError(
                f"corrupt output stream at byte {pos}: {msg_type!r} header has "
                f"{len(parts)} fields, expected {expected}: {header_line!r}"
            )

        _validate_iso_timestamp(
            parts[2],
            f"corrupt output stream at byte {pos}: {msg_type!r} header",
        )
        try:
            payload_size = int(parts[0])
        except ValueError:
            raise ValueError(
                f"corrupt output stream at byte {pos}: payload_size "
                f"{parts[0]!r} is not an integer"
            ) from None

        if payload_size < 0:
            raise ValueError(
                f"corrupt output stream at byte {pos}: negative payload_size "
                f"{payload_size}"
            )
        # A corrupt-but-well-formed header can declare an arbitrarily large
        # payload_size. read_exact rejects such sizes for the streaming
        # readers; enforce the same bound here so an absurd size fails fast
        # instead of being silently treated as an incomplete write below
        # (which would freeze every reader of this file at this record).
        if payload_size > MAX_PAYLOAD_BYTES:
            raise ValueError(
                f"corrupt output stream at byte {pos}: payload_size "
                f"{payload_size} exceeds MAX_PAYLOAD_BYTES "
                f"({MAX_PAYLOAD_BYTES})"
            )

        payload_start = header_end + 1
        payload_end = payload_start + payload_size

        # If payload extends beyond available data, stop (incomplete write)
        if payload_end > len(content_bytes):
            break

        # Extract payload
        if payload_size > 0:
            try:
                payload = content_bytes[payload_start:payload_end].decode('utf-8')
            except UnicodeDecodeError:
                raise ValueError(
                    f"corrupt output stream at byte {payload_start}: payload is not valid UTF-8"
                ) from None
        else:
            payload = ""

        header = ParsedHeader.from_parts(parts)
        yield (header, payload, payload_end)

        pos = payload_end


# ============================================================================
# Streaming parser (incremental, byte-positioned reads)
# ============================================================================

@dataclass(frozen=True)
class StreamedRecord:
    """One fully parsed, validated wire record read from a byte stream.

    Unlike the buffer-based ``parse_output_messages`` tuple, this carries the
    exact byte offsets of the record so a caller can commit a position after
    it has fully processed the record.

    Attributes:
        header: The parsed, validated :class:`ParsedHeader`.
        payload: The decoded payload string ('' when payload_size == 0).
        start_pos: Byte offset where this record begins.
        next_pos: Byte offset immediately after this record's payload -- the
            position to commit once the record has been consumed.
    """
    header: ParsedHeader
    payload: str
    start_pos: int
    next_pos: int


# Bounded maximum size of a single newline-terminated header line. Wire headers
# are structurally small; a header fragment that grows beyond this without a
# newline is treated as corruption rather than buffered indefinitely. This keeps
# a stream read bounded per record (it never reads the whole file).
MAX_HEADER_BYTES = 16 * 1024

# Maximum accepted payload_size declared by a wire header. Blocking readers
# wait for the declared byte count to arrive; a corrupt-but-well-formed
# header can declare an arbitrarily large payload and make the agent stall
# until the timeout instead of failing fast. The positioned parsers
# (parse_output_messages / parse_streamed_records) enforce the same bound so
# an absurd size is never silently treated as an incomplete write (which
# would freeze every reader of the file at that record). 1 GB is far above
# any legitimate protocol payload, so anything larger is treated as
# corruption and rejected immediately.
MAX_PAYLOAD_BYTES = 1024 * 1024 * 1024


def _read_bounded_header_line(binary_file) -> Tuple[Optional[bytes], int]:
    """Read a newline-terminated header line from a positioned binary file.

    Returns ``(header_line_bytes, byte_count_read)`` where ``header_line_bytes``
    includes the terminating ``\\n``, or ``(None, ...)`` when no terminating
    newline is present within ``MAX_HEADER_BYTES + 1`` bytes.

    Raises:
        ValueError: The header exceeded ``MAX_HEADER_BYTES`` without a newline
            (a structurally impossible header fragment -> corruption).
    """
    data = binary_file.read(MAX_HEADER_BYTES + 1)
    nl = data.find(b'\n')
    if nl == -1:
        if len(data) > MAX_HEADER_BYTES:
            raise ValueError(
                f"corrupt output stream: header exceeds MAX_HEADER_BYTES "
                f"({MAX_HEADER_BYTES}) without terminating newline"
            )
        # EOF before newline: legitimate partial write; caller rewinds & waits.
        return None, len(data)
    return data[:nl + 1], nl + 1


def parse_streamed_records(binary_file, *, start_pos: int = 0) -> Generator[StreamedRecord, None, None]:
    """Streamingly parse wire records from a positioned binary file object.

    Reads only data at or after ``start_pos`` and yields one fully validated
    record at a time. Positions are byte offsets into the file.

    Partial-write handling:
        - A trailing header without a terminating newline is not consumed; the
          file is rewound to ``start_pos`` and iteration stops.
        - A trailing payload that is shorter than the header's declared size is
          not consumed; the file is rewound to ``start_pos`` and iteration stops.

    Corrupt handling (fail fast, never silently skipped):
        - A completed header that is not valid UTF-8.
        - A completed header whose field count differs from
          ``expected_field_count``.
        - A payload_size that is not an integer, is negative, or exceeds
          ``MAX_PAYLOAD_BYTES``.
        - A header line longer than ``MAX_HEADER_BYTES`` without a newline.
        - Invalid UTF-8 in a completed payload.

    Earlier complete records are already yielded (and may have been committed by
    the caller) before a later corrupt or partial record is encountered; the
    offending record is simply never yielded.

    Args:
        binary_file: An open binary file object positioned as the caller chooses
            (this function seeks to ``start_pos`` itself).
        start_pos: Byte offset into the stream to begin parsing.

    Yields:
        :class:`StreamedRecord` per fully parsed, validated record.
    """
    binary_file.seek(start_pos)
    while True:
        record_start = binary_file.tell()

        header_line_bytes, header_len = _read_bounded_header_line(binary_file)
        if header_line_bytes is None:
            # Legitimate partial header (EOF before newline) -> wait.
            binary_file.seek(record_start)
            return

        try:
            header_line = header_line_bytes[:-1].decode('utf-8').rstrip('\r')
        except UnicodeDecodeError:
            raise ValueError(
                f"corrupt output stream at byte {record_start}: header is not valid UTF-8"
            ) from None

        parts = header_line.split(',')
        msg_type = parts[1] if len(parts) > 1 else ""
        expected = expected_field_count(msg_type)
        if len(parts) != expected:
            raise ValueError(
                f"corrupt output stream at byte {record_start}: {msg_type!r} header "
                f"has {len(parts)} fields, expected {expected}: {header_line!r}"
            )

        _validate_iso_timestamp(
            parts[2],
            f"corrupt output stream at byte {record_start}: {msg_type!r} header",
        )
        try:
            payload_size = int(parts[0])
        except ValueError:
            raise ValueError(
                f"corrupt output stream at byte {record_start}: payload_size "
                f"{parts[0]!r} is not an integer"
            ) from None
        if payload_size < 0:
            raise ValueError(
                f"corrupt output stream at byte {record_start}: negative payload_size "
                f"{payload_size}"
            )
        # A corrupt-but-well-formed header can declare an arbitrarily large
        # payload_size. read_exact rejects such sizes for the streaming
        # readers; enforce the same bound here BEFORE the payload read below,
        # otherwise an absurd size is silently treated as a legitimate
        # partial payload (waiting forever and buffering the rest of the
        # file) instead of failing fast.
        if payload_size > MAX_PAYLOAD_BYTES:
            raise ValueError(
                f"corrupt output stream at byte {record_start}: payload_size "
                f"{payload_size} exceeds MAX_PAYLOAD_BYTES "
                f"({MAX_PAYLOAD_BYTES})"
            )

        payload_start = record_start + header_len
        # Re-seek to the exact payload start (the bounded header read may have
        # pulled in bytes beyond the header line), then read exactly the payload.
        binary_file.seek(payload_start)
        payload_bytes = binary_file.read(payload_size)
        if len(payload_bytes) < payload_size:
            # Legitimate partial payload -> wait.
            binary_file.seek(record_start)
            return

        if payload_size > 0:
            try:
                payload = payload_bytes.decode('utf-8')
            except UnicodeDecodeError:
                raise ValueError(
                    f"corrupt output stream at byte {payload_start}: payload is not valid UTF-8"
                ) from None
        else:
            payload = ""

        header = ParsedHeader.from_parts(parts)
        next_pos = payload_start + payload_size
        yield StreamedRecord(header, payload, record_start, next_pos)
        # Next iteration continues at next_pos; record_start will be next_pos.
