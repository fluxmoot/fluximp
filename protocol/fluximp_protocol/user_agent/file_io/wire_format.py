#!/usr/bin/env python3
"""
Wire format encode/decode functions for the file I/O user-agent protocol.

Consolidates the wire format functions used by both agent and user sides.
Agent-side functions write size-prefixed payloads with metadata headers.
Client-side functions build headers for input files.

Also contains file parsing utilities for recovering conversation state.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Dict, FrozenSet, Optional, Tuple, Generator, List

from fluximp_protocol.user_agent.file_io.defs import (
    IOFilenameError,
    MessageType,
    MAX_PAYLOAD_BYTES,
    _read_bounded_header_line,
    _validate_iso_timestamp,
    expected_field_count,
    extract_fork_turn_from_filename,
    extract_fork_turn_from_filename_strict,
    monotonic_stamp,
    parse_output_messages,
    parse_utc_timestamp,
    utc_now_iso,
)
from fluximp_protocol.user_agent.fork_turn_state import (
    ForkTurnState,
    ForkTurnStateError,
    IOFactKind,
    IOTurnFact,
    IOTurnStatus,
)

# ============================================================================
# Wire Format Functions
# ============================================================================

def _validate_header_field(name: str, value: Optional[str]) -> str:
    """Validate a string destined for a comma-separated wire-format header.

    The wire format uses comma/newline as structural separators, so a value
    containing either would desynchronize parsing. Because both sides of the
    stream are controlled, this is enforced on the writing side by raising
    rather than attempting to escape (escaping has no reader-side counterpart).

    ``None`` is a legitimate "no value" (e.g. an optional API field absent
    from the response) and is serialized as the empty string, which the
    reader round-trips back to ``None``.

    Args:
        name: Field name for the error message.
        value: String value to validate.

    Returns:
        The validated value, unchanged (empty string for ``None``).

    Raises:
        ValueError: If ``value`` is neither ``None`` nor a ``str`` (a
            non-string's rendered form round-trips only by accident: the
            separator checks below would raise a bare TypeError for numbers
            -- an exception class the upstream corruption paths do not
            recognize -- or pass a dict/list whose repr then embeds commas
            and desynchronizes the header field count on read); or if
            ``value`` contains a comma or newline.
    """
    if value is None:
        return ""
    if not isinstance(value, str):
        # A non-string cannot be validated for round-trip: it would either
        # crash this guard with a bare TypeError (numbers) instead of the
        # corruption-signal ValueError, or pass the `in` checks outright
        # (a dict/list from provider metadata) and be f-string-rendered
        # with embedded commas -- a header with more fields than the
        # declared message type, which every reader rejects on the next
        # scan and which permanently poisons the output file. Fail fast.
        raise ValueError(
            f"header field {name!r} must be a string or None, got "
            f"{type(value).__name__}: {value!r}"
        )
    if value == 'None':
        raise ValueError(
            f"header field {name!r} is the string 'None', which would be "
            f"silently corrupted on read; use '' for an absent value"
        )
    for ch in (",", "\n", "\r"):
        if ch in value:
            raise ValueError(
                f"header field {name!r} contains {ch!r}, which would corrupt "
                f"the wire-format header: {value!r}"
            )
    return value


def _validate_tags(tags: List[str]) -> List[str]:
    """Validate the ``+``-separated ``tags`` header field.

    The wire format uses comma/newline as structural separators and the
    ``tags`` field is embedded directly into the comma-separated header line
    (see :func:`write_model_message_to_user`), so a tag containing a comma,
    newline, or carriage return would desynchronize parsing exactly like any
    other header field. Tags are also joined with ``+`` and split on ``+`` by
    the reader, so a tag containing ``+`` cannot be represented unambiguously.
    Because both sides of the stream are controlled, this is enforced on the
    writing side by raising rather than attempting to escape (escaping has no
    reader-side counterpart) -- mirroring :func:`_validate_header_field`.
    """
    if isinstance(tags, str):
        raise ValueError(
            f"tags must be a list of tag strings, not a single string: {tags!r}"
        )
    if not isinstance(tags, (list, tuple)):
        raise ValueError(
            f"tags must be a list or tuple of tag strings, got "
            f"{type(tags).__name__}: {tags!r}"
        )
    for tag in tags:
        if not isinstance(tag, str):
            raise ValueError(f"tag must be a string, got {tag!r}")
        for ch in ("+", ",", "\n", "\r"):
            if ch in tag:
                raise ValueError(
                    f"tag {tag!r} contains {ch!r}, which would corrupt the "
                    f"wire-format tags header field"
                )
    return tags

def _validate_int_field(name: str, value) -> int:
    """Coerce an integer header field (token count / ``created``) to ``int``.

    The wire-format reader parses these fields with ``int()`` and fails fast on
    a value it cannot parse (see ``ParsedHeader._int``), so the writer must emit
    a value that round-trips. A float or other non-int value from the provider
    metadata would otherwise produce a header the reader rejects as corruption.
    ``None`` is not expected here (the callers resolve absent values to ``0``)
    but is normalized to ``0`` defensively.

    Raises:
        ValueError: If ``value`` cannot be coerced to a non-boolean integer.
    """
    if value is None:
        return 0
    if isinstance(value, bool):
        raise ValueError(
            f"header field {name!r} is a boolean, not an integer: {value!r}"
        )
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(
                f"header field {name!r} is not a valid integer: {value!r}"
            )
        value = int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"header field {name!r} is not a valid integer: {value!r}"
        ) from None


def _first_present_int(name: str, *sources: Tuple[str, Dict]) -> int:
    """Resolve an integer header field from the first source that has it.

    Each source is an ordered ``(key, mapping)`` pair. The first mapping that
    explicitly contains ``key`` supplies the value, so an explicit top-level
    ``0`` wins over a later nested fallback (an ``or``-chain would defer an
    explicit falsy value to later sources and mis-report the caller's
    requested count). An explicitly present ``None`` defers like an absent
    key. If no source has the key, ``0`` is returned.

    Args:
        name: Field name used in error messages.
        sources: Ordered ``(key, mapping)`` pairs, most specific first.

    Returns:
        The resolved value coerced via :func:`_validate_int_field`.

    Raises:
        ValueError: If the resolved value cannot be coerced to an integer.
    """
    for key, mapping in sources:
        if key in mapping and mapping[key] is not None:
            return _validate_int_field(name, mapping[key])
    return 0

def _stamp_for(handle: BinaryIO, file_id: Optional[str] = None) -> str:
    """Per-file monotonic header stamp for a write to ``handle``.

    The agent-side writers call this instead of ``utc_now_iso()`` so every
    record written to a stream carries a strictly-increasing header timestamp
    (``monotonic_stamp``, §7.7). ``file_id`` is the protocol-known file path
    when the caller has it; otherwise it is derived from the handle's ``name``
    (the path for a real file) or a stable per-handle id (e.g. ``BytesIO`` in
    tests), so the per-file key is stable for the stream's lifetime.
    """
    if file_id is not None:
        return monotonic_stamp(file_id)
    name = getattr(handle, "name", None)
    key = str(name) if name else f"<handle:{id(handle)}>"
    return monotonic_stamp(key)


def write_request_header(handle: BinaryIO, start_time: str, agent_type: str,
                          provider: str, requested_model: str,
                          file_id: Optional[str] = None):
    """Write a request header to the output stream when API call starts.

    Format: 0,request,<timestamp>,<agent_type>,<provider>,<requested_model>,<start_time_iso>\n
    Args:
        handle: Binary writable handle to the output stream
        start_time: ISO format timestamp when request started
        agent_type: The agent type (e.g., 'coder', 'scrutinator')
        provider: The model provider name
        requested_model: The model name being requested
        file_id: The protocol-known file path, used to key the per-file
            monotonic header stamp (defaults to the handle's name).
    """
    timestamp = _stamp_for(handle, file_id)
    agent_type = _validate_header_field("agent_type", agent_type)
    provider = _validate_header_field("provider", provider)
    requested_model = _validate_header_field("requested_model", requested_model)
    if start_time:
        start_time = parse_utc_timestamp(start_time).isoformat()
    start_time = _validate_header_field("start_time", start_time)
    header = f"0,request,{timestamp},{agent_type},{provider},{requested_model},{start_time}\n"
    content_bytes = header.encode('utf-8')
    handle.write(content_bytes)
    handle.flush()


def write_response(handle: BinaryIO, content: str, reasoning_content: Optional[str],
                   metadata: dict, file_id: Optional[str] = None):
    """Write a response message to the output file.

    Written immediately when the LLM API response arrives, before tool execution.
    Carries all response metadata (deduplicated from other messages) and the
    LLM's text content and reasoning via subheader-encoded payload.

    Token counts may be nested under ``metadata['token_usage']`` (LangChain/OpenAI
    convention) or at the top level; the first location that explicitly carries
    a key wins (including an explicit ``0``, via :func:`_first_present_int`),
    and OpenAI's ``prompt_tokens_details``/``completion_tokens_details``
    sub-dicts are used as fallbacks for cached/reasoning tokens.

    Header format:
        <payload_size>,response,<timestamp>,<end_time>,<actual_model>,<finish_reason>,
        <prompt_tokens>,<completion_tokens>,<total_tokens>,<cached_tokens>,
        <reasoning_tokens>,<system_fingerprint>,<created>

    Payload format (subheader-encoded):
        <sub_size>,content\n<sub_size bytes of content>
        <sub_size>,reasoning\n<sub_size bytes of reasoning>

    Only non-empty subpayloads are included. The outer payload_size is the
    total byte count of all subheaders + subpayload text bytes.
    """
    # token_usage (and the *_details sub-dicts) must be dicts. Fail fast here,
    # before any write, rather than raising an AttributeError mid-flight or
    # writing a malformed response to the output stream.
    token_usage = metadata.get("token_usage")
    if token_usage is None:
        token_usage = {}
    if not isinstance(token_usage, dict):
        raise TypeError(
            f"metadata['token_usage'] must be a dict, got "
            f"{type(token_usage).__name__}: {token_usage!r}"
        )
    prompt_details = token_usage.get("prompt_tokens_details")
    if prompt_details is None:
        prompt_details = {}
    if not isinstance(prompt_details, dict):
        raise TypeError(
            f"token_usage['prompt_tokens_details'] must be a dict, got "
            f"{type(prompt_details).__name__}: {prompt_details!r}"
        )
    completion_details = token_usage.get("completion_tokens_details")
    if completion_details is None:
        completion_details = {}
    if not isinstance(completion_details, dict):
        raise TypeError(
            f"token_usage['completion_tokens_details'] must be a dict, got "
            f"{type(completion_details).__name__}: {completion_details!r}"
        )

    _end = metadata.get("end_time")
    end_time = _end if _end is not None else utc_now_iso()
    _actual = metadata.get("actual_model")
    _model = metadata.get("model")
    actual_model = _actual if _actual is not None else (_model if _model is not None else "")
    finish_reason = metadata.get("finish_reason", "")
    prompt_tokens = _first_present_int(
        "prompt_tokens",
        ("prompt_tokens", metadata),
        ("prompt_tokens", token_usage),
    )
    completion_tokens = _first_present_int(
        "completion_tokens",
        ("completion_tokens", metadata),
        ("completion_tokens", token_usage),
    )
    total_tokens = _first_present_int(
        "total_tokens",
        ("total_tokens", metadata),
        ("total_tokens", token_usage),
    )
    cached_tokens = _first_present_int(
        "cached_tokens",
        ("cached_tokens", metadata),
        ("cached_tokens", token_usage),
        ("cached_tokens", prompt_details),
    )
    reasoning_tokens = _first_present_int(
        "reasoning_tokens",
        ("reasoning_tokens", metadata),
        ("reasoning_tokens", token_usage),
        ("reasoning_tokens", completion_details),
    )
    system_fingerprint = metadata.get("system_fingerprint", "")
    created = _validate_int_field(
        "created",
        metadata.get("created", 0),
    )

    # Build subpayloads
    payload_parts = []

    if content:
        if not content.endswith('\n'):
            content += '\n'
        content_bytes = content.encode('utf-8')
        subheader = f"{len(content_bytes)},content\n"
        payload_parts.append(subheader.encode('utf-8') + content_bytes)

    if reasoning_content:
        if not reasoning_content.endswith('\n'):
            reasoning_content += '\n'
        reasoning_bytes = reasoning_content.encode('utf-8')
        subheader = f"{len(reasoning_bytes)},reasoning\n"
        payload_parts.append(subheader.encode('utf-8') + reasoning_bytes)

    # Calculate total payload size
    payload_data = b''.join(payload_parts)
    payload_size = len(payload_data)

    # Build header
    timestamp = _stamp_for(handle, file_id)
    if end_time:
        end_time = parse_utc_timestamp(end_time).isoformat()
    end_time = _validate_header_field("end_time", end_time)
    actual_model = _validate_header_field("actual_model", actual_model)
    finish_reason = _validate_header_field("finish_reason", finish_reason)
    system_fingerprint = _validate_header_field("system_fingerprint", system_fingerprint)
    header = (
        f"{payload_size},response,{timestamp},"
        f"{end_time},{actual_model},{finish_reason},"
        f"{prompt_tokens},{completion_tokens},{total_tokens},"
        f"{cached_tokens},{reasoning_tokens},"
        f"{system_fingerprint},{created}\n"
    )

    # Write header + payload
    full_content = header.encode('utf-8') + payload_data
    handle.write(full_content)
    handle.flush()


def parse_response_payload(payload: str) -> Dict:
    """Parse subheader-encoded response payload.

    Format: <sub_size>,<text_type>\n<sub_size bytes of text>[<sub_size>,<text_type>\n...]
    Each subpayload is: subheader line (size,type) followed by exactly sub_size bytes of text.

    Returns:
        Dict mapping text_type to text string (e.g., {"content": "...", "reasoning": "..."})

    Raises:
        ValueError: If the payload is corrupt (malformed sub-header, unparseable
            or negative size, or a size that overruns the payload). Fails fast
            on a corrupt stream rather than silently returning a partial dict.
    """
    result = {}
    raw = payload.encode('utf-8')
    pos = 0

    while pos < len(raw):
        # Find the next newline (end of subheader)
        nl = raw.find(b'\n', pos)
        if nl == -1:
            raise ValueError(
                f"corrupt response payload at byte {pos}: sub-header has no newline"
            )

        try:
            subheader = raw[pos:nl].decode('utf-8')
        except UnicodeDecodeError:
            raise ValueError(
                f"corrupt response payload at byte {pos}: sub-header is not "
                f"valid UTF-8"
            ) from None
        comma = subheader.find(',')
        if comma == -1:
            raise ValueError(
                f"corrupt response payload at byte {pos}: sub-header missing "
                f"comma separator: {subheader!r}"
            )

        try:
            sub_size = int(subheader[:comma])
        except ValueError:
            raise ValueError(
                f"corrupt response payload at byte {pos}: sub_size "
                f"{subheader[:comma]!r} is not an integer"
            ) from None
        if sub_size < 0:
            raise ValueError(
                f"corrupt response payload at byte {pos}: negative sub_size "
                f"{sub_size}"
            )

        text_type = subheader[comma + 1:]
        text_start = nl + 1
        text_end = text_start + sub_size

        if text_end > len(raw):
            raise ValueError(
                f"corrupt response payload at byte {pos}: sub_size {sub_size} "
                f"overruns payload of {len(raw)} bytes"
            )

        try:
            text = raw[text_start:text_end].decode('utf-8')
        except UnicodeDecodeError:
            raise ValueError(
                f"corrupt response payload at byte {pos}: subpayload is not "
                f"valid UTF-8"
            ) from None
        result[text_type] = text

        pos = text_end

    return result


def write_payload(handle: BinaryIO, payload_type: str, payload: Optional[str],
                  execution_successful: int = 0, file_id: Optional[str] = None):
    """Write a size-prefixed payload to the output stream.

    Simplified format with only common header fields plus type-specific fields:
        Common: <payload_size>,<payload_type>,<timestamp>
        tool_use_result adds: ,<execution_successful>

    Args:
        handle: Binary writable handle to the output stream
        payload_type: Type of payload (e.g., 'tool_use', 'tool_use_result')
        payload: The actual content to write (None for empty payload)
        execution_successful: Truthy if tool execution succeeded (serialized as 'true'/'false'; legacy '1'/'0' accepted when parsing)
    """
    if payload is not None:
        # Ensure payload ends with newline for proper separation
        if payload and not payload.endswith('\n'):
            payload += '\n'
        # Calculate byte length of payload (UTF-8 encoded)
        payload_bytes = len(payload.encode('utf-8'))
    else:
        payload_bytes = 0

    # Build header with common fields + type-specific fields
    payload_type = _validate_header_field("payload_type", payload_type)
    # Fail fast on a payload_type this writer cannot represent correctly:
    # write_payload emits a 3-field header for every type except
    # tool_use_result (4 fields). A known type with a different layout
    # (request, response, model_message_to_user, system_agent_*,
    # tool_permission_response, ...) would otherwise be written here as a
    # 3-field record that every reader rejects as corrupt -- permanently
    # poisoning the output file. An unknown type is likewise corruption
    # (expected_field_count raises).
    allowed_field_count = 4 if payload_type == "tool_use_result" else 3
    field_count = expected_field_count(payload_type)
    if field_count != allowed_field_count:
        raise ValueError(
            f"payload_type {payload_type!r} requires a {field_count}-field "
            f"header and cannot be written by write_payload; use its "
            f"dedicated writer"
        )
    timestamp = _stamp_for(handle, file_id)
    if payload_type == "tool_use_result":
        success_str = "true" if execution_successful else "false"
        header = f"{payload_bytes},{payload_type},{timestamp},{success_str}\n"
    else:
        header = f"{payload_bytes},{payload_type},{timestamp}\n"

    full_content = header
    if payload is not None:
        full_content += payload

    # Write in binary mode to ensure consistent Unix-style line endings on all platforms
    content_bytes = full_content.encode('utf-8')
    handle.write(content_bytes)
    handle.flush()


def write_model_message_to_user(handle: BinaryIO, payload: Optional[str],
                                tags: List[str] = None,
                                format: str = "markdown",
                                file_id: Optional[str] = None):
    """Write a model_message_to_user payload to the output stream.

    Header format:
        <payload_size>,model_message_to_user,<timestamp>,<tags>,<format>

    Tags are multiple values joined with '+' (e.g., 'findings_report+work_completion_summary').
    Format is 'plain_text' or 'markdown'.

    Args:
        handle: Binary writable handle to the output stream
        payload: The message content to write (None for empty payload)
        tags: List of tags indicating what the message contains
        format: Format of the message content ('plain_text' or 'markdown')
    """
    if payload is not None:
        # Ensure payload ends with newline for proper separation
        if payload and not payload.endswith('\n'):
            payload += '\n'
        payload_bytes = len(payload.encode('utf-8'))
    else:
        payload_bytes = 0

    # Encode tags as '+'-separated string
    if tags:
        _validate_tags(tags)
    format = _validate_header_field("format", format)
    tags_str = '+'.join(tags) if tags else ''

    timestamp = _stamp_for(handle, file_id)
    header = f"{payload_bytes},model_message_to_user,{timestamp},{tags_str},{format}\n"

    full_content = header
    if payload is not None:
        full_content += payload

    content_bytes = full_content.encode('utf-8')
    handle.write(content_bytes)
    handle.flush()




def write_system_agent_started(handle: BinaryIO, agent_type: str, subdir: str,
                               file_id: Optional[str] = None):
    """Write a system_agent_started message to the primary output stream.
    
    Zero-payload message with all fields in the comma-separated header.
    
    Header format:
        0,system_agent_started,<timestamp>,<agent_type>,<subdir>
    
    Args:
        handle: Binary writable handle to the primary output stream
        agent_type: The type of system agent (e.g., 'scrutinator', 'summarizer')
        subdir: The subdirectory name for this agent run
    """
    timestamp = _stamp_for(handle, file_id)
    agent_type = _validate_header_field("agent_type", agent_type)
    subdir = _validate_header_field("subdir", subdir)
    header = f"0,system_agent_started,{timestamp},{agent_type},{subdir}\n"
    content_bytes = header.encode('utf-8')
    handle.write(content_bytes)
    handle.flush()


def write_system_agent_stopped(handle: BinaryIO, agent_type: str,
                               success: str, error: str,
                               file_id: Optional[str] = None):
    """Write a system_agent_stopped message to the primary output stream.

    The error text is carried as the record payload (raw UTF-8) so free-form
    diagnostics can contain any character -- including comma, newline, or
    carriage return -- without corrupting the header. The subdir is not
    included since it was already provided in the corresponding
    system_agent_started message.

    Header format:
        <payload_size>,system_agent_stopped,<timestamp>,<agent_type>,<success>
    Payload:
        <error bytes, raw UTF-8> (empty payload when success)

    Args:
        handle: Binary writable handle to the primary output stream
        agent_type: The type of system agent (e.g., 'scrutinator', 'summarizer')
        success: "true" or "false"
        error: Error string (empty if no error). Carried as the payload, so no
            validation or truncation is required.
    """
    timestamp = _stamp_for(handle, file_id)
    agent_type = _validate_header_field("agent_type", agent_type)
    success_norm = str(success).strip().lower()
    success_str = "true" if success_norm in ("true", "1", "yes") else "false"

    # The error text is the payload: raw UTF-8 with no header-field validation.
    # Normalize like the other payload writers (append a trailing newline for
    # text payloads) so readers receive a complete, well-formed payload.
    if error is None:
        error = ""
    else:
        error = str(error)
    if error and not error.endswith('\n'):
        error += '\n'
    payload_bytes = len(error.encode('utf-8'))

    header = f"{payload_bytes},system_agent_stopped,{timestamp},{agent_type},{success_str}\n"
    full_content = header + error
    content_bytes = full_content.encode('utf-8')
    handle.write(content_bytes)
    handle.flush()

def read_exact(input_handle, num_bytes, timeout, poll_interval=0.05):
    """Read exactly ``num_bytes`` from a streaming binary handle.

    The peer writes the header and payload in separate ``write()`` calls, so
    the payload may not yet be fully appended when the header has been read.
    This loops between reading and waiting until the full byte count is
    accumulated. Raises :class:`TimeoutError` if ``timeout`` (in seconds)
    elapses first. Callers must only decode after this returns, so the
    payload is always complete.

    Args:
        input_handle: File handle opened for reading (binary mode).
        num_bytes: Exact number of bytes to read.
        timeout: Maximum seconds to wait for the full payload.
        poll_interval: Seconds to sleep between reads while waiting.

    Returns:
        Exactly ``num_bytes`` bytes.

    Raises:
        ValueError: If ``num_bytes`` exceeds ``MAX_PAYLOAD_BYTES`` (corrupt
            stream -- fail fast rather than blocking on an impossible payload).
        TimeoutError: If ``timeout`` elapses before the payload is complete.
            The handle is seeked back to its position before this call --
            which for the drain helpers in agent.py is the record start, but
            for the streaming readers is only the payload start because
            read_payload / request_permission consumed the record's header
            first. Those callers rewind over the header themselves in their
            TimeoutError handler, so the whole-record convention still
            holds: a caller that handles the timeout and retries re-reads
            the entire record instead of resuming mid-payload, which would
            misparse payload bytes as a header.
    """
    assert timeout is not None, "read_exact requires a non-None timeout"
    if num_bytes > MAX_PAYLOAD_BYTES:
        raise ValueError(
            f"corrupt stream: payload_size {num_bytes} exceeds "
            f"MAX_PAYLOAD_BYTES ({MAX_PAYLOAD_BYTES})"
        )
    if num_bytes <= 0:
        return b""
    chunks = []
    remaining = num_bytes
    start = time.monotonic()
    start_pos = input_handle.tell()
    while remaining > 0:
        chunk = input_handle.read(remaining)
        if chunk:
            chunks.append(chunk)
            remaining -= len(chunk)
            continue
        # No data available yet (streaming append): wait, then retry. We sleep
        # BEFORE enforcing the timeout so even a zero remaining budget
        # (recomputed right before this call) gives the peer's separate payload
        # write a chance to land instead of raising TimeoutError on the very
        # first empty read.
        time.sleep(poll_interval)
        if time.monotonic() - start > timeout:
            # Do not leave the stream mid-payload (same convention as the
            # drain helpers in agent.py): restore the stream to the position
            # before this payload. Callers that consumed this record's header
            # first (read_payload / request_permission) rewind over the
            # header in their own TimeoutError handler, so a caller that
            # handles the timeout and retries re-reads the whole record
            # instead of misparsing payload bytes as a header.
            input_handle.seek(start_pos)
            raise TimeoutError(
                f"payload read timed out: expected {num_bytes} bytes, "
                f"got {num_bytes - remaining} before timeout"
            )
    return b"".join(chunks)

def read_payload(input_handle, timeout) -> Tuple[str, str]:
    """Read a size-prefixed payload from an input file handle.

    Format: <payload_bytes>,<payload_type>,<timestamp>\n<payload>

    Reads from the current file position (streaming mode). The peer writes the
    header and payload in separate writes, so the file may be observed before
    either is complete. The header is read with :func:`read_streaming_header`,
    which waits until the header is newline-terminated (seeking back over
    partial header writes lacking a trailing newline) and fails fast on a
    corrupt stream. The payload is then read with :func:`read_exact`, which
    waits until the full byte count has arrived. A single ``timeout`` budget
    is shared across both
    waits, so the payload is only decoded once the entire message is complete.

    Args:
        input_handle: File handle opened for reading (binary mode)
        timeout: Maximum seconds to wait for the full message; a
            ``TimeoutError`` is raised if it elapses first.

    Returns:
        Tuple of (payload_type, payload_content)

    Raises:
        ValueError: If the file format is invalid
        TimeoutError: If the full message does not arrive within ``timeout``
    """
    assert timeout is not None, "read_payload requires a non-None timeout"
    # Wait for a complete, valid header. read_streaming_header seeks back over
    # partial header writes (no terminating newline yet) and fails fast on a
    # corrupt stream (wrong field count, unparseable/negative payload size, or
    # invalid UTF-8).
    start = time.monotonic()
    # Track the position before the header read: the header is consumed by
    # the read_streaming_header call below, so on a payload timeout the
    # rewind below must restore the record start -- the position before the
    # HEADER -- rather than stopping at the payload start read_exact rewinds
    # to (see read_exact's whole-record convention).
    record_start = input_handle.tell()
    parts = read_streaming_header(input_handle, timeout=timeout)
    msg_type = parts[1]
    payload_size = int(parts[0])

    # Read the payload with the remaining budget so the header wait plus the
    # payload wait cannot exceed the caller's timeout.
    remaining = max(0.0, timeout - (time.monotonic() - start))
    try:
        payload_bytes = read_exact(input_handle, payload_size, timeout=remaining)
    except TimeoutError:
        # The header of this record is already consumed, and read_exact
        # seeks back only to its own start (the payload start). Rewind to
        # before the header as well so the stream state matches "this record
        # not consumed" (the same convention as the drain helpers in
        # agent.py) and a caller that handles the timeout and retries reads
        # the whole record again instead of misparsing payload bytes as a
        # header.
        input_handle.seek(record_start)
        raise

    payload = payload_bytes.decode('utf-8')

    return msg_type, payload


def read_streaming_header(input_handle, timeout, poll_interval=0.05):
    """Block until a complete, valid header line is available on ``input_handle``.

    The peer appends the header and payload in separate writes, so a header line
    may be only partially present when first observed. This waits (seeking back
    and re-reading) until the header has its expected number of comma-separated
    fields, mirroring how :func:`read_exact` waits for payload bytes.

    * No terminating newline yet -> the peer is still writing; wait and retry
      until ``timeout`` (or ``TimeoutError``).
    * No newline within ``MAX_HEADER_BYTES`` -> an impossible (over-long)
      header fragment; corruption -- the bounded read fails fast in
      ``_read_bounded_header_line`` with ``ValueError`` exactly like
      ``parse_streamed_records`` (a corrupt stream is never buffered
      indefinitely).
    * Newline present but wrong field count for ``msg_type`` -> the stream is
      corrupt; raise ``ValueError`` immediately (fail fast).
    * Correct field count but unparseable/negative size -> corrupt; raise
      ``ValueError``.

    Args:
        input_handle: File handle opened for reading (binary mode).
        timeout: Maximum seconds to wait for a complete header.
        poll_interval: Seconds to sleep between re-reads while waiting.

    Returns:
        The comma-separated header fields as a list of strings.

    Raises:
        TimeoutError: If a complete header does not arrive within ``timeout``.
        ValueError: If a complete header is corrupt (wrong field count, an
            unparseable/negative payload size, or a header line exceeding
            ``MAX_HEADER_BYTES`` without a terminating newline).
    """
    assert timeout is not None, "read_streaming_header requires a non-None timeout"
    start = time.monotonic()
    while True:
        pos = input_handle.tell()
        # Bounded header read (mirrors parse_streamed_records): via
        # _read_bounded_header_line, a header fragment exceeding
        # MAX_HEADER_BYTES without a terminating newline is structurally
        # impossible wire data -- corruption -- and fails fast there instead
        # of buffering the rest of the file via an unbounded readline().
        header_line, header_len = _read_bounded_header_line(input_handle)
        if header_line is None:
            # No terminating newline within the bound: either nothing has
            # been appended since the last record, or the peer is still
            # writing a partial header. Seek back so the partial header is
            # NOT consumed, and wait until the newline arrives -- a header is
            # complete only once it is newline-terminated (the writers always
            # emit the trailing '\n' in the same write as the rest of the
            # header).
            input_handle.seek(pos)
            if time.monotonic() - start > timeout:
                raise TimeoutError("header read timed out")
            time.sleep(poll_interval)
            continue

        # Re-position exactly after the header line: the bounded read may
        # have pulled bytes beyond the newline (the beginning of the payload)
        # into the handle, so reset the logical position to the payload start
        # before the caller's payload wait -- same as parse_streamed_records.
        input_handle.seek(pos + header_len)

        try:
            line_str = header_line.decode('utf-8').rstrip('\r\n')
        except UnicodeDecodeError:
            # Invalid UTF-8 in a header means the stream is corrupt -- fail
            # fast with ValueError rather than a UnicodeDecodeError.
            raise ValueError(
                f"corrupt stream: header is not valid UTF-8"
            ) from None
        parts = line_str.split(',')

        msg_type = parts[1] if len(parts) > 1 else ""
        expected = expected_field_count(msg_type)

        # The header is newline-terminated, so it is complete. A field count
        # that differs from what the type requires is corruption -- fail fast.
        # (A partial header is handled above by the newline check, which waits
        # instead.)
        if len(parts) != expected:
            raise ValueError(
                f"corrupt stream: {msg_type!r} header has {len(parts)} fields, "
                f"expected {expected}: {line_str!r}"
            )

        _validate_iso_timestamp(parts[2], f"corrupt stream: {msg_type!r} header")

        # Correct field count: the size must still be a valid non-negative int,
        # otherwise the stream is corrupt -- fail fast.
        try:
            size = int(parts[0])
        except ValueError:
            raise ValueError(
                f"corrupt stream: payload_size {parts[0]!r} is not an integer "
                f"in header {line_str!r}"
            ) from None
        if size < 0:
            raise ValueError(
                f"corrupt stream: negative payload_size {size} in header {line_str!r}"
            )

        return parts

# ============================================================================
# Message dataclass
# ============================================================================

@dataclass
class Message:
    """A single message to send to the agent."""
    msg_type: MessageType
    content: str


# ============================================================================
# Wire format builders
# ============================================================================

def build_header(payload: bytes, msg_type: MessageType, timestamp: str) -> str:
    """Build a header string for input files.

    Format: <payload_size>,<msg_type>,<timestamp>\n

    Args:
        payload: The message payload as bytes
        msg_type: The type of message
        timestamp: The header timestamp (computed by the caller via
            ``monotonic_stamp`` so input-file records carry strictly-increasing
            per-file stamps, §7.7).

    Returns:
        Header string (without payload -- caller must write payload separately)
    """
    return f"{len(payload)},{msg_type.value},{timestamp}\n"


def build_permission_response_header(payload: bytes, approved: bool, timestamp: str) -> str:
    """Build a header string for tool permission response.

    Format: <payload_size>,tool_permission_response,<timestamp>,approved\n

    Args:
        payload: The message payload as bytes
        approved: Boolean indicating whether the permission is approved
        timestamp: The header timestamp (computed by the caller via
            ``monotonic_stamp`` so input-file records carry strictly-increasing
            per-file stamps, §7.7).

    Returns:
        Header string (without payload -- caller must write payload separately)
    """
    approved_str = "true" if approved else "false"
    return f"{len(payload)},tool_permission_response,{timestamp},{approved_str}\n"


def build_cancel_turn_header(timestamp: str) -> str:
    """Build a header string for cancel_turn.

    Format: 0,cancel_turn,<timestamp>\n

    Args:
        timestamp: The header timestamp (computed by the caller via
            ``monotonic_stamp`` so input-file records carry strictly-increasing
            per-file stamps, §7.7).

    Returns:
        Header string (no payload for cancel_turn)
    """
    return f"0,cancel_turn,{timestamp}\n"

# ============================================================================
# File Parsing Functions
# ============================================================================





# Terminal-normal message markers (the agent-specific completion markers).
# Terminal messages are mutually exclusive per agent class.
_TERMINAL_NORMAL_TYPES = frozenset({
    MessageType.MODEL_MESSAGE_TO_USER,
    MessageType.SCRUTINATOR_REPORT,
    MessageType.CONDITIONAL_REPORT,
    MessageType.SUMMARY,
})
_TERMINAL_NORMAL_VALUES = frozenset(m.value for m in _TERMINAL_NORMAL_TYPES)


def expected_terminal_marker_for_agent(agent_type: str) -> MessageType:
    """The exclusive terminal-normal marker the given agent class produces.

    Terminal messages are mutually exclusive per agent class: conditional
    agents (``conditional*``) report with ``CONDITIONAL_REPORT``; the
    scrutinator with ``SCRUTINATOR_REPORT``; the summarizer with ``SUMMARY``;
    and the primary coder agents with ``MODEL_MESSAGE_TO_USER``.
    """
    at = (agent_type or "").lower()
    if at.startswith("conditional"):
        return MessageType.CONDITIONAL_REPORT
    if at.startswith("scrutinator"):
        return MessageType.SCRUTINATOR_REPORT
    if at.startswith("summarizer"):
        return MessageType.SUMMARY
    return MessageType.MODEL_MESSAGE_TO_USER


def classify_output_status(output_file: Path,
                           expected_terminal_marker: Optional[MessageType] = None) -> IOTurnStatus:
    """Classify a primary output file's current content into an OUT status.

    A ``MAX_REQUESTS_REACHED`` or ``infrastructure_error`` header type yields
    ``COMPLETE_ERROR``; a ``CANCEL_TURN_ACK`` yields ``CANCELLED``; a terminal
    normal message yields ``COMPLETE_NORMAL``; otherwise the output is
    ``INCOMPLETE``. Error and cancellation outcomes are universal and take
    precedence over a normal message.

    ``expected_terminal_marker`` is the exclusive terminal-normal marker this
    agent class is expected to produce (see :func:`expected_terminal_marker_for_agent`),
    or ``None`` for a permissive scan (GUI display / tests). Because terminal
    messages are mutually exclusive per agent, a strict scan accepts exactly
    that marker as ``COMPLETE_NORMAL`` and raises ``ForkTurnStateError`` for any
     other terminal-normal marker (invalid on-disk state).

    A strict scan also fails fast with ``ForkTurnStateError`` when the file
    ends in a torn (incomplete) trailing record: strict scans run only at
    agent startup, where the output file is static (the previous agent
    process is dead, and only agent processes write ``.out.txt`` files, one
    flushed write per record), so a torn tail can only exist if the prior
    process died mid-record. A permissive scan (``None``) tolerates the
    trailing partial record because the GUI scans conversations whose agent
    may be mid-write.
    """
    if not output_file.is_file():
        return IOTurnStatus.INCOMPLETE
    content_bytes = output_file.read_bytes()
    has_normal = False
    has_error = False
    has_cancel = False
    parsed_pos = 0
    for header, payload, next_pos in parse_output_messages(content_bytes):
        parsed_pos = next_pos
        if (header.is_type(MessageType.MAX_REQUESTS_REACHED)
                or header.is_type(MessageType.INFRASTRUCTURE_ERROR)):
            has_error = True
        elif header.is_type(MessageType.CANCEL_TURN_ACK):
            has_cancel = True
        elif header.msg_type in _TERMINAL_NORMAL_VALUES:
            if (expected_terminal_marker is not None
                    and header.msg_type != expected_terminal_marker.value):
                raise ForkTurnStateError(
                    f"unexpected terminal marker {header.msg_type!r} in primary "
                    f"output {output_file.name} (expected "
                    f"{expected_terminal_marker.value!r})"
                )
            has_normal = True
    # Strict-scan torn-tail detection: parse_output_messages stops silently
    # on an incomplete trailing record (a legitimate partial write for a
    # live writer), but the file is static in a strict scan -- the startup
    # scan runs before this process opens any output handle, and only the
    # agent process writes .out.txt files, one flushed write per record --
    # so a parsed position short of the file end means the file ends in a
    # torn record that can only exist if the prior process died mid-record.
    # Fail fast instead of misclassifying: INCOMPLETE when the torn tail
    # precedes every terminal marker would let a recovery run reopen this
    # file and append records AFTER the torn tail, and a terminal status
    # when the torn tail follows one would hide the corruption forever.
    # Failing fast here is also what keeps the torn-tail-with-data-after-it
    # state from ever forming (no recovery run can start over a torn file).
    if expected_terminal_marker is not None and parsed_pos < len(content_bytes):
        raise ForkTurnStateError(
            f"corrupt output stream: {output_file.name} ends in an "
            f"incomplete record at byte {parsed_pos} (file size "
            f"{len(content_bytes)}); a torn tail only exists if the prior "
            f"process died mid-record"
        )
    if has_error:
        return IOTurnStatus.COMPLETE_ERROR
    if has_cancel:
        return IOTurnStatus.CANCELLED
    if has_normal:
        return IOTurnStatus.COMPLETE_NORMAL
    return IOTurnStatus.INCOMPLETE


def _is_coordinate_shaped(filename: str, kind: IOFactKind) -> bool:
    """True if ``filename`` is a candidate primary protocol filename.

    ``.end`` files are ``{a}.{b}.end`` (3 dot-separated parts); ``.in.txt`` and
    ``.out.txt`` files are ``{a}.{b}.in.txt`` / ``{a}.{b}.out.txt`` (4 parts).
    A name that matches the primary suffix but carries extra components
    (``0.1.2.in.txt``, ``0.1.tmp.out.txt``, ``0.1.2.end``) is also a candidate
    so the strict parser rejects it with :class:`IOFilenameError` -- it is a
    malformed primary filename, not an unrelated file. Unrelated names that do
    not match the primary suffix shape (``garbage.end``, ``notes.out.txt``,
    ``notes.in.txt``) are excluded before the strict parser runs.
    Coordinate-shaped but malformed names (``-1.0.out.txt``, ``00.05.in.txt``,
    ``01.2.end``) are kept for the same reason.
    """
    parts = filename.split(".")
    if kind is IOFactKind.END:
        return len(parts) >= 3 and parts[-1] == "end"
    return len(parts) >= 4 and parts[-2] in ("in", "out") and parts[-1] == "txt"

def _scan_io_facts_raw(io_path: Path,
                       expected_terminal_marker: Optional[MessageType] = None) -> FrozenSet[IOTurnFact]:
    """Raw filename/content scanner.

    Returns the unordered fact set WITHOUT running whole-snapshot
    reconciliation. Used by :func:`scan_io_facts` and tests.
    """
    facts = set()
    # Broad suffix globs gather candidate files; a shape filter then keeps only
    # names matching the primary suffix ({fork}.{turn}.in.txt / .out.txt / .end),
    # including extra-component names (0.1.2.in.txt, 0.1.tmp.out.txt,
    # 0.1.2.end) so the strict parser rejects them as malformed. Everything
    # else -- stray or temporary files that do not match the primary suffix
    # (e.g. ``notes.out.txt``, ``garbage.end``) -- is ignored before the strict
    # parser runs, so it can never crash the snapshot scan. Coordinate-shaped
    # but malformed names (-1.0.out.txt, 00.05.in.txt, 01.2.end) are kept for
    # the same reason.
    for glob_pattern, kind in (
        ("*.in.txt", IOFactKind.IN),
        ("*.out.txt", IOFactKind.OUT),
        ("*.end", IOFactKind.END),
    ):
        for path in io_path.glob(glob_pattern):
            if not path.is_file():
                continue
            if not _is_coordinate_shaped(path.name, kind):
                continue
            fork, turn = extract_fork_turn_from_filename_strict(path.name)
            if kind is IOFactKind.OUT:
                status = classify_output_status(path, expected_terminal_marker)
            else:
                status = IOTurnStatus.NOT_APPLICABLE
            facts.add(IOTurnFact(fork, turn, kind, status))
    return frozenset(facts)


def scan_io_facts(io_path: Path,
                  expected_terminal_marker: Optional[MessageType] = None) -> ForkTurnState:
    """Return a fully validated and reconciled ForkTurnState from disk.

    ``expected_terminal_marker`` forwards to :func:`classify_output_status`;
    ``None`` is a permissive scan. A strict scan (non-``None`` marker)
    additionally fails fast on an output file that ends in a torn
    (incomplete) trailing record -- see :func:`classify_output_status`.
    """
    raw = _scan_io_facts_raw(io_path, expected_terminal_marker)
    return ForkTurnState(raw).reconcile_snapshot()
