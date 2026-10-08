
#!/usr/bin/env python3
"""
TurnResult and accumulate_message for tracking agent turn state.

Shared dispatch logic for message-type classification and token-count
extraction.  Used by both the GUI conversation manager and test utilities.
"""

import json
from dataclasses import dataclass, field
from typing import Optional

from fluximp_protocol.user_agent.file_io.defs import MessageType, ParsedHeader


@dataclass
class TurnResult:
    """Result of parsing an output file.

    Contains both completion status and token statistics.
    """
    is_complete: bool = False
    completion_type: Optional[str] = None
    has_tool_use: bool = False
    requests: int = 0
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cached_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    tool_permission_request: Optional[dict] = None


class MaxRequestsReachedError(Exception):
    """Raised when the agent hits the maximum request limit."""
    pass


def accumulate_message(result: TurnResult, header: ParsedHeader, payload: str) -> None:
    """Accumulate a single parsed message into a TurnResult.

    Accepts a pre-configured TurnResult and a parsed header/payload pair,
    then checks the header type to update the appropriate TurnResult fields.

    Args:
        result: TurnResult to accumulate into.
        header: Parsed message header.
        payload: Raw payload string for the message.
    """
    # Count request headers
    if header.is_request:
        result.requests += 1
        return

    # Dispatch on message type
    if header.is_type(MessageType.RESPONSE):
        pass
    elif header.is_type(MessageType.MODEL_MESSAGE_TO_USER):
        result.is_complete = True
        result.completion_type = header.msg_type
    elif header.is_type(MessageType.SCRUTINATOR_REPORT):
        result.is_complete = True
        result.completion_type = header.msg_type
    elif header.is_type(MessageType.CONDITIONAL_REPORT):
        result.is_complete = True
        result.completion_type = header.msg_type
    elif header.is_type(MessageType.SUMMARY):
        result.is_complete = True
        result.completion_type = header.msg_type
    elif header.is_type(MessageType.MAX_REQUESTS_REACHED):
        result.is_complete = True
        result.completion_type = header.msg_type
    elif header.is_type(MessageType.INFRASTRUCTURE_ERROR):
        result.is_complete = True
        result.completion_type = header.msg_type
    elif header.is_type(MessageType.TOOL_USE):
        result.has_tool_use = True
    elif header.is_type(MessageType.TOOL_USE_RESULT):
        result.has_tool_use = True
    elif header.is_type(MessageType.TOOL_PERMISSION_REQUEST):
        try:
            permission_request = json.loads(payload)
        except json.JSONDecodeError:
            # A corrupt tool_permission_request payload must fail fast rather
            # than being silently dropped (leaving tool_permission_request None).
            raise ValueError(
                f"corrupt output: tool_permission_request payload is not "
                f"valid JSON: {payload!r}"
            ) from None
        if not isinstance(permission_request, dict):
            raise ValueError(
                f"corrupt output: tool_permission_request payload is not "
                f"a JSON object: {payload!r}"
            )
        result.tool_permission_request = permission_request
    elif header.is_type(MessageType.TOOL_PERMISSION_RESPONSE_ACK):
        pass
    elif header.is_type(MessageType.CANCEL_TURN_ACK):
        pass
    elif header.is_type(MessageType.SYSTEM_AGENT_STARTED):
        pass
    elif header.is_type(MessageType.SYSTEM_AGENT_STOPPED):
        pass
    else:
        raise ValueError(f"Unrecognized message type '{header.msg_type}' in output file")

    # Accumulate token counts from response messages only
    if header.is_response:
        if header.prompt_tokens is not None:
            result.prompt_tokens = (result.prompt_tokens or 0) + header.prompt_tokens
        if header.completion_tokens is not None:
            result.completion_tokens = (result.completion_tokens or 0) + header.completion_tokens
        if header.total_tokens is not None:
            result.total_tokens = (result.total_tokens or 0) + header.total_tokens
        if header.cached_tokens is not None:
            result.cached_tokens = (result.cached_tokens or 0) + header.cached_tokens
        if header.reasoning_tokens is not None:
            result.reasoning_tokens = (result.reasoning_tokens or 0) + header.reasoning_tokens
