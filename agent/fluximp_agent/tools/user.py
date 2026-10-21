from fluximp_agent.tools.tool_definitions import ToolResult

# Valid tags for send_message
VALID_TAGS = frozenset({
    'findings_report',
    'user_feedback_requested',
    'complete_plan',
    'plan_update',
    'work_completion_summary',
    'resolved_reported_issues',
})

VALID_FORMATS = frozenset({'plain_text', 'markdown'})



def _has_successful_tool_use_since_last_human(state_service) -> bool:
    """Check if any successful non-send_message tool has been used since the last human message."""
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    if hasattr(state_service, "get_state"):
        state = state_service.get_state()
    else:
        state = getattr(state_service, "_state", None)
    if state is None:
        return False

    messages = state.get("messages", [])
    if not messages:
        return False

    # Find the index of the last HumanMessage
    last_human_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if isinstance(msg, HumanMessage):
            last_human_idx = i
            break

    if last_human_idx == -1:
        # No human message found - check all messages
        last_human_idx = 0

    # Build mapping of tool_call_id -> tool_name from AIMessages after last human message
    tool_call_id_to_name = {}
    for i in range(last_human_idx + 1, len(messages)):
        msg = messages[i]
        if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tc in msg.tool_calls:
                tc_id = tc.get("id", "")
                tc_name = tc.get("name", "")
                if tc_id:
                    tool_call_id_to_name[tc_id] = tc_name

    # Check for successful tool results from non-send_message tools
    for i in range(last_human_idx + 1, len(messages)):
        msg = messages[i]

        if isinstance(msg, ToolMessage):
            # API protocol tool result
            if getattr(msg, 'execution_successful', True):
                tool_name = tool_call_id_to_name.get(msg.tool_call_id, "")
                if tool_name != "send_message":
                    return True

    return False


def send_message(content: str, tags: list, format: str = "markdown", **kwargs) -> ToolResult:
    """
    Display a message to the user.

    This is the REQUIRED mechanism for the model to communicate text directly to the user.
    All explanations, answers, acknowledgments, and any user-facing text MUST go through this tool.
    Free-form text output is NOT allowed - you MUST use this tool to communicate.

    Args:
        content: The message text to display to the user
        tags: List of tags indicating what the message contains. Must have one or more of:
            findings_report, user_feedback_requested, complete_plan, plan_update, work_completion_summary
        format: Format of the message content: 'plain_text' or 'markdown' (default: 'markdown')

    Returns:
        ToolResult - with success=True and content=the message
    """
    # Validate tags
    invalid_tags = [t for t in tags if t not in VALID_TAGS]
    if invalid_tags:
        return ToolResult(success=False, content=f"Invalid tags: {invalid_tags}. Valid tags are: {sorted(VALID_TAGS)}")
    if not tags:
        return ToolResult(success=False, content="tags must contain at least one tag")

    # Validate format
    if format not in VALID_FORMATS:
        return ToolResult(success=False, content=f"Invalid format: '{format}'. Valid formats are: {sorted(VALID_FORMATS)}")


    # Require at least one successful tool use before work_completion_summary
    if "work_completion_summary" in tags:
        state_service = kwargs.get("state_service")
        if state_service is None or not _has_successful_tool_use_since_last_human(state_service):
            return ToolResult(
                success=False,
                content="Cannot use 'work_completion_summary' tag without performing any work. "
                        "At least one successful tool use (other than send_message) is required "
                        "since the last human message before claiming work completion."
            )

    return ToolResult(
        success=True,
        content=content,
        state_updates={"send_message_tags": tags, "send_message_format": format}
    )
