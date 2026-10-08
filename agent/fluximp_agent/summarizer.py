"""One-shot summarizer agent for condensing message blocks.

Provides the SummarizerCapuchin agent configuration and supporting
utilities for formatting messages and running the summarizer.
"""

import operator

from typing import Annotated

from fluximp_agent.model_config import ModelConfig, get_model_capabilities
from fluximp_agent.agent_config import AgentConfig, ArgDef, ToolDef
from fluximp_agent.tools.tool_definitions import ToolResult
from fluximp_agent.oneshot_agent import OneShotState


def submit_summary(summary: str, **kwargs) -> ToolResult:
    """Submit a summary of the message block.

    Args:
        summary: A concise summary of the message block.

    Returns:
        ToolResult with the summary text.
    """
    if not summary or not summary.strip():
        return ToolResult(success=False, content="ERROR: Summary cannot be empty.")
    return ToolResult(success=True, content=summary.strip())


def format_messages_for_summarization(messages, retained_indices, summary_map=None, summarized_indices=None, summary_contents=None, start_idx: int = 0):
    """Format messages as plain text for the summarizer agent.

    Each message gets a header with:
    - Index (or index range for prior summaries)
    - Status: TO BE SUMMARIZED, RETAINED EXACTLY, or PRIOR SUMMARY | RETAINED EXACTLY
    - Type (human, ai, or tool; system messages are excluded)

    AI messages get subheaders for content, reasoning, and tool uses.

    Prior summaries are shown once at their start index with the summary text,
    and all other indices covered by that range are skipped entirely.

    Args:
        messages: List of messages to format (the full message list)
        retained_indices: Set of message indices that will be retained exactly
        summary_map: Optional dict mapping start index to (start, end) for
            messages that are prior summaries
        summarized_indices: Optional set of message indices being summarized
            in this run
        summary_contents: Optional dict mapping start index of a summary range
            to the summary text
        start_idx: Index in `messages` from which to start formatting.

    Returns:
        Formatted text string
    """
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage

    if summary_map is None:
        summary_map = {}
    if summarized_indices is None:
        summarized_indices = set()
    if summary_contents is None:
        summary_contents = {}

    # Build mapping from any covered index to its range start
    covered_to_range_start = {}
    for start, (s, e) in summary_map.items():
        for idx in range(s, e + 1):
            covered_to_range_start[idx] = start

    lines = []
    outputted_summaries = set()

    for i, msg in enumerate(messages):
        if i < start_idx:
            continue
        if isinstance(msg, SystemMessage):
            continue

        # Check if this index is part of a prior summary range
        if i in covered_to_range_start:
            range_start = covered_to_range_start[i]
            # Only output the summary once, at the start index of the range
            if range_start not in outputted_summaries:
                outputted_summaries.add(range_start)
                s, e = summary_map[range_start]
                status = "PRIOR SUMMARY | RETAINED EXACTLY"
                index_str = f"{s}–{e}"
                lines.append(f"--- Message {index_str} | {status} | ai ---")
                summary_text = summary_contents.get(range_start, "")
                if summary_text:
                    lines.append(summary_text)
                lines.append("")
            # Skip all indices covered by prior summary ranges
            continue

        if isinstance(msg, HumanMessage):
            msg_type = "human"
        elif isinstance(msg, AIMessage):
            msg_type = "ai"
        elif isinstance(msg, ToolMessage):
            msg_type = "tool"
        else:
            msg_type = "unknown"

        # Determine status
        if i in summarized_indices:
            status = "TO BE SUMMARIZED"
        else:
            status = "RETAINED EXACTLY"

        index_str = str(i)
        lines.append(f"--- Message {index_str} | {status} | {msg_type} ---")

        if isinstance(msg, AIMessage):
            if msg.content:
                lines.append("--- Content ---")
                lines.append(str(msg.content))

            reasoning = (
                getattr(msg, 'reasoning_content', None)
                or getattr(msg, 'additional_kwargs', {}).get('reasoning')
            )
            if reasoning:
                lines.append("--- Reasoning ---")
                lines.append(str(reasoning))

            tool_calls = getattr(msg, 'tool_calls', None)
            if tool_calls:
                lines.append("--- Tool Uses ---")
                for tc in tool_calls:
                    name = tc.get('name', 'unknown')
                    args = tc.get('args', {})
                    args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
                    lines.append(f"  {name}({args_str})")
        else:
            content = str(msg.content) if hasattr(msg, 'content') and msg.content else ""
            if content:
                lines.append(content)

        lines.append("")

    return "\n".join(lines)




class SummarizerCapuchin(AgentConfig):
    """One-shot agent for summarizing a block of messages.

    Receives formatted message text and produces a concise summary
    that minimizes redundant information relative to messages that
    will be retained exactly.
    """

    agent_type = "summarizer"

    def __init__(self, model_config: ModelConfig):
        caps = get_model_capabilities(model_config)
        tool_names = [
            "submit_summary",
        ]
        super().__init__(
            model_config=model_config,
            tool_names=tool_names,
            use_tool_messages=caps.able_to_receive_tool_messages_at_all,
            use_api_protocol_for_tool_functions=not caps.able_to_use_python_like_protocol_for_tool_function,
            use_tool_message_for_no_tool_error=caps.able_to_recieve_tool_messages_for_no_tool_use_errors,
        )

    def get_tool_defintions(self):
        return [
            ToolDef(
                name="submit_summary",
                description="Submit your summary of the message block. Once you use this tool, the conversation ends.",
                category="communication",
                args=[
                    ArgDef(
                        name="summary",
                        arg_type="string",
                        description="A concise summary of the message block, minimizing redundant information relative to messages marked as RETAINED EXACTLY.",
                        tags=frozenset({'content'})
                    )
                ],
                func=submit_summary,
                is_state_modifying=False
            )
        ]

    def get_system_prompt_tool_use_example_lines(self) -> list:
        return [
            "TOOL FUNCTION USAGE EXAMPLES",
            "",
            'submit_summary(summary="""',
            '   The agent searched for files matching *.py and found 3 results...',
            '   """)',
        ]

    def get_system_prompt_preamble(self, **kwargs) -> str:
        return """Your job is to summarize the messages in a conversation between a human/user and ai/assistant.
        
You will receive the conversation messages formatted as plain text in single human role message. Each message has a header showing:
- The message's index in the message sequence (or index range if it is a prior summary)
- Whether it is TO BE SUMMARIZED by you into on message in the conversation or if it will be RETAINED EXACTLY in the conversation.
- The message's role/type: human, ai, or tool

AI messages have subheaders for:
- Content: the text the agent wrote
- Reasoning: the agent's internal reasoning (if present)
- Tool Uses: tool function calls the agent made

Messages marked RETAINED EXACTLY will appear verbatim in the final context alongside your summary.
Messages marked PRIOR SUMMARY | RETAINED EXACTLY are previous summaries that will also appear verbatim.

Your summary should:
1. Be written from the first-person perspective
2. Be concise but complete enough for continuity
3. Minimize redundant information relative to messages marked RETAINED EXACTLY since those will be seen verbatim
4. Capture insights and key decisions in the messages marked as TO BE SUMMARIZED that are no included in the message marked RETAINED EXACTLY

Avoid including information that is already present in messages marked RETAINED EXACTLY.  For example the contents of a send_message tool use are always retained exactly, so you should not include restatements of what was said to the user via send_message in your summary.

The submit_summary tool is the only tool available to you.  You may see other tool uses in the conversation messages you are summarizing, but those tools are not available to you.

You must use the submit_summary tool to submit your summary. Once you use submit_summary, the conversation ends."""



# ---------------------------------------------------------------------------
# Summarizer state
# ---------------------------------------------------------------------------

class SummarizerState(OneShotState):
    """Minimal state for the summarizer agent.

    Extends OneShotState with summarizer-specific output fields.
    """
    summary_outputs: Annotated[list, operator.add]


def run_summarizer_file_io(filesystem_service, model_config, io_dir,
                            max_requests=3, max_transient_retries=3,
                            input_timeout=120.0, permission_timeout=120.0):
    """Run the summarizer agent in file-based I/O mode (subprocess entry point).

    Delegates to run_oneshot_file_io with summarizer-specific configuration.
    See oneshot_agent.run_oneshot_file_io for full documentation.
    """
    from fluximp_agent.oneshot_agent import run_oneshot_file_io
    run_oneshot_file_io(
        filesystem_service=filesystem_service,
        model_config=model_config,
        io_dir=io_dir,
        agent_config=SummarizerCapuchin(model_config),
        agent_name='SUMMARIZER',
        required_tool_name='submit_summary',
        state_class=SummarizerState,
        extra_output_keys=['summary_outputs'],
        payload_type='summary',
        max_requests=max_requests,
        max_transient_retries=max_transient_retries,
        input_timeout=input_timeout,
        permission_timeout=permission_timeout,
    )
