"""Shared utilities for one-shot agents (scrutinator, summarizer, etc.).

Provides common state, node factories, routing functions, and graph
construction helpers used by agents that follow the one-shot pattern:
receive input -> call LLM with retry -> execute a single submit tool -> end.
"""

from typing import Annotated, Any, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import AIMessage, HumanMessage
from fluximp_agent.agent_node import BaseAgentNode
from fluximp_agent.tools_node import (
    BaseParseToolCallsNode,
    BaseExecuteToolCallNode,
    make_error_msg_object,
    make_skip_notification_msg_object,
)


class OneShotState(TypedDict):
    """Base state for one-shot agents.

    Includes the minimal fields needed by the agent and tools node
    functions.  Specialized agents can extend this with additional
    output fields (e.g. scrutinator_report_outputs).
    """
    messages: Annotated[list, add_messages]
    request_count: int
    tool_success: bool
    pending_tool_calls: list
    tool_execution_context: Optional[Any]
    # Declared in the base state schema because LangGraph silently filters
    # every node's returned dict down to the keys declared in the state
    # schema: the agent node does return "response_metadata", but without
    # this key that dict entry is dropped before reaching the updates
    # stream consumed by run_oneshot_file_io, leaving write_response to
    # fall back to its defaults (all token counts 0, empty model /
    # finish_reason, fallback end_time).  Mirrors the primary agent's
    # FSAgentState.response_metadata.
    response_metadata: Optional[dict]


def oneshot_should_continue(state) -> str:
    """Route after agent node: continue to tools if AIMessage, else end.

    Truncated reasoning-only detection is deferred to ParseToolCallsNode.
    """
    messages = state.get("messages", [])
    if not messages:
        return "end"

    last_message = messages[-1]
    if not isinstance(last_message, AIMessage):
        return "end"

    # Route all AI messages to parse_tool_calls.
    # ParseToolCallsNode handles truncated-reasoning-only detection.
    return "continue"


def oneshot_route_after_parse(state) -> str:
    """Route after parse_tool_calls: execute if calls pending, else back to agent."""
    if state.get("pending_tool_calls"):
        return "execute"
    return "agent"


def oneshot_route_after_execute(state) -> str:
    """Route after execute_tool_call: end if tool succeeded, else back to agent."""
    if state.get("tool_success", False):
        return "end"
    return "agent"


class OneshotAgentNode(BaseAgentNode):
    """Agent node for one-shot agents (scrutinator, summarizer, etc.).
    
    Extends BaseAgentNode with protocol support for request lifecycle
    notifications. Uses the base class's simple message building and
    response processing without transformers, context injection, or
    summarization.
    """
    
    def __init__(self, agent_config, registry, filesystem_service,
                 agent_name=None, max_requests=5, max_transient_retries=8,
                 protocol=None):
        super().__init__(
            filesystem_service=filesystem_service,
            agent_config=agent_config,
            registry=registry,
            max_requests=max_requests,
            max_transient_retries=max_transient_retries,
            agent_name=agent_name,
        )
        self.protocol = protocol
    
    def _on_request_start(self, start_dt):
        """Notify protocol of request start."""
        if self.protocol is not None:
            self.protocol.on_request_start(start_dt)


def create_oneshot_agent_node(agent_config, registry, filesystem_service,
                               agent_name, max_requests=5, max_transient_retries=8,
                               protocol=None):
    """Create a simple agent node for a one-shot agent.
    
    Uses OneshotAgentNode which inherits all LLM invocation logic,
    retry, and response processing from BaseAgentNode, while
    customizing for the simpler OneShotState schema.
    
    Args:
        agent_config: AgentConfig instance containing model/API settings.
        registry: ToolRegistry for the agent.
        filesystem_service: FileSystemService instance.
        agent_name: Display name for log messages (e.g. "SCRUTINATOR", "SUMMARIZER").
        max_requests: Maximum API requests allowed.
        max_transient_retries: Maximum retries for transient API errors.
        protocol: Optional AgentProtocol for writing request headers.
    """
    return OneshotAgentNode(
        agent_config=agent_config,
        registry=registry,
        filesystem_service=filesystem_service,
        agent_name=agent_name,
        max_requests=max_requests,
        max_transient_retries=max_transient_retries,
        protocol=protocol,
    )


class OneShotParseToolCallsNode(BaseParseToolCallsNode):
    """Parse tool calls for one-shot agents.
    
    Enforces the constraint that one-shot agents should only call their
    single required tool once per request. Multiple tool calls are
    treated as an error.
    """
    
    def __init__(self, registry, required_tool_name, agent_protocol=None, extra_output_keys=None, max_tool_calls_per_response=None):
        super().__init__(registry, agent_protocol, max_tool_calls_per_response=max_tool_calls_per_response)
        self.required_tool_name = required_tool_name
        self.extra_output_keys = extra_output_keys or []
    
    def _base_result(self, **overrides):
        """Build a base result dict with tool_success=False and empty extra keys."""
        result = {"tool_success": False}
        result.update(overrides)
        result.update({key: [] for key in self.extra_output_keys})
        return result
    
    # ââ Hook implementations ââââââââââââââââââââââââââââââââââââââââââââ
    
    def build_empty_result(self, state):
        return self._base_result(messages=[])
    
    def build_not_ai_result(self, state):
        return self._base_result(messages=[])
    
    def build_no_tool_calls_api_result(self, state, no_tool_count, error_msg_obj, warning_messages):
        return self._base_result(messages=[error_msg_obj] + warning_messages)
    
    def build_no_tool_calls_text_result(self, state, error_msg_obj, invalid_history):
        return self._base_result(messages=[error_msg_obj])
    
    def build_fatal_parsing_error_result(self, state, error_msg_obj):
        return self._base_result(messages=[error_msg_obj])
    
    def build_duplicate_result(self, state, valid_calls, error_msg_obj):
        return self._base_result(messages=[error_msg_obj])
    
    def build_missing_args_result(self, state, valid_calls, error_msg_obj):
        return self._base_result(messages=[error_msg_obj])
    
    def build_multiple_calls_error_result(self, state, valid_calls, error_msg_obj):
        return self._base_result(messages=[error_msg_obj])
    
    def build_unavailable_tool_result(self, state, valid_calls, error_msg_obj):
        return self._base_result(messages=[error_msg_obj])
    
    def build_valid_result(self, state, pending, context):
        result = {
            "messages": [],
            "pending_tool_calls": pending,
            "tool_execution_context": context,
            "tool_success": False,
        }
        result.update({key: [] for key in self.extra_output_keys})
        return result
    

    def build_truncated_reasoning_result(self, state, max_reasoning_count):
        return self._base_result(messages=[])
    
    def build_max_reasoning_error_result(self, state, max_reasoning_count):
        error_msg = self.generate_max_reasoning_error(max_reasoning_count)
        error_msg += f"\n\nYou must use the {self.required_tool_name} tool to submit your result."
        error_msg_obj = make_error_msg_object(error_msg, no_tool_error=True)
        return self._base_result(messages=[error_msg_obj])

    def validate_call_count(self, valid_calls):
        """Reject multiple tool calls for one-shot agents."""
        if len(valid_calls) > 1:
            error_msg = (
                f"You may only call one tool per response. "
                f"You must use the {self.required_tool_name} tool to submit your result."
            )
            return make_error_msg_object(error_msg, no_tool_error=True)
        return None
    
    # ââ Error message generation overrides ââââââââââââââââââââââââââââââ
    
    def generate_no_tool_calls_api_error(self, no_tool_count):
        return f"You must use the {self.required_tool_name} tool to submit your result."
    
    def generate_no_tool_calls_api_warnings(self, no_tool_count):
        return []
    
    def generate_duplicate_error_msg(self, valid_calls):
        return (
            f"Error: This tool call is identical to your previous call. "
            f"The operation has already been completed. "
            f"You must use the {self.required_tool_name} tool to submit your result."
        )
    
    def generate_no_tool_calls_text_base_error(self, ai_content):
        return f"You must use the {self.required_tool_name} tool to submit your result."


class OneShotExecuteToolCallNode(BaseExecuteToolCallNode):
    """Execute a single tool call for one-shot agents.
    
    Since one-shot agents only have one tool call per request,
    execution is always final (no loop for remaining calls).
    """
    
    def __init__(self, registry, agent_protocol=None, extra_output_keys=None):
        super().__init__(registry, agent_protocol)
        self.extra_output_keys = extra_output_keys or []
    
    def _base_result(self, **overrides):
        """Build a base result dict with tool_success=False and empty extra keys."""
        result = {"tool_success": False}
        result.update(overrides)
        result.update({key: [] for key in self.extra_output_keys})
        return result
    
    # ââ Hook implementations ââââââââââââââââââââââââââââââââââââââââââââ
    
    def build_empty_result(self, state):
        return self._base_result(messages=[])
    
    def accumulate_result(self, state, result, tool_name, remaining):
        """Build updates dict from a successful tool execution."""
        updates = {
            "messages": result.get("messages", []),
            "pending_tool_calls": remaining,  # Always empty for one-shot
            "tool_execution_context": state.get("tool_execution_context"),
            "tool_success": False,  # Will be set to True in finalize
        }
        
        # Collect extra output keys from result
        for key in self.extra_output_keys:
            if key in result:
                updates[key] = result[key]
        
        return updates
    
    def build_tool_error_result(self, state, error_msg_obj):
        result = {
            "messages": [error_msg_obj],
            "pending_tool_calls": [],
            "tool_execution_context": None,
            "tool_success": False,
        }
        result.update({key: [] for key in self.extra_output_keys})
        return result
    
    def finalize(self, state, updates):
        """Set finalization fields when tool execution is done."""
        context = state.get("tool_execution_context") or {}
        
        # Set tool_success
        updates["tool_success"] = True
        
        # Append partial match error message if any
        partial_error_msg_obj = context.get("partial_match_error_msg_obj")
        if partial_error_msg_obj is not None:
            messages = list(updates.get("messages", []))
            messages.append(partial_error_msg_obj)
            updates["messages"] = messages
        
        # Inform the agent if tool calls were skipped due to the per-response limit
        skipped_count = context.get("skipped_tool_call_count", 0)
        if skipped_count:
            limit = context.get("max_tool_calls_per_response")
            skip_msg = make_skip_notification_msg_object(
                f"Note: {skipped_count} tool call(s) beyond the maximum of {limit} were not processed."
            )
            messages = list(updates.get("messages", []))
            messages.append(skip_msg)
            updates["messages"] = messages
        
        # Clear context
        updates["tool_execution_context"] = None
    
    def build_execute_result(self, state, updates, is_final):
        # For one-shot agents, just return the updates dict directly
        return updates


def create_oneshot_parse_node(registry, required_tool_name, protocol=None, extra_output_keys=None, max_tool_calls_per_response=None):
    """Create a parse tool calls node for a one-shot agent.

    Args:
        registry: ToolRegistry for the agent.
        required_tool_name: Name of the required tool (used in error messages).
        protocol: Optional AgentProtocol for writing tool events.
        extra_output_keys: Optional list of extra keys to collect from tool
            results (e.g. ["scrutinator_report_outputs"]).
        max_tool_calls_per_response: Optional per-response limit. One-shot
            agents always pass None because they enforce exactly one call.
    """
    return OneShotParseToolCallsNode(
        registry=registry,
        required_tool_name=required_tool_name,
        agent_protocol=protocol,
        extra_output_keys=extra_output_keys,
        max_tool_calls_per_response=max_tool_calls_per_response
    )


def create_oneshot_execute_node(registry, protocol=None, extra_output_keys=None):
    """Create an execute tool call node for a one-shot agent.

    Args:
        registry: ToolRegistry for the agent.
        protocol: Optional AgentProtocol for writing tool events.
        extra_output_keys: Optional list of extra keys to collect from tool
            results (e.g. ["scrutinator_report_outputs"]).
    """
    return OneShotExecuteToolCallNode(
        registry=registry,
        agent_protocol=protocol,
        extra_output_keys=extra_output_keys
    )


def build_oneshot_graph(state_class, agent_node, parse_node, execute_node,
                         should_continue=None, route_after_parse=None, route_after_execute=None):
    """Build a standard one-shot LangGraph graph.

    Creates a graph with agent, parse_tool_calls, and execute_tool_call nodes,
    wired with the standard one-shot routing.

    Args:
        state_class: The TypedDict class for the graph state.
        agent_node: The agent node function.
        parse_node: The parse tool calls node function.
        execute_node: The execute tool call node function.
        should_continue: Routing function after agent node (default: oneshot_should_continue).
        route_after_parse: Routing function after parse node (default: oneshot_route_after_parse).
        route_after_execute: Routing function after execute node (default: oneshot_route_after_execute).

    Returns:
        Compiled LangGraph graph.
    """
    from langgraph.graph import StateGraph, START, END

    if should_continue is None:
        should_continue = oneshot_should_continue
    if route_after_parse is None:
        route_after_parse = oneshot_route_after_parse
    if route_after_execute is None:
        route_after_execute = oneshot_route_after_execute

    graph = StateGraph(state_class)
    graph.add_node("agent", agent_node)
    graph.add_node("parse_tool_calls", parse_node)
    graph.add_node("execute_tool_call", execute_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {"continue": "parse_tool_calls", "end": END}
    )
    graph.add_conditional_edges(
        "parse_tool_calls",
        route_after_parse,
        {"execute": "execute_tool_call", "agent": "agent"}
    )
    graph.add_conditional_edges(
        "execute_tool_call",
        route_after_execute,
        {"agent": "agent", "end": END}
    )

    return graph.compile()



def run_oneshot_file_io(
    filesystem_service,
    model_config,
    io_dir,
    agent_config,
    agent_name,
    required_tool_name,
    state_class,
    extra_output_keys,
    payload_type,
    max_requests=5,
    max_transient_retries=8,
    input_timeout=120.0,
    permission_timeout=120.0,
):
    """Run a one-shot agent in file-based I/O mode (subprocess entry point).

    Creates a LangGraph graph using shared one-shot node factories
    and runs it with the file-based user agent protocol for communication
    with the primary agent and AgentService.

    Unlike the main agent's file_io_mode(), this does not handle checkpointing,
    time travel, mid-turn recovery, cancel detection, or fork handling.  A
    one-shot agent processes a single input and produces a single output.

    Args:
        filesystem_service: FileSystemService instance for file operations
        model_config: ModelConfig for the LLM
        io_dir: Directory containing the conversation files
        agent_config: AgentConfig instance (e.g. SummarizerCapuchin, ScrutinatorMacaque)
        agent_name: Display name for log messages (e.g. "SUMMARIZER", "SCRUTINATOR")
        required_tool_name: Name of the required tool (e.g. "submit_summary", "submit_report")
        state_class: The TypedDict class for the graph state
        extra_output_keys: List of extra output keys to collect from tool results
        payload_type: Payload type string for write_with_metadata (e.g. "summary", "scrutinator_report")
        max_requests: Maximum API requests allowed
        max_transient_retries: Maximum retries for transient API errors
        input_timeout: Timeout in seconds for waiting on input files
        permission_timeout: Timeout in seconds for permission responses
    """
    import sys
    import time
    from pathlib import Path
    from fluximp_agent.tool_registry import create_tool_registry
    from fluximp_protocol.user_agent.file_io.agent import FileIOAgentProtocol
    from fluximp_protocol.user_agent.base import MaxRequestsExceeded
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    io_path = Path(io_dir).resolve()
    if not io_path.exists():
        print(f"Error: I/O directory does not exist: {io_path}")
        return

    # Create the FileIOAgentProtocol
    protocol = FileIOAgentProtocol(
        io_path=io_path,
        agent_type=agent_config.agent_type,
        provider=model_config.provider.name,
        requested_model=model_config.model,
        input_timeout=input_timeout,
        permission_timeout=permission_timeout
    )

    try:
        # Wait for input (single turn, fork=0, turn=0)
        fork = 0
        turn = 0
        input_wait_start = time.time()
        while True:
            result = protocol.check_for_new_input(fork, turn)
            if result and (result.input_file or result.end_file):
                break
            time.sleep(0.05)
            if time.time() - input_wait_start > input_timeout:
                print(f"Timeout: No input file received within {input_timeout}s")
                return

        # Check for end signal
        if result.end_file is not None and result.end_file.exists():
            print(f"End signal found at fork {fork}, turn {turn}. Stopping.")
            return

        # Mark input file as seen so we don't process it again
        protocol.mark_seen(result.input_file)

        # Open handles and read input
        protocol.open_handles(result.input_file, result.output_file)
        msg_type, user_input = protocol.read_user_message()

        assert msg_type == 'user_message'

        print(f"\n{'='*60}")
        print(f"FORK {fork}, TURN {turn}")
        print(f"{'='*60}")
        print(f"Input: {user_input[:100]}...")

        # Create the registry
        registry = create_tool_registry(filesystem_service, agent_config)

        # Create agent and tools nodes using shared factory
        agent_node = create_oneshot_agent_node(
            agent_config, registry, filesystem_service,
            agent_name=agent_name,
            max_requests=max_requests,
            max_transient_retries=max_transient_retries,
            protocol=protocol
        )
        parse_node = create_oneshot_parse_node(
            registry,
            required_tool_name=required_tool_name,
            protocol=protocol,
            extra_output_keys=extra_output_keys
        )
        execute_node = create_oneshot_execute_node(
            registry,
            protocol=protocol,
            extra_output_keys=extra_output_keys
        )

        # Build the graph using shared helper
        compiled = build_oneshot_graph(state_class, agent_node, parse_node, execute_node)

        # Stream the graph, writing events to the protocol
        initial_state = {
            "messages": [SystemMessage(content=""), HumanMessage(content=user_input)],
            "request_count": 0,
            "tool_success": False,
            "pending_tool_calls": [],
            "tool_execution_context": None,
            "response_metadata": None,
        }
        for key in extra_output_keys:
            initial_state[key] = []

        stream = compiled.stream(initial_state, stream_mode="updates")

        for chunk in stream:
            for node_name, state_update in chunk.items():
                # Write response events for AIMessages from the agent node
                if node_name == "agent":
                    # Capture this chunk's metadata from the state update.
                    # state_update["response_metadata"] carries 'model' (the key
                    # the protocol's response writer reads for actual_model);
                    # the raw AIMessage.response_metadata uses 'model_name', so
                    # passing it wrote an empty model field.
                    chunk_metadata = None
                    if "response_metadata" in state_update:
                        chunk_metadata = state_update["response_metadata"]
                    for msg in state_update.get("messages", []):
                        if isinstance(msg, AIMessage):
                            reasoning = msg.additional_kwargs.get("reasoning_content")
                            if not reasoning:
                                reasoning = msg.additional_kwargs.get("reasoning")
                            protocol.write_response(
                                msg.content,
                                reasoning_content=reasoning,
                                metadata=chunk_metadata or {}
                            )

                # Write output events from the tools node
                for key in extra_output_keys:
                    if key in state_update:
                        for content in state_update[key]:
                            protocol.write_with_metadata(payload_type, content)

        print(f"\n{'='*60}")
        print(f"{agent_name} file I/O mode completed.")
        print(f"{'='*60}")

    except MaxRequestsExceeded as e:
        protocol.write_max_requests_reached(str(e))
        print(f"Max requests reached: {e}")

    except Exception as e:
        print(f"Error in {agent_name} file I/O mode: {e}")
        import traceback
        traceback.print_exc()

        # Record the failure on the output stream (when a handle is open) so a
        # reader classifies the turn as infrastructure_error instead of seeing a
        # truncated stream; the finally block below closes the handles.
        try:
            if protocol._output_handle:
                import io as _io
                tb_buffer = _io.StringIO()
                traceback.print_exc(file=tb_buffer)
                error_content = (
                    f"\nERROR:\n{'-' * 40}\n{str(e)}\n\n"
                    f"TRACEBACK:\n{tb_buffer.getvalue()}"
                )
                protocol.write_with_metadata("infrastructure_error", error_content)
                print(error_content)
        except Exception as write_error:
            print(f"Failed to write error to output file: {write_error}")

        sys.stdout.flush()
        sys.exit(1)

    finally:
        protocol.close_handles()
