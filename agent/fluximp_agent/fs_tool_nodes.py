"""Filesystem-specific tool node implementations for the LangGraph agent pipeline.

Provides ParseToolCallsNode, ExecuteToolCallNode, their factory functions,
and create_should_continue for the filesystem-based agents (consultant, code,
developer).
"""

from fluximp_agent.fs_agent_state import AgentState, AGENT_STATE_CONFIG
from langchain_core.messages import AIMessage, ToolMessage
from fluximp_agent.tools_node import (
    BaseExecuteToolCallNode,
    BaseParseToolCallsNode,
    Services,
    make_error_msg_object,
    make_skip_notification_msg_object,
)
from fluximp_agent.tool_call_omission import (
    compute_api_omission_transformer,
    compute_text_omission_transformer,
)



def _promote_pending_to_dict(state):
    """Promote the pending omission transformer to the active dict.
    
    Reads tool_call_omission_transformers and pending_tool_call_omission_transformer
    from state, adds the pending transformer to the dict (if any), and returns
    the updated dict. This ensures the pending transformer is promoted regardless
    of whether the execute node runs (error paths skip the execute node).
    """
    active = dict(state.get("tool_call_omission_transformers", {}))
    pending = state.get("pending_tool_call_omission_transformer")
    if pending is not None:
        active[pending.message_index] = pending
    return active

class ParseToolCallsNode(BaseParseToolCallsNode):
    """Parse tool calls from an AIMessage and set up execution context.
    
    Handles parsing, validation, duplicate detection, and error handling
    for both API and text protocol tool calls. Does NOT execute any tools -
    it sets up pending_tool_calls and tool_execution_context for the
    ExecuteToolCallNode.
    
    On error (no tool calls, parsing errors, duplicates, missing args),
    returns a state update with error messages and empty pending_tool_calls.
    route_after_parse will route back to the agent node.
    
    On success, returns a state update with pending_tool_calls populated
    and tool_execution_context set. route_after_parse will route to
    execute_tool_call.
    """
    
    def __init__(self, registry, agent_protocol, state_config=None, max_tool_calls_per_response=None):
        super().__init__(registry, agent_protocol, max_tool_calls_per_response=max_tool_calls_per_response)
        self.state_config = state_config or AGENT_STATE_CONFIG
    
    # ââ Private helpers for building state updates âââââââââââââââââââââââââ
    
    def _build_result(self, state, updates, node_name):
        return self.state_config.build_full_update(state, updates, node_name)
    
    def _error_updates(self, state, messages, **overrides):
        """Build updates dict for error cases."""
        fields = {
            "messages": messages,
            "pending_tool_calls": [],
            "tool_execution_context": None,
            "last_tools_executed": [],
            "last_tool_success": False,
            "consecutive_no_tool_count": 0,
            "consecutive_max_reasoning_count": 0,
            "invalid_tool_calls_history": [],
            "last_valid_calls": None,
            "tool_call_omission_transformers": _promote_pending_to_dict(state),
            "pending_tool_call_omission_transformer": None,
            "file_operations_this_turn": [],
            "file_views_added_this_turn": [],
        }
        fields.update(overrides)
        return fields
    
    def _valid_updates(self, state, pending, context):
        """Build updates dict for valid calls case."""
        return {
            "messages": [],
            "pending_tool_calls": pending,
            "tool_execution_context": context,
            "last_tools_executed": [],
            "last_tool_success": False,
            "consecutive_no_tool_count": 0,
            "consecutive_max_reasoning_count": 0,
            "invalid_tool_calls_history": [],
            "last_valid_calls": None,  # Set in execute node finalization
            "tool_call_omission_transformers": _promote_pending_to_dict(state),  # Updated in execute node finalization
            "pending_tool_call_omission_transformer": None,
            "file_operations_this_turn": [],
            "file_views_added_this_turn": [],
        }
    
    # ââ Hook implementations ââââââââââââââââââââââââââââââââââââââââââââââ
    
    def build_empty_result(self, state):
        return self._build_result(state, {"messages": []}, "parse_tool_calls(empty)")
    
    def build_not_ai_result(self, state):
        return self._build_result(state, {"messages": []}, "parse_tool_calls(not_ai)")
    
    def build_no_tool_calls_api_result(self, state, no_tool_count, error_msg_obj, warning_messages):
        messages = [error_msg_obj] + warning_messages
        max_reasoning_count = state.get("consecutive_max_reasoning_count", 0)
        return self._build_result(state,
            self._error_updates(state, messages,
                consecutive_no_tool_count=no_tool_count,
                consecutive_max_reasoning_count=max_reasoning_count),
            "parse_tool_calls(no_tool_calls_api)")
    
    def build_no_tool_calls_text_result(self, state, error_msg_obj, invalid_history):
        max_reasoning_count = state.get("consecutive_max_reasoning_count", 0)
        return self._build_result(state,
            self._error_updates(state, [error_msg_obj],
                invalid_tool_calls_history=invalid_history,
                consecutive_max_reasoning_count=max_reasoning_count),
            "parse_tool_calls(invalid_calls)")
    
    def build_fatal_parsing_error_result(self, state, error_msg_obj):
        return self._build_result(state,
            self._error_updates(state, [error_msg_obj]),
            "parse_tool_calls(fatal_parsing_errors)")
    
    def build_duplicate_result(self, state, valid_calls, error_msg_obj):
        return self._build_result(state,
            self._error_updates(state, [error_msg_obj], last_valid_calls=valid_calls),
            "parse_tool_calls(duplicate_call)")
    
    def build_missing_args_result(self, state, valid_calls, error_msg_obj):
        return self._build_result(state,
            self._error_updates(state, [error_msg_obj], last_valid_calls=valid_calls),
            "parse_tool_calls(missing_args)")
    
    def build_multiple_calls_error_result(self, state, valid_calls, error_msg_obj):
        return self._build_result(state,
            self._error_updates(state, [error_msg_obj], last_valid_calls=valid_calls),
            "parse_tool_calls(multiple_calls)")
    
    def build_unavailable_tool_result(self, state, valid_calls, error_msg_obj):
        return self._build_result(state,
            self._error_updates(state, [error_msg_obj]),
            "parse_tool_calls(unavailable_tool)")
    
    def build_valid_result(self, state, pending, context):
        node_name = "parse_tool_calls(api_valid)" if self.use_api_protocol else "parse_tool_calls(text_valid)"
        return self._build_result(state,
            self._valid_updates(state, pending, context),
            node_name)
    

    def build_truncated_reasoning_result(self, state, max_reasoning_count):
        return self._build_result(state, {"messages": [], "consecutive_max_reasoning_count": max_reasoning_count}, "parse_tool_calls(truncated_reasoning)")
    
    def build_max_reasoning_error_result(self, state, max_reasoning_count):
        """Build error result for excessive consecutive max-reasoning responses."""
        no_tool_count = state.get("consecutive_no_tool_count", 0) + 1
        error_msg = self.generate_max_reasoning_error(max_reasoning_count)
        error_msg += "\n\n" + self.generate_no_tool_calls_api_error(no_tool_count)
        error_msg_obj = make_error_msg_object(error_msg, no_tool_error=True)
        warning_messages = self.generate_no_tool_calls_api_warnings(no_tool_count)
        messages = [error_msg_obj] + warning_messages
        return self._build_result(state,
            self._error_updates(state, messages,
                consecutive_no_tool_count=no_tool_count,
                consecutive_max_reasoning_count=max_reasoning_count),
            "parse_tool_calls(max_reasoning_error)")

    def _arg_value_omission_enabled(self):
        """Return True when the 'omit_arg_values' experiment is active.

        A registry without an experiments service (e.g. lightweight test
        registries) is treated as enabled so the historical always-on
        behavior is preserved.
        """
        experiments_service = getattr(self.registry, '_experiments_service', None)
        if experiments_service is None:
            return True
        return experiments_service.is_enabled('omit_arg_values')

    def compute_api_omission_transformer(self, last_message, valid_calls, messages, keep_first_n_calls=None):
        return compute_api_omission_transformer(
            last_message, valid_calls, messages, self.registry,
            keep_first_n_calls=keep_first_n_calls,
            omit_arg_values=self._arg_value_omission_enabled())
    
    def compute_text_omission_transformer(self, last_message, tc_dict_edits, tc_name_edits,
                                          content_edits, reasoning_edits, content_was_stripped,
                                          messages, keep_first_n_calls=None, call_span_deletions=None):
        return compute_text_omission_transformer(
            last_message, tc_dict_edits, tc_name_edits,
            content_edits, reasoning_edits, content_was_stripped, messages,
            keep_first_n_calls=keep_first_n_calls,
            call_span_deletions=call_span_deletions,
            omit_arg_values=self._arg_value_omission_enabled())


class ExecuteToolCallNode(BaseExecuteToolCallNode):
    """Execute a single tool call and update state.
    
    Pops the first call from pending_tool_calls, executes it, and
    accumulates results. When the last call is executed (or a ToolError
    occurs), finalization logic runs to set metadata fields.
    
    On unhandled exception, the exception propagates and LangGraph
    checkpoints the successful prior calls, preventing duplicate
    side effects on resume.
    """
    
    def __init__(self, registry, agent_protocol, state_config=None):
        super().__init__(registry, agent_protocol)
        self.state_config = state_config or AGENT_STATE_CONFIG
    
    # ââ Hook implementations ââââââââââââââââââââââââââââââââââââââââââââââ
    
    def update_state_reference(self, state):
        """Update state service reference for in-place state mutations."""
        if self.registry._state_service is not None:
            self.registry._state_service.update_state_reference(state)
    
    def build_empty_result(self, state):
        return self.state_config.build_full_update(
            state, {"messages": []}, "execute_tool_call(empty)")
    
    def accumulate_result(self, state, result, tool_name, remaining):
        """Build updates dict from a successful tool execution."""
        updates = {}
        
        # New messages from the tool (only new, for add_messages reducer)
        updates["messages"] = result.get("messages", [])
        
        # Replacement fields from tool result
        for field in ("file_operations", "operation_position", "file_view_windows",
                      "search_list_view", "last_scrutinized_position", "deleted_file_paths"):
            if field in result:
                updates[field] = result[field]
        
        # Accumulated fields - read from state, append, return complete
        file_ops_this_turn = list(state.get("file_operations_this_turn", []))
        if "file_operations_this_turn" in result:
            file_ops_this_turn.extend(result["file_operations_this_turn"])
        updates["file_operations_this_turn"] = file_ops_this_turn
        
        # Accumulate file_views_added_this_turn (same pattern as file_operations_this_turn)
        file_views_added = list(state.get("file_views_added_this_turn", []))
        if "file_views_added_this_turn" in result:
            file_views_added.extend(result["file_views_added_this_turn"])
        updates["file_views_added_this_turn"] = file_views_added
        
        last_tools = list(state.get("last_tools_executed", []))
        last_tools.append(tool_name)
        updates["last_tools_executed"] = last_tools
        
        if "send_message_outputs" in result:
            updates["send_message_outputs"] = result["send_message_outputs"]
        
        if "conditional_report_outputs" in result:
            updates["conditional_report_outputs"] = result["conditional_report_outputs"]
        
        # Pending tool calls - remaining after this one
        updates["pending_tool_calls"] = remaining
        
        # Tool execution context - pass through from state
        updates["tool_execution_context"] = state.get("tool_execution_context")
        
        return updates
    
    def build_tool_error_result(self, state, error_msg_obj):
        """Build result for a ToolError during execution."""
        updates = {
            "messages": [error_msg_obj],
            "pending_tool_calls": [],  # Skip remaining calls
            "file_operations_this_turn": list(state.get("file_operations_this_turn", [])),
            "last_tools_executed": list(state.get("last_tools_executed", [])),
            "tool_execution_context": state.get("tool_execution_context"),
        }
        
        # Finalize with error
        self.finalize(state, updates)
        
        return self.state_config.build_full_update(state, updates, "execute_tool_call(tool_error)")
    
    def finalize(self, state, updates):
        """Set finalization fields when all tool calls are done."""
        context = state.get("tool_execution_context") or {}
        
        updates["last_valid_calls"] = context.get("valid_calls")
        # Promote pending transformer to active dict
        current_active = dict(state.get("tool_call_omission_transformers", {}))
        pending = state.get("pending_tool_call_omission_transformer")
        if pending is not None:
            current_active[pending.message_index] = pending
        updates["tool_call_omission_transformers"] = current_active
        updates["pending_tool_call_omission_transformer"] = context.get("omission_transformer")
        
        last_tools = updates.get("last_tools_executed", state.get("last_tools_executed", []))
        updates["last_tool_success"] = len(last_tools) > 0
        
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
        node_name = "execute_tool_call(final)" if is_final else "execute_tool_call(continue)"
        return self.state_config.build_full_update(state, updates, node_name)


def create_parse_tool_calls_node(services, agent_config=None, state_config=None):
    """Create a ParseToolCallsNode bound to a services dict."""
    if agent_config is None:
        agent_config = getattr(services["registry"], "_agent_config", None)
    max_calls = getattr(agent_config, "max_tool_calls_per_response", None) if agent_config else None
    return ParseToolCallsNode(
        registry=services["registry"],
        agent_protocol=services["agent_protocol"],
        state_config=state_config,
        max_tool_calls_per_response=max_calls,
    )


def create_execute_tool_call_node(services, state_config=None):
    """Create an ExecuteToolCallNode bound to a services dict."""
    return ExecuteToolCallNode(
        registry=services["registry"],
        agent_protocol=services["agent_protocol"],
        state_config=state_config
    )


def create_should_continue(services: Services):
    """Create a should_continue function bound to a services dict."""
    def should_continue(state: AgentState) -> str:
        """Determine if we should continue to tools or end."""
        messages = state.get("messages", [])
        if not messages:
            return "end"

        last_message = messages[-1]

        # If last message is from tool, we're done with this turn
        if isinstance(last_message, ToolMessage):
            return "end"

        # If last message is not from AI, can't have tool calls
        if not isinstance(last_message, AIMessage):
            return "end"

        # Route all AI messages to parse_tool_calls.
        # ParseToolCallsNode handles truncated-reasoning-only detection.
        return "continue"

    return should_continue
