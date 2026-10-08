"""Tools node for the LangGraph agent pipeline.

Provides base classes for tool call parsing and execution, error message
utilities, and arg-omission helpers. Filesystem-specific implementations
live in fs_tool_nodes.
"""

import uuid
from fluximp_agent.tool_call_omission import strip_think_blocks, compute_omitted_args, find_think_block_ranges, map_stripped_pos_to_original, OMISSION_PLACEHOLDER_RE

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from rapidfuzz import fuzz
from typing_extensions import TypedDict

from fluximp_agent.tool_func_parser import detect_xml_tags
from fluximp_agent.tools.common import ToolError
from fluximp_agent.agent_config import AgentConfig
from fluximp_agent.tool_registry import ToolRegistry
from fluximp_protocol.user_agent.base import AgentProtocol


class Services(TypedDict, total=False):
    """Dictionary of services available to graph node functions.
    
    Attributes:
        registry: ToolRegistry for tool execution and parsing
        agent_protocol: AgentProtocol for writing events directly to the output stream
    """
    registry: ToolRegistry
    agent_protocol: AgentProtocol

def make_error_msg_object(error_msg: str, no_tool_error: bool = False):
    # Always create a ToolMessage with markers that the ToolMessageCompatibilityTransformer
    # will use to decide whether to convert to HumanMessage for models that need it.
    additional_kwargs = {'human_prefix': 'Tool error:\n'}
    if no_tool_error:
        additional_kwargs['no_tool_use_error'] = True
    return make_tool_error_obj(str(uuid.uuid4()), error_msg, additional_kwargs=additional_kwargs)


def make_skip_notification_msg_object(notification_msg: str):
    """Create a ToolMessage for a skipped-tool-call notification.

    The message has no matching tool call in the preceding AIMessage, so it
    is marked with no_tool_use_error.  ToolMessageCompatibilityTransformer
    will convert it to a prefixed HumanMessage for models/providers that
    cannot consume unmatched ToolMessages.
    """
    additional_kwargs = {
        'human_prefix': 'Tool notification:\n',
        'no_tool_use_error': True,
    }
    return make_tool_error_obj(
        str(uuid.uuid4()),
        notification_msg,
        additional_kwargs=additional_kwargs,
        execution_successful=True,
    )




def generate_partial_match_error(partial_matches, registry: ToolRegistry) -> str:
    """Generate an error message for partial tool call matches.

    Args:
        partial_matches: List of PartialMatch objects from the parser
        registry: ToolRegistry for looking up tool examples

    Returns:
        Error message string explaining what went wrong
    """
    error_msg = ""

    tool_names_for_invalid = set()
    tool_missing_args = []
    for pm in partial_matches:
        tool_names_for_invalid.add(pm.tool_name)
        if pm.extracted_args:
            tool_missing_args.append(pm.missing_args)

    tool_names_for_invalid = list(tool_names_for_invalid)
    tool_names_for_invalid.sort()

    # Collect any XML/HTML tags detected in partial matches
    all_detected_tags = set()
    for pm in partial_matches:
        if pm.detected_tags:
            all_detected_tags.update(pm.detected_tags)

    # Append tag warning if any tags detected
    if all_detected_tags:
        error_msg += f"\nYour response contains XML/HTML tags: {', '.join(sorted(all_detected_tags))}"
        error_msg += "\nNever use XML/HTML tags, angle brackets, or markup syntax."

    # Collect any JSON keys detected in partial matches
    all_detected_json = set()
    for pm in partial_matches:
        if pm.detected_json:
            all_detected_json.update(pm.detected_json)

    # Append JSON warning if any JSON detected
    if all_detected_json:
        error_msg += f"\nYour response contains JSON format with keys: {', '.join(sorted(all_detected_json))}"
        error_msg += "\nNever use JSON format for tool calls."

    if len(tool_names_for_invalid) > 1:
        error_msg += f"You appear to be trying to call the tool function(s) {', '.join(tool_names_for_invalid)} but your using an invalid format."
    else:
        most_missing_args = None
        for ma in tool_missing_args:
            if ma:
                if most_missing_args is None:
                    most_missing_args = ma
                else:
                    if len(ma) > len(most_missing_args):
                        most_missing_args = ma
        if most_missing_args:
            error_msg += f"You appear to be trying to call the tool function(s) {', '.join(tool_names_for_invalid)} but you haven't provided some required arguments: {', '.join(most_missing_args)}."
        else:
            error_msg += f"You appear to be trying to call the tool function(s) {', '.join(tool_names_for_invalid)} but your using an invalid format."
    # Append relevant examples to help the model correct its format
    error_msg += "\n\nCorrect examples:\n"
    if tool_names_for_invalid:
        for name in sorted(tool_names_for_invalid):
            example = registry.get_example(name)
            if example:
                error_msg += f"  {example}\n"
    else:
        error_msg += registry.get_all_examples()

    return error_msg


def make_tool_error_obj(tool_call_id, error_msg, additional_kwargs=None, execution_successful=False):
    """Create a ToolMessage for a tool error.

    Args:
        tool_call_id: The tool call ID for ToolMessage
        error_msg: The error message content
        additional_kwargs: Optional dict of markers for ToolMessageCompatibilityTransformer
            (e.g. 'human_prefix', 'no_tool_use_error')
        execution_successful: Optional bool for execution_successful flag
            (default False because this helper is primarily for errors)

    Returns:
        ToolMessage with the error content and markers
    """
    return ToolMessage(tool_call_id=tool_call_id, content=error_msg,
                       execution_successful=execution_successful, additional_kwargs=additional_kwargs or {})


def validate_required_args(tool_name, args, registry):
    """Validate that all required arguments are present for a tool.

    Args:
        tool_name: Name of the tool to validate
        args: Dict of arguments provided
        registry: ToolRegistry for looking up tool definitions

    Returns:
        Error message string if required args are missing, or None if valid.
    """
    if tool_name not in registry._tools:
        return None  # Unknown tool, let execution handle it

    tool_def = registry._tools[tool_name]
    required_args = [arg.name for arg in tool_def.args if arg.required]
    missing_args = [arg for arg in required_args if arg not in args]
    if missing_args:
        return f"Error: Tool '{tool_name}' is missing required argument(s): {', '.join(missing_args)}"
    return None




# ââ Per-tool-call pipeline ââââââââââââââââââââââââââââââââââââââââââ

def validate_no_placeholders(tool_name, args, registry):
    """Validate that content-tagged args do not contain omission placeholders.
    
    Checks string values of args with 'content' tags for patterns matching
    omission placeholder strings.
    
    Args:
        tool_name: Name of the tool to validate
        args: Dict of arguments provided
        registry: ToolRegistry for looking up tool definitions
    
    Returns:
        Error message string if a placeholder is detected, None if valid.
    """
    tool_def = registry._tools.get(tool_name)
    if not tool_def:
        return None
    
    for arg_def in tool_def.args:
        if arg_def.name in args and arg_def.tags and 'content' in arg_def.tags:
            value = args[arg_def.name]
            if isinstance(value, str) and OMISSION_PLACEHOLDER_RE.search(value):
                return (
                    f"Argument '{arg_def.name}' contains an omission placeholder. "
                    f"These placeholders appear in your conversation history where "
                    f"content was removed to save context. "
                    f"Never copy these placeholders as argument values. "
                    f"If you genuinely need to add text matching this pattern, "
                    f"split it across two tool uses."
                )
    return None


class BaseParseToolCallsNode:
    """Base class for parsing tool calls from an AIMessage.
    
    Handles parsing, validation, duplicate detection, and error handling
    for both API and text protocol tool calls. Does NOT execute any tools -
    it sets up pending_tool_calls and tool_execution_context for the
    execute node.
    
    Subclasses must implement hook methods to build results appropriate
    for their state schema.
    """
    
    def __init__(self, registry, agent_protocol=None, max_tool_calls_per_response=None):
        self.registry = registry
        self.agent_protocol = agent_protocol
        self.agent_config = registry._agent_config
        self.use_api_protocol = self.agent_config.use_api_protocol_for_tool_functions
        self.max_tool_calls_per_response = max_tool_calls_per_response
    
    def _maybe_truncate(self, valid_calls, context):
        """Truncate valid_calls to the configured per-response limit.
        
        Validation (duplicate detection, missing args, unavailable tools, and
        validate_call_count) is performed on the full parsed set before this
        method is called.  Only the first N calls are returned for execution;
        skipped calls are recorded in context and reported after execution.
        
        Returns:
            (executed_calls, skipped_count) tuple.
        """
        limit = self.max_tool_calls_per_response
        if limit is None or len(valid_calls) <= limit:
            return list(valid_calls), 0
        executed = list(valid_calls[:limit])
        skipped = len(valid_calls) - limit
        context["skipped_tool_call_count"] = skipped
        context["max_tool_calls_per_response"] = limit
        return executed, skipped
    
    def __call__(self, state):
        messages = state.get("messages", [])
        if not messages:
            return self.build_empty_result(state)
        
        last_message = messages[-1]
        if not isinstance(last_message, AIMessage):
            return self.build_not_ai_result(state)
        
        # Check for truncated reasoning-only response
        finish_reason = ""
        if hasattr(last_message, 'response_metadata') and last_message.response_metadata:
            finish_reason = last_message.response_metadata.get('finish_reason', '')
        if finish_reason == 'length':
            has_reasoning = bool(
                last_message.additional_kwargs.get("reasoning_content") or
                last_message.additional_kwargs.get("reasoning")
            )
            if has_reasoning and not self._has_any_tool_calls(last_message):
                max_reasoning_count = state.get("consecutive_max_reasoning_count", 0) + 1
                if max_reasoning_count <= 2:
                    return self.build_truncated_reasoning_result(state, max_reasoning_count)
                else:
                    return self.build_max_reasoning_error_result(state, max_reasoning_count)
        
        if self.use_api_protocol:
            return self._handle_api_protocol(state, messages, last_message)
        else:
            return self._handle_text_protocol(state, messages, last_message)
    
    # ââ Hook methods (subclass must implement) ââââââââââââââââââââââââââ
    
    def build_empty_result(self, state):
        """Build result when there are no messages."""
        raise NotImplementedError
    
    def build_not_ai_result(self, state):
        """Build result when last message is not an AIMessage."""
        raise NotImplementedError
    
    def build_no_tool_calls_api_result(self, state, no_tool_count, error_msg_obj, warning_messages):
        """Build result for API protocol response with no tool calls."""
        raise NotImplementedError
    
    def build_no_tool_calls_text_result(self, state, error_msg_obj, invalid_history):
        """Build result for text protocol response with no valid tool calls."""
        raise NotImplementedError
    
    def build_fatal_parsing_error_result(self, state, error_msg_obj):
        """Build result for fatal parsing errors in text protocol."""
        raise NotImplementedError
    
    def build_duplicate_result(self, state, valid_calls, error_msg_obj):
        """Build result when duplicate tool calls are detected."""
        raise NotImplementedError
    
    def build_missing_args_result(self, state, valid_calls, error_msg_obj):
        """Build result when required args are missing."""
        raise NotImplementedError
    
    def build_multiple_calls_error_result(self, state, valid_calls, error_msg_obj):
        """Build result when too many tool calls are detected."""
        raise NotImplementedError

    def build_unavailable_tool_result(self, state, valid_calls, error_msg_obj):
        """Build result when the agent tries to use a real harness tool
        that is not available to this agent."""
        raise NotImplementedError
    
    def build_valid_result(self, state, pending, context):
        """Build result for successful parse with pending calls."""
        raise NotImplementedError
    
    # ââ Hook methods with default implementations âââââââââââââââââââââââ
    

    def build_truncated_reasoning_result(self, state, max_reasoning_count):
        """Build result for truncated reasoning-only response (within tolerance)."""
        raise NotImplementedError
    
    def build_max_reasoning_error_result(self, state, max_reasoning_count):
        """Build error result when consecutive max-reasoning exceeds threshold."""
        raise NotImplementedError

    def validate_call_count(self, valid_calls):
        """Return error_msg_obj if too many calls, None if OK. Default: no constraint."""
        return None
    
    def compute_api_omission_transformer(self, last_message, valid_calls, messages, keep_first_n_calls=None):
        """Compute omission transformer for API protocol. Override if needed."""
        return None
    
    def compute_text_omission_transformer(self, last_message, tc_dict_edits, tc_name_edits,
                                          content_edits, reasoning_edits, content_was_stripped,
                                          messages, keep_first_n_calls=None, call_span_deletions=None):
        """Compute omission transformer for text protocol. Override if needed."""
        return None
    
    def generate_no_tool_calls_api_error(self, no_tool_count):
        """Generate error message for API protocol no-tool-calls case."""
        error_msg = "Your response did not contain any valid tool function usages. You must use tool functions in every response.  To communicate with the user you use the send_message tool functions."
        if no_tool_count >= 2:
            error_msg += f"\n\n[Attempt {no_tool_count}] You have made {no_tool_count} consecutive responses without using any tools."
        return error_msg
    
    def generate_no_tool_calls_api_warnings(self, no_tool_count):
        """Generate warning messages for repeated no-tool-calls. Returns list of messages."""
        if no_tool_count >= 3:
            return [make_error_msg_object(
                "STOP IGNORING THE TOOL ERRORS. You are being told repeatedly that you "
                "must use tools in every response, yet you continue to respond without using tools. "
                "This is the ONLY way to communicate. Use send_message() to talk to the user.",
                no_tool_error=True
            )]
        return []
    
    def generate_max_reasoning_error(self, max_reasoning_count):
        """Generate error message for excessive consecutive max-reasoning responses."""
        error_msg = (
            "Your response hit the maximum completion token limit while generating reasoning. "
            "Your reasoning was truncated before you could produce any tool calls. "
            "You must use tool functions in every response."
        )
        if max_reasoning_count >= 3:
            ordinal = 'th' if max_reasoning_count > 3 else 'rd'
            error_msg += (
                f"\n\nThis is your {max_reasoning_count}{ordinal} "
                f"consecutive response that hit the token limit with only reasoning and no tool calls. "
                f"You are stuck in a reasoning loop. Limit your reasoning and use tools now."
            )
        return error_msg
    
    def generate_duplicate_error_msg(self, valid_calls):
        """Generate error message for duplicate tool calls."""
        return (
            "Error: This tool call is identical to your previous call. "
            "The operation has already been completed. "
            'Use send_message(content="...") to communicate with the user.'
        )
    
    def generate_no_tool_calls_text_base_error(self, ai_content):
        """Generate base error message when no tool calls and no partial matches."""
        error_msg = "Your response did not contain any valid tool function usages. You must use tool functions in every response.  To communicate with the user you use the send_message tool functions."
        detected_tags = detect_xml_tags(ai_content)
        if detected_tags:
            error_msg += f"Your response contains XML/HTML tags: {', '.join(detected_tags)}\n"
            error_msg += "Never use XML/HTML tags, angle brackets, or markup syntax.\n\n"
        return error_msg
    
    # ââ API protocol path âââââââââââââââââââââââââââââââââââââââââââââââ
    
    def _handle_api_protocol(self, state, messages, last_message):
        if not last_message.tool_calls:
            print("DEBUG _handle_api_protocol no tool calls")
            return self._handle_no_tool_calls_api(state)
        
        all_calls = [(tc.get("name", ""), tc.get("args", {})) for tc in last_message.tool_calls]

        # ââ Per-call validation: process calls in order, stop at first invalid ââ
        valid_prefix = []
        first_invalid = None  # (error_type, error_msg_obj) or None

        for api_call, tool_call in zip(all_calls, last_message.tool_calls):
            tool_name, args = api_call
            tool_call_id = tool_call.get("id", str(uuid.uuid4()))
            error_type, error_msg_obj = self._validate_api_call(tool_name, args, tool_call_id)
            if error_type is not None:
                print(f"DEBUG _handle_api_protocol per-call validation failed: {error_type}")
                first_invalid = (error_type, error_msg_obj)
                break
            valid_prefix.append(api_call)

        last_message.additional_kwargs['fluximp_valid_tool_calls'] = list(valid_prefix)

        # ââ Case 1: First call was invalid â no valid prefix to process ââ
        if first_invalid is not None and not valid_prefix:
            error_type, error_msg_obj = first_invalid
            if error_type == "unavailable":
                print(f"DEBUG _handle_api_protocol first call unavailable tool")
                return self.build_unavailable_tool_result(state, all_calls, error_msg_obj)
            else:
                print(f"DEBUG _handle_api_protocol first call invalid: {error_type}")
                return self.build_missing_args_result(state, all_calls, error_msg_obj)

        # ââ Batch-level checks on the valid prefix ââ
        # (These have the same behavior as before but now operate on just the valid prefix.)

        # Duplicate detection
        if self._is_duplicate_call(state, valid_prefix):
            print("DEBUG _handle_api_protocol duplicate call")
            return self._handle_duplicate(state, valid_prefix)

        # Too many calls
        multiple_error = self.validate_call_count(valid_prefix)
        if multiple_error is not None:
            print(f"DEBUG _handle_api_protocol too many tool calls: {multiple_error}")
            return self.build_multiple_calls_error_result(state, valid_prefix, multiple_error)

        # ââ Truncate to per-response execution limit ââ
        context = {
            "partial_match_error_msg_obj": None,
        }

        # A later call was invalid while a valid prefix preceded it (the rejected
        # case).  That invalid call is omitted from the stored AI message by the
        # omission transformer below (keep_first_n_calls), so we must NOT defer
        # its error message here.  Leaving it would produce a message referencing
        # a tool call that no longer exists in the preceding AI message -- confusing
        # for the model and, for providers that require tool-message ids to match a
        # preceding tool use (use_api_protocol_for_tool_functions=True with
        # use_tool_message_for_no_tool_error=False), a bad-request error.

        executed_calls, skipped_count = self._maybe_truncate(valid_prefix, context)
        if skipped_count:
            print(f"DEBUG _handle_api_protocol truncated to {len(executed_calls)} calls, skipped {skipped_count}")

        last_message.additional_kwargs['fluximp_valid_tool_calls'] = list(executed_calls)

        # ââ Compute omission transformer ââ
        # If calls were rejected (partial) or truncated, drop them from the stored message.
        rejected = first_invalid is not None and len(valid_prefix) < len(all_calls)
        keep_first_n_calls = len(executed_calls) if (skipped_count > 0 or rejected) else None
        omission_transformer = self.compute_api_omission_transformer(
            last_message, valid_prefix, messages,
            keep_first_n_calls=keep_first_n_calls)

        # ââ Build pending_tool_calls ââ
        pending = []
        print("DEBUG _handle_api_protocol valid calls:")
        for api_call, tool_call in zip(executed_calls, last_message.tool_calls):
            tool_name, args = api_call
            tool_call_id = tool_call.get("id", str(uuid.uuid4()))
            print(f"  {tool_name} {tool_call_id}")
            pending.append({
                "tool_name": tool_name,
                "args": args,
                "tool_call_id": tool_call_id,
            })

        context.update({
            "valid_calls": executed_calls,
            "all_valid_calls": valid_prefix,
            "omission_transformer": omission_transformer,
        })

        return self.build_valid_result(state, pending, context)
    
    def _handle_no_tool_calls_api(self, state):
        no_tool_count = state.get("consecutive_no_tool_count", 0) + 1
        error_msg = self.generate_no_tool_calls_api_error(no_tool_count)
        error_msg_obj = make_error_msg_object(error_msg, no_tool_error=True)
        warning_messages = self.generate_no_tool_calls_api_warnings(no_tool_count)
        return self.build_no_tool_calls_api_result(state, no_tool_count, error_msg_obj, warning_messages)
    
    # ââ Text protocol path ââââââââââââââââââââââââââââââââââââââââââââââ
    
    def _handle_text_protocol(self, state, messages, last_message):
        all_errors = []
        all_valid_calls = []
        all_partial_matches = []
        content_parts = []
        unavailable_real_tools = set()
        
        tc_dict_edits = []
        tc_name_edits = []
        content_call_spans = []
        content_call_edits = []
        reasoning_call_spans = []
        reasoning_call_edits = []
        content_removed_ranges = None
        content_was_stripped = False
        thinking_count = len(last_message.tool_calls) if last_message.tool_calls else 0
        
        if last_message.tool_calls:
            for i, tool_call in enumerate(last_message.tool_calls):
                call_str = tool_call.get("name", "")
                call_args = tool_call.get("args", {})
                if call_str:
                    if call_str in self.registry.supported_tool_names and call_str not in self.registry._tools:
                        unavailable_real_tools.add(call_str)
                    elif call_str in self.registry._tools and call_args:
                        error_msg = validate_required_args(call_str, call_args, self.registry)
                        if error_msg:
                            all_errors.append({'tool_name': call_str, 'error': error_msg})
                            break
                        all_valid_calls.append((call_str, call_args))
                        content_parts.append(call_str)
                        omitted = compute_omitted_args(call_str, self.registry, call_args)
                        if omitted:
                            tc_dict_edits.append((i, omitted))
                    else:
                        errors, valid_calls, partial_matches = self.registry.parse_tool_calls(call_str)
                        all_errors.extend(errors)
                        for tool_name, args, arg_spans, call_span in valid_calls:
                            all_valid_calls.append((tool_name, args))
                            omitted = compute_omitted_args(tool_name, self.registry, args)
                            if omitted:
                                tc_name_edits.append((i, arg_spans, omitted))
                        all_partial_matches.extend(partial_matches)
                        content_parts.append(call_str)
        
        if last_message.content:
            raw_content = strip_think_blocks(last_message.content)
            content_was_stripped = (raw_content != last_message.content)
            content_removed_ranges = find_think_block_ranges(last_message.content) if content_was_stripped else None
            errors, valid_calls, partial_matches = self.registry.parse_tool_calls(raw_content)
            all_errors.extend(errors)
            for tool_name, args, arg_spans, call_span in valid_calls:
                all_valid_calls.append((tool_name, args))
                content_call_spans.append(call_span)
                global_idx = thinking_count + len(content_call_spans) - 1
                omitted = compute_omitted_args(tool_name, self.registry, args)
                if omitted:
                    content_call_edits.append((global_idx, arg_spans, omitted))
            all_partial_matches.extend(partial_matches)
            content_parts.append(raw_content)

            # Check for real harness tools that are not available to this agent
            sup_errors, sup_valid_calls, sup_partial_matches = self.registry.parse_tool_calls_with_supported_tools(raw_content)
            self._collect_unavailable_from_parse(sup_errors, sup_valid_calls, sup_partial_matches, unavailable_real_tools)
        
        # Disable parsing content from reasoning for now.  May be causing more problems then it solves.
        if False and not all_errors and not all_valid_calls and not all_partial_matches:
            reasoning_content = last_message.additional_kwargs.get('reasoning_content')
            if not reasoning_content:
                reasoning_content = last_message.additional_kwargs.get('reasoning')
            if reasoning_content:
                errors, valid_calls, partial_matches = self.registry.parse_tool_calls(reasoning_content)
                all_errors.extend(errors)
                for tool_name, args, arg_spans, call_span in valid_calls:
                    all_valid_calls.append((tool_name, args))
                    reasoning_call_spans.append(call_span)
                    global_idx = thinking_count + len(content_call_spans) + len(reasoning_call_spans) - 1
                    omitted = compute_omitted_args(tool_name, self.registry, args)
                    if omitted:
                        reasoning_call_edits.append((global_idx, arg_spans, omitted))
                all_partial_matches.extend(partial_matches)
        
        errors = all_errors
        valid_calls = all_valid_calls
        partial_matches = all_partial_matches
        ai_content = "\n".join(content_parts)

        # If the agent tried to use real harness tools that are not available to
        # it, give a specific error before falling through to generic handling.
        last_message.additional_kwargs['fluximp_valid_tool_calls'] = list(valid_calls) if valid_calls else []
        if unavailable_real_tools:
            unavailable = sorted(unavailable_real_tools)
            print(f"DEBUG _handle_text_protocol unavailable tools: {unavailable}")
            error_msg_obj = self._build_unavailable_tool_error_obj(unavailable)
            return self.build_unavailable_tool_result(state, valid_calls, error_msg_obj)
        
        
        # Check for omission placeholders in content-tagged args
        for tool_name, args in valid_calls:
            placeholder_error = validate_no_placeholders(tool_name, args, self.registry)
            if placeholder_error:
                errors.append({'tool_name': tool_name, 'error': placeholder_error})
                break

        # Check for fatal parsing errors first
        if errors:
            error_msgs = []
            for error in errors:
                error_msgs.append(f"Error in {error['tool_name']}: {error['error']}")
            combined_error = "\n\n".join(error_msgs)
            print(f"DEBUG _handle_text_protocol fatal parsing errors: {combined_error}")
            error_msg_obj = make_error_msg_object(combined_error)
            return self.build_fatal_parsing_error_result(state, error_msg_obj)
        
        
        if not valid_calls:
            print("DEBUG _handle_text_protocol no tool calls; partial_matches:", partial_matches)
            return self._handle_no_tool_calls_text(state, ai_content, partial_matches)

        if self._is_duplicate_call(state, valid_calls):
            print("DEBUG _handle_text_protocol duplicate call")
            return self._handle_duplicate(state, valid_calls)
        
        # Check for too many calls
        multiple_error = self.validate_call_count(valid_calls)
        if multiple_error is not None:
            print(f"DEBUG _handle_text_protocol too many tool calls: {multiple_error}")
            return self.build_multiple_calls_error_result(state, valid_calls, multiple_error)
        
        # Truncate to per-response execution limit after validation
        context = {
            "partial_match_error_msg_obj": None,
        }
        executed_calls, skipped_count = self._maybe_truncate(valid_calls, context)
        if skipped_count:
            print(f"DEBUG _handle_text_protocol truncated to {len(executed_calls)} calls, skipped {skipped_count}")
        
        last_message.additional_kwargs['fluximp_valid_tool_calls'] = list(executed_calls)
        executed_count = len(executed_calls)
        content_edits = [(arg_spans, omitted) for idx, arg_spans, omitted in content_call_edits if idx < executed_count]
        reasoning_edits = [(arg_spans, omitted) for idx, arg_spans, omitted in reasoning_call_edits if idx < executed_count]
        content_span_deletions = []
        for idx, span in enumerate(content_call_spans):
            if thinking_count + idx >= executed_count:
                start, end = span
                if content_removed_ranges:
                    start = map_stripped_pos_to_original(start, content_removed_ranges)
                    end = map_stripped_pos_to_original(end, content_removed_ranges)
                content_span_deletions.append((start, end))
        reasoning_span_deletions = [span for idx, span in enumerate(reasoning_call_spans) if thinking_count + len(content_call_spans) + idx >= executed_count]
        call_span_deletions = {"content": content_span_deletions, "reasoning": reasoning_span_deletions}
        keep_first_n_calls = self.max_tool_calls_per_response if skipped_count > 0 else None
        omission_transformer = self.compute_text_omission_transformer(
            last_message, tc_dict_edits, tc_name_edits, content_edits,
            reasoning_edits, content_was_stripped, messages,
            keep_first_n_calls=keep_first_n_calls,
            call_span_deletions=call_span_deletions)
        # Build pending_tool_calls
        pending = []
        print("DEBUG _handle_text_protocol valid calls:")
        for tool_name, args in executed_calls:
            print(f"  {tool_name}")
            tool_call_id = str(uuid.uuid4())
            pending.append({
                "tool_name": tool_name,
                "args": args,
                "tool_call_id": tool_call_id,
            })
        
        # Pre-compute partial match error message object
        partial_match_error_msg_obj = None
        if partial_matches:
            partial_error_msg = generate_partial_match_error(partial_matches, self.registry)
            print(f"DEBUG _handle_text_protocol partial match error: {partial_error_msg}")
            partial_match_error_msg_obj = make_error_msg_object(partial_error_msg)
        
        context.update({
            "valid_calls": executed_calls,
            "all_valid_calls": valid_calls,
            "omission_transformer": omission_transformer,
            "partial_match_error_msg_obj": partial_match_error_msg_obj,
        })
        
        return self.build_valid_result(state, pending, context)
    
    def _handle_no_tool_calls_text(self, state, ai_content, partial_matches):
        """Handle text protocol response with no valid tool calls."""
        invalid_history = state.get("invalid_tool_calls_history", []).copy()
        invalid_history.append(ai_content)
        
        count = len(invalid_history)
        error_msg = ""
        
        # Check similarity to previous call using rapidfuzz
        if count >= 2:
            similarity = fuzz.ratio(invalid_history[-1], invalid_history[-2])
            if similarity >= 60:
                error_msg += f"You appear to be stuck in a loop.\nYou have made {count} consecutive highly similar invalid call attempts.\n"
            if similarity == 100:
                error_msg += "Your invalid tool call attempt is exactly identical to your previous invalid call attempt.\n"
            elif similarity >= 60:
                error_msg += f"Your invalid tool call attempt is {similarity:.0f}% similar to your previous invalid call attempt.\n"
        
        if partial_matches:
            error_msg += generate_partial_match_error(partial_matches, self.registry)
        else:
            error_msg += self.generate_no_tool_calls_text_base_error(ai_content)
        
        error_msg_obj = make_error_msg_object(error_msg)
        
        return self.build_no_tool_calls_text_result(state, error_msg_obj, invalid_history)
    
    def _handle_duplicate(self, state, valid_calls):
        error_msg = self.generate_duplicate_error_msg(valid_calls)
        error_msg_obj = make_error_msg_object(error_msg, no_tool_error=True)
        return self.build_duplicate_result(state, valid_calls, error_msg_obj)
    
    # ââ Shared helpers ââââââââââââââââââââââââââââââââââââââââââââââââââ
    
    def _is_duplicate_call(self, state, valid_calls):
        last_valid_calls = state.get("last_valid_calls")
        is_undo_redo_only = all(name in ("undo", "redo") for name, _ in valid_calls)
        return valid_calls == last_valid_calls and not is_undo_redo_only



    def _has_any_tool_calls(self, last_message):
        """Check if a message has any tool calls, using protocol-aware detection.

        For API protocol: just check last_message.tool_calls (the provider puts
        them there natively).

        For text protocol: tool calls can appear in tool_calls, content, or
        reasoning, so we need to check all three sources.
        """
        if self.use_api_protocol:
            return bool(last_message.tool_calls)
        else:
            # Text protocol: check all three sources
            if last_message.tool_calls:
                return True
            if last_message.content:
                raw = strip_think_blocks(last_message.content)
                errors, valid_calls, partial_matches = self.registry.parse_tool_calls(raw)
                if valid_calls or partial_matches:
                    return True
            reasoning = (last_message.additional_kwargs.get('reasoning_content') or
                         last_message.additional_kwargs.get('reasoning'))
            if reasoning:
                errors, valid_calls, partial_matches = self.registry.parse_tool_calls(reasoning)
                if valid_calls or partial_matches:
                    return True
            return False

    def _detect_unavailable_real_tools(self, tool_names):
        """Return names that are supported by the harness but not available to this agent.

        Filters to real tool names the harness recognizes.  Unknown tool names
        are intentionally ignored so that callers can still apply the normal
        invalid-format / no-tool-call handling for garbage input.
        """
        supported = self.registry.supported_tool_names
        available = set(self.registry._tools.keys())
        return sorted({name for name in tool_names if name and name in supported and name not in available})

    def _build_unavailable_tool_error_obj(self, unavailable_names):
        """Build an error message object for real-but-unavailable tool usage."""
        names_str = ", ".join(f"`{name}`" for name in unavailable_names)
        error_msg = (
            f"You tried to use the tool(s) {names_str} which are not available to you. "
            f"You may only use the tools shown in your system prompt."
        )
        return make_error_msg_object(error_msg)

    def _collect_unavailable_from_parse(self, errors, valid_calls, partial_matches, unavailable_real_tools):
        """Update unavailable_real_tools set from a parse result.

        ``errors``, ``valid_calls``, and ``partial_matches`` should come from
        parsing against the full set of harness-supported tools.
        """
        for tool_name, args, arg_spans, call_span in valid_calls:
            if tool_name not in self.registry._tools:
                unavailable_real_tools.add(tool_name)
        for pm in partial_matches:
            if pm.tool_name and pm.tool_name in self.registry.supported_tool_names and pm.tool_name not in self.registry._tools:
                unavailable_real_tools.add(pm.tool_name)

    def _validate_api_call(self, tool_name, args, tool_call_id):
        """Validate a single API tool call.

        Returns (error_type, error_msg_obj) if invalid, (None, None) if valid.
        error_type is 'unavailable', 'missing_args', or 'placeholder'.
        """
        # 1. Check unavailable real tool
        unavailable = self._detect_unavailable_real_tools([tool_name])
        if unavailable:
            names_str = ", ".join(f"`{name}`" for name in unavailable)
            error_msg = (
                f"You tried to use the tool(s) {names_str} which are not available to you. "
                f"You may only use the tools shown in your system prompt."
            )
            error_msg_obj = make_tool_error_obj(tool_call_id, error_msg)
            return ("unavailable", error_msg_obj)

        # 2. Check missing required args
        error_msg = validate_required_args(tool_name, args, self.registry)
        if error_msg:
            error_msg_obj = make_tool_error_obj(tool_call_id, error_msg)
            return ("missing_args", error_msg_obj)

        # 3. Check placeholders in content-tagged args
        placeholder_error = validate_no_placeholders(tool_name, args, self.registry)
        if placeholder_error:
            error_msg_obj = make_tool_error_obj(tool_call_id, placeholder_error)
            return ("placeholder", error_msg_obj)

        return (None, None)

class BaseExecuteToolCallNode:
    """Base class for executing a single tool call and updating state.
    
    Pops the first call from pending_tool_calls, executes it, and
    accumulates results. When the last call is executed (or a ToolError
    occurs), finalization logic runs to set metadata fields.
    
    Subclasses must implement hook methods to build results appropriate
    for their state schema.
    """
    
    def __init__(self, registry, agent_protocol=None):
        self.registry = registry
        self.agent_protocol = agent_protocol
        self.agent_config = registry._agent_config
    
    def __call__(self, state):
        pending = state.get("pending_tool_calls", [])
        if not pending:
            print("DEBUG BaseExecuteToolCallNode no pending tool calls")
            return self.build_empty_result(state)
        
        # Pop first call
        call = pending[0]
        remaining = pending[1:]
        is_final = len(remaining) == 0
        
        tool_name = call["tool_name"]
        args = dict(call["args"])  # Mutable copy - checkpoint deserialization may produce read-only dicts
        tool_call_id = call.get("tool_call_id", str(uuid.uuid4()))

        print(f"DEBUG BaseExecuteToolCallNode tool_name: {tool_name} is_final: {is_final} tool_call_id: {tool_call_id}")
        
        # Update state reference for tool functions that need it
        self.update_state_reference(state)
        
        # Write tool_use event
        self._write_tool_use_event(tool_name, (tool_name, args))
        
        # Execute tool
        try:
            result = self.registry.execute(tool_name, args, state, tool_call_id)
        except ToolError as e:
            tool_error_msg = f"ERROR executing tool '{tool_name}': {str(e)}"
            print(f"DEBUG BaseExecuteToolCallNode tool execution failed: {tool_error_msg}")
            self._write_tool_use_result_error(tool_error_msg)
            error_msg_obj = make_tool_error_obj(tool_call_id, tool_error_msg)
            return self.build_tool_error_result(state, error_msg_obj)
        
        # Success - accumulate results
        updates = self.accumulate_result(state, result, tool_name, remaining)
        
        # Finalize if last call
        if is_final:
            self.finalize(state, updates)
        
        print(f"DEBUG BaseExecuteToolCallNode tool execution succeeded")
        
        # Write success event
        self._write_tool_use_result_success(updates)
        
        return self.build_execute_result(state, updates, is_final)
    
    # ââ Hook methods (subclass must implement) ââââââââââââââââââââââââââ
    
    def build_empty_result(self, state):
        """Build result when there are no pending tool calls."""
        raise NotImplementedError
    
    def accumulate_result(self, state, result, tool_name, remaining):
        """Build updates dict from a successful tool execution."""
        raise NotImplementedError
    
    def build_tool_error_result(self, state, error_msg_obj):
        """Build result for a ToolError during execution."""
        raise NotImplementedError
    
    def finalize(self, state, updates):
        """Set finalization fields when all tool calls are done."""
        raise NotImplementedError
    
    def build_execute_result(self, state, updates, is_final):
        """Build the final result dict from updates."""
        raise NotImplementedError
    
    # ââ Hook methods with default implementations âââââââââââââââââââââââ
    
    def update_state_reference(self, state):
        """Update state service reference. Override in subclasses that need it."""
        pass
    
    # ââ Protocol event helpers ââââââââââââââââââââââââââââââââââââââââââ
    
    def _write_tool_use_event(self, tool_name, call_info):        
        print("DEBUG BaseExecuteToolCallNode._write_tool_use_event")
        if self.agent_protocol is None:
            return
        if not self.agent_protocol.has_pending_request(tool_name):
            self.agent_protocol.write_with_metadata('tool_use', '', tool_calls=[call_info])
    
    def _write_tool_use_result_error(self, error_msg):
        if self.agent_protocol is None:
            return
        self.agent_protocol.write_with_metadata("tool_use_result", error_msg,
                                                 execution_successful=0)
    
    def _write_tool_use_result_success(self, updates):
        if self.agent_protocol is None:
            return
        messages = updates.get("messages", [])
        if not messages:
            return
        last_msg = messages[-1]
        if isinstance(last_msg, ToolMessage):
            is_success = getattr(last_msg, "execution_successful", True)
            self.agent_protocol.write_with_metadata("tool_use_result", last_msg.content,
                                                     execution_successful=1 if is_success else 0)


# ââ Per-tool-call pipeline routing ââââââââââââââââââââââââââââââââââ

def route_after_parse(state):
    """Route after parse_tool_calls: execute if calls pending, else back to agent."""
    if state.get("pending_tool_calls"):
        return "execute"
    return "agent"


def route_after_execute(state):
    """Route after execute_tool_call: continue if more calls, else route to agent/end."""
    if state.get("pending_tool_calls"):
        return "continue"
    
    # All calls done - check for terminal tools
    last_tools = state.get("last_tools_executed", [])
    last_success = state.get("last_tool_success", False)
    terminal_tools = {"send_message", "submit_report"}
    if any(tool in terminal_tools for tool in last_tools) and last_success:
        return "end"
    return "agent"
