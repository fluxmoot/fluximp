"""Tool call omission transforms for the LangGraph agent pipeline.

Provides utilities for omitting content/validation arg values from tool call
messages sent back to the agent, keeping the agent's context window lean.
Arg-value omission is optional (see the omit_arg_values parameter); the same
transforms also drop whole skipped/rejected calls from the message.
Also provides think-block stripping and position-mapping helpers used by
the omission transforms.
"""

import regex
from typing import Any

# Regex to detect omission placeholder strings in arg values.
# Matches: <523 characters omitted as ...> or <characters omitted as ...>
# Used by parser and validation layers to catch LLMs copying placeholders.
OMISSION_PLACEHOLDER_RE = regex.compile(r'<(?:\d+\s+)?characters?\s+omitted\s+as\s+')

# Tools whose content args create new files (vs. modifying existing ones).
# Used to select the appropriate omission template.
_NEW_FILE_TOOLS = frozenset({'write_file', 'create_file'})


def make_content_omission_string(original_value, tool_name):
    """Build omission string for a content-tagged arg.

    Selects 'new_file' vs 'modified_file' category based on tool_name.
    Returns the replacement string if the original value meets the 2x length
    threshold (len(original) >= 2 * len(replacement)), otherwise returns None.
    """
    length = len(original_value)
    if tool_name in _NEW_FILE_TOOLS:
        template = '<{length} characters omitted as the contents of newly created files are automatically added to the views>'
    else:
        template = '<{length} characters omitted as modifications to files are automatically added to the views and the cumulative diff>'
    replacement = template.format(length=length)
    if length >= 2 * len(replacement):
        return replacement
    return None


def make_validation_omission_string(original_value):
    """Build omission string for a validation-tagged arg.

    Returns the replacement string if the original value meets the 2x length
    threshold (len(original) >= 2 * len(replacement)), otherwise returns None.
    """
    length = len(original_value)
    template = '<{length} characters omitted as anchor lines are noise after the tool was used>'
    replacement = template.format(length=length)
    if length >= 2 * len(replacement):
        return replacement
    return None

def strip_think_blocks(content: str) -> str:
    """Remove think/reasoning blocks from AI message content.

    Handles formats like:
    - ```think\ncontent\n```
    - 

    Args:
        content: The AI message content to clean

    Returns:
        Content with think blocks removed
    """
    # regex imported at module level

    # XML-style think tags
    # 
    open_tag = "<" + "think>"
    close_tag = "</" + "think>"
    pattern = open_tag + ".*?" + close_tag
    modified_content = regex.sub(pattern, "", content, flags=regex.DOTALL)
    if modified_content != content:
        return modified_content
    return content


def find_think_block_ranges(content: str) -> list:
    """Find all think blocks and return (start, end) pairs in original content.
    
    Used for mapping positions from stripped content back to original content.
    """
    open_tag = "<" + "think>"
    close_tag = "</" + "think>"
    pattern = open_tag + ".*?" + close_tag
    return [(m.start(), m.end()) for m in regex.finditer(pattern, content, flags=regex.DOTALL)]


def map_stripped_pos_to_original(stripped_pos: int, removed_ranges: list) -> int:
    """Map a position in stripped content to the corresponding position in original content.
    
    Args:
        stripped_pos: Position in the stripped content
        removed_ranges: List of (start, end) pairs for removed blocks, sorted by start
    
    Returns:
        Corresponding position in the original content
    """
    offset = 0
    for start, end in removed_ranges:
        if stripped_pos + offset >= start:
            offset += end - start
        else:
            break
    return stripped_pos + offset


def apply_arg_omissions(text: str, calls_info: list, removed_ranges: list = None) -> str:
    """Apply arg omissions to a text string using arg_spans from multiple tool calls.
    
    Args:
        text: The text to modify
        calls_info: List of (arg_spans, omitted_args) tuples, one per tool call.
            arg_spans: Dict mapping arg_name to (start, end) positions in the parsed text
            omitted_args: Dict mapping arg_name to omission replacement string
        removed_ranges: Think block ranges for position mapping (if text was stripped)
    
    Returns:
        Modified text with tagged arg values replaced by omission strings
    """
    # Collect all edits
    edits = []
    
    for arg_spans, omitted_args in calls_info:
        for arg_name, replacement in omitted_args.items():
            if arg_name not in arg_spans:
                continue
            start, end = arg_spans[arg_name]
            
            # Map position if needed
            if removed_ranges:
                orig_start = map_stripped_pos_to_original(start, removed_ranges)
                orig_end = map_stripped_pos_to_original(end, removed_ranges)
            else:
                orig_start, orig_end = start, end
            
            edits.append((orig_start, orig_end, replacement))
    
    if not edits:
        return text
    
    # Sort by start position descending (back-to-front) so earlier positions stay valid
    edits.sort(key=lambda x: x[0], reverse=True)
    
    # Apply edits
    result = text
    for start, end, replacement in edits:
        result = result[:start] + replacement + result[end:]
    
    return result


class ToolCallOmissionTransformer:
    """Transformer that returns a pre-computed omission version of an AI message.
    
    The omission version has content/validation tagged arg values omitted
    directly in the source text (content, reasoning, or tool_calls args dict).
    
    The omission message is pre-computed where all the 
    arg_spans and source text information is available.
    """
    
    
    def __init__(self, omission_message: Any, message_index: int = -1):
        """Initialize with a pre-computed omission message.
        
        Args:
            omission_message: The pre-computed AIMessage with omitted arg values
            message_index: Index of the AI message in the messages list (for direct attachment)
        """
        self.omission_message = omission_message
        self.message_index = message_index
    
    def __call__(self, msg: Any) -> Any:
        """Return the pre-computed omission message.
        
        Args:
            msg: The original AI message (ignored)
            
        Returns:
            The pre-computed omission message with omitted arg values
        """
        return self.omission_message


def compute_omitted_args(tool_name: str, registry, original_values: dict = None) -> dict:
    """Compute the omitted args dict for a tool call.
    
    Returns a dict mapping arg_name to omission_string for args that should be omitted.
    Only includes args with 'content' or 'validation' tags whose original values
    meet the 2x length threshold (original is at least twice as long as replacement).
    
    Args:
        tool_name: Name of the tool
        registry: ToolRegistry for looking up tool definitions
        original_values: Dict of original arg values (required for dynamic omission
            strings with character counts). If not provided, returns empty dict.
    
    Returns:
        Dict mapping arg_name to omission replacement string
    """
    if not original_values:
        return {}
    
    tool_def = registry._tools.get(tool_name)
    if not tool_def:
        return {}
    omitted = {}
    for arg_def in tool_def.args:
        if arg_def.tags and arg_def.name in original_values:
            value = original_values[arg_def.name]
            if not isinstance(value, str):
                continue
            if 'content' in arg_def.tags:
                replacement = make_content_omission_string(value, tool_name)
                if replacement:
                    omitted[arg_def.name] = replacement
            elif 'validation' in arg_def.tags:
                replacement = make_validation_omission_string(value)
                if replacement:
                    omitted[arg_def.name] = replacement
    return omitted


# ââ Standalone omission transformer functions ââ

def compute_api_omission_transformer(last_message, valid_calls, messages, registry, keep_first_n_calls=None, omit_arg_values=True):
    """Compute omission transformer for API protocol tool calls.
    
    Creates a copy of the last AIMessage with content/validation tagged
    arg values omitted, for use as an omission-only version.
    
    Args:
        keep_first_n_calls: If set, the returned omission message will only
            retain the first N tool calls from the original message. This is
            used when additional parsed calls were skipped due to the per-
            response execution limit.
        omit_arg_values: When False, arg values are left intact and only the
            call-dropping (keep_first_n_calls) behavior is applied.
    """
    api_tc_dict_edits = []
    if omit_arg_values:
        for i, (tool_name, args) in enumerate(valid_calls):
            omitted = compute_omitted_args(tool_name, registry, args)
            if omitted:
                api_tc_dict_edits.append((i, omitted))
    
    if api_tc_dict_edits or keep_first_n_calls is not None:
        omission_msg = last_message.model_copy(deep=True)
        if keep_first_n_calls is not None:
            omission_msg.tool_calls = omission_msg.tool_calls[:keep_first_n_calls]
        for tc_index, omitted_args in api_tc_dict_edits:
            if tc_index < len(omission_msg.tool_calls):
                for arg_name, replacement in omitted_args.items():
                    if arg_name in omission_msg.tool_calls[tc_index]["args"]:
                        omission_msg.tool_calls[tc_index]["args"][arg_name] = replacement
        return ToolCallOmissionTransformer(omission_msg, message_index=len(messages) - 1)
    
    return None


def compute_text_omission_transformer(last_message, tc_dict_edits, tc_name_edits,
                                       content_edits, reasoning_edits, content_was_stripped,
                                       messages, keep_first_n_calls=None, call_span_deletions=None,
                                       omit_arg_values=True):
    """Compute omission transformer for text protocol tool calls.
    
    Creates a copy of the last AIMessage with content/validation tagged
    arg values omitted in all content sources (tool_calls, content, reasoning).

    Args:
        keep_first_n_calls: If set, slice omission_msg.tool_calls to the first N
            entries (used for calls from tool_calls/thinking that were skipped).
        call_span_deletions: Optional dict mapping source name ("content" or
            "reasoning") to a list of (start, end) spans of whole calls that
            should be removed entirely. Whole-call deletion is applied before
            arg-value omission so that skipped calls disappear from history.
        omit_arg_values: When False, the arg-value edits are dropped and only
            whole-call deletion / keep_first_n_calls is applied.
    """
    if not omit_arg_values:
        tc_dict_edits = []
        tc_name_edits = []
        content_edits = []
        reasoning_edits = []
    has_edits = bool(tc_dict_edits or tc_name_edits or content_edits or reasoning_edits)
    has_deletions = bool(call_span_deletions and any(call_span_deletions.values()))
    if not has_edits and not has_deletions and keep_first_n_calls is None:
        return None
    
    omission_msg = last_message.model_copy(deep=True)
    
    # Slice thinking/API tool_calls when truncation occurred
    if keep_first_n_calls is not None:
        omission_msg.tool_calls = omission_msg.tool_calls[:keep_first_n_calls]
    
    # Apply tool_calls dict edits (fully parsed)
    for tc_index, omitted_args in tc_dict_edits:
        if tc_index < len(omission_msg.tool_calls):
            for arg_name, replacement in omitted_args.items():
                if arg_name in omission_msg.tool_calls[tc_index]["args"]:
                    omission_msg.tool_calls[tc_index]["args"][arg_name] = replacement
    
    # Apply tool_calls name edits (not fully parsed)
    for tc_index, arg_spans, omitted_args in tc_name_edits:
        if tc_index < len(omission_msg.tool_calls):
            omission_msg.tool_calls[tc_index]["name"] = apply_arg_omissions(
                omission_msg.tool_calls[tc_index]["name"], [(arg_spans, omitted_args)])
    
    def delete_spans(text, spans):
        """Remove the given (start, end) spans from text, processing end-to-start."""
        if not spans:
            return text
        chars = list(text)
        for start, end in sorted(spans, key=lambda s: s[0], reverse=True):
            if start < 0 or end > len(chars) or start >= end:
                continue
            del chars[start:end]
        return "".join(chars)
    
    # Whole-call deletion must run before arg-value omission.  Deleted spans
    # belong to skipped calls (later in the text), while edited spans belong to
    # executed calls (earlier in the text).  delete_spans processes spans
    # end-to-start, so positions for kept/executed calls remain valid.
    # Remove whole skipped calls before applying arg-value omissions
    if call_span_deletions:
        if "content" in call_span_deletions:
            omission_msg.content = delete_spans(omission_msg.content, call_span_deletions["content"])
        if "reasoning" in call_span_deletions and call_span_deletions["reasoning"]:
            reasoning_key = 'reasoning_content' if 'reasoning_content' in omission_msg.additional_kwargs else 'reasoning'
            if reasoning_key in omission_msg.additional_kwargs:
                omission_msg.additional_kwargs[reasoning_key] = delete_spans(
                    omission_msg.additional_kwargs[reasoning_key], call_span_deletions["reasoning"])
    
    # Apply content edits
    if content_edits:
        think_ranges = find_think_block_ranges(last_message.content) if content_was_stripped else None
        omission_msg.content = apply_arg_omissions(omission_msg.content, content_edits, think_ranges)
    
    # Apply reasoning edits
    if reasoning_edits:
        reasoning_key = 'reasoning_content' if 'reasoning_content' in omission_msg.additional_kwargs else 'reasoning'
        omission_msg.additional_kwargs[reasoning_key] = apply_arg_omissions(
            omission_msg.additional_kwargs[reasoning_key], reasoning_edits)
    
    return ToolCallOmissionTransformer(omission_msg, message_index=len(messages) - 1)
