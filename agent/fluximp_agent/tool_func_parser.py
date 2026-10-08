#!/usr/bin/env python3
"""
Tool Function Call Parser

This module provides parsing functionality for tool function calls from AI message content.
It supports multiple formats and provides detailed error information for partial matches.
"""

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any
import regex

from fluximp_agent.agent_config import ToolDef, ArgDef
from fluximp_agent.tool_call_omission import OMISSION_PLACEHOLDER_RE

@dataclass
class PartialMatch:
    """Represents a partially matched tool call that couldn't be fully parsed."""
    tool_name: str
    pattern_matched: str  # Which pattern got closest
    extracted_args: Dict[str, any]  # Successfully parsed args
    missing_args: List[str]  # Required args not found
    parse_error: str  # Why parsing failed
    raw_match_text: str  # The substring that looked like a tool call
    detected_tags: List[str] = None  # XML/HTML tags detected in content
    detected_json: List[str] = None  # JSON keys/structure detected in content
    start_pos: int = -1  # Start position in content (for filtering nested matches)
    end_pos: int = -1  # End position in content (for filtering nested matches)


def validate_arg_types(args: Dict[str, Any], tool_def: ToolDef) -> Optional[str]:
    """Validate that parsed arguments match their expected arg_types.
    
    Args:
        args: Dictionary of parsed argument values
        tool_def: ToolDef containing ArgDef definitions with arg_type info
        
    Returns:
        Error message string if validation fails, None if validation passes
    """
    for arg_def in tool_def.args:
        arg_name = arg_def.name
        expected_type = arg_def.arg_type
        
        # Skip if arg not present (optional args may be missing)
        if arg_name not in args:
            continue
        
        value = args[arg_name]
        
        if expected_type == "integer":
            # Note: bool is subclass of int in Python, so exclude it explicitly
            if not isinstance(value, int) or isinstance(value, bool):
                return f"Argument '{arg_name}' expected integer but got {type(value).__name__}: {repr(value)}"
        elif expected_type == "boolean":
            if not isinstance(value, bool):
                return f"Argument '{arg_name}' expected boolean but got {type(value).__name__}: {repr(value)}"
        elif expected_type == "array":
            if not isinstance(value, list):
                return f"Argument '{arg_name}' expected array but got {type(value).__name__}: {repr(value)}"
        elif expected_type == "string":
            if not isinstance(value, str):
                return f"Argument '{arg_name}' expected string but got {type(value).__name__}: {repr(value)}"
    
    return None


def detect_xml_tags(content: str) -> List[str]:
    """Detect XML/HTML tags in content.
    
    Args:
        content: The text to analyze
        
    Returns:
        List of unique tag names found (e.g., ['minimax:tool_call', 'invoke', 'parameter'])
    """
    tags = set()
    # Match opening tags: <tag> or <tag attr="value">
    for match in regex.finditer(r'<([a-zA-Z_][a-zA-Z0-9_:]*)(?:\s|>|/>)', content):
        tags.add(match.group(1))
    # Match closing tags: </tag>
    for match in regex.finditer(r'</([a-zA-Z_][a-zA-Z0-9_:]*)>', content):
        tags.add(match.group(1))
    return sorted(list(tags))


def _check_global_invalid_formats(content: str, tools: Dict[str, Dict[str, List[str]]]) -> Optional[PartialMatch]:
    """Check for invalid formats that are not tool-specific.
    
    These checks run once before per-tool parsing to avoid generating
    duplicate partial matches for every tool.
    
    Args:
        content: The AI message content to parse
        tools: Dict mapping tool_name to {'required': [...], 'optional': [...]}
        
    Returns:
        PartialMatch if an invalid format is detected, None otherwise
    """
    # Helper function to find tool names mentioned in content
    def _find_mentioned_tool(content: str, tools: Dict[str, Dict[str, List[str]]]) -> str:
        """Find tool names mentioned in content with word boundaries."""
        for tool_name in tools.keys():
            # Match tool name with word boundaries - must be surrounded by whitespace, punctuation, or special chars
            # Allow: space, tab, newline, ., ,, :, (, ), <, >, /, ", ', = but NOT _ (since tool names use _)
            # Pattern: (?<![a-zA-Z0-9_])tool_name(?![a-zA-Z0-9_])
            if regex.search(r'(?<![a-zA-Z0-9_])' + regex.escape(tool_name) + r'(?![a-zA-Z0-9_])', content):
                return tool_name
        return ""
    
    # 1. Check for literal 'tool_name' placeholder (AI used placeholder instead of actual tool name)
    # This is language-agnostic - detects the literal string regardless of surrounding text
    # Pattern 6 is VALID: tool_name(actual_tool, ...) - "actual_tool" is a bare word followed by comma
    # Placeholder is INVALID: tool_name(arg="...") - first thing after ( is arg=value
    # First check if this is Pattern 6 (valid)
    pattern6_match = regex.search(r'\btool_name\s*\(\s*[a-zA-Z_][a-zA-Z0-9_]*\s*,', content)
    if not pattern6_match:
        # Not Pattern 6, check for placeholder pattern
        placeholder_match = regex.search(r'\btool_name\s*\(', content)
        if placeholder_match:
            # Count occurrences
            tool_name_calls = regex.findall(r'\btool_name\s*\(', content)
            start_pos = placeholder_match.start()
            end_pos = placeholder_match.end()
            # Check for multiple placeholders (demonstration pattern)
            if len(tool_name_calls) > 1:
                return PartialMatch(
                    tool_name="",
                    pattern_matched="Invalid Format",
                    extracted_args={},
                    missing_args=[],
                    parse_error="Multiple 'tool_name' placeholders detected. Replace with actual tool names like 'write_file', 'send_message', 'edit_file', etc.",
                    raw_match_text="Multiple tool_name(...)",
                    start_pos=start_pos,
                    end_pos=end_pos
                )
            # Single tool_name placeholder
            return PartialMatch(
                tool_name="",
                pattern_matched="Invalid Format",
                extracted_args={},
                missing_args=[],
                parse_error="Replace 'tool_name' with the actual tool name you want to use (e.g., 'write_file', 'edit_file', 'send_message')",
                raw_match_text="tool_name(...)",
                start_pos=start_pos,
                end_pos=end_pos
            )
    
    # 2. Check for XML markup at content level (not within quoted strings)
    # Pattern matches XML tags: <tag>, </tag>, <tag/>, or <tag attr="value">
    detected_tags = detect_xml_tags(content)
    xml_pattern = regex.search(r'<[a-zA-Z_][a-zA-Z0-9_:]*(?:\s+[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*"[^"]*")*\s*/?>.*?</[a-zA-Z_][a-zA-Z0-9_:]*>', content, regex.DOTALL)
    if not xml_pattern:
        # Also check for self-closing or opening tags without closing
        xml_pattern = regex.search(r'<[a-zA-Z_][a-zA-Z0-9_:]*(?:\s+[a-zA-Z_][a-zA-Z0-9_]*\s*=\s*"[^"]*")*\s*/?>', content)
    if xml_pattern:
        raw_match = xml_pattern.group(0)[:100]  # First 100 chars
        start_pos = xml_pattern.start()
        end_pos = xml_pattern.end()
        mentioned_tool = _find_mentioned_tool(content, tools)
        if mentioned_tool:
            parse_error = f"You appear to be trying to call the tool function '{mentioned_tool}' but you're using an invalid format. Use standard format: {mentioned_tool}(arg1=\"value\", arg2=\"value\")"
        else:
            parse_error = "Tool call uses unsupported XML format. Use standard format: tool_name(arg1=\"value\", arg2=\"value\")"
        return PartialMatch(
            tool_name=mentioned_tool,
            pattern_matched="Invalid Format",
            extracted_args={},
            missing_args=[],
            parse_error=parse_error,
            raw_match_text=raw_match,
            detected_tags=detected_tags,
            start_pos=start_pos,
            end_pos=end_pos
        )
    
    # 3. Check for JSON format at content level
    # Look for JSON-like content with braces and quoted strings
    json_like_pattern = regex.search(r'\{[^}]*"[^"]+"\s*:\s*[^}]*\}', content, regex.DOTALL)
    if json_like_pattern:
        raw_match = json_like_pattern.group(0)[:100]  # First 100 chars
        start_pos = json_like_pattern.start()
        end_pos = json_like_pattern.end()
        # Extract JSON keys found in the content
        detected_json_keys = []
        for key_match in regex.finditer(r'"([a-zA-Z_][a-zA-Z0-9_]*)"\s*:', content):
            detected_json_keys.append(key_match.group(1))
        
        # Look for any of our tool names in double quotes within the content
        mentioned_tool = ""
        for tool_name in tools.keys():
            # Check if "tool_name" appears in the content (as a JSON string value)
            if regex.search(r'"' + regex.escape(tool_name) + r'"', content):
                mentioned_tool = tool_name
                break
        
        if mentioned_tool:
            parse_error = f"You appear to be trying to call the tool function '{mentioned_tool}' but you're using an invalid format. Use standard format: {mentioned_tool}(arg1=\"value\", arg2=\"value\")"
        else:
            parse_error = "Tool call uses unsupported JSON format. Use standard format: tool_name(arg1=\"value\", arg2=\"value\")"
        return PartialMatch(
            tool_name=mentioned_tool,
            pattern_matched="Invalid Format",
            extracted_args={},
            missing_args=[],
            parse_error=parse_error,
            raw_match_text=raw_match,
            detected_json=detected_json_keys,
            start_pos=start_pos,
            end_pos=end_pos
        )
    
    # 4. Check for square-bracket markup at content level (BBCode style, etc.)
    # Pattern matches: [TAG]...[/TAG] or [TAG attr="value"]...[/TAG] or [TAG/]
    # Only match if there's a closing tag OR self-closing syntax with /
    bracket_pattern = regex.search(r'\[[A-Z_][A-Z0-9_]*(?:\s+[^\]]+)?\](.*?)\[/[A-Z_][A-Z0-9_]*\]', content, regex.DOTALL)
    if not bracket_pattern:
        # Check for self-closing bracket tags: [TAG/] or [TAG attr="value" /]
        # Require the / to distinguish from simple [WORD] patterns like [FAIL] or [PASS]
        bracket_pattern = regex.search(r'\[[A-Z_][A-Z0-9_]*(?:\s+[^\]]+)?/\]', content)
    if bracket_pattern:
        raw_match = bracket_pattern.group(0)[:100]  # First 100 chars
        start_pos = bracket_pattern.start()
        end_pos = bracket_pattern.end()
        mentioned_tool = _find_mentioned_tool(content, tools)
        if mentioned_tool:
            parse_error = f"You appear to be trying to call the tool function '{mentioned_tool}' but you're using an invalid format. Use standard format: {mentioned_tool}(arg1=\"value\", arg2=\"value\")"
        else:
            parse_error = "Tool call uses unsupported markup format. Use standard format: tool_name(arg1=\"value\", arg2=\"value\")"
        return PartialMatch(
            tool_name=mentioned_tool,
            pattern_matched="Invalid Format",
            extracted_args={},
            missing_args=[],
            parse_error=parse_error,
            raw_match_text=raw_match,
            start_pos=start_pos,
            end_pos=end_pos
        )
    
    return None


def parse_tool_call(content: str, tools: Dict[str, ToolDef]) -> Tuple[List[Dict], List[Tuple[str, Dict, Dict[str, Tuple[int, int]], Tuple[int, int]]], List[PartialMatch]]:
    """Parse tool calls from content for multiple tools.
    
    This is the public API for parsing tool calls. It iterates over all
    provided tools and returns valid calls and partial matches.
    
    Args:
        content: The AI message content to parse
        tools: Dict mapping tool_name to ToolDef instances
        
    Returns:
        Tuple of (errors, valid_calls, partial_matches):
        - errors: List of error dicts with 'tool_name', 'error', 'raw_text' for fatal parsing errors
        - valid_calls: List of (tool_name, args_dict, arg_spans, call_span) tuples for fully parsed calls
          where arg_spans maps arg_name -> (start, end) in original content,
          and call_span is (start, end) of the entire tool call in original content.
        - partial_matches: List of PartialMatch objects for diagnostics
        
    Note: If errors is truthy, valid_calls will be empty. Errors indicate fatal
    parsing issues (e.g., unescaped quotes) that prevent reliable parsing.
    """
    errors = []
    valid_calls_with_positions = []
    partial_matches = []
    
    # First check for global invalid formats (not tool-specific)
    # This adds at most ONE partial match for format issues, then continues parsing
    # Convert tools to simplified dict for backward compatibility with _check_global_invalid_formats
    tools_simple = {
        tool_name: {
            'required': [arg.name for arg in tool_def.args if arg.required],
            'optional': [arg.name for arg in tool_def.args if not arg.required],
            'numeric': [arg.name for arg in tool_def.args if arg.arg_type in ('integer', 'number')]
        }
        for tool_name, tool_def in tools.items()
    }
    global_invalid = _check_global_invalid_formats(content, tools_simple)
    if global_invalid:
        partial_matches.append(global_invalid)
    
    for tool_name, tool_def in tools.items():
        arg_names = [arg.name for arg in tool_def.args if arg.required]
        optional_args = [arg.name for arg in tool_def.args if not arg.required]
        results = _parse_tool_call(content, tool_name, arg_names, optional_args, tools_simple, errors)
        # results is now a list of (args_dict, start, end) or PartialMatch objects
        for result in results:
            if result is None:
                pass  # No match
            elif isinstance(result, tuple):
                # (args_dict, start, end, arg_spans)
                args_dict, start, end, arg_spans = result
                
                # Validate argument types using ToolDef
                type_error = validate_arg_types(args_dict, tool_def)
                if type_error:
                    partial = PartialMatch(
                        tool_name=tool_name,
                        pattern_matched="Type Validation",
                        extracted_args=args_dict.copy(),
                        missing_args=[],
                        parse_error=type_error,
                        raw_match_text=content[start:end] if start != -1 else "",
                        start_pos=start,
                        end_pos=end
                    )
                    partial_matches.append(partial)
                    continue  # Skip adding to valid_calls_with_positions
                
                valid_calls_with_positions.append((tool_name, args_dict, start, end, arg_spans))
            elif isinstance(result, PartialMatch):
                partial_matches.append(result)
    
    # If there are fatal parsing errors, return empty valid_calls
    if errors:
        return errors, [], partial_matches
    
    # Filter out nested tool calls (tool calls inside another tool's parameter value)
    # Keep positions for now to filter partial matches
    valid_calls_filtered = _filter_nested_calls_with_positions(valid_calls_with_positions)
    
    # Sort by position in the text to preserve execution order
    valid_calls_filtered.sort(key=lambda x: x[2] if x[2] != -1 else float('inf'))
    
    # Filter out partial matches that fall within valid tool call spans
    # These are likely false positives from content inside string parameters
    filtered_partial_matches = _filter_nested_partial_matches(partial_matches, valid_calls_filtered)
    
    # Preserve arg_spans and call_span in valid_calls for final return
    valid_calls = [(tool_name, args, arg_spans, (start, end)) for tool_name, args, start, end, arg_spans in valid_calls_filtered]
    
    return [], valid_calls, filtered_partial_matches


def _find_tool_match_position(content: str, tool_name: str) -> Optional[Tuple[int, int]]:
    """Find the start and end position of a tool call in content.
    
    Args:
        content: The content to search
        tool_name: The tool name to find
        
    Returns:
        Tuple of (start, end) positions or None if not found
    """
    # Look for tool_name followed by parentheses
    pattern = regex.compile(r'(?<![a-zA-Z0-9_])' + regex.escape(tool_name) + r'\s*\(')
    match = pattern.search(content)
    if match:
        start = match.start()
        # Find the closing parenthesis
        paren_count = 0
        i = match.end() - 1  # Start from the opening paren
        while i < len(content):
            if content[i] == '(':
                paren_count += 1
            elif content[i] == ')':
                paren_count -= 1
                if paren_count == 0:
                    return (start, i + 1)
            i += 1
    return None


def _filter_nested_calls(calls_with_positions: List[Tuple[str, Dict, int, int, Dict[str, Tuple[int, int]]]]) -> List[Tuple[str, Dict]]:
    """Filter out tool calls that are nested inside another tool call's span.
    
    This prevents false positives where a tool name is mentioned inside
    another tool's parameter value (e.g., undo() mentioned in send_message content).
    
    Args:
        calls_with_positions: List of (tool_name, args_dict, start, end, arg_spans) tuples
        
    Returns:
        Filtered list of calls with nested calls removed
    """
    if len(calls_with_positions) <= 1:
        return [(tool_name, args) for tool_name, args, _, _, _ in calls_with_positions]
    
    # Filter out calls that are nested inside another call
    filtered = []
    for i, (tool_name, args, start, end, _) in enumerate(calls_with_positions):
        if start == -1:  # No position found, include it
            filtered.append((tool_name, args))
            continue

        is_nested = False
        for j, (other_name, _, other_start, other_end, _) in enumerate(calls_with_positions):
            if i != j and other_start != -1 and start >= other_start and end <= other_end:
                # This call is nested inside another call
                is_nested = True
                break
        if not is_nested:
            filtered.append((tool_name, args))

    return filtered


def _filter_nested_calls_with_positions(calls_with_positions: List[Tuple[str, Dict, int, int, Dict[str, Tuple[int, int]]]]) -> List[Tuple[str, Dict, int, int, Dict[str, Tuple[int, int]]]]:
    """Filter out tool calls that are nested inside another tool call's span.
    
    Same as _filter_nested_calls but keeps positions for further processing.
    
    Args:
        calls_with_positions: List of (tool_name, args_dict, start, end, arg_spans) tuples
        
    Returns:
        Filtered list of calls with nested calls removed, keeping positions
    """
    if len(calls_with_positions) <= 1:
        return calls_with_positions
    
    # Filter out calls that are nested inside another call
    filtered = []
    for i, (tool_name, args, start, end, arg_spans) in enumerate(calls_with_positions):
        if start == -1:  # No position found, include it
            filtered.append((tool_name, args, start, end, arg_spans))
            continue
            
        is_nested = False
        for j, (other_name, _, other_start, other_end, _) in enumerate(calls_with_positions):
            if i != j and other_start != -1 and start >= other_start and end <= other_end:
                # This call is nested inside another call
                is_nested = True
                break
        if not is_nested:
            filtered.append((tool_name, args, start, end, arg_spans))
    
    return filtered


def _filter_nested_partial_matches(partial_matches: List[PartialMatch], valid_calls: List[Tuple[str, Dict, int, int, Dict[str, Tuple[int, int]]]]) -> List[PartialMatch]:
    """Filter out partial matches that fall within valid tool call spans.
    
    This prevents false positives where the global format checker detects
    patterns (like [FAIL], [PASS]) that are actually content inside valid
    tool call string parameters.
    
    Also filters out global "Invalid Format" partial matches (XML/JSON detection)
    when there are valid tool calls, since these are false positives when at
    least one tool was successfully parsed.
    
    Args:
        partial_matches: List of PartialMatch objects from parser
        valid_calls: List of (tool_name, args_dict, start, end, arg_spans) tuples for valid calls
        
    Returns:
        Filtered list of partial matches with nested ones removed
    """
    if not partial_matches:
        return partial_matches
    
    # If we have valid tool calls, filter out ALL "Invalid Format" partial matches
    # These are false positives when at least one tool was successfully parsed.
    # The reasoning: if the model produced valid tool calls in the standard format,
    # any XML/JSON format errors are false positives from the global checker.
    if valid_calls:
        partial_matches = [
            pm for pm in partial_matches
            if pm.pattern_matched != "Invalid Format"
        ]
    
    if not valid_calls:
        return partial_matches
    
    filtered = []
    for pm in partial_matches:
        # If partial match has no position info:
        # - When there are valid calls: filter it out (can't verify it's not nested)
        # - When no valid calls: keep it for backwards compatibility
        if pm.start_pos == -1 or pm.end_pos == -1:
            if not valid_calls:
                filtered.append(pm)
            continue
        
        # Check if this partial match falls within any valid tool call span
        # Check if this partial match overlaps any valid tool call span
        # Overlap means the partial match's range intersects the valid call's range.
        # This catches both fully-nested matches AND matches that extend past the
        # valid call boundary (which happens when the partial match's end position
        # is computed independently from its own paren balancing).
        is_overlapping = False
        is_before_valid = False
        for _, _, valid_start, valid_end, _ in valid_calls:
            # Two ranges overlap if: start1 < end2 AND start2 < end1
            if pm.start_pos < valid_end and pm.end_pos > valid_start:
                # This partial match overlaps a valid tool call - likely a false positive
                is_overlapping = True
                break
            # Check if partial match ends right before a valid tool call
            # (e.g.,  remarks immediately before write_file(...))
            if pm.end_pos <= valid_start and pm.end_pos > valid_start - 2:
                is_before_valid = True
                break
        
        if not is_overlapping and not is_before_valid:
            filtered.append(pm)
    
    return filtered


def _detect_arg_quote_types(args_str: str, valid_arg_names: set) -> Dict[str, str]:
    """Detect the outer quote type for each arg=value pair in args_str.
    
    Scans each arg_name=value pair and determines whether the value is delimited
    by single-quotes or double-quotes, based on the first quote character after =.
    Only detects args whose = sign is outside any quoted string, to avoid
    false matches from content like file_path="..." inside a new_content value.
    
    Args:
        args_str: The content inside the tool call parentheses
        valid_arg_names: Set of valid argument names for this tool
        
    Returns:
        Dict mapping arg_name to its quote type (single-quote, double-quote,
        triple-single, or triple-double as specific quote characters).
    """
    arg_quote_types = {}
    in_double_quote = False
    in_single_quote = False
    i = 0
    while i < len(args_str):
        char = args_str[i]
        if char == '\\' and i + 1 < len(args_str):
            i += 2
            continue
        if not in_single_quote and i + 2 < len(args_str) and args_str[i:i+3] == '"""':
            in_double_quote = not in_double_quote
            i += 3
            continue
        if not in_double_quote and i + 2 < len(args_str) and args_str[i:i+3] == "'''":
            in_single_quote = not in_single_quote
            i += 3
            continue
        if not in_double_quote and not in_single_quote:
            if char == '"':
                in_double_quote = True
                i += 1
                continue
            elif char == "'":
                in_single_quote = True
                i += 1
                continue
        elif in_double_quote and char == '"':
            in_double_quote = False
            i += 1
            continue
        elif in_single_quote and char == "'":
            in_single_quote = False
            i += 1
            continue
        if not in_double_quote and not in_single_quote:
            for name in valid_arg_names:
                if args_str[i:i+len(name)] == name:
                    after_name = i + len(name)
                    rest = args_str[after_name:].lstrip()
                    if rest.startswith('='):
                        eq_end = after_name + (len(args_str) - after_name - len(args_str[after_name:].lstrip())) + 1
                        pos = eq_end
                        while pos < len(args_str) and args_str[pos] in ' \t':
                            pos += 1
                        if pos < len(args_str):
                            qchar = args_str[pos]
                            if qchar == '"':
                                if pos + 2 < len(args_str) and args_str[pos:pos+3] == '"""':
                                    arg_quote_types[name] = '"""'
                                else:
                                    arg_quote_types[name] = '"'
                            elif qchar == "'":
                                if pos + 2 < len(args_str) and args_str[pos:pos+3] == "'''":
                                    arg_quote_types[name] = "'''"
                                else:
                                    arg_quote_types[name] = "'"
                        break
        i += 1
    return arg_quote_types


def _parse_tool_call(content: str, tool_name: str, arg_names: List[str], optional_args: List[str], tools: Dict[str, Dict[str, List[str]]], errors: List[Dict] = None) -> List[Union[Tuple[Dict, int, int, Dict[str, Tuple[int, int]]], PartialMatch, None]]:
    """Parse tool calls from various formats.
    
    Supports:
    1. Standard: tool_name(arg1="...", arg2="...") - finds ALL occurrences
    2. tool_name: format: tool_name: tool_name\narg1: value\narg2: value (single only)
    3. tool_name= format: tool_name=tool_name\narg1=value\narg2=value (single only)
    4. argN/valueN format: arg1: param_name\nvalue1: value (single only)
    5. argN= format: arg1="param_name", arg2="value", ... (single only)
    6. tool_name(...) format: tool_name(tool_name, arg1="...", ...) (single only)

    Args:
        content: The AI message content to parse
        tool_name: The name of the tool to look for
        arg_names: List of required argument names in order
        optional_args: List of optional argument names
        tools: Dict mapping tool_name to {'required': [...], 'optional': [...]}
        errors: List to collect fatal parsing errors (e.g., unescaped quotes)

    Returns:
        List of results, each being:
        - Tuple[Dict, int, int, Dict[str, Tuple[int, int]]] with extracted arguments,
          position (start, end), and arg_spans mapping arg_name -> (start, end)
          in the original content if fully parsed
        - PartialMatch if partially parsed
        - None if not detected at all
    """
    if errors is None:
        errors = []
    import codecs
    import ast
    
    # Fields that may contain escape sequences like \n, \t, etc.
    escape_fields = {'content', 'diff_text', 'new_content', 'start_line_content', 'end_line_content'}
    
    def decode_escapes(value: str, field_name: str) -> str:
        """Decode escape sequences for text fields."""
        if field_name in escape_fields:
            try:
                if '\\"' in value:
                    value = value.replace('\\"', '"')
                if "\\'" in value:
                    value = value.replace("\\'", "'")
                return codecs.decode(value, 'unicode_escape')
            except (ValueError, UnicodeDecodeError):
                pass
        return value
    
    def parse_value(value: str, field_name: str) -> any:
        """Parse a value, handling lists, numbers, and strings."""
        # Try to parse as a list first
        if value.strip().startswith('[') and value.strip().endswith(']'):
            try:
                parsed = ast.literal_eval(value)
                # Reject Python-only literals not valid in JSON (e.g. Ellipsis)
                json.dumps(parsed)
                return parsed
            except (ValueError, SyntaxError, TypeError):
                pass
        # Try to parse as int
        if field_name in ['start_line', 'end_line', 'max_count', 'offset', 'limit']:
            try:
                return int(value)
            except ValueError:
                pass
        # Decode escapes for text fields
        return decode_escapes(value, field_name)
    
    def create_partial(pattern_name: str, args: Dict, raw_text: str, parse_error: str, start_pos: int = -1, end_pos: int = -1) -> PartialMatch:
        """Create a PartialMatch from current parsing state."""
        missing = [name for name in arg_names if name not in args]
        return PartialMatch(
            tool_name=tool_name,
            pattern_matched=pattern_name,
            extracted_args=args.copy(),
            missing_args=missing,
            parse_error=parse_error,
            raw_match_text=raw_text[:200] if len(raw_text) > 200 else raw_text,
            start_pos=start_pos,
            end_pos=end_pos
        )
    
    def all_required_present(args: Dict) -> bool:
        """Check if all required arguments are present."""
        return all(name in args for name in arg_names)
    
    best_partial: Optional[PartialMatch] = None
    args: Dict[str, any] = {}
    
    # Initialize results list for Pattern 1 (used by other patterns to skip)
    pattern1_results = []

    # Pattern 6: tool_name(tool_name, arg1="...", arg2=...) - recursive with balanced parens
    # Only try if Pattern 1 didn't find anything
    if not pattern1_results:
        pattern6 = regex.compile(r'tool_name\s*\(\s*' + regex.escape(tool_name) + r'\s*,\s*(.*)\)', regex.DOTALL)
        match = pattern6.search(content)
        if match:
            raw_match = match.group(0)
            args_str = match.group(1)
            
            for name in arg_names:
                val_match = regex.search(r'\b' + regex.escape(name) + r'\s*=\s*(\[[^\]]*\])', args_str)
                if val_match:
                    args[name] = parse_value(val_match.group(1), name)
                    continue
                val_match = regex.search(r'\b' + regex.escape(name) + r'\s*=\s*"([^"]*)"', args_str)
                if val_match:
                    args[name] = parse_value(val_match.group(1), name)
                elif name in ['start_line', 'end_line', 'max_count', 'offset', 'limit']:
                    val_match = regex.search(r'\b' + regex.escape(name) + r'\s*=\s*(\d+)', args_str)
                    if val_match:
                        args[name] = int(val_match.group(1))
            
            if all_required_present(args):
                # Pattern 6 only finds one call, return with position -1
                return [(args, -1, -1, {})]
            
            partial = create_partial("Pattern 6", args, raw_match, "incomplete arguments in tool_name(tool_name, ...) format")
            if len(args) > len(best_partial.extracted_args) if best_partial else True:
                best_partial = partial
            args = {}

    # Pattern 1: Standard tool_name(arg1="...", arg2=...)
    # Use finditer to find ALL occurrences of this tool
    pattern1_results = []
    pattern1 = regex.compile(r'(?<![a-zA-Z0-9_])' + regex.escape(tool_name) + r'\s*\(', regex.DOTALL)
    
    for match in pattern1.finditer(content):
        args: Dict[str, any] = {}
        arg_spans: Dict[str, Tuple[int, int]] = {}
        call_has_error = False  # Track if this tool call has a fatal parsing error
        start_pos = match.end()
        match_start = match.start()
        
        # Find the matching closing parenthesis, skipping over quoted strings
        paren_count = 1
        i = start_pos
        in_double_quote = False
        in_single_quote = False
        in_triple_double = False
        in_triple_single = False
        escape_next = False

        while i < len(content) and paren_count > 0:
            char = content[i]

            # Handle escape sequences
            if escape_next:
                escape_next = False
                i += 1
                continue

            if char == '\\':
                escape_next = True
                i += 1
                continue

            # Handle triple-quoted strings
            if not in_double_quote and not in_single_quote:
                if i + 2 < len(content) and content[i:i+3] == '"""':
                    in_triple_double = not in_triple_double
                    i += 3
                    continue
                if i + 2 < len(content) and content[i:i+3] == "'''":
                    in_triple_single = not in_triple_single
                    i += 3
                    continue

            # Handle single-quoted strings (only if not in triple quotes)
            if not in_triple_double and not in_triple_single:
                if char == '"' and not in_single_quote:
                    in_double_quote = not in_double_quote
                elif char == "'" and not in_double_quote:
                    in_single_quote = not in_single_quote

            # When inside a single/double-quoted string, also track triple-quote
            # boundaries of the OPPOSITE character type so that ) inside embedded
            # triple-quoted content isn't counted as closing the tool call.
            # Only track the opposite type: """ inside '...', ''' inside "..."
            if in_single_quote:
                if not in_triple_double and i + 2 < len(content) and content[i:i+3] == '"""':
                    in_triple_double = True
                    i += 3
                    continue
                if in_triple_double and i + 2 < len(content) and content[i:i+3] == '"""':
                    in_triple_double = False
                    i += 3
                    continue
            if in_double_quote:
                if not in_triple_single and i + 2 < len(content) and content[i:i+3] == "'''":
                    in_triple_single = True
                    i += 3
                    continue
                if in_triple_single and i + 2 < len(content) and content[i:i+3] == "'''":
                    in_triple_single = False
                    i += 3
                    continue

            # Only count parentheses when not inside any string
            if not in_double_quote and not in_single_quote and not in_triple_double and not in_triple_single:
                if char == '(':
                    paren_count += 1
                elif char == ')':
                    paren_count -= 1

            i += 1
         
        match_end = i
        raw_match = content[match_start:match_end]
        args_str = content[start_pos:match_end-1]  # Exclude the outer parens
        
        # NOTE: List value extraction (name=[...]) is deferred to AFTER the single-pass
        # quoted-string extraction below, so that we can use extracted_spans to skip
        # matches that fall inside already-extracted quoted string values (e.g.,
        # tool_calls=[...] appearing inside content="""...""" should NOT be
        # extracted as a separate argument).
        
        # Build list of valid arg names for this tool
        valid_arg_names = set(arg_names) | set(optional_args) | {'paths', 'globs', 'max_count', 'offset', 'limit',
                                                                 'start_line', 'end_line', 'line_numbers', 'file_path',
                                                                 'content', 'diff_text', 'new_content', 'patterns'}
        
        # Single-pass extraction of quoted string args.
        # For each arg_name, find its occurrences (arg_name=) in args_str and
        # inspect the delimiter right after = to determine quote type (single,
        # double, triple-single, triple-double), then extract the value with the
        # correct regex for that delimiter type. This avoids spurious matches
        # from quote characters inside values of the other type.

        # Precompile regexes for each delimiter type
        triple_double_re = regex.compile(r'"""(.*?)"""', regex.DOTALL)
        triple_double_greedy_re = regex.compile(r'"""(.*)"""', regex.DOTALL)
        triple_single_re = regex.compile(r"'''(.*?)'''", regex.DOTALL)
        double_quote_re = regex.compile(r'"((?:\\.|[^"\\])*)"')
        single_quote_re = regex.compile(r"'((?:\\.|[^'\\])*)'")

        # Helper: count triple-quote delimiters to detect unterminated strings
        def _count_triple_quote_delimiters(s):
            tdq = 0
            tsq = 0
            in_dq = False
            in_sq = False
            in_tdq = False
            in_tsq = False
            idx = 0
            while idx < len(s):
                if s[idx] == '\\' and idx + 1 < len(s):
                    idx += 2
                    continue

                # Handle triple quotes outside any regular-quoted string
                if not in_dq and not in_sq:
                    if idx + 2 < len(s) and s[idx:idx+3] == '"""':
                        tdq += 1
                        in_tdq = not in_tdq
                        idx += 3
                        continue
                    if idx + 2 < len(s) and s[idx:idx+3] == "'''":
                        tsq += 1
                        in_tsq = not in_tsq
                        idx += 3
                        continue

                # Handle regular quotes outside triple-quoted strings
                if not in_tdq and not in_tsq:
                    if not in_dq and not in_sq:
                        if s[idx] == '"':
                            in_dq = True
                        elif s[idx] == "'":
                            in_sq = True
                    elif in_dq and s[idx] == '"':
                        in_dq = False
                    elif in_sq and s[idx] == "'":
                        in_sq = False

                # Track opposite-type triple-quote boundaries inside regular-quoted strings
                if in_sq:
                    if not in_tdq and idx + 2 < len(s) and s[idx:idx+3] == '"""':
                        tdq += 1
                        in_tdq = True
                        idx += 3
                        continue
                    if in_tdq and idx + 2 < len(s) and s[idx:idx+3] == '"""':
                        in_tdq = False
                        idx += 3
                        continue
                if in_dq:
                    if not in_tsq and idx + 2 < len(s) and s[idx:idx+3] == "'''":
                        tsq += 1
                        in_tsq = True
                        idx += 3
                        continue
                    if in_tsq and idx + 2 < len(s) and s[idx:idx+3] == "'''":
                        in_tsq = False
                        idx += 3
                        continue

                idx += 1
            return tdq, tsq

        # Track value spans of already-extracted args to filter out
        # arg_name= occurrences inside other arg values
        extracted_spans = []

        # Find all arg_name= occurrences in args_str, ordered by position,
        # then extract each one's value based on the delimiter after =.
        # Process in position order so that extracted_spans correctly
        # filters out arg_name= occurrences inside other arg values.
        all_arg_occurrences = []
        for arg_name in valid_arg_names:
            if arg_name in args:
                continue
            arg_re = regex.compile(r'(?<![a-zA-Z0-9_])' + regex.escape(arg_name) + r'\s*=\s*')
            for arg_match in arg_re.finditer(args_str):
                all_arg_occurrences.append((arg_match.start(), arg_name, arg_match))

        # Sort by position in args_str
        all_arg_occurrences.sort(key=lambda x: x[0])

        # Track if an omission placeholder was detected in an unquoted arg value
        placeholder_arg_name = None
        for _, arg_name, arg_match in all_arg_occurrences:
            if arg_name in args:
                continue

            # Skip if this arg_name= occurrence is inside an already-extracted arg's value
            match_start_pos = arg_match.start()
            inside_extracted = False
            for span_start, span_end in extracted_spans:
                if span_start <= match_start_pos < span_end:
                    inside_extracted = True
                    break
            if inside_extracted:
                continue

            after_eq = arg_match.end()
            rest_after_eq = args_str[after_eq:].lstrip()
            leading_ws = len(args_str[after_eq:]) - len(rest_after_eq)
            value_start = after_eq + leading_ws

            if value_start >= len(args_str):
                continue

            # Determine delimiter type from the actual characters after =
            delim = None
            if args_str[value_start:value_start+3] == '"""':
                delim = '"""'
            elif args_str[value_start:value_start+3] == "'''":
                delim = "'''"
            elif args_str[value_start] == '"':
                delim = '"'
            elif args_str[value_start] == "'":
                delim = "'"
            else:
                # Check for omission placeholder pattern (unquoted value)
                if args_str[value_start] == '<' and OMISSION_PLACEHOLDER_RE.search(args_str[value_start:]):
                    placeholder_arg_name = arg_name
                continue

            # Extract value using the regex for this delimiter type
            content_after = args_str[value_start:]
            value = None
            val_match = None

            if delim == '"""':
                val_match = triple_double_re.match(content_after)
                if val_match:
                    value = val_match.group(1)
                    if arg_name in ('new_content', 'content', 'diff_text'):
                        greedy_match = triple_double_greedy_re.match(content_after)
                        if greedy_match:
                            value = greedy_match.group(1)
                            val_match = greedy_match
            elif delim == "'''":
                val_match = triple_single_re.match(content_after)
                if val_match:
                    value = val_match.group(1)
            elif delim == '"':
                val_match = double_quote_re.match(content_after)
                if val_match:
                    value = val_match.group(1)
            elif delim == "'":
                val_match = single_quote_re.match(content_after)
                if val_match:
                    value = val_match.group(1)

            if val_match is None or value is None:
                continue

            # Validate: check what comes after the matched value
            val_end = value_start + val_match.end()
            remainder = args_str[val_end:]
            stripped_remainder = remainder.lstrip()

            if delim == '"""':
                if stripped_remainder.startswith('"""'):
                    next_comma = remainder.find(',')
                    next_paren = remainder.find(')')
                    next_terminator = len(remainder)
                    if next_comma != -1:
                        next_terminator = min(next_terminator, next_comma)
                    if next_paren != -1:
                        next_terminator = min(next_terminator, next_paren)
                    if stripped_remainder.find('"""') < next_terminator:
                        errors.append({
                            "tool_name": tool_name,
                            "error": f"Argument '{arg_name}' contains unescaped triple-double-quotes (\"\"\"). "
                                     f"When content includes triple quotes, use triple-single-quotes (''') as the outer delimiter, "
                                     f"or escape the inner quotes as \\\"\\\"\\\".",
                            "raw_text": raw_match[:200] if len(raw_match) > 200 else raw_match
                        })
                        call_has_error = True
                        break
                else:
                    # For greedy matches on large content args, check if there's
                    # an unescaped """ before the terminator in the remainder.
                    if arg_name in ('new_content', 'content', 'diff_text'):
                        ri = 0
                        while ri < len(remainder):
                            next_tdq = remainder.find('"""', ri)
                            if next_tdq == -1:
                                break
                            num_bs = 0
                            bj = next_tdq - 1
                            while bj >= 0 and remainder[bj] == '\\':
                                num_bs += 1
                                bj -= 1
                            if num_bs % 2 == 0:
                                next_comma = remainder.find(',')
                                next_paren = remainder.find(')')
                                next_term = len(remainder)
                                if next_comma != -1:
                                    next_term = min(next_term, next_comma)
                                if next_paren != -1:
                                    next_term = min(next_term, next_paren)
                                if next_tdq < next_term:
                                    errors.append({
                                        "tool_name": tool_name,
                                        "error": f"Argument '{arg_name}' contains unescaped triple-double-quotes (\"\"\"). "
                                                 f"When content includes triple quotes, use triple-single-quotes (''') as the outer delimiter, "
                                                 f"or escape the inner quotes as \\\"\\\"\\\".",
                                        "raw_text": raw_match[:200] if len(raw_match) > 200 else raw_match
                                    })
                                    call_has_error = True
                                break
                            ri = next_tdq + 3
                        if call_has_error:
                            break
                # Check for truncated response: count triple-dq delimiters for odd count.
                if arg_name in ('new_content', 'content', 'diff_text') and stripped_remainder and not call_has_error:
                    tdq_count = 0
                    _in_dq = False
                    _in_sq = False
                    _in_tdq = False
                    _in_tsq = False
                    _idx = 0
                    while _idx < len(args_str):
                        if args_str[_idx] == '\\' and _idx + 1 < len(args_str):
                            _idx += 2
                            continue
                        if not _in_dq and not _in_sq:
                            if args_str[_idx:_idx+3] == '"""':
                                tdq_count += 1
                                _in_tdq = not _in_tdq
                                _idx += 3
                                continue
                            if args_str[_idx:_idx+3] == "'''":
                                _in_tsq = not _in_tsq
                                _idx += 3
                                continue
                        if not _in_tdq and not _in_tsq:
                            if not _in_dq and not _in_sq:
                                if args_str[_idx] == '"':
                                    _in_dq = True
                                elif args_str[_idx] == "'":
                                    _in_sq = True
                            elif _in_dq and args_str[_idx] == '"':
                                _in_dq = False
                            elif _in_sq and args_str[_idx] == "'":
                                _in_sq = False
                        if _in_sq:
                            if not _in_tdq and args_str[_idx:_idx+3] == '"""':
                                tdq_count += 1
                                _in_tdq = True
                                _idx += 3
                                continue
                            if _in_tdq and args_str[_idx:_idx+3] == '"""':
                                _in_tdq = False
                                _idx += 3
                                continue
                        if _in_dq:
                            if not _in_tsq and args_str[_idx:_idx+3] == "'''":
                                _in_tsq = True
                                _idx += 3
                                continue
                            if _in_tsq and args_str[_idx:_idx+3] == "'''":
                                _in_tsq = False
                                _idx += 3
                                continue
                        _idx += 1
                    if tdq_count % 2 == 1:
                        errors.append({
                            "tool_name": tool_name,
                            "error": f"Argument '{arg_name}' has unterminated triple-quoted string (\"\"\"). "
                                     f"Your response appears to be truncated - the content after \"\"\" was cut off.",
                            "raw_text": raw_match[:200] if len(raw_match) > 200 else raw_match
                        })
                        call_has_error = True
                        break
            elif delim == "'''":
                r = 0
                while r < len(remainder):
                    next_triple = remainder.find("'''", r)
                    if next_triple == -1:
                        break
                    num_backslashes = 0
                    j = next_triple - 1
                    while j >= 0 and remainder[j] == '\\':
                        num_backslashes += 1
                        j -= 1
                    if num_backslashes % 2 == 0:
                        next_comma = remainder.find(',')
                        next_paren = remainder.find(')')
                        next_terminator = len(remainder)
                        if next_comma != -1:
                            next_terminator = min(next_terminator, next_comma)
                        if next_paren != -1:
                            next_terminator = min(next_terminator, next_paren)
                        if next_triple < next_terminator:
                            errors.append({
                                "tool_name": tool_name,
                                "error": f"Argument '{arg_name}' contains unescaped triple-single-quotes ('''). "
                                         f"When content includes triple quotes, use triple-double-quotes (\"\"\") as the outer delimiter, "
                                         f"or escape the inner quotes as \\'\\'\\'.",
                                "raw_text": raw_match[:200] if len(raw_match) > 200 else raw_match
                            })
                            call_has_error = True
                        break
                    r = next_triple + 3
            elif delim == '"':
                if stripped_remainder and stripped_remainder[0] not in ',)':
                    errors.append({
                        "tool_name": tool_name,
                        "error": f"Argument '{arg_name}' contains unescaped double-quotes (\"). "
                                 f"When content includes double-quotes, use single-quotes (') as the outer delimiter, "
                                 f"or escape the inner quotes as \\\".",
                        "raw_text": raw_match[:200] if len(raw_match) > 200 else raw_match
                    })
                    call_has_error = True
                    break
            elif delim == "'":
                if stripped_remainder and stripped_remainder[0] not in ',)':
                    errors.append({
                        "tool_name": tool_name,
                        "error": f"Argument '{arg_name}' contains unescaped single-quotes ('). "
                                 f"When content includes single-quotes, use double-quotes (\") as the outer delimiter, "
                                 f"or escape the inner quotes as \\'.",
                        "raw_text": raw_match[:200] if len(raw_match) > 200 else raw_match
                    })
                    call_has_error = True
                    break

            if call_has_error:
                break

            args[arg_name] = decode_escapes(value, arg_name)

            # Track the value span for filtering nested arg_name= matches
            extracted_spans.append((value_start, value_start + val_match.end()))
            # Record absolute position of arg value in original content
            abs_value_start = start_pos + value_start
            abs_value_end = start_pos + value_start + val_match.end()
            arg_spans[arg_name] = (abs_value_start, abs_value_end)

            # Convert numeric string values to int for known numeric args
            _numeric_arg_names = {'start_line', 'end_line', 'max_count', 'offset', 'limit', 'insert_line'}
            if arg_name in _numeric_arg_names and isinstance(args[arg_name], str):
                try:
                    args[arg_name] = int(args[arg_name])
                except ValueError:
                    pass

            if call_has_error:
                break

        if call_has_error:
            continue

        # Check for unterminated triple-quoted strings indicating truncated response.
        # Only check arg positions that correspond to valid tool args that we
        # failed to extract. If the delimiter was triple-quoted and the regex
        # couldn't find the closing triple-quote, the response was truncated.
        for arg_name in valid_arg_names:
            if arg_name in args:
                continue
            arg_re_check = regex.compile(r'(?<![a-zA-Z0-9_])' + regex.escape(arg_name) + r'\s*=\s*')
            for arg_match_check in arg_re_check.finditer(args_str):
                after_eq_check = arg_match_check.end()
                rest_check = args_str[after_eq_check:].lstrip()
                ws_check = len(args_str[after_eq_check:]) - len(rest_check)
                vs_check = after_eq_check + ws_check
                if vs_check + 2 < len(args_str):
                    delim_check = args_str[vs_check:vs_check+3]
                    if delim_check == '"""' or delim_check == "'''":
                        errors.append({
                            "tool_name": tool_name,
                            "error": f"Argument '{arg_name}' has unterminated triple-quoted string ({delim_check}). "
                                     f"Your response appears to be truncated - the content after {delim_check} was cut off.",
                            "raw_text": raw_match[:200] if len(raw_match) > 200 else raw_match
                        })
                        call_has_error = True
                        break
            if call_has_error:
                break

        if call_has_error:
            continue

        # Extract list values, skipping matches inside already-extracted arg values
        # This was deferred from before the single-pass extraction so we can use
        # extracted_spans to avoid extracting name=[...] patterns that appear
        # inside quoted string values (e.g., tool_calls=[...] inside content="""...""")
        for val_match in regex.finditer(r'(\w+)\s*=\s*(\[[\s\S]*?\])', args_str):
            name = val_match.group(1)
            if name in args:
                continue
            if name not in valid_arg_names:
                continue
            # Skip if this match is inside an already-extracted arg's value
            match_start_pos = val_match.start()
            inside_extracted = False
            for span_start, span_end in extracted_spans:
                if span_start <= match_start_pos < span_end:
                    inside_extracted = True
                    break
            if inside_extracted:
                continue
            args[name] = parse_value(val_match.group(2), name)
            # Track the value span for filtering subsequent matches
            extracted_spans.append((val_match.start(2), val_match.end(2)))
            # Record absolute position of arg value in original content
            abs_value_start = start_pos + val_match.start(2)
            abs_value_end = start_pos + val_match.end(2)
            arg_spans[name] = (abs_value_start, abs_value_end)

        # Match numeric values for known numeric params
        # Use explicit 'numeric' list from tools dict, or fall back to hardcoded list for backward compatibility
        tool_info = tools.get(tool_name, {})
        if 'numeric' in tool_info:
            numeric_args = tool_info['numeric']
        else:
            # Fallback to hardcoded list for backward compatibility (e.g., test cases without type info)
            numeric_args = ['start_line', 'end_line', 'max_count', 'offset', 'limit']
        for name in numeric_args:
            if name in args:
                continue
            # Try unquoted numeric value, skipping matches inside extracted spans
            for val_match in regex.finditer(r'\b' + regex.escape(name) + r'\s*=\s*(\d+)', args_str):
                match_start_pos = val_match.start()
                inside_extracted = False
                for span_start, span_end in extracted_spans:
                    if span_start <= match_start_pos < span_end:
                        inside_extracted = True
                        break
                if not inside_extracted:
                    args[name] = int(val_match.group(1))
                    extracted_spans.append((val_match.start(1), val_match.end(1)))
                    abs_value_start = start_pos + val_match.start(1)
                    abs_value_end = start_pos + val_match.end(1)
                    arg_spans[name] = (abs_value_start, abs_value_end)
                    break
            if name in args:
                continue
            # Try quoted numeric value, skipping matches inside extracted spans
            _quoted_num_pat = regex.compile(
                r'\b' + regex.escape(name) + r'\s*=\s*[\x27\x22](\d+)[\x27\x22]'
            )
            for val_match in _quoted_num_pat.finditer(args_str):
                match_start_pos = val_match.start()
                inside_extracted = False
                for span_start, span_end in extracted_spans:
                    if span_start <= match_start_pos < span_end:
                        inside_extracted = True
                        break
                if not inside_extracted:
                    args[name] = int(val_match.group(1))
                    extracted_spans.append((val_match.start(1), val_match.end(1)))
                    abs_value_start = start_pos + val_match.start(1)
                    abs_value_end = start_pos + val_match.end(1)
                    arg_spans[name] = (abs_value_start, abs_value_end)
                    break

        # Accept if all required args are present
        if all_required_present(args):
            pattern1_results.append((args, match_start, match_end, arg_spans))
        else:
            if placeholder_arg_name:
                placeholder_error = (
                    f"Argument '{placeholder_arg_name}' appears to be an omission placeholder, not actual content. "
                    f"These placeholders (like '<N characters omitted as ...>') appear in your "
                    f"conversation history where content was removed to save context. "
                    f"Never copy these placeholders \u2014 provide the actual content instead."
                )
                partial = create_partial("Pattern 1", args, raw_match, placeholder_error, match_start, match_end)
            else:
                partial = create_partial("Pattern 1", args, raw_match, "incomplete arguments in standard format", match_start, match_end)
            if len(args) > len(best_partial.extracted_args) if best_partial else True:
                best_partial = partial
    
    # If we found any valid Pattern 1 results, return them
    if pattern1_results:
        return pattern1_results

    # Pattern 5: arg1="param_name", arg2="value", arg3="param_name", arg4="value"...
    # Only try if Pattern 1 didn't find anything
    if not pattern1_results:
        pattern5 = regex.compile(r'tool_name\s*=\s*["\']' + regex.escape(tool_name) + r'["\']')
        if pattern5.search(content):
            arg_pattern = regex.compile(r'arg(\d+)\s*=\s*(?:"([^"]+)"|\'([^\']+)\'|(\S+))')
            arg_pairs = arg_pattern.findall(content)
            i = 0
            while i < len(arg_pairs):
                arg_num, param_name_quoted, param_name_single, param_name_unquoted = arg_pairs[i]
                param_name = param_name_quoted or param_name_single or param_name_unquoted
                if i + 1 < len(arg_pairs):
                    arg_num2, param_value_quoted, param_value_single, param_value_unquoted = arg_pairs[i + 1]
                    param_value = param_value_quoted or param_value_single or param_value_unquoted
                    if param_name in arg_names or param_name in optional_args:
                        if param_name in ['start_line', 'end_line', 'max_count', 'offset', 'limit']:
                            try:
                                args[param_name] = int(param_value)
                            except ValueError:
                                args[param_name] = param_value
                        else:
                            args[param_name] = param_value
                i += 2
            
            if all_required_present(args):
                # Pattern 5 only finds one call, return with position -1
                return [(args, -1, -1, {})]
            
            partial = create_partial("Pattern 5", args, "argN format", "incomplete argN=value pairs")
            if len(args) > len(best_partial.extracted_args) if best_partial else True:
                best_partial = partial
            args = {}

    # Pattern 2: tool_name: tool_name\narg1: value\narg2: value
    # Only try if Pattern 1 didn't find anything
    if not pattern1_results:
        pattern2 = regex.compile(r'tool_name\s*:\s*' + regex.escape(tool_name), regex.IGNORECASE)
        pattern2_match = pattern2.search(content)
        if pattern2_match:
            for name in arg_names:
                # Try list pattern first: name: [...] (multiline)
                val_match = regex.search(regex.escape(name) + r'\s*:\s*(\[[^\]]*\])', content, regex.DOTALL)
                if val_match:
                    args[name] = parse_value(val_match.group(1), name)
                    continue
                val_match = regex.search(regex.escape(name) + r'\s*:\s*"([^"]*)"', content)
                if val_match:
                    args[name] = parse_value(val_match.group(1), name)
                else:
                    val_match = regex.search(regex.escape(name) + r"\s*:\s*'([^']*)'", content)
                    if val_match:
                        args[name] = parse_value(val_match.group(1), name)
                    else:
                        # Try unquoted value (for numbers or simple strings)
                        val_match = regex.search(regex.escape(name) + r'\s*:\s*([^\n]+)', content)
                        if val_match:
                            val = val_match.group(1).strip()
                            if name in ['start_line', 'end_line', 'max_count', 'offset', 'limit']:
                                try:
                                    args[name] = int(val)
                                except ValueError:
                                    args[name] = val
                            else:
                                args[name] = val
            
            if all_required_present(args):
                # Pattern 2 only finds one call, return with position -1
                return [(args, -1, -1, {})]
            
            partial = create_partial("Pattern 2", args, f"tool_name: {tool_name}", "incomplete tool_name: format")
            if len(args) > len(best_partial.extracted_args) if best_partial else True:
                best_partial = partial
            args = {}

    # Pattern 3a: tool_name=func_name\nparam=value format (multiline key=value pairs)
    # Only try if Pattern 1 didn't find anything
    if not pattern1_results:
        pattern3a = regex.compile(r'tool_name\s*=\s*' + regex.escape(tool_name) + r'(?:\s*\(\s*\))?(?:\n|$)', regex.IGNORECASE)
        if pattern3a.search(content):
            for name in arg_names:
                # Try list pattern first: name=[...]
                val_match = regex.search(regex.escape(name) + r'\s*=\s*(\[[^\]]*\])', content)
                if val_match:
                    args[name] = parse_value(val_match.group(1), name)
                    continue
                # Try quoted string
                val_match = regex.search(regex.escape(name) + r'\s*=\s*"([^"]*)"', content)
                if val_match:
                    args[name] = parse_value(val_match.group(1), name)
                elif name in ['start_line', 'end_line', 'max_count', 'offset', 'limit']:
                    # Try numeric value
                    val_match = regex.search(regex.escape(name) + r'\s*=\s*(\d+)', content)
                    if val_match:
                        args[name] = int(val_match.group(1))
            
            if all_required_present(args):
                # Pattern 3a only finds one call, return with position -1
                return [(args, -1, -1, {})]
            
            partial = create_partial("Pattern 3a", args, f"tool_name={tool_name}", "incomplete tool_name= format")
            if len(args) > len(best_partial.extracted_args) if best_partial else True:
                best_partial = partial
            args = {}
    # Pattern 3b: tool_name=func_name(args) - inline function call format
    # Only try if Pattern 1 didn't find anything
    if not pattern1_results:
        pattern3b = regex.compile(r'tool_name\s*=\s*' + regex.escape(tool_name) + r'\s*\((.*)\)', regex.DOTALL)
        match = pattern3b.search(content)
        if match:
            raw_match = match.group(0)
            args_str = match.group(1)
            
            for name in arg_names:
                # Try list pattern first: name=[...]
                val_match = regex.search(r'\b' + regex.escape(name) + r'\s*=\s*(\[[^\]]*\])', args_str)
                if val_match:
                    args[name] = parse_value(val_match.group(1), name)
                    continue
                # Try quoted string
                val_match = regex.search(r'\b' + regex.escape(name) + r'\s*=\s*"([^"]*)"', args_str)
                if val_match:
                    args[name] = parse_value(val_match.group(1), name)
                elif name in ['start_line', 'end_line', 'max_count', 'offset', 'limit']:
                    # Try numeric value
                    val_match = regex.search(r'\b' + regex.escape(name) + r'\s*=\s*(\d+)', args_str)
                    if val_match:
                        args[name] = int(val_match.group(1))
            if all_required_present(args):
                # Pattern 3b only finds one call, return with position -1
                return [(args, -1, -1, {})]
            
            partial = create_partial("Pattern 3b", args, raw_match, "incomplete arguments in tool_name=func(args) format")
            if len(args) > len(best_partial.extracted_args) if best_partial else True:
                best_partial = partial
            args = {}
    # Pattern 4: arg1: param_name\nvalue1: value\narg2: param_name\nvalue2: value
    # OR arg1: param_name=value (combined format)
    # Only try if Pattern 1 didn't find anything
    if not pattern1_results and regex.search(r'tool_name\s*:\s*' + regex.escape(tool_name), content, regex.IGNORECASE):
        arg_value_pairs = {}
        
        # First try combined format: argN: param_name=value
        for match in regex.finditer(r'arg(\d+)\s*:\s*(\w+)\s*=\s*([^\n]+)', content):
            arg_num = match.group(1)
            param_name = match.group(2)
            param_value = match.group(3).strip()
            # Remove quotes if present
            if (param_value.startswith('"') and param_value.endswith('"')) or \
               (param_value.startswith("'") and param_value.endswith("'")):
                param_value = param_value[1:-1]
            if param_name in ['start_line', 'end_line', 'max_count', 'offset', 'limit']:
                try:
                    arg_value_pairs[param_name] = int(param_value)
                except ValueError:
                    arg_value_pairs[param_name] = param_value
            else:
                arg_value_pairs[param_name] = param_value
        
        # If no combined format found, try separate argN/valueN format
        if not arg_value_pairs:
            for match in regex.finditer(r'arg(\d+)\s*:\s*(\w+)', content):
                arg_num = match.group(1)
                param_name = match.group(2)
                # Look for valueN
                val_match = regex.search(r'value' + arg_num + r'\s*:\s*"([^"]*)"', content)
                if val_match:
                    arg_value_pairs[param_name] = val_match.group(1)
                else:
                    val_match = regex.search(r'value' + arg_num + r"\s*:\s*'([^']*)'", content)
                    if val_match:
                        arg_value_pairs[param_name] = val_match.group(1)
                    else:
                        val_match = regex.search(r'value' + arg_num + r'\s*:\s*([^\n]+)', content)
                        if val_match:
                            val = val_match.group(1).strip()
                            if param_name in ['start_line', 'end_line', 'max_count', 'offset', 'limit']:
                                try:
                                    arg_value_pairs[param_name] = int(val)
                                except ValueError:
                                    arg_value_pairs[param_name] = val
                            else:
                                arg_value_pairs[param_name] = val

        # Check if we found all required args
        if all(name in arg_value_pairs for name in arg_names):
            # Pattern 4 only finds one call, return with position -1
            return [(arg_value_pairs, -1, -1, {})]
        
        partial = create_partial("Pattern 4", arg_value_pairs, f"argN/valueN for {tool_name}", "incomplete argN:value pairs")
        if len(arg_value_pairs) > len(best_partial.extracted_args) if best_partial else True:
            best_partial = partial

    # Pattern 7: Any text prefix followed by tool_name(args) at the end
    # Only try if Pattern 1 didn't find anything
    if not pattern1_results:
        pattern7 = regex.compile(
            r'(?<![a-zA-Z0-9_])' + regex.escape(tool_name) + r'\s*\((.*)\)\s*$',
            regex.DOTALL
        )
        match = pattern7.search(content)
        if match:
            raw_match = match.group(0)
            match_start = match.start()
            match_end = match.end()
            args_str = match.group(1)
            
            # Extract all arg=value pairs found in the call (both required and optional)
            # Match list values: name=[...]
            for val_match in regex.finditer(r'(\w+)\s*=\s*(\[[^\]]*\])', args_str):
                name = val_match.group(1)
                args[name] = parse_value(val_match.group(2), name)
            
# Match triple-quoted strings
            for val_match in regex.finditer(r'(\w+)\s*=\s*"""(.*)"""', args_str, regex.DOTALL):
                name = val_match.group(1)
                args[name] = val_match.group(2)
            for val_match in regex.finditer(r"(\w+)\s*=\s='''(.*?)'''", args_str, regex.DOTALL):
                name = val_match.group(1)
                args[name] = val_match.group(2)
            for val_match in regex.finditer(r"(\w+)\s*=\s*'''(.*?)'''", args_str, regex.DOTALL):
                name = val_match.group(1)
                args[name] = val_match.group(2)
            
            # Match double-quoted strings: name="..." (but not triple-quoted)
            for val_match in regex.finditer(r'(\w+)\s*=\s*"((?:\\.|[^"\\])*)"', args_str):
                name = val_match.group(1)
                if name not in args:  # Don't overwrite triple-quoted values
                    args[name] = decode_escapes(val_match.group(2), name)
            
            # Match single-quoted strings: name='...' (but not triple-quoted)
            for val_match in regex.finditer(r"(\w+)\s*=\s*'((?:\\.|[^'\\])*)'", args_str):
                name = val_match.group(1)
                if name not in args:  # Don't overwrite triple-quoted values
                    args[name] = decode_escapes(val_match.group(2), name)
            
            # Match numeric values for known numeric params
            for name in ['start_line', 'end_line', 'max_count', 'offset', 'limit']:
                val_match = regex.search(r'\b' + regex.escape(name) + r'\s*=\s*(\d+)', args_str)
                if val_match:
                    args[name] = int(val_match.group(1))
            
            # Check if all required args are present
            if all(name in args for name in arg_names):
                # Pattern 7 only finds one call, return with position
                return [(args, match_start, match_end, {})]
            
            partial = create_partial("Pattern 7", args, raw_match, "incomplete arguments at end-of-text format")
            if len(args) > len(best_partial.extracted_args) if best_partial else True:
                best_partial = partial
            args = {}

    # Check for unsupported markup format (e.g., .edit_file_lines: from XML/markup formats)
    invalid_pattern = regex.search(r'\.' + regex.escape(tool_name) + r':', content)
    if invalid_pattern:
        raw_match = invalid_pattern.group(0)
        return [create_partial("Invalid Format", {}, raw_match, "Tool call uses unsupported markup format (functions.tool_name:ID {...})")]
    
    # Return best partial match if we found anything, else empty list
    if best_partial:
        return [best_partial]
    
    return []
