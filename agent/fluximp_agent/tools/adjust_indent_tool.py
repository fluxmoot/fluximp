#!/usr/bin/env python3
"""
Adjust indent tool functions for file operations.

This module contains the adjust_indent tool functions used by the LangGraph-based coding agent.
These functions implement indent level adjustment capabilities for blocks of code,
with optional content validation.
"""

import math
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from fluximp_agent.tools.common import FileOperation, compress, lines_match
from fluximp_agent.filesystem_service import check_line_mode_file_size
from fluximp_agent.tools.tool_definitions import ToolResult
from fluximp_agent.tool_call_omission import OMISSION_PLACEHOLDER_RE

def _placeholder_error(arg_name, value):
    """Return specific error if validation arg matches omission placeholder pattern."""
    if OMISSION_PLACEHOLDER_RE.search(value):
        return f"The validation line you provided for '{arg_name}' matches an omission placeholder pattern. You may have copied a placeholder from your conversation history instead of the actual file content.\n"
    return None


def _parse_indent_level(line: str, spaces_per_indent: int) -> int:
    """Parse the indent level of a line.
    
    Tab characters count as 1 indent level each.
    Spaces are grouped by spaces_per_indent to form indent levels.
    Any remaining spaces are rounded up to 1 level.
    Mixed tabs and spaces on the same line are handled by summing
    the levels from each.
    
    Args:
        line: The line to parse
        spaces_per_indent: Number of spaces per indent level
        
    Returns:
        The indent level as an integer
    """
    level = 0
    space_count = 0
    
    for ch in line:
        if ch == '\t':
            if space_count > 0:
                # Accumulated spaces before this tab
                level += math.ceil(space_count / spaces_per_indent)
                space_count = 0
            level += 1
        elif ch == ' ':
            space_count += 1
        else:
            # Reached non-whitespace
            break
    
    # Handle remaining spaces after last tab (or all-spaces indent)
    if space_count > 0:
        level += math.ceil(space_count / spaces_per_indent)
    
    return level


def _has_tabs_in_indent(line: str) -> bool:
    """Check if a line uses tabs in its leading whitespace."""
    for ch in line:
        if ch == '\t':
            return True
        elif ch not in ' ':
            break
    return False


def _reconstruct_indent(level: int, use_tabs: bool, spaces_per_indent: int) -> str:
    """Reconstruct the indent string for a given level.
    
    Args:
        level: Number of indent levels
        use_tabs: If True, use tabs; otherwise use spaces
        spaces_per_indent: Number of spaces per indent level (only used when use_tabs is False)
        
    Returns:
        The indent string
    """
    if use_tabs:
        return '\t' * level
    else:
        return ' ' * (level * spaces_per_indent)


def _adjust_line_indent(line: str, indent_delta: int, spaces_per_indent: int) -> str:
    """Adjust the indent level of a single line.
    
    Blank lines (empty or whitespace-only) are left unchanged.
    
    Args:
        line: The line to adjust
        indent_delta: Number of indent levels to change (negative = decrease, positive = increase)
        spaces_per_indent: Number of spaces per indent level
        
    Returns:
        The line with adjusted indent
    """
    if not line or line.isspace():
        # Don't change blank/whitespace-only lines
        return line
    
    current_level = _parse_indent_level(line, spaces_per_indent)
    new_level = max(0, current_level + indent_delta)
    
    use_tabs = _has_tabs_in_indent(line)
    
    # Find where the non-whitespace content starts
    content_start = 0
    for i, ch in enumerate(line):
        if ch not in ' \t':
            content_start = i
            break
    else:
        # Line is all whitespace - return as-is
        return line
    
    content = line[content_start:]
    new_indent = _reconstruct_indent(new_level, use_tabs, spaces_per_indent)
    
    return new_indent + content


def adjust_indent_tool(file_path: str, start_line: int, end_line: int,
                       indent_delta: int, spaces_per_indent: int = 4,
                       **kwargs) -> ToolResult:
    """Adjust the indent level of a block of code - STATE-MODIFYING tool.

    Changes the indent level of lines from start_line to end_line (1-indexed, inclusive)
    by indent_delta levels. Negative values decrease indent, positive values increase it.
    
    Tab characters count as 1 indent level each. Spaces are grouped by spaces_per_indent
    to form indent levels, with any remainder rounded up. Mixed tabs and spaces are
    handled on a per-line basis.

    Stores compressed before/after content for reliable undo/redo.

    When the file_view_windows experiment is active, this tool also adds the
    affected region (+/- 5 lines) to the file view windows.

    Args:
        file_path: Source file path
        start_line: First line to adjust (1-indexed, inclusive)
        end_line: Last line to adjust (1-indexed, inclusive)
        indent_delta: Number of indent levels to change (negative = decrease, positive = increase)
        spaces_per_indent: Number of spaces per indent level (default: 4)

    Returns:
        ToolResult - with success status, content message, and state_updates dict
    """
    filesystem_service = kwargs.get('filesystem_service')
    state_service = kwargs.get('state_service')
    experiments_service = kwargs.get('experiments_service')

    # Validate path
    path, error = filesystem_service.validate_path(file_path)
    if error:
        return ToolResult(success=False, content=error)

    # Check file exists
    if not path.exists() or not path.is_file():
        return ToolResult(success=False, content=f"Error: File '{file_path}' not found")

    size_error = check_line_mode_file_size(path, file_path)
    if size_error:
        return ToolResult(success=False, content=size_error)
    # Read original content
    try:
        original_content = path.read_text(encoding='utf-8')
    except Exception as e:
        return ToolResult(success=False, content=f"Error reading file: {e}")

    # Store compressed before content
    compressed_before = compress(original_content)

    # Split into lines
    original_lines = original_content.splitlines(keepends=True)
    total_lines = len(original_lines)

    # Validate line numbers
    if start_line < 1:
        start_line = 1
    if end_line < start_line:
        end_line = start_line
    if start_line > total_lines:
        return ToolResult(success=False, content=f"Error: start_line {start_line} is beyond file length ({total_lines} lines)")
    if end_line > total_lines:
        end_line = total_lines

    # Validate spaces_per_indent
    if spaces_per_indent < 1:
        return ToolResult(success=False, content=f"Error: spaces_per_indent must be at least 1, got {spaces_per_indent}")

    # Adjust indent for each line in the range
    modified_lines = list(original_lines)
    for i in range(start_line - 1, end_line):
        modified_lines[i] = _adjust_line_indent(original_lines[i], indent_delta, spaces_per_indent)

    modified_content = ''.join(modified_lines)

    # Write modified content
    try:
        path.write_text(modified_content, encoding='utf-8', newline='')
    except Exception as e:
        return ToolResult(success=False, content=f"Error writing file: {e}")

    # Store compressed after content
    compressed_after = compress(modified_content)

    # Create operation record
    op = FileOperation(
        operation_type="adjust_indent",
        timestamp=datetime.now(timezone.utc).isoformat(),
        file_path=file_path,
        compressed_before=compressed_before,
        compressed_after=compressed_after
    )

    # Build state update
    state_update = state_service.build_new_file_operation(op)

    # Adjust file view windows for the indent change if experiment is active
    file_view_update = {}
    if experiments_service and experiments_service.is_enabled("file_view_windows") and state_service:
        file_view_update = state_service.build_file_view_windows_edit_adjustment(
            file_path=file_path,
            start_line=start_line,
            old_line_count=end_line - start_line + 1,
            new_line_count=end_line - start_line + 1,
            source="edit"
        )

    direction = "increased" if indent_delta > 0 else "decreased"
    abs_delta = abs(indent_delta)
    level_word = "level" if abs_delta == 1 else "levels"

    return ToolResult(
        success=True,
        content=f"Indent {direction} by {abs_delta} {level_word} for lines {start_line}-{end_line} in '{file_path}'. The requested operation is complete.",
        state_updates={**state_update, **file_view_update}
    )


def adjust_indent_tool_validated(file_path: str, start_line: int, start_line_content: str,
                                  end_line: int, end_line_content: str,
                                  indent_delta: int, spaces_per_indent: int = 4,
                                  **kwargs) -> ToolResult:
    """Adjust indent with content validation - STATE-MODIFYING tool.

    Validates that the provided start_line_content and end_line_content match the actual
    source file content (whitespace-stripped comparison). If mismatch, checks +/-1 lines
    and auto-adjusts if exactly one neighbor matches.

    Changes the indent level of lines from start_line to end_line (1-indexed, inclusive)
    by indent_delta levels. Negative values decrease indent, positive values increase it.
    
    Tab characters count as 1 indent level each. Spaces are grouped by spaces_per_indent
    to form indent levels, with any remainder rounded up. Mixed tabs and spaces are
    handled on a per-line basis.

    Args:
        file_path: Source file path
        start_line: First line to adjust (1-indexed, inclusive)
        start_line_content: Expected content of the start line in source (for validation)
        end_line: Last line to adjust (1-indexed, inclusive)
        end_line_content: Expected content of the end line in source (for validation)
        indent_delta: Number of indent levels to change (negative = decrease, positive = increase)
        spaces_per_indent: Number of spaces per indent level (default: 4)

    Returns:
        ToolResult - with success status, content message, and state_updates dict
    """
    filesystem_service = kwargs.get('filesystem_service')
    state_service = kwargs.get('state_service')
    experiments_service = kwargs.get('experiments_service')
    

    # Validate path
    path, error = filesystem_service.validate_path(file_path)
    if error:
        return ToolResult(success=False, content=error)

    # Check file exists
    if not path.exists() or not path.is_file():
        return ToolResult(success=False, content=f"Error: File '{file_path}' not found")

    size_error = check_line_mode_file_size(path, file_path)
    if size_error:
        return ToolResult(success=False, content=size_error)
    # Read original content
    try:
        original_content = path.read_text(encoding='utf-8')
    except Exception as e:
        return ToolResult(success=False, content=f"Error reading file: {e}")

    # Store compressed before content
    compressed_before = compress(original_content)

    # Split into lines
    original_lines = original_content.splitlines(keepends=False)
    total_lines = len(original_lines)

    # Apply multiline fix experiment: some LLMs provide multiple lines for line content
    # parameters instead of just one line. Only apply if the original content doesn't
    # match, since the content may correctly be a single line. Use split('\n') instead
    # of splitlines() to avoid splitting on \r, \v, \f, etc.
    if experiments_service and experiments_service.is_enabled('line_content_multiline_fix'):
        start_content_lines = start_line_content.split('\n', 1)
        if len(start_content_lines) > 1:
            start_idx = start_line - 1
            if not lines_match(start_idx, start_line_content, original_lines):
                start_line_content = start_content_lines[0]
        
        end_content_lines = end_line_content.rsplit('\n', 1)
        if len(end_content_lines) > 1:
            end_idx = end_line - 1
            if not lines_match(end_idx, end_line_content, original_lines):
                end_line_content = end_content_lines[-1]

    # ================================================================
    # Validate source start_line and end_line
    # ================================================================
    alignment_errors = []
    start_adjustments = []
    end_adjustments = []

    original_start_line = start_line
    original_end_line = end_line

    # Validate start_line
    if start_line < 1:
        start_line = 1
    if start_line > total_lines:
        return ToolResult(success=False, content=f"Error: start_line {start_line} is beyond file length ({total_lines} lines)")

    start_idx = start_line - 1
    if not lines_match(start_idx, start_line_content, original_lines):
        match_above = lines_match(start_idx - 1, start_line_content, original_lines)
        match_below = lines_match(start_idx + 1, start_line_content, original_lines)
        if match_above and not match_below:
            start_line = start_line - 1
            start_adjustments.append(f"start_line adjusted from {original_start_line} to {start_line}")
        elif match_below and not match_above:
            start_line = start_line + 1
            start_adjustments.append(f"start_line adjusted from {original_start_line} to {start_line}")
        else:
            actual = original_lines[start_idx] if 0 <= start_idx < total_lines else "<out of range>"
            alignment_errors.append(
                f"You provided start_line_content '{start_line_content}' but the content at line {original_start_line} "
                f"in the file is currently '{actual}'. Remember: you must provide the exact current content "
                f"of the line and only the one line, not what you intend it to be after adjusting indent.\n"
            )

    # Validate end_line
    if end_line < start_line:
        end_line = start_line
    if end_line > total_lines:
        end_line = total_lines

    end_idx = end_line - 1
    if not lines_match(end_idx, end_line_content, original_lines):
        match_above = lines_match(end_idx - 1, end_line_content, original_lines)
        match_below = lines_match(end_idx + 1, end_line_content, original_lines)
        if match_above and not match_below:
            end_line = end_line - 1
            end_adjustments.append(f"end_line adjusted from {original_end_line} to {end_line}")
        elif match_below and not match_above:
            end_line = end_line + 1
            end_adjustments.append(f"end_line adjusted from {original_end_line} to {end_line}")
        else:
            actual = original_lines[end_idx] if 0 <= end_idx < total_lines else "<out of range>"
            alignment_errors.append(
                f"You provided end_line_content '{end_line_content}' but the content at line {original_end_line} "
                f"in the file is currently '{actual}'. Remember: you must provide the exact current content "
                f"of the line and only the one line, not what you intend it to be after adjusting indent.\n"
            )

    # Re-validate adjusted line numbers
    if start_line < 1:
        start_line = 1
    if end_line < start_line:
        end_line = start_line
    if start_line > total_lines:
        return ToolResult(success=False, content=f"Error: start_line {start_line} is beyond file length ({total_lines} lines)")
    if end_line > total_lines:
        end_line = total_lines

    # Check for omission placeholders in validation args
    if alignment_errors:
        _new_errors = []
        for err in alignment_errors:
            _replaced = False
            for _arg_name, _arg_value in [('start_line_content', start_line_content), ('end_line_content', end_line_content)]:
                if _arg_name in err:
                    _ph_err = _placeholder_error(_arg_name, _arg_value)
                    if _ph_err:
                        _new_errors.append(_ph_err)
                        _replaced = True
                        break
            if not _replaced:
                _new_errors.append(err)
        alignment_errors = _new_errors

    # If any alignment errors, return failure with file views
    if alignment_errors:
        file_view_update = {}
        if experiments_service and experiments_service.is_enabled("file_view_windows") and state_service:
            from fluximp_agent.state_service import FileViewRequest
            view_start = max(1, min(original_start_line, original_end_line) - 5)
            view_end = min(total_lines, max(original_start_line, original_end_line) + 5)
            view_limit = view_end - view_start + 1
            file_view_update = state_service.build_file_view_windows_update(
                [FileViewRequest(file_path=file_path, offset=view_start, limit=view_limit)],
                source="edit"
            )
        return ToolResult(success=False, content=''.join(alignment_errors), state_updates=file_view_update)

    # Validate spaces_per_indent
    if spaces_per_indent < 1:
        return ToolResult(success=False, content=f"Error: spaces_per_indent must be at least 1, got {spaces_per_indent}")

    # ================================================================
    # Perform the indent adjustment
    # ================================================================
    # We need to work with lines that preserve their line endings
    original_lines_with_ends = original_content.splitlines(keepends=True)
    modified_lines = list(original_lines_with_ends)
    
    for i in range(start_line - 1, end_line):
        modified_lines[i] = _adjust_line_indent(original_lines_with_ends[i], indent_delta, spaces_per_indent)

    modified_content = ''.join(modified_lines)

    # Write modified content
    try:
        path.write_text(modified_content, encoding='utf-8', newline='')
    except Exception as e:
        return ToolResult(success=False, content=f"Error writing file: {e}")

    # Store compressed after content
    compressed_after = compress(modified_content)

    # Create operation record
    op = FileOperation(
        operation_type="adjust_indent",
        timestamp=datetime.now(timezone.utc).isoformat(),
        file_path=file_path,
        compressed_before=compressed_before,
        compressed_after=compressed_after
    )

    # Build state update
    state_update = state_service.build_new_file_operation(op)

    # Adjust file view windows for the indent change if experiment is active
    file_view_update = {}
    if experiments_service and experiments_service.is_enabled("file_view_windows") and state_service:
        file_view_update = state_service.build_file_view_windows_edit_adjustment(
            file_path=file_path,
            start_line=start_line,
            old_line_count=end_line - start_line + 1,
            new_line_count=end_line - start_line + 1,
            source="edit"
        )

    direction = "increased" if indent_delta > 0 else "decreased"
    abs_delta = abs(indent_delta)
    level_word = "level" if abs_delta == 1 else "levels"

    # Build success message with adjustment notes
    base_message = f"Indent {direction} by {abs_delta} {level_word} for lines {start_line}-{end_line} in '{file_path}'. The requested operation is complete."
    all_adjustments = start_adjustments + end_adjustments
    if all_adjustments:
        adjustment_note = " Note: " + "; ".join(all_adjustments)
        success_message = base_message + adjustment_note
    else:
        success_message = base_message

    return ToolResult(
        success=True,
        content=success_message,
        state_updates={**state_update, **file_view_update}
    )


def _set_exact_whitespace(line: str, count: int, whitespace_type: str) -> str:
    """Set the leading whitespace of a line to exactly count characters of whitespace_type.
    
    Blank lines (empty or whitespace-only) are left unchanged.
    
    Args:
        line: The line to modify
        count: Number of whitespace characters to prefix
        whitespace_type: Either 'space' or 'tab'
        
    Returns:
        The line with exact leading whitespace
    """
    if not line or line.isspace():
        return line
    
    # Find where non-whitespace content starts
    content_start = 0
    for i, ch in enumerate(line):
        if ch not in ' \t':
            content_start = i
            break
    else:
        # Line is all whitespace
        return line
    
    content = line[content_start:]
    char = '\t' if whitespace_type == 'tab' else ' '
    return char * count + content


def adjust_indent_exact_tool(file_path: str, start_line: int, end_line: int,
                              count_leading_whitespace: int, whitespace_type: str = "space",
                              **kwargs) -> ToolResult:
    """Set exact leading whitespace on a block of code - STATE-MODIFYING tool.

    Changes the leading whitespace of lines from start_line to end_line (1-indexed, inclusive)
    to have exactly count_leading_whitespace characters of whitespace_type.
    Blank lines are left unchanged.

    Stores compressed before/after content for reliable undo/redo.

    When the file_view_windows experiment is active, this tool also adds the
    affected region (+/- 5 lines) to the file view windows.

    Args:
        file_path: Source file path
        start_line: First line to adjust (1-indexed, inclusive)
        end_line: Last line to adjust (1-indexed, inclusive)
        count_leading_whitespace: Exact number of whitespace characters to set at the start of each line
        whitespace_type: Type of whitespace to use - 'space' or 'tab' (default: 'space')

    Returns:
        ToolResult - with success status, content message, and state_updates dict
    """
    filesystem_service = kwargs.get('filesystem_service')
    state_service = kwargs.get('state_service')
    experiments_service = kwargs.get('experiments_service')

    # Validate path
    path, error = filesystem_service.validate_path(file_path)
    if error:
        return ToolResult(success=False, content=error)

    # Check file exists
    if not path.exists() or not path.is_file():
        return ToolResult(success=False, content=f"Error: File '{file_path}' not found")

    size_error = check_line_mode_file_size(path, file_path)
    if size_error:
        return ToolResult(success=False, content=size_error)
    # Read original content
    try:
        original_content = path.read_text(encoding='utf-8')
    except Exception as e:
        return ToolResult(success=False, content=f"Error reading file: {e}")

    # Store compressed before content
    compressed_before = compress(original_content)

    # Split into lines
    original_lines = original_content.splitlines(keepends=True)
    total_lines = len(original_lines)

    # Validate line numbers
    if start_line < 1:
        start_line = 1
    if end_line < start_line:
        end_line = start_line
    if start_line > total_lines:
        return ToolResult(success=False, content=f"Error: start_line {start_line} is beyond file length ({total_lines} lines)")
    if end_line > total_lines:
        end_line = total_lines

    # Validate count_leading_whitespace
    if count_leading_whitespace < 0:
        return ToolResult(success=False, content=f"Error: count_leading_whitespace must be non-negative, got {count_leading_whitespace}")

    # Validate whitespace_type
    if whitespace_type not in ('space', 'tab'):
        return ToolResult(success=False, content=f"Error: whitespace_type must be 'space' or 'tab', got '{whitespace_type}'")

    # Set exact whitespace for each line in the range
    modified_lines = list(original_lines)
    for i in range(start_line - 1, end_line):
        modified_lines[i] = _set_exact_whitespace(original_lines[i], count_leading_whitespace, whitespace_type)

    modified_content = ''.join(modified_lines)

    # Write modified content
    try:
        path.write_text(modified_content, encoding='utf-8', newline='')
    except Exception as e:
        return ToolResult(success=False, content=f"Error writing file: {e}")

    # Store compressed after content
    compressed_after = compress(modified_content)

    # Create operation record
    op = FileOperation(
        operation_type="adjust_indent_exact",
        timestamp=datetime.now(timezone.utc).isoformat(),
        file_path=file_path,
        compressed_before=compressed_before,
        compressed_after=compressed_after
    )

    # Build state update
    state_update = state_service.build_new_file_operation(op)

    # Adjust file view windows for the indent change if experiment is active
    file_view_update = {}
    if experiments_service and experiments_service.is_enabled("file_view_windows") and state_service:
        file_view_update = state_service.build_file_view_windows_edit_adjustment(
            file_path=file_path,
            start_line=start_line,
            old_line_count=end_line - start_line + 1,
            new_line_count=end_line - start_line + 1,
            source="edit"
        )

    char_word = "tab" if whitespace_type == 'tab' else "space"
    char_plural = "tabs" if whitespace_type == 'tab' else "spaces"
    count_word = char_word if count_leading_whitespace == 1 else char_plural

    return ToolResult(
        success=True,
        content=f"Set leading whitespace to {count_leading_whitespace} {count_word} for lines {start_line}-{end_line} in '{file_path}'. The requested operation is complete.",
        state_updates={**state_update, **file_view_update}
    )


def adjust_indent_exact_tool_validated(file_path: str, start_line: int, start_line_content: str,
                                        end_line: int, end_line_content: str,
                                        count_leading_whitespace: int, whitespace_type: str = "space",
                                        **kwargs) -> ToolResult:
    """Set exact leading whitespace with content validation - STATE-MODIFYING tool.

    Validates that the provided start_line_content and end_line_content match the actual
    source file content (whitespace-stripped comparison). If mismatch, checks +/-1 lines
    and auto-adjusts if exactly one neighbor matches.

    Changes the leading whitespace of lines from start_line to end_line (1-indexed, inclusive)
    to have exactly count_leading_whitespace characters of whitespace_type.
    Blank lines are left unchanged.

    Args:
        file_path: Source file path
        start_line: First line to adjust (1-indexed, inclusive)
        start_line_content: Expected content of the start line in source (for validation)
        end_line: Last line to adjust (1-indexed, inclusive)
        end_line_content: Expected content of the end line in source (for validation)
        count_leading_whitespace: Exact number of whitespace characters to set at the start of each line
        whitespace_type: Type of whitespace to use - 'space' or 'tab' (default: 'space')

    Returns:
        ToolResult - with success status, content message, and state_updates dict
    """
    filesystem_service = kwargs.get('filesystem_service')
    state_service = kwargs.get('state_service')
    experiments_service = kwargs.get('experiments_service')
    

    # Validate path
    path, error = filesystem_service.validate_path(file_path)
    if error:
        return ToolResult(success=False, content=error)

    # Check file exists
    if not path.exists() or not path.is_file():
        return ToolResult(success=False, content=f"Error: File '{file_path}' not found")

    size_error = check_line_mode_file_size(path, file_path)
    if size_error:
        return ToolResult(success=False, content=size_error)
    # Read original content
    try:
        original_content = path.read_text(encoding='utf-8')
    except Exception as e:
        return ToolResult(success=False, content=f"Error reading file: {e}")

    # Store compressed before content
    compressed_before = compress(original_content)

    # Split into lines
    original_lines = original_content.splitlines(keepends=False)
    total_lines = len(original_lines)

    # Apply multiline fix experiment: some LLMs provide multiple lines for line content
    # parameters instead of just one line. Only apply if the original content doesn't
    # match, since the content may correctly be a single line. Use split('\n') instead
    # of splitlines() to avoid splitting on \r, \v, \f, etc.
    if experiments_service and experiments_service.is_enabled('line_content_multiline_fix'):
        start_content_lines = start_line_content.split('\n', 1)
        if len(start_content_lines) > 1:
            start_idx = start_line - 1
            if not lines_match(start_idx, start_line_content, original_lines):
                start_line_content = start_content_lines[0]
        
        end_content_lines = end_line_content.rsplit('\n', 1)
        if len(end_content_lines) > 1:
            end_idx = end_line - 1
            if not lines_match(end_idx, end_line_content, original_lines):
                end_line_content = end_content_lines[-1]

    # ================================================================
    # Validate source start_line and end_line
    # ================================================================
    alignment_errors = []
    start_adjustments = []
    end_adjustments = []

    original_start_line = start_line
    original_end_line = end_line

    # Validate start_line
    if start_line < 1:
        start_line = 1
    if start_line > total_lines:
        return ToolResult(success=False, content=f"Error: start_line {start_line} is beyond file length ({total_lines} lines)")

    start_idx = start_line - 1
    if not lines_match(start_idx, start_line_content, original_lines):
        match_above = lines_match(start_idx - 1, start_line_content, original_lines)
        match_below = lines_match(start_idx + 1, start_line_content, original_lines)
        if match_above and not match_below:
            start_line = start_line - 1
            start_adjustments.append(f"start_line adjusted from {original_start_line} to {start_line}")
        elif match_below and not match_above:
            start_line = start_line + 1
            start_adjustments.append(f"start_line adjusted from {original_start_line} to {start_line}")
        else:
            actual = original_lines[start_idx] if 0 <= start_idx < total_lines else "<out of range>"
            alignment_errors.append(
                f"You provided start_line_content '{start_line_content}' but the content at line {original_start_line} "
                f"in the file is currently '{actual}'. Remember: you must provide the exact current content "
                f"of the line and only the one line, not what you intend it to be after adjusting indent.\n"
            )

    # Validate end_line
    if end_line < start_line:
        end_line = start_line
    if end_line > total_lines:
        end_line = total_lines

    end_idx = end_line - 1
    if not lines_match(end_idx, end_line_content, original_lines):
        match_above = lines_match(end_idx - 1, end_line_content, original_lines)
        match_below = lines_match(end_idx + 1, end_line_content, original_lines)
        if match_above and not match_below:
            end_line = end_line - 1
            end_adjustments.append(f"end_line adjusted from {original_end_line} to {end_line}")
        elif match_below and not match_above:
            end_line = end_line + 1
            end_adjustments.append(f"end_line adjusted from {original_end_line} to {end_line}")
        else:
            actual = original_lines[end_idx] if 0 <= end_idx < total_lines else "<out of range>"
            alignment_errors.append(
                f"You provided end_line_content '{end_line_content}' but the content at line {original_end_line} "
                f"in the file is currently '{actual}'. Remember: you must provide the exact current content "
                f"of the line and only the one line, not what you intend it to be after adjusting indent.\n"
            )

    # Re-validate adjusted line numbers
    if start_line < 1:
        start_line = 1
    if end_line < start_line:
        end_line = start_line
    if start_line > total_lines:
        return ToolResult(success=False, content=f"Error: start_line {start_line} is beyond file length ({total_lines} lines)")
    if end_line > total_lines:
        end_line = total_lines

    # If any alignment errors, return failure with file views
    # Check for omission placeholders in validation args
    if alignment_errors:
        _new_errors = []
        for err in alignment_errors:
            _replaced = False
            for _arg_name, _arg_value in [('start_line_content', start_line_content), ('end_line_content', end_line_content)]:
                if _arg_name in err:
                    _ph_err = _placeholder_error(_arg_name, _arg_value)
                    if _ph_err:
                        _new_errors.append(_ph_err)
                        _replaced = True
                        break
            if not _replaced:
                _new_errors.append(err)
        alignment_errors = _new_errors

    if alignment_errors:
        file_view_update = {}
        if experiments_service and experiments_service.is_enabled("file_view_windows") and state_service:
            from fluximp_agent.state_service import FileViewRequest
            view_start = max(1, min(original_start_line, original_end_line) - 5)
            view_end = min(total_lines, max(original_start_line, original_end_line) + 5)
            view_limit = view_end - view_start + 1
            file_view_update = state_service.build_file_view_windows_update(
                [FileViewRequest(file_path=file_path, offset=view_start, limit=view_limit)],
                source="edit"
            )
        return ToolResult(success=False, content=''.join(alignment_errors), state_updates=file_view_update)

    # Validate count_leading_whitespace
    if count_leading_whitespace < 0:
        return ToolResult(success=False, content=f"Error: count_leading_whitespace must be non-negative, got {count_leading_whitespace}")

    # Validate whitespace_type
    if whitespace_type not in ('space', 'tab'):
        return ToolResult(success=False, content=f"Error: whitespace_type must be 'space' or 'tab', got '{whitespace_type}'")

    # ================================================================
    # Perform the whitespace adjustment
    # ================================================================
    # We need to work with lines that preserve their line endings
    original_lines_with_ends = original_content.splitlines(keepends=True)
    modified_lines = list(original_lines_with_ends)
    
    for i in range(start_line - 1, end_line):
        modified_lines[i] = _set_exact_whitespace(original_lines_with_ends[i], count_leading_whitespace, whitespace_type)

    modified_content = ''.join(modified_lines)

    # Write modified content
    try:
        path.write_text(modified_content, encoding='utf-8', newline='')
    except Exception as e:
        return ToolResult(success=False, content=f"Error writing file: {e}")

    # Store compressed after content
    compressed_after = compress(modified_content)

    # Create operation record
    op = FileOperation(
        operation_type="adjust_indent_exact",
        timestamp=datetime.now(timezone.utc).isoformat(),
        file_path=file_path,
        compressed_before=compressed_before,
        compressed_after=compressed_after
    )

    # Build state update
    state_update = state_service.build_new_file_operation(op)

    # Adjust file view windows for the indent change if experiment is active
    file_view_update = {}
    if experiments_service and experiments_service.is_enabled("file_view_windows") and state_service:
        file_view_update = state_service.build_file_view_windows_edit_adjustment(
            file_path=file_path,
            start_line=start_line,
            old_line_count=end_line - start_line + 1,
            new_line_count=end_line - start_line + 1,
            source="edit"
        )

    char_word = "tab" if whitespace_type == 'tab' else "space"
    char_plural = "tabs" if whitespace_type == 'tab' else "spaces"
    count_word = char_word if count_leading_whitespace == 1 else char_plural

    # Build success message with adjustment notes
    base_message = f"Set leading whitespace to {count_leading_whitespace} {count_word} for lines {start_line}-{end_line} in '{file_path}'. The requested operation is complete."
    all_adjustments = start_adjustments + end_adjustments
    if all_adjustments:
        adjustment_note = " Note: " + "; ".join(all_adjustments)
        success_message = base_message + adjustment_note
    else:
        success_message = base_message

    return ToolResult(
        success=True,
        content=success_message,
        state_updates={**state_update, **file_view_update}
    )
