from datetime import datetime, timezone

from fluximp_agent.filesystem_service import check_line_mode_file_size
from fluximp_agent.tools.tool_definitions import ToolResult
from fluximp_agent.tools.common import compress, FileOperation, lines_match
from fluximp_agent.tool_call_omission import OMISSION_PLACEHOLDER_RE
def _placeholder_error(arg_name, value):
    """Return specific error if validation arg matches omission placeholder pattern."""
    if OMISSION_PLACEHOLDER_RE.search(value):
        return f"The validation line you provided for '{arg_name}' matches an omission placeholder pattern. You may have copied a placeholder from your conversation history instead of the actual file content.\n"
    return None


def edit_file_lines_tool(file_path: str, start_line: int, end_line: int, new_content: str,
                          **kwargs) -> ToolResult:
    """Edit specific lines in a file - STATE-MODIFYING tool.

    Replaces lines from start_line to end_line (1-indexed, inclusive) with new_content.
    Stores compressed before/after content for reliable undo/redo.

    When the file_view_windows experiment is active, this tool also adds the
    edited region (+/- 5 lines) to the file view windows.

    Args:
        file_path: Target file
        start_line: First line to replace (1-indexed)
        end_line: Last line to replace (1-indexed, inclusive)
        new_content: Replacement text

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

    # Ensure end_line doesn't exceed file length
    if end_line > total_lines:
        end_line = total_lines

    # Convert to 0-indexed
    start_idx = start_line - 1
    end_idx = end_line  # exclusive

    # Split new content into lines
    new_lines = new_content.splitlines(keepends=False)
    # Add newlines if not present
    new_lines = [line + '\n' for line in new_lines]

    # Build modified lines
    modified_lines = original_lines[:start_idx] + new_lines + original_lines[end_idx:]
    modified_content = ''.join(modified_lines)

    # Write modified content
    try:
        path.write_text(modified_content, encoding='utf-8', newline='')
        #print(f"DEBUG modified_content: {modified_content}")
    except Exception as e:
        return ToolResult(success=False, content=f"Error writing file: {e}")

    # Store compressed after content
    compressed_after = compress(modified_content)

    # Create operation record
    op = FileOperation(
        operation_type="edit_file_lines",
        timestamp=datetime.now(timezone.utc).isoformat(),
        file_path=file_path,
        compressed_before=compressed_before,
        compressed_after=compressed_after
    )

    # Build state update
    state_update = state_service.build_new_file_operation(op)

    # Adjust file view windows for the edit if experiment is active
    file_view_update = {}
    if experiments_service and experiments_service.is_enabled("file_view_windows") and state_service:
        file_view_update = state_service.build_file_view_windows_edit_adjustment(
            file_path=file_path,
            start_line=start_line,
            old_line_count=end_line - start_line + 1,
            new_line_count=len(new_lines),
            source="edit"
        )

    return ToolResult(
        success=True,
        content=f"Lines {start_line}-{end_line} in '{file_path}' have been edited successfully. The requested operation is complete.",
        state_updates={**state_update, **file_view_update}
    )


def edit_file_lines_tool_validated(file_path: str, start_line: int, start_line_content: str,
                                   end_line: int, end_line_content: str, new_content: str,
                                   **kwargs) -> ToolResult:
    """Edit specific lines with content validation - STATE-MODIFYING tool.

    Validates that the provided start_line_content and end_line_content match
    the actual file content (whitespace-stripped comparison). If mismatch,
    checks Â±1 lines and auto-adjusts line numbers if exactly one neighbor matches.

    Args:
        file_path: Target file
        start_line: First line to replace (1-indexed)
        start_line_content: Expected content of start line (for validation)
        end_line: Last line to replace (1-indexed, inclusive)
        end_line_content: Expected content of end line (for validation)
        new_content: Replacement text

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

    # Split into lines
    original_lines = original_content.splitlines(keepends=False)
    total_lines = len(original_lines)

    # Apply multiline fix experiment: some LLMs provide multiple lines for line content
    # parameters instead of just one line. Only apply if the original content doesn't
    # match, since the content may correctly be a single line. Use split('\n') instead
    # of splitlines() to avoid splitting on \r, \v, \f, etc.
    if experiments_service and experiments_service.is_enabled('line_content_multiline_fix'):
        start_line_content_lines = start_line_content.split('\n', 1)
        if len(start_line_content_lines) > 1:
            start_idx = start_line - 1
            if not lines_match(start_idx, start_line_content, original_lines):
                start_line_content = start_line_content_lines[0]
        
        end_line_content_lines = end_line_content.rsplit('\n', 1)
        if len(end_line_content_lines) > 1:
            end_idx = end_line - 1
            if not lines_match(end_idx, end_line_content, original_lines):
                end_line_content = end_line_content_lines[-1]

    # Validate and potentially adjust start_line
    start_adjustments = []
    original_start_line = start_line
    start_idx = start_line - 1  # Convert to 0-indexed

    alignment_errors = []

    if not lines_match(start_idx, start_line_content, original_lines):
        # Check ±1 lines
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
            alignment_errors.append(f"You provided '{start_line_content}' but the content at line {original_start_line} in the file is currently '{actual}'. Remember: you must provide the exact current content of the line and only the one line, not what you intend it to be after editing.\n")

    # Validate and potentially adjust end_line
    end_adjustments = []
    original_end_line = end_line
    end_idx = end_line - 1  # Convert to 0-indexed

    if not lines_match(end_idx, end_line_content, original_lines):
        # Check ±1 lines
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
            alignment_errors.append(f"You provided '{end_line_content}' but the content at line {original_end_line} in the file is currently '{actual}'. Remember: you must provide the exact current content of the line and only the one line, not what you intend it to be after editing.\n")

    # Check for omission placeholders in validation args
    if alignment_errors:
        for _arg_name, _arg_value in [('start_line_content', start_line_content), ('end_line_content', end_line_content)]:
            _ph_err = _placeholder_error(_arg_name, _arg_value)
            if _ph_err:
                alignment_errors = [_ph_err]
                break
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

    # Validate adjusted line numbers
    if start_line < 1:
        start_line = 1
    if end_line < start_line:
        end_line = start_line
    if start_line > total_lines:
        return ToolResult(success=False, content=f"Error: start_line {start_line} is beyond file length ({total_lines} lines)")
    if end_line > total_lines:
        end_line = total_lines

    # Convert to 0-indexed for editing
    start_idx = start_line - 1
    end_idx = end_line  # exclusive

    # Split new content into lines
    new_lines = new_content.splitlines(keepends=False)
    # Add newlines if not present
    new_lines = [line + '\n' for line in new_lines]

    # Build modified lines
    original_lines_with_newlines = [line + '\n' for line in original_lines]
    modified_lines = original_lines_with_newlines[:start_idx] + new_lines + original_lines_with_newlines[end_idx:]
    modified_content = ''.join(modified_lines)

    # Write modified content
    try:
        path.write_text(modified_content, encoding='utf-8', newline='')
        #print(f"DEBUG modified_content: {modified_content}")
    except Exception as e:
        return ToolResult(success=False, content=f"Error writing file: {e}")

    # Store compressed before/after content
    compressed_before = compress(original_content)
    compressed_after = compress(modified_content)

    # Create operation record
    op = FileOperation(
        operation_type="edit_file_lines",
        timestamp=datetime.now(timezone.utc).isoformat(),
        file_path=file_path,
        compressed_before=compressed_before,
        compressed_after=compressed_after
    )

    # Build state update
    state_update = state_service.build_new_file_operation(op)

    # Adjust file view windows for the edit if experiment is active
    file_view_update = {}
    if experiments_service and experiments_service.is_enabled("file_view_windows") and state_service:
        file_view_update = state_service.build_file_view_windows_edit_adjustment(
            file_path=file_path,
            start_line=start_line,
            old_line_count=end_line - start_line + 1,
            new_line_count=len(new_lines),
            source="edit"
        )

    # Build success message with adjustment notes
    base_message = f"Lines {start_line}-{end_line} in '{file_path}' have been edited successfully. The requested operation is complete."
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


def _set_exact_whitespace_first_line(content: str, count: int, whitespace_type: str) -> str:
    """Set the leading whitespace of the first line of content to exactly count characters of whitespace_type.
    
    Args:
        content: The multi-line content to modify
        count: Number of whitespace characters to prefix on the first line
        whitespace_type: Either 'space' or 'tab'
        
    Returns:
        The content with exact leading whitespace on the first line
    """
    lines = content.splitlines(keepends=False)
    if not lines:
        return content
    
    first_line = lines[0]
    # Find where non-whitespace content starts in the first line
    content_start = 0
    for i, ch in enumerate(first_line):
        if ch not in ' \t':
            content_start = i
            break
    else:
        # First line is all whitespace - replace it entirely
        content_start = len(first_line)
    
    actual_content = first_line[content_start:]
    char = '\t' if whitespace_type == 'tab' else ' '
    lines[0] = char * count + actual_content
    
    return '\n'.join(lines)


def edit_file_lines_tool_validated_first_line_whitespace(file_path: str, start_line: int, start_line_content: str,
                                                         end_line: int, end_line_content: str, new_content: str,
                                                         count_leading_whitespace_first_line: int,
                                                         whitespace_type_first_line: str = "space",
                                                         **kwargs) -> ToolResult:
    """Edit specific lines with content validation and first-line whitespace control - STATE-MODIFYING tool.

    Validates that the provided start_line_content and end_line_content match
    the actual file content (whitespace-stripped comparison). If mismatch,
    checks Â±1 lines and auto-adjusts line numbers if exactly one neighbor matches.
    
    Applies exact whitespace control to the first line of new_content to work around
    infrastructure issues that may strip leading whitespace.

    Args:
        file_path: Target file
        start_line: First line to replace (1-indexed)
        start_line_content: Expected content of start line (for validation)
        end_line: Last line to replace (1-indexed, inclusive)
        end_line_content: Expected content of end line (for validation)
        new_content: Replacement text
        count_leading_whitespace_first_line: Exact number of whitespace chars for first line
        whitespace_type_first_line: Type of whitespace - 'space' or 'tab' (default: 'space')

    Returns:
        ToolResult - with success status, content message, and state_updates dict
    """
    # Apply whitespace control to the first line of new_content
    new_content = _set_exact_whitespace_first_line(
        new_content, 
        count_leading_whitespace_first_line, 
        whitespace_type_first_line
    )
    
    # Delegate to the existing validated function
    return edit_file_lines_tool_validated(
        file_path=file_path,
        start_line=start_line,
        start_line_content=start_line_content,
        end_line=end_line,
        end_line_content=end_line_content,
        new_content=new_content,
        **kwargs
    )
