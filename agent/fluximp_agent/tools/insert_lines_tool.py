#!/usr/bin/env python3
"""
Insert lines tool functions for file operations.

This module contains the insert_lines tool functions used by the LangGraph-based coding agent.
These functions implement line insertion capabilities with optional content validation.
"""

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


def insert_lines_tool(file_path: str, insert_line: int, new_content: str,
                      **kwargs) -> ToolResult:
    """Insert lines into a file at the specified position - STATE-MODIFYING tool.

    Inserts new_content BEFORE the specified insert_line (1-indexed).
    Use insert_line=1 to insert at the beginning.
    Use insert_line=total_lines+1 to append at the end.
    
    Stores compressed before/after content for reliable undo/redo.

    When the file_view_windows experiment is active, this tool also adds the
    inserted region (+/- 5 lines) to the file view windows.

    Args:
        file_path: Target file
        insert_line: Line number BEFORE which to insert (1-indexed)
        new_content: Text to insert (will be split into lines)

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
    if insert_line < 1:
        insert_line = 1
    if insert_line > total_lines + 2:
        return ToolResult(success=False, content=f"Error: insert_line {insert_line} is beyond file length + 2 ({total_lines + 2} max)")
    if insert_line > total_lines + 1:
        insert_line = total_lines + 1

    # Convert to 0-indexed insert position
    insert_idx = insert_line - 1

    # Split new content into lines
    new_lines = new_content.splitlines(keepends=False)
    # Add newlines if not present
    new_lines = [line + '\n' for line in new_lines]

    # Build modified lines
    modified_lines = original_lines[:insert_idx] + new_lines + original_lines[insert_idx:]
    modified_content = ''.join(modified_lines)

    # Write modified content
    try:
        path.write_text(modified_content, encoding='utf-8', newline='')
        #print(f"DEBUG modified_content: {modified_content}")
    except Exception as e:
        return ToolResult(success=False, content=f"Error writing file: {e}")

    # Store compressed after content
    compressed_after = compress(modified_content)

    # Calculate the line range of inserted content
    inserted_start_line = insert_line
    inserted_end_line = insert_line + len(new_lines) - 1

    # Create operation record
    op = FileOperation(
        operation_type="insert_lines",
        timestamp=datetime.now(timezone.utc).isoformat(),
        file_path=file_path,
        compressed_before=compressed_before,
        compressed_after=compressed_after
    )

    # Build state update
    state_update = state_service.build_new_file_operation(op)

    # Adjust file view windows for the insertion if experiment is active
    file_view_update = {}
    if experiments_service and experiments_service.is_enabled("file_view_windows") and state_service:
        file_view_update = state_service.build_file_view_windows_edit_adjustment(
            file_path=file_path,
            start_line=insert_line,
            old_line_count=0,
            new_line_count=len(new_lines),
            source="edit"
        )

    return ToolResult(
        success=True,
        content=f"{len(new_lines)} line(s) have been inserted into '{file_path}' before line {insert_line}. The requested operation is complete.",
        state_updates={**state_update, **file_view_update}
    )


def insert_lines_tool_validated(file_path: str, insert_line: int, anchor_line_content: str,
                                new_content: str, **kwargs) -> ToolResult:
    """Insert lines with content validation - STATE-MODIFYING tool.

    Validates that the provided anchor_line_content matches the actual file content
    at insert_line (whitespace-stripped comparison). If mismatch, checks Â±1 lines
    and auto-adjusts insert_line if exactly one neighbor matches.

    Inserts new_content BEFORE the (possibly adjusted) insert_line.

    Args:
        file_path: Target file
        insert_line: Line number BEFORE which to insert (1-indexed)
        anchor_line_content: Expected content of the anchor line (for validation)
        new_content: Text to insert (will be split into lines)

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
        anchor_content_lines = anchor_line_content.split('\n', 1)
        if len(anchor_content_lines) > 1:
            anchor_idx = insert_line - 1
            if not lines_match(anchor_idx, anchor_line_content, original_lines):
                anchor_line_content = anchor_content_lines[0]

    # Validate line numbers
    if insert_line < 1:
        insert_line = 1
    if insert_line > total_lines + 2:
        return ToolResult(success=False, content=f"Error: insert_line {insert_line} is beyond file length + 2 ({total_lines + 2} max)")
    if insert_line > total_lines + 1:
        insert_line = total_lines + 1

    # Track adjustments and alignment errors
    adjustments = []
    original_insert_line = insert_line

    # Validate anchor line content (if not appending at end)
    if insert_line <= total_lines:
        anchor_idx = insert_line - 1        
        
        if not lines_match(anchor_idx, anchor_line_content, original_lines):
            # Check neighbors (±1)
            matches = []
            
            # Check line before
            if lines_match(anchor_idx - 1, anchor_line_content, original_lines):
                matches.append(insert_line - 1)
            
            # Check line after
            if lines_match(anchor_idx + 1, anchor_line_content, original_lines):
                matches.append(insert_line + 1)
            
            alignment_error = None
            if len(matches) == 1:
                insert_line = matches[0]
                adjustments.append(f"Adjusted insert_line from {original_insert_line} to {insert_line} to match anchor content")
            elif len(matches) == 0:
                alignment_error = f"You provided '{anchor_line_content}' but the content at line {insert_line} in the file is currently '{original_lines[anchor_idx]}'. Remember: you must provide the exact current content of the line and only the one line, not what you intend it to be after editing."
            else:
                alignment_error = f"You provided '{anchor_line_content}' but the content at line {insert_line} in the file is currently '{original_lines[anchor_idx]}'. The anchor content matches multiple lines ({matches}). Cannot auto-adjust."

            # Check for omission placeholder in anchor_line_content
            if alignment_error:
                _ph_err = _placeholder_error('anchor_line_content', anchor_line_content)
                if _ph_err:
                    alignment_error = _ph_err

            # Handle alignment error with file view auto-add
            if alignment_error:
                file_view_update = {}
                if experiments_service and experiments_service.is_enabled("file_view_windows") and state_service:
                    from fluximp_agent.state_service import FileViewRequest
                    view_start = max(1, insert_line - 5)
                    view_end = min(total_lines, insert_line + 5)
                    view_limit = view_end - view_start + 1
                    file_view_update = state_service.build_file_view_windows_update(
                        [FileViewRequest(file_path=file_path, offset=view_start, limit=view_limit)],
                        source="edit"
                    )
                return ToolResult(success=False, content=alignment_error, state_updates=file_view_update)

    # Convert to 0-indexed insert position
    insert_idx = insert_line - 1

    # Split new content into lines
    new_lines = new_content.splitlines(keepends=False)
    # Add newlines if not present
    new_lines = [line + '\n' for line in new_lines]

    # Build modified lines
    original_lines_with_ends = original_content.splitlines(keepends=True)
    modified_lines = original_lines_with_ends[:insert_idx] + new_lines + original_lines_with_ends[insert_idx:]
    modified_content = ''.join(modified_lines)

    # Write modified content
    try:
        path.write_text(modified_content, encoding='utf-8', newline='')
        #print(f"DEBUG modified_content: {modified_content}")
    except Exception as e:
        return ToolResult(success=False, content=f"Error writing file: {e}")

    # Store compressed after content
    compressed_after = compress(modified_content)

    # Calculate the line range of inserted content
    inserted_start_line = insert_line
    inserted_end_line = insert_line + len(new_lines) - 1

    # Create operation record
    op = FileOperation(
        operation_type="insert_lines",
        timestamp=datetime.now(timezone.utc).isoformat(),
        file_path=file_path,
        compressed_before=compressed_before,
        compressed_after=compressed_after
    )

    # Build state update
    state_update = state_service.build_new_file_operation(op)

    # Adjust file view windows for the insertion if experiment is active
    file_view_update = {}
    if experiments_service and experiments_service.is_enabled("file_view_windows") and state_service:
        file_view_update = state_service.build_file_view_windows_edit_adjustment(
            file_path=file_path,
            start_line=insert_line,
            old_line_count=0,
            new_line_count=len(new_lines),
            source="edit"
        )

    # Build success message with adjustment notes
    base_message = f"{len(new_lines)} line(s) have been inserted into '{file_path}' before line {insert_line}. The requested operation is complete."
    if adjustments:
        adjustment_note = " Note: " + "; ".join(adjustments)
        success_message = base_message + adjustment_note
    else:
        success_message = base_message

    return ToolResult(
        success=True,
        content=success_message,
        state_updates={**state_update, **file_view_update}
    )


def insert_lines_tool_validated_first_line_whitespace(file_path: str, insert_line: int, anchor_line_content: str,
                                                       new_content: str, count_leading_whitespace_first_line: int,
                                                       whitespace_type_first_line: str = "space",
                                                       **kwargs) -> ToolResult:
    """Insert lines with content validation and first-line whitespace control - STATE-MODIFYING tool.

    Validates that the provided anchor_line_content matches the actual file content
    at insert_line (whitespace-stripped comparison). If mismatch, checks ÃÂ±1 lines
    and auto-adjusts insert_line if exactly one neighbor matches.
    
    Applies exact whitespace control to the first line of new_content to work around
    infrastructure issues that may strip leading whitespace.

    Args:
        file_path: Target file
        insert_line: Line number BEFORE which to insert (1-indexed)
        anchor_line_content: Expected content of the anchor line (for validation)
        new_content: Text to insert (will be split into lines)
        count_leading_whitespace_first_line: Exact number of whitespace chars for first line
        whitespace_type_first_line: Type of whitespace - 'space' or 'tab' (default: 'space')

    Returns:
        ToolResult - with success status, content message, and state_updates dict
    """
    # Apply whitespace control to the first line of new_content
    from fluximp_agent.tools.edit_file_lines import _set_exact_whitespace_first_line
    new_content = _set_exact_whitespace_first_line(
        new_content, 
        count_leading_whitespace_first_line, 
        whitespace_type_first_line
    )
    
    # Delegate to the existing validated function
    return insert_lines_tool_validated(
        file_path=file_path,
        insert_line=insert_line,
        anchor_line_content=anchor_line_content,
        new_content=new_content,
        **kwargs
    )
