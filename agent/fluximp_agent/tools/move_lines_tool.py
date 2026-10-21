#!/usr/bin/env python3
"""
Move lines tool functions for file operations.

This module contains the move_lines tool functions used by the LangGraph-based coding agent.
These functions implement line moving capabilities within and between files,
with optional content validation for source and destination.
"""

from datetime import datetime, timezone
from typing import Dict, Any, Optional
import uuid

from fluximp_agent.tools.common import FileOperation, compress, lines_match
from fluximp_agent.filesystem_service import check_line_mode_file_size
from fluximp_agent.tools.tool_definitions import ToolResult
from fluximp_agent.tool_call_omission import OMISSION_PLACEHOLDER_RE

def _placeholder_error(arg_name, value):
    """Return specific error if validation arg matches omission placeholder pattern."""
    if OMISSION_PLACEHOLDER_RE.search(value):
        return f"The validation line you provided for '{arg_name}' matches an omission placeholder pattern. You may have copied a placeholder from your conversation history instead of the actual file content.\n"
    return None


def move_lines_tool(file_path: str, start_line: int, end_line: int,
                    insert_line: int, dest_file_path: str = None,
                    **kwargs) -> ToolResult:
    """Move lines from one location to another - STATE-MODIFYING tool.

    Moves lines from start_line to end_line (1-indexed, inclusive) from the source file
    to BEFORE insert_line in the destination file. If dest_file_path is not provided,
    lines are moved within the same file. If the destination file does not exist, it will
    be created with leading blank lines as needed to place the moved content at the
    requested line.

    For same-file moves, the implementation first copies the lines, then deletes them
    from their original position. This ensures correct line number handling when the
    destination is after the source.

    Stores compressed before/after content for reliable undo/redo.

    When the file_view_windows experiment is active, this tool also adds the
    affected regions (+/- 5 lines) to the file view windows.

    Args:
        file_path: Source file path
        start_line: First line to move (1-indexed, inclusive)
        end_line: Last line to move (1-indexed, inclusive)
        insert_line: Line number BEFORE which to insert in destination (1-indexed)
        dest_file_path: Destination file path (defaults to same as source)

    Returns:
        ToolResult - with success status, content message, and state_updates dict
    """
    filesystem_service = kwargs.get('filesystem_service')
    state_service = kwargs.get('state_service')
    experiments_service = kwargs.get('experiments_service')

    # Default dest_file_path to file_path (same file)
    if dest_file_path is None:
        dest_file_path = file_path

    same_file = (file_path == dest_file_path)

    # Validate source path
    src_path, error = filesystem_service.validate_path(file_path)
    if error:
        return ToolResult(success=False, content=error)

    # Check source file exists
    if not src_path.exists() or not src_path.is_file():
        return ToolResult(success=False, content=f"Error: Source file '{file_path}' not found")

    size_error = check_line_mode_file_size(src_path, file_path)
    if size_error:
        return ToolResult(success=False, content=size_error)
    # Read source file content
    try:
        src_content = src_path.read_text(encoding='utf-8')
    except Exception as e:
        return ToolResult(success=False, content=f"Error reading source file: {e}")

    src_lines = src_content.splitlines(keepends=False)
    src_total_lines = len(src_lines)

    # Validate source line numbers
    if start_line < 1:
        start_line = 1
    if end_line < start_line:
        end_line = start_line
    if start_line > src_total_lines:
        return ToolResult(success=False, content=f"Error: start_line {start_line} is beyond source file length ({src_total_lines} lines)")
    if end_line > src_total_lines:
        end_line = src_total_lines

    # Extract the lines to move
    lines_to_move = src_lines[start_line - 1:end_line]

    if same_file:
        # Same-file move: copy then delete approach
        # This ensures line numbers match the current file state

        compressed_before = compress(src_content)

        # Step 1: Copy - Insert the lines at the destination
        # Adjust insert_line for the current file state
        if insert_line < 1:
            insert_line = 1
        if insert_line > src_total_lines + 2:
            return ToolResult(success=False, content=f"Error: insert_line {insert_line} is beyond file length + 2 ({src_total_lines + 2} max)")
        if insert_line > src_total_lines + 1:
            insert_line = src_total_lines + 1

        insert_idx = insert_line - 1

        # Build intermediate state: with lines inserted
        lines_with_newlines = [line + '\n' for line in src_lines]
        new_lines_with_newlines = [line + '\n' for line in lines_to_move]

        intermediate_lines = lines_with_newlines[:insert_idx] + new_lines_with_newlines + lines_with_newlines[insert_idx:]

        # Step 2: Delete - Remove the original lines
        # The original lines are now at a shifted position if insert was before them
        # We need to figure out where the original lines ended up after the insert
        if insert_line <= start_line:
            # Insert was before or at the source: original lines shifted down by len(lines_to_move)
            delete_start = start_line - 1 + len(lines_to_move)
            delete_end = end_line - 1 + len(lines_to_move)
        else:
            # Insert was after the source: original lines are still at same position
            delete_start = start_line - 1
            delete_end = end_line - 1

        # Remove the original lines from the intermediate state
        final_lines = intermediate_lines[:delete_start] + intermediate_lines[delete_end + 1:]
        modified_content = ''.join(final_lines)

        # Write modified content
        try:
            src_path.write_text(modified_content, encoding='utf-8', newline='')
        except Exception as e:
            return ToolResult(success=False, content=f"Error writing file: {e}")

        compressed_after = compress(modified_content)

        # Calculate where the moved lines ended up
        if insert_line <= start_line:
            moved_start = insert_line
        else:
            # Lines were removed first (conceptually), so insert position shifts
            moved_start = insert_line - (end_line - start_line + 1)

        moved_end = moved_start + len(lines_to_move) - 1

        # Create operation record
        op = FileOperation(
            operation_type="move_lines",
            timestamp=datetime.now(timezone.utc).isoformat(),
            file_path=file_path,
            compressed_before=compressed_before,
            compressed_after=compressed_after
        )

        # Build state update
        state_update = state_service.build_new_file_operation(op)

        # Adjust file view windows for the same-file move if experiment is active
        file_view_update = {}
        if experiments_service and experiments_service.is_enabled("file_view_windows") and state_service:
            N = end_line - start_line + 1
            # First, adjust for the deletion at the source position
            deletion_update = state_service.build_file_view_windows_edit_adjustment(
                file_path=file_path,
                start_line=start_line,
                old_line_count=N,
                new_line_count=0,
                source="edit"
            )
            # Compute the adjusted insert position after deletion
            adjusted_insert = insert_line if insert_line <= start_line else insert_line - N
            # Then, adjust for the insertion at the destination position
            updated_windows = deletion_update.get("file_view_windows", state_service.file_view_windows)
            file_view_update = state_service.build_file_view_windows_edit_adjustment(
                file_path=file_path,
                start_line=adjusted_insert,
                old_line_count=0,
                new_line_count=N,
                source="edit",
                current_windows=updated_windows
            )

        return ToolResult(
            success=True,
            content=f"Lines {start_line}-{end_line} in '{file_path}' have been moved to before line {insert_line}. The requested operation is complete.",
            state_updates={**state_update, **file_view_update}
        )

    else:
        # Cross-file move
        # Validate destination path
        dest_path, error = filesystem_service.validate_path(dest_file_path)
        if error:
            return ToolResult(success=False, content=error)

        # Check if destination file exists or needs to be created
        dest_is_new = not dest_path.exists() or not dest_path.is_file()
        if dest_is_new:
            # Create parent directories if needed
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            dest_content = ''
            dest_lines = []
            dest_total_lines = 0
        else:
            size_error = check_line_mode_file_size(dest_path, dest_file_path)
            if size_error:
                return ToolResult(success=False, content=size_error)
            # Read destination file content
            try:
                dest_content = dest_path.read_text(encoding='utf-8')
            except Exception as e:
                return ToolResult(success=False, content=f"Error reading destination file: {e}")
            dest_lines = dest_content.splitlines(keepends=False)
            dest_total_lines = len(dest_lines)

        # Validate insert_line for destination
        if insert_line < 1:
            insert_line = 1
        if dest_is_new:
            # For new files, any insert_line >= 1 is valid; blank lines will be added as needed
            pass
        else:
            if insert_line > dest_total_lines + 2:
                return ToolResult(success=False, content=f"Error: insert_line {insert_line} is beyond destination file length + 2 ({dest_total_lines + 2} max)")
            if insert_line > dest_total_lines + 1:
                insert_line = dest_total_lines + 1

        # Compress before states
        src_compressed_before = compress(src_content)
        dest_compressed_before = compress(dest_content) if dest_content else None

        # Insert lines into destination
        new_lines_with_newlines = [line + '\n' for line in lines_to_move]
        if dest_is_new:
            # Build new file with leading blank lines and moved content
            leading_blank_lines = ['\n'] * (insert_line - 1)
            modified_dest_lines = leading_blank_lines + new_lines_with_newlines
            modified_dest_content = ''.join(modified_dest_lines)
        else:
            insert_idx = insert_line - 1
            dest_lines_with_newlines = [line + '\n' for line in dest_lines]
            modified_dest_lines = dest_lines_with_newlines[:insert_idx] + new_lines_with_newlines + dest_lines_with_newlines[insert_idx:]
            modified_dest_content = ''.join(modified_dest_lines)

        # Delete lines from source
        src_lines_with_newlines = [line + '\n' for line in src_lines]
        modified_src_lines = src_lines_with_newlines[:start_line - 1] + src_lines_with_newlines[end_line:]
        modified_src_content = ''.join(modified_src_lines)

        # Write destination file
        try:
            dest_path.write_text(modified_dest_content, encoding='utf-8', newline='')
        except Exception as e:
            return ToolResult(success=False, content=f"Error writing destination file: {e}")

        # Write source file
        try:
            src_path.write_text(modified_src_content, encoding='utf-8', newline='')
        except Exception as e:
            # Try to roll back destination write
            try:
                dest_path.write_text(dest_content, encoding='utf-8', newline='')
            except:
                pass
            return ToolResult(success=False, content=f"Error writing source file: {e}")

        src_compressed_after = compress(modified_src_content)
        dest_compressed_after = compress(modified_dest_content)

        # Calculate where the moved lines ended up in destination
        moved_start = insert_line
        moved_end = moved_start + len(lines_to_move) - 1

        # Create operation records (source first, then destination)
        # Use atomic_group_id to link cross-file operations for undo/redo
        group_id = str(uuid.uuid4())
        src_op = FileOperation(
            operation_type="move_lines",
            timestamp=datetime.now(timezone.utc).isoformat(),
            file_path=file_path,
            compressed_before=src_compressed_before,
            compressed_after=src_compressed_after,
            atomic_group_id=group_id
        )

        dest_op = FileOperation(
            operation_type="move_lines",
            timestamp=datetime.now(timezone.utc).isoformat(),
            file_path=dest_file_path,
            compressed_before=dest_compressed_before,
            compressed_after=dest_compressed_after,
            atomic_group_id=group_id
        )

        # Build state updates for both files atomically
        state_update = state_service.build_new_file_operations([src_op, dest_op])

        # Adjust file view windows for the cross-file move if experiment is active
        file_view_update = {}
        if experiments_service and experiments_service.is_enabled("file_view_windows") and state_service:
            N = end_line - start_line + 1
            # First, adjust for the deletion in source file
            deletion_update = state_service.build_file_view_windows_edit_adjustment(
                file_path=file_path,
                start_line=start_line,
                old_line_count=N,
                new_line_count=0,
                source="edit"
            )
            # Then, adjust for the insertion in destination file
            updated_windows = deletion_update.get("file_view_windows", state_service.file_view_windows)
            file_view_update = state_service.build_file_view_windows_edit_adjustment(
                file_path=dest_file_path,
                start_line=insert_line,
                old_line_count=0,
                new_line_count=N,
                source="edit",
                current_windows=updated_windows
            )

        return ToolResult(
            success=True,
            content=f"Lines {start_line}-{end_line} moved from '{file_path}' to before line {insert_line} in '{dest_file_path}'" + (" (new file created)" if dest_is_new else "") + f". The requested operation is complete.",
            state_updates={**state_update, **file_view_update}
        )


def move_lines_tool_validated(file_path: str, start_line: int, start_line_content: str,
                              end_line: int, end_line_content: str,
                              insert_line: int, anchor_line_content: str,
                              dest_file_path: str = None,
                              **kwargs) -> ToolResult:
    """Move lines with content validation - STATE-MODIFYING tool.

    Validates that the provided start_line_content and end_line_content match the actual
    source file content (whitespace-stripped comparison), and that anchor_line_content
    matches the destination insert line. If mismatch, checks +/-1 lines and auto-adjusts
    if exactly one neighbor matches.

    Moves lines from start_line to end_line (1-indexed, inclusive) from the source file
    to BEFORE insert_line in the destination file. If dest_file_path is not provided,
    lines are moved within the same file. If the destination file does not exist, it will
    be created with leading blank lines as needed to place the moved content at the
    requested line.

    For same-file moves, the implementation first copies the lines, then deletes them
    from their original position. This ensures correct line number handling when the
    destination is after the source.

    Args:
        file_path: Source file path
        start_line: First line to move (1-indexed, inclusive)
        start_line_content: Expected content of the start line in source (for validation)
        end_line: Last line to move (1-indexed, inclusive)
        end_line_content: Expected content of the end line in source (for validation)
        insert_line: Line number BEFORE which to insert in destination (1-indexed)
        anchor_line_content: Expected content of the anchor line in destination (for validation)
        dest_file_path: Destination file path (defaults to same as source). If the file
            does not exist, it will be created.

    Returns:
        ToolResult - with success status, content message, and state_updates dict
    """
    filesystem_service = kwargs.get('filesystem_service')
    state_service = kwargs.get('state_service')
    experiments_service = kwargs.get('experiments_service')
    
    # Default dest_file_path to file_path (same file)
    if dest_file_path is None:
        dest_file_path = file_path

    same_file = (file_path == dest_file_path)

    # Validate source path
    src_path, error = filesystem_service.validate_path(file_path)
    if error:
        return ToolResult(success=False, content=error)

    # Check source file exists
    if not src_path.exists() or not src_path.is_file():
        return ToolResult(success=False, content=f"Error: Source file '{file_path}' not found")

    size_error = check_line_mode_file_size(src_path, file_path)
    if size_error:
        return ToolResult(success=False, content=size_error)
    # Read source file content
    try:
        src_content = src_path.read_text(encoding='utf-8')
    except Exception as e:
        return ToolResult(success=False, content=f"Error reading source file: {e}")

    src_lines = src_content.splitlines(keepends=False)
    src_total_lines = len(src_lines)

    # Apply multiline fix experiment for start/end_line_content (source file):
    # some LLMs provide multiple lines for line content parameters instead of just
    # one line. Only apply if the original content doesn't match, since the content
    # may correctly be a single line. Use split('\n') instead of splitlines() to
    # avoid splitting on \r, \v, \f, etc.
    if experiments_service and experiments_service.is_enabled('line_content_multiline_fix'):
        start_content_lines = start_line_content.split('\n', 1)
        if len(start_content_lines) > 1:
            start_idx = start_line - 1
            if not lines_match(start_idx, start_line_content, src_lines):
                start_line_content = start_content_lines[0]
        
        end_content_lines = end_line_content.rsplit('\n', 1)
        if len(end_content_lines) > 1:
            end_idx = end_line - 1
            if not lines_match(end_idx, end_line_content, src_lines):
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
    if start_line > src_total_lines:
        return ToolResult(success=False, content=f"Error: start_line {start_line} is beyond source file length ({src_total_lines} lines)")

    start_idx = start_line - 1
    if not lines_match(start_idx, start_line_content, src_lines):
        match_above = lines_match(start_idx - 1, start_line_content, src_lines)
        match_below = lines_match(start_idx + 1, start_line_content, src_lines)
        if match_above and not match_below:
            start_line = start_line - 1
            start_adjustments.append(f"start_line adjusted from {original_start_line} to {start_line}")
        elif match_below and not match_above:
            start_line = start_line + 1
            start_adjustments.append(f"start_line adjusted from {original_start_line} to {start_line}")
        else:
            actual = src_lines[start_idx] if 0 <= start_idx < src_total_lines else "<out of range>"
            alignment_errors.append(
                f"You provided start_line_content '{start_line_content}' but the content at line {original_start_line} "
                f"in the source file is currently '{actual}'. Remember: you must provide the exact current content "
                f"of the line and only the one line, not what you intend it to be after moving.\n"
            )

    # Validate end_line
    if end_line < start_line:
        end_line = start_line
    if end_line > src_total_lines:
        end_line = src_total_lines

    end_idx = end_line - 1
    if not lines_match(end_idx, end_line_content, src_lines):
        match_above = lines_match(end_idx - 1, end_line_content, src_lines)
        match_below = lines_match(end_idx + 1, end_line_content, src_lines)
        if match_above and not match_below:
            end_line = end_line - 1
            end_adjustments.append(f"end_line adjusted from {original_end_line} to {end_line}")
        elif match_below and not match_above:
            end_line = end_line + 1
            end_adjustments.append(f"end_line adjusted from {original_end_line} to {end_line}")
        else:
            actual = src_lines[end_idx] if 0 <= end_idx < src_total_lines else "<out of range>"
            alignment_errors.append(
                f"You provided end_line_content '{end_line_content}' but the content at line {original_end_line} "
                f"in the source file is currently '{actual}'. Remember: you must provide the exact current content "
                f"of the line and only the one line, not what you intend it to be after moving.\n"
            )

    # Re-validate adjusted line numbers
    if start_line < 1:
        start_line = 1
    if end_line < start_line:
        end_line = start_line
    if start_line > src_total_lines:
        return ToolResult(success=False, content=f"Error: start_line {start_line} is beyond source file length ({src_total_lines} lines)")
    if end_line > src_total_lines:
        end_line = src_total_lines

    # ================================================================
    # Validate destination anchor_line_content
    # ================================================================
    dest_lines = None
    dest_total_lines = 0
    dest_content = None

    if same_file:
        dest_lines = src_lines
        dest_total_lines = src_total_lines
        dest_content = src_content
        dest_is_new = False
    else:
        dest_path, error = filesystem_service.validate_path(dest_file_path)
        if error:
            return ToolResult(success=False, content=error)
        dest_is_new = not dest_path.exists() or not dest_path.is_file()
        if dest_is_new:
            # Create parent directories if needed
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            dest_content = None
            dest_lines = []
            dest_total_lines = 0
        else:
            size_error = check_line_mode_file_size(dest_path, dest_file_path)
            if size_error:
                return ToolResult(success=False, content=size_error)
            try:
                dest_content = dest_path.read_text(encoding='utf-8')
            except Exception as e:
                return ToolResult(success=False, content=f"Error reading destination file: {e}")
            dest_lines = dest_content.splitlines(keepends=False)
    dest_total_lines = len(dest_lines)

    # Apply multiline fix experiment for anchor_line_content (destination file):
    # some LLMs provide multiple lines for line content parameters instead of just
    # one line. Only apply if the original content doesn't match, since the content
    # may correctly be a single line. Use split('\n') instead of splitlines() to
    # avoid splitting on \r, \v, \f, etc.
    if experiments_service and experiments_service.is_enabled('line_content_multiline_fix'):
        anchor_content_lines = anchor_line_content.split('\n', 1)
        if len(anchor_content_lines) > 1:
            anchor_idx = insert_line - 1
            if not lines_match(anchor_idx, anchor_line_content, dest_lines):
                anchor_line_content = anchor_content_lines[0]

    anchor_adjustments = []
    original_insert_line = insert_line

    # Validate insert_line range
    if insert_line < 1:
        insert_line = 1
    if dest_is_new:
        # For new files, any insert_line >= 1 is valid; blank lines will be added as needed
        pass
    else:
        if insert_line > dest_total_lines + 2:
            return ToolResult(success=False, content=f"Error: insert_line {insert_line} is beyond destination file length + 2 ({dest_total_lines + 2} max)")
        if insert_line > dest_total_lines + 1:
            insert_line = dest_total_lines + 1

    # Validate anchor content (if not appending at end and destination exists)
    if not dest_is_new and insert_line <= dest_total_lines:
        anchor_idx = insert_line - 1

        if not lines_match(anchor_idx, anchor_line_content, dest_lines):
            # Check +/-1 lines
            matches = []
            if lines_match(anchor_idx - 1, anchor_line_content, dest_lines):
                matches.append(insert_line - 1)
            if lines_match(anchor_idx + 1, anchor_line_content, dest_lines):
                matches.append(insert_line + 1)

            if len(matches) == 1:
                insert_line = matches[0]
                anchor_adjustments.append(f"insert_line adjusted from {original_insert_line} to {insert_line} to match anchor content")
            elif len(matches) == 0:
                alignment_errors.append(
                    f"You provided anchor_line_content '{anchor_line_content}' but the content at line {insert_line} "
                    f"in the destination file is currently '{dest_lines[anchor_idx]}'. Remember: you must provide the "
                    f"exact current content of the line, not what you intend it to be after moving.\n"
                )
            else:
                alignment_errors.append(
                    f"You provided anchor_line_content '{anchor_line_content}' but the content at line {insert_line} "
                    f"in the destination file is currently '{dest_lines[anchor_idx]}'. The anchor content matches "
                    f"multiple lines ({matches}). Cannot auto-adjust.\n"
                )

    # If any alignment errors, return failure with file views
    # Check for omission placeholders in validation args
    if alignment_errors:
        _new_errors = []
        for err in alignment_errors:
            _replaced = False
            for _arg_name, _arg_value in [('start_line_content', start_line_content), ('end_line_content', end_line_content), ('anchor_line_content', anchor_line_content)]:
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
            requests = []
            # Show source area around the problem
            view_start = max(1, min(original_start_line, original_end_line) - 5)
            view_end = min(src_total_lines, max(original_start_line, original_end_line) + 5)
            view_limit = view_end - view_start + 1
            requests.append(FileViewRequest(file_path=file_path, offset=view_start, limit=view_limit))
            # Show destination area if different file
            if not same_file:
                d_view_start = max(1, original_insert_line - 5)
                d_view_end = min(dest_total_lines, original_insert_line + 5)
                d_view_limit = d_view_end - d_view_start + 1
                requests.append(FileViewRequest(file_path=dest_file_path, offset=d_view_start, limit=d_view_limit))
            file_view_update = state_service.build_file_view_windows_update(requests, source="edit")
        return ToolResult(success=False, content=''.join(alignment_errors), state_updates=file_view_update)

    # ================================================================
    # Perform the move
    # ================================================================
    # Extract the lines to move
    lines_to_move = src_lines[start_line - 1:end_line]

    if same_file:
        # Same-file move: copy then delete approach
        compressed_before = compress(src_content)

        # Step 1: Copy - Insert the lines at the destination
        insert_idx = insert_line - 1
        lines_with_newlines = [line + '\n' for line in src_lines]
        new_lines_with_newlines = [line + '\n' for line in lines_to_move]

        intermediate_lines = lines_with_newlines[:insert_idx] + new_lines_with_newlines + lines_with_newlines[insert_idx:]

        # Step 2: Delete - Remove the original lines
        if insert_line <= start_line:
            # Insert was before or at the source: original lines shifted down by len(lines_to_move)
            delete_start = start_line - 1 + len(lines_to_move)
            delete_end = end_line - 1 + len(lines_to_move)
        else:
            # Insert was after the source: original lines are still at same position
            delete_start = start_line - 1
            delete_end = end_line - 1

        final_lines = intermediate_lines[:delete_start] + intermediate_lines[delete_end + 1:]
        modified_content = ''.join(final_lines)

        # Write modified content
        try:
            src_path.write_text(modified_content, encoding='utf-8', newline='')
        except Exception as e:
            return ToolResult(success=False, content=f"Error writing file: {e}")

        compressed_after = compress(modified_content)

        # Calculate where the moved lines ended up
        if insert_line <= start_line:
            moved_start = insert_line
        else:
            moved_start = insert_line - (end_line - start_line + 1)

        moved_end = moved_start + len(lines_to_move) - 1

        # Create operation record
        op = FileOperation(
            operation_type="move_lines",
            timestamp=datetime.now(timezone.utc).isoformat(),
            file_path=file_path,
            compressed_before=compressed_before,
            compressed_after=compressed_after
        )

        # Build state update
        state_update = state_service.build_new_file_operation(op)

        # Adjust file view windows for the same-file move if experiment is active
        file_view_update = {}
        if experiments_service and experiments_service.is_enabled("file_view_windows") and state_service:
            N = end_line - start_line + 1
            # First, adjust for the deletion at the source position
            deletion_update = state_service.build_file_view_windows_edit_adjustment(
                file_path=file_path,
                start_line=start_line,
                old_line_count=N,
                new_line_count=0,
                source="edit"
            )
            # Compute the adjusted insert position after deletion
            adjusted_insert = insert_line if insert_line <= start_line else insert_line - N
            # Then, adjust for the insertion at the destination position
            updated_windows = deletion_update.get("file_view_windows", state_service.file_view_windows)
            file_view_update = state_service.build_file_view_windows_edit_adjustment(
                file_path=file_path,
                start_line=adjusted_insert,
                old_line_count=0,
                new_line_count=N,
                source="edit",
                current_windows=updated_windows
            )

        # Build success message with adjustment notes
        base_message = f"Lines {start_line}-{end_line} in '{file_path}' have been moved to before line {insert_line}. The requested operation is complete."
        all_adjustments = start_adjustments + end_adjustments + anchor_adjustments
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

    else:
        # Cross-file move
        dest_path, _ = filesystem_service.validate_path(dest_file_path)

        # Validate insert_line for destination (already validated anchor above)
        if insert_line < 1:
            insert_line = 1
        if dest_is_new:
            # For new files, any insert_line >= 1 is valid
            pass
        else:
            if insert_line > dest_total_lines + 2:
                return ToolResult(success=False, content=f"Error: insert_line {insert_line} is beyond destination file length + 2 ({dest_total_lines + 2} max)")
            if insert_line > dest_total_lines + 1:
                insert_line = dest_total_lines + 1

        # Compress before states
        src_compressed_before = compress(src_content)
        dest_compressed_before = compress(dest_content) if dest_content else None

        # Insert lines into destination
        new_lines_with_newlines = [line + '\n' for line in lines_to_move]
        if dest_is_new:
            # Build new file with leading blank lines and moved content
            leading_blank_lines = ['\n'] * (insert_line - 1)
            modified_dest_lines = leading_blank_lines + new_lines_with_newlines
            modified_dest_content = ''.join(modified_dest_lines)
        else:
            insert_idx = insert_line - 1
            dest_lines_with_newlines = [line + '\n' for line in dest_lines]
            modified_dest_lines = dest_lines_with_newlines[:insert_idx] + new_lines_with_newlines + dest_lines_with_newlines[insert_idx:]
            modified_dest_content = ''.join(modified_dest_lines)

        # Delete lines from source
        src_lines_with_newlines = [line + '\n' for line in src_lines]
        modified_src_lines = src_lines_with_newlines[:start_line - 1] + src_lines_with_newlines[end_line:]
        modified_src_content = ''.join(modified_src_lines)

        # Write destination file
        try:
            dest_path.write_text(modified_dest_content, encoding='utf-8', newline='')
        except Exception as e:
            return ToolResult(success=False, content=f"Error writing destination file: {e}")

        # Write source file
        try:
            src_path.write_text(modified_src_content, encoding='utf-8', newline='')
        except Exception as e:
            # Try to roll back destination write
            if dest_is_new:
                try:
                    dest_path.unlink()
                except:
                    pass
            else:
                try:
                    dest_path.write_text(dest_content, encoding='utf-8', newline='')
                except:
                    pass
            return ToolResult(success=False, content=f"Error writing source file: {e}")

        src_compressed_after = compress(modified_src_content)
        dest_compressed_after = compress(modified_dest_content)

        # Calculate where the moved lines ended up in destination
        moved_start = insert_line
        moved_end = moved_start + len(lines_to_move) - 1

        # Create operation records
        # Use atomic_group_id to link cross-file operations for undo/redo
        group_id = str(uuid.uuid4())
        src_op = FileOperation(
            operation_type="move_lines",
            timestamp=datetime.now(timezone.utc).isoformat(),
            file_path=file_path,
            compressed_before=src_compressed_before,
            compressed_after=src_compressed_after,
            atomic_group_id=group_id
        )

        dest_op = FileOperation(
            operation_type="move_lines",
            timestamp=datetime.now(timezone.utc).isoformat(),
            file_path=dest_file_path,
            compressed_before=dest_compressed_before,
            compressed_after=dest_compressed_after,
            atomic_group_id=group_id
        )

        # Build state updates for both files atomically
        state_update = state_service.build_new_file_operations([src_op, dest_op])

        # Adjust file view windows for the cross-file move if experiment is active
        file_view_update = {}
        if experiments_service and experiments_service.is_enabled("file_view_windows") and state_service:
            N = end_line - start_line + 1
            # First, adjust for the deletion in source file
            deletion_update = state_service.build_file_view_windows_edit_adjustment(
                file_path=file_path,
                start_line=start_line,
                old_line_count=N,
                new_line_count=0,
                source="edit"
            )
            # Then, adjust for the insertion in destination file
            updated_windows = deletion_update.get("file_view_windows", state_service.file_view_windows)
            file_view_update = state_service.build_file_view_windows_edit_adjustment(
                file_path=dest_file_path,
                start_line=insert_line,
                old_line_count=0,
                new_line_count=N,
                source="edit",
                current_windows=updated_windows
            )

        # Build success message with adjustment notes
        base_message = f"Lines {start_line}-{end_line} moved from '{file_path}' to before line {insert_line} in '{dest_file_path}'" + (" (new file created)" if dest_is_new else "") + f". The requested operation is complete."
        all_adjustments = start_adjustments + end_adjustments + anchor_adjustments
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
