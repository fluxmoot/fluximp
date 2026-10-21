from typing import Tuple, Optional
from datetime import datetime, timezone

from fluximp_agent.filesystem_service import check_line_mode_file_size
from fluximp_agent.tools.tool_definitions import ToolResult
from fluximp_agent.tools.common import compress, FileOperation
from fluximp_agent.context_managment_constants import VIEW_BUDGET_DESC
from fluximp_agent.state_service import normalize_file_path


def _delete_file(file_path: str, filesystem_service) -> Tuple[bool, Optional[str]]:
    """Delete a file if it exists."""
    path, error = filesystem_service.validate_path(file_path)
    if error:
        return (False, error)
    
    try:
        if path.exists():
            path.unlink()
        return (True, None)
    except Exception as e:
        return (False, f"Error deleting file '{file_path}': {str(e)}")


def _write_file(file_path: str, content: str, filesystem_service) -> Tuple[bool, Optional[str]]:
    """Write content to file, creating directories if needed."""
    path, error = filesystem_service.validate_path(file_path)
    if error:
        return (False, error)
    
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8', newline='')
        return (True, None)
    except Exception as e:
        return (False, f"Error writing file '{file_path}': {str(e)}")


def delete_file_tool(file_path: str, **kwargs) -> ToolResult:
    """Delete a file - STATE-MODIFYING tool.

    When the file_view_windows experiment is active, also removes any
    file view windows for the deleted file.

    Args:
        file_path: Path to the file to delete

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

    # Store compressed content before deletion for undo
    compressed_before = None
    if path.exists() and path.is_file():
        try:
            size_error = check_line_mode_file_size(path, file_path)
            if size_error:
                return ToolResult(success=False, content=size_error)
            content = path.read_text(encoding='utf-8')
            compressed_before = compress(content)
        except Exception as e:
            return ToolResult(success=False, content=f"Error reading file before deletion: {e}")
    else:
        return ToolResult(success=False, content=f"Error: File '{file_path}' not found")

    # Delete the file
    success, error = _delete_file(file_path, filesystem_service)
    if not success:
        return ToolResult(success=False, content=error)

    # Create operation record
    op = FileOperation(
        operation_type="delete_file",
        timestamp=datetime.now(timezone.utc).isoformat(),
        file_path=file_path,
        compressed_before=compressed_before,
        compressed_after=None
    )

    # Build state update
    state_update = state_service.build_new_file_operation(op)

    # Build file view windows removal if experiment is active
    file_view_update = {}
    if experiments_service and experiments_service.is_enabled("file_view_windows") and state_service:
        file_view_update = state_service.build_file_view_windows_removal([file_path])

    # Track deleted file path for create_file experiment
    deleted_file_update = {}
    if experiments_service and experiments_service.is_enabled("create_file") and state_service:
        deleted_file_update = state_service.build_deleted_file_paths_update(file_path)

    return ToolResult(
        success=True,
        content=f"File '{file_path}' has been successfully deleted. The requested operation is complete.",
        state_updates={**state_update, **file_view_update, **deleted_file_update}
    )
def write_file_tool(file_path: str, content: str, **kwargs) -> ToolResult:
    f"""Write content to a file - STATE-MODIFYING tool.

    Creates directories if needed.

    When the file_view_windows experiment is active, this tool also adds the
    written file to the file view windows (first 2500 lines). File view windows
    are automatically closed when total size exceeds {VIEW_BUDGET_DESC}.

    Args:
        file_path: Target file path
        content: Content to write

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

    # Store compressed content before writing (for undo)
    compressed_before = None
    if path.exists() and path.is_file():
        try:
            size_error = check_line_mode_file_size(path, file_path)
            if size_error:
                return ToolResult(success=False, content=size_error)
            old_content = path.read_text(encoding='utf-8')
            compressed_before = compress(old_content)
        except Exception as e:
            return ToolResult(success=False, content=f"Error reading existing file: {e}")

    # Write the file
    success, error = _write_file(file_path, content, filesystem_service)
    if not success:
        return ToolResult(success=False, content=error)

    # Store compressed content after writing (for redo)
    compressed_after = compress(content)

    # Create operation record
    op = FileOperation(
        operation_type="write_file",
        timestamp=datetime.now(timezone.utc).isoformat(),
        file_path=file_path,
        compressed_before=compressed_before,
        compressed_after=compressed_after
    )

    # Build state update
    state_update = state_service.build_new_file_operation(op)

    # Build file view windows update if experiment is active
    file_view_update = {}
    if experiments_service and experiments_service.is_enabled("file_view_windows") and state_service:
        from fluximp_agent.state_service import FileViewRequest
        # Add window for the entire file (limit=None means to end of file)
        file_view_update = state_service.build_file_view_windows_update(
            [FileViewRequest(file_path=file_path, offset=1, limit=None)],
            source="edit"
        )

    return ToolResult(
        success=True,
        content=f"File '{file_path}' has been written successfully with {len(content)} characters. The requested operation is complete.",
        state_updates={**state_update, **file_view_update}
    )

def create_file_tool(file_path: str, content: str, **kwargs) -> ToolResult:
    """Create a new file - STATE-MODIFYING tool.

    Only works for files that do NOT currently exist and were NOT deleted during
    this session. If a file was deleted during the session, it will be automatically
    restored to its pre-deletion state and the tool will fail with a message telling
    the agent to use edit tools instead.

    When the file_view_windows experiment is active, this tool also adds the
    created file to the file view windows (first 2500 lines).

    Args:
        file_path: Target file path (must not exist)
        content: Content to write

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

    # Check 1: Does the file exist on disk?
    if path.exists() and path.is_file():
        return ToolResult(
            success=False,
            content=f"Error: File '{file_path}' already exists. Use edit_file_lines or insert_lines to modify existing files."
        )

    # Check 2: Was the file deleted during this session?
    deleted_paths = state_service.deleted_file_paths if state_service else frozenset()
    norm_file_path = normalize_file_path(file_path, filesystem_service)
    deleted_norm = {normalize_file_path(p, filesystem_service) for p in deleted_paths}
    if norm_file_path in deleted_norm:
        # Find the most recent delete_file operation for this file_path
        file_ops = state_service.file_operations if state_service else []
        op_position = state_service.operation_position if state_service else -1
        restored_content = None
        
        for i in range(min(op_position, len(file_ops) - 1), -1, -1):
            op = file_ops[i]
            if op.operation_type == "delete_file" and op.file_path is not None and normalize_file_path(op.file_path, filesystem_service) == norm_file_path:
                if op.compressed_before:
                    from fluximp_agent.tools.common import decompress
                    restored_content = decompress(op.compressed_before)
                break
        
        if restored_content is not None:
            # Write the restored content back to disk
            success, error = _write_file(file_path, restored_content, filesystem_service)
            if not success:
                return ToolResult(success=False, content=error)
            
            # Record the restoration as a create_file operation with compressed_before=None
            # (file did not exist on disk when this operation started)
            compressed_after = compress(restored_content)
            
            op = FileOperation(
                operation_type="create_file",
                timestamp=datetime.now(timezone.utc).isoformat(),
                file_path=file_path,
                compressed_before=None,
                compressed_after=compressed_after
            )
            
            # Build state updates
            state_update = state_service.build_new_file_operation(op)
            
            # Build file view windows update if experiment is active
            file_view_update = {}
            if experiments_service and experiments_service.is_enabled("file_view_windows") and state_service:
                from fluximp_agent.state_service import FileViewRequest
                file_view_update = state_service.build_file_view_windows_update(
                    [FileViewRequest(file_path=file_path, offset=1, limit=None)],
                    source="edit"
                )
            
            return ToolResult(
                success=False,
                content=f"File '{file_path}' was deleted during this session and has been restored to its state before deletion. Use edit_file_lines, insert_lines, or move_lines to modify it instead of deleting and recreating.",
                state_updates={**state_update, **file_view_update}
            )
        else:
            # File was deleted but we couldn't find the content - shouldn't happen normally
            return ToolResult(
                success=False,
                content=f"Error: File '{file_path}' was deleted during this session and cannot be recreated with create_file. Use edit_file_lines, insert_lines, or move_lines to modify it after restoring."
            )

    # Check 3: File does not exist and was not deleted -> create it
    # Write the file
    success, error = _write_file(file_path, content, filesystem_service)
    if not success:
        return ToolResult(success=False, content=error)

    # Store compressed content after writing
    compressed_after = compress(content)

    # Create operation record
    op = FileOperation(
        operation_type="create_file",
        timestamp=datetime.now(timezone.utc).isoformat(),
        file_path=file_path,
        compressed_before=None,
        compressed_after=compressed_after
    )

    # Build state update
    state_update = state_service.build_new_file_operation(op)

    # Build file view windows update if experiment is active
    file_view_update = {}
    if experiments_service and experiments_service.is_enabled("file_view_windows") and state_service:
        from fluximp_agent.state_service import FileViewRequest
        file_view_update = state_service.build_file_view_windows_update(
            [FileViewRequest(file_path=file_path, offset=1, limit=None)],
            source="edit"
        )

    return ToolResult(
        success=True,
        content=f"File '{file_path}' has been created successfully with {len(content)} characters. The requested operation is complete.",
        state_updates={**state_update, **file_view_update}
    )
