"""File operation reversal, application, and recording.

Provides the low-level machinery for undo/redo and time travel:
- reverse_operation: reverse a single FileOperation on disk
- apply_operation: apply or reverse a FileOperation in a given direction
- new_operation: record a new FileOperation in agent state
"""

from typing import Optional, Tuple

from fluximp_agent.tools.common import FileOperation, decompress
from fluximp_agent.tools.filesystem_modify import _delete_file, _write_file
from fluximp_agent.filesystem_service import FileSystemService
from fluximp_agent.fs_agent_state import AgentState




# ============================================================================
# Operation Reversal and Application
# ============================================================================
def reverse_operation(op: FileOperation, state: AgentState, filesystem_service: FileSystemService) -> Tuple[bool, Optional[str]]:
    """Reverse a file operation to implement undo functionality.

    Uses low-level file operation functions (prefixed with _) to
    directly modify files on disk. These are NOT the same as the
    LangGraph tool functions.

    Args:
        op: The file operation to reverse
        state: Current agent state
        filesystem_service: The FileSystemService instance

    Returns:
        Tuple of (success, error_message)
    """
    print(f"DEBUG reverse_operation: Reversing {op.operation_type} on {op.file_path}")
    print(f"DEBUG reverse_operation: compressed_before={op.compressed_before is not None}, compressed_after={op.compressed_after is not None}")

    if op.operation_type == "delete_file":
        # Restore compressed_before content
        if op.compressed_before:
            return _write_file(op.file_path, decompress(op.compressed_before), filesystem_service)
        return (True, None)  # Nothing to restore
    elif op.operation_type in ["write_file", "create_file"]:
        # Restore compressed_before content (previous state before write/create)
        if op.compressed_before:
            print(f"DEBUG reverse_operation: Restoring compressed_before content")
            return _write_file(op.file_path, decompress(op.compressed_before), filesystem_service)
        else:
            # File didn't exist before, so delete it
            print(f"DEBUG reverse_operation: No compressed_before, deleting file")
            result = _delete_file(op.file_path, filesystem_service)
            print(f"DEBUG reverse_operation: _delete_file result: {result}")
            return result
    elif op.operation_type == "edit_file_lines":
        # Restore compressed_before content
        if op.compressed_before:
            return _write_file(op.file_path, decompress(op.compressed_before), filesystem_service)
        return (True, None)
    elif op.operation_type == "insert_lines":
        # Restore compressed_before content
        if op.compressed_before:
            return _write_file(op.file_path, decompress(op.compressed_before), filesystem_service)
        return (True, None)
    elif op.operation_type == "move_lines":
        # Restore compressed_before content
        if op.compressed_before:
            return _write_file(op.file_path, decompress(op.compressed_before), filesystem_service)
        return (True, None)
    elif op.operation_type == "adjust_indent":
        # Restore compressed_before content
        if op.compressed_before:
            return _write_file(op.file_path, decompress(op.compressed_before), filesystem_service)
        return (True, None)
    elif op.operation_type == "adjust_indent_exact":
        # Restore compressed_before content
        if op.compressed_before:
            return _write_file(op.file_path, decompress(op.compressed_before), filesystem_service)
        return (True, None)
    elif op.operation_type == "apply_patch":
        # Restore the file's pre-patch content, or delete it if it didn't exist
        # before the patch. Content-based, like the other editing tools.
        if op.compressed_before:
            return _write_file(op.file_path, decompress(op.compressed_before), filesystem_service)
        return _delete_file(op.file_path, filesystem_service)
    elif op.operation_type in ["undo", "redo"]:
        # Skip undo/redo operations
        return (True, None)

    raise ValueError(f"reverse_operation: Unknown operation type '{op.operation_type}' for file '{op.file_path}'")


def apply_operation(op: FileOperation, direction: str, filesystem_service: FileSystemService) -> Tuple[bool, Optional[str]]:
    """Apply or reverse a file operation.

    Uses low-level file operation functions (prefixed with _) to
    directly modify files on disk.

    Args:
        op: The file operation to apply
        direction: "forward" to apply, "backward" to reverse
        filesystem_service: The FileSystemService instance

    Returns:
        Tuple of (success, error_message)
    """
    if op.operation_type == "delete_file":
        if direction == "forward":
            return _delete_file(op.file_path, filesystem_service)
        else:
            # Restore from compressed_before
            if op.compressed_before:
                return _write_file(op.file_path, decompress(op.compressed_before), filesystem_service)
            return (True, None)
    elif op.operation_type in ["write_file", "create_file"]:
        if direction == "forward":
            # Use compressed_after when going forward
            return _write_file(op.file_path, decompress(op.compressed_after), filesystem_service)
        else:
            # Use compressed_before when going backward
            if op.compressed_before:
                return _write_file(op.file_path, decompress(op.compressed_before), filesystem_service)
            else:
                # File didn't exist before, delete it
                return _delete_file(op.file_path, filesystem_service)
    elif op.operation_type == "edit_file_lines":
        if direction == "forward":
            # Use compressed_after when going forward
            if op.compressed_after:
                return _write_file(op.file_path, decompress(op.compressed_after), filesystem_service)
            return (True, None)
        else:
            # Use compressed_before when going backward
            if op.compressed_before:
                return _write_file(op.file_path, decompress(op.compressed_before), filesystem_service)
            return (True, None)
    elif op.operation_type == "insert_lines":
        if direction == "forward":
            # Use compressed_after when going forward
            if op.compressed_after:
                return _write_file(op.file_path, decompress(op.compressed_after), filesystem_service)
            return (True, None)
        else:
            # Use compressed_before when going backward
            if op.compressed_before:
                return _write_file(op.file_path, decompress(op.compressed_before), filesystem_service)
            return (True, None)
    elif op.operation_type == "move_lines":
        if direction == "forward":
            # Use compressed_after when going forward
            if op.compressed_after:
                return _write_file(op.file_path, decompress(op.compressed_after), filesystem_service)
            return (True, None)
        else:
            # Use compressed_before when going backward
            if op.compressed_before:
                return _write_file(op.file_path, decompress(op.compressed_before), filesystem_service)
            return (True, None)
    elif op.operation_type == "adjust_indent":
        if direction == "forward":
            # Use compressed_after when going forward
            if op.compressed_after:
                return _write_file(op.file_path, decompress(op.compressed_after), filesystem_service)
            return (True, None)
        else:
            # Use compressed_before when going backward
            if op.compressed_before:
                return _write_file(op.file_path, decompress(op.compressed_before), filesystem_service)
            return (True, None)
    elif op.operation_type == "adjust_indent_exact":
        if direction == "forward":
            # Use compressed_after when going forward
            if op.compressed_after:
                return _write_file(op.file_path, decompress(op.compressed_after), filesystem_service)
            return (True, None)
        else:
            # Use compressed_before when going backward
            if op.compressed_before:
                return _write_file(op.file_path, decompress(op.compressed_before), filesystem_service)
            return (True, None)
    elif op.operation_type == "apply_patch":
        if direction == "forward":
            # Use compressed_after when going forward (delete if file was removed)
            if op.compressed_after:
                return _write_file(op.file_path, decompress(op.compressed_after), filesystem_service)
            return _delete_file(op.file_path, filesystem_service)
        else:
            # Use compressed_before when going backward
            if op.compressed_before:
                return _write_file(op.file_path, decompress(op.compressed_before), filesystem_service)
            return _delete_file(op.file_path, filesystem_service)
    elif op.operation_type in ["undo", "redo"]:
        # Skip undo/redo operations
        return (True, None)

    return (True, None)


def new_operation(state: AgentState, op: FileOperation) -> dict:
    """Add a new file operation.
    
    Note: Truncation of redo history is handled in tools_node before
    accumulation, since each tool reads from stale state.
    """
    print(f"DEBUG new_operation: Adding {op.operation_type}")

    return {
        "file_operations": [op],  # For undo/redo stack
        "file_operations_this_turn": [op],  # For time travel checkpoint
        "operation_position_delta": 1,  # Forward operation
    }
