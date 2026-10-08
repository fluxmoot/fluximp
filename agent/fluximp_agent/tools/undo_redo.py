"""Undo and redo tool implementations.

These tools are registered in the ToolRegistry and called during agent execution
to reverse or re-apply file operations.
"""

from datetime import datetime, timezone

from fluximp_agent.tools.common import FileOperation
from fluximp_agent.tools.tool_definitions import ToolResult
from fluximp_agent.tools.file_operations import reverse_operation, apply_operation
from fluximp_agent.fs_agent_state import AgentState


def _execute_undo(state: AgentState, **kwargs) -> ToolResult:
    """Execute undo operation within the graph.
    
    When undoing, operations that share the same atomic_group_id are undone
    together as an atomic unit. This ensures that multi-file operations
    (e.g., cross-file move_lines) are undone atomically. Operations without
    an atomic_group_id are undone individually.
    """
    position = state.get("operation_position", -1)
    file_operations = state.get("file_operations", [])

    # Get services from kwargs (injected by ToolRegistry)
    filesystem_service = kwargs.get('filesystem_service')
    state_service = kwargs.get('state_service')

    print(f"DEBUG _execute_undo: position={position}, total_ops={len(file_operations)}")
    for i, op in enumerate(file_operations):
        print(f"DEBUG _execute_undo: file_operations[{i}] = {op.operation_type} on {op.file_path}")

    if position < 0:
        print(f"DEBUG _execute_undo: Nothing to undo (position < 0)")
        return ToolResult(success=False, content="Nothing to undo")

    op = file_operations[position]
    print(f"DEBUG _execute_undo: Attempting to undo operation at position {position}: {op.operation_type} on {op.file_path}")

    # Skip undo/redo operations - walk back to find the last real file operation
    while op.operation_type in ["undo", "redo"]:
        print(f"DEBUG _execute_undo: Skipping {op.operation_type} at position {position}")
        position -= 1
        if position < 0:
            print(f"DEBUG _execute_undo: Nothing to undo after skipping undo/redo ops")
            return ToolResult(success=False, content="Nothing to undo")
        op = file_operations[position]

    # Find all operations in the same atomic group
    group_ops = [op]
    
    if op.atomic_group_id is not None:
        # This operation is part of an atomic group - find all ops with the same group_id
        group_id = op.atomic_group_id
        group_ops = [gop for gop in file_operations 
                     if gop.atomic_group_id == group_id 
                     and gop.operation_type not in ["undo", "redo"]]
    # else: standalone operation, group_ops = [op]

    print(f"DEBUG _execute_undo: Undoing {len(group_ops)} atomic operations with group_id {op.atomic_group_id}")
    
    # Reverse all operations in the group
    for group_op in group_ops:
        print(f"DEBUG _execute_undo: Reversing {group_op.operation_type} on {group_op.file_path}")
        reverse_operation(group_op, state, filesystem_service)

    # Create undo operation record
    undo_op = FileOperation(
        operation_type="undo",
        timestamp=datetime.now(timezone.utc).isoformat(),
        file_path=op.file_path,
        operation_position=position
    )

    # The new position should be before the entire atomic group
    # Find the earliest position of any op in the group
    group_positions = [i for i, gop in enumerate(file_operations) 
                       if gop in group_ops and gop.operation_type not in ["undo", "redo"]]
    first_op_position = min(group_positions) if group_positions else position
    new_position = first_op_position - 1
    print(f"DEBUG _execute_undo: Created undo_op for {len(group_ops)} atomic ops, new position will be {new_position}")

    # Build state updates
    state_updates = {
        "file_operations_this_turn": [undo_op],  # track for time travel checkpoint
    }

    # Add state update from state_service
    if state_service is not None:
        # Handle undo past scrutiny point
        last_scrutinized = min(state_service.last_scrutinized_position, new_position)
        state_updates.update({
            "operation_position": new_position,
            "last_scrutinized_position": last_scrutinized,
        })

    if len(group_ops) == 1:
        content_msg = f"Undid {op.operation_type} on {op.file_path} file. The requested operation is complete."
    else:
        files_desc = ", ".join(f"{g_op.operation_type} on {g_op.file_path}" for g_op in group_ops)
        content_msg = f"Undid {len(group_ops)} atomic operations: {files_desc}. The requested operation is complete."

    return ToolResult(
        success=True,
        content=content_msg,
        state_updates=state_updates
    )

def _execute_redo(state: AgentState, **kwargs) -> ToolResult:
    """Execute redo operation within the graph.
    
    When redoing, operations that share the same atomic_group_id are redone
    together as an atomic unit. This ensures that multi-file operations
    (e.g., cross-file move_lines) are redone atomically. Operations without
    an atomic_group_id are redone individually.
    """
    position = state.get("operation_position", -1)
    file_operations = state.get("file_operations", [])

    # Get services from kwargs (injected by ToolRegistry)
    filesystem_service = kwargs.get('filesystem_service')
    state_service = kwargs.get('state_service')

    if position >= len(file_operations) - 1:
        return ToolResult(success=False, content="Nothing to redo")

    position += 1
    op = file_operations[position]

    # Skip undo/redo operations
    if op.operation_type in ["undo", "redo"]:
        return ToolResult(success=False, content="Nothing to redo")

    # Find all operations in the same atomic group
    group_ops = [op]
    
    if op.atomic_group_id is not None:
        # This operation is part of an atomic group - find all ops with the same group_id
        group_id = op.atomic_group_id
        group_ops = [gop for gop in file_operations 
                     if gop.atomic_group_id == group_id 
                     and gop.operation_type not in ["undo", "redo"]]
    # else: standalone operation, group_ops = [op]

    # Apply all operations in the group forward
    for group_op in group_ops:
        apply_operation(group_op, "forward", filesystem_service)

    # Create redo operation record
    redo_op = FileOperation(
        operation_type="redo",
        timestamp=datetime.now(timezone.utc).isoformat(),
        file_path=op.file_path,
        operation_position=position
    )

    # The new position should be at the end of the entire atomic group
    # Find the latest position of any op in the group
    group_positions = [i for i, gop in enumerate(file_operations) 
                       if gop in group_ops and gop.operation_type not in ["undo", "redo"]]
    last_op_position = max(group_positions) if group_positions else position
    new_position = last_op_position

    # Build state updates
    state_updates = {
        "file_operations_this_turn": [redo_op],  # track for time travel checkpoint
    }

    # Add state update from state_service
    if state_service is not None:
        state_updates.update({
            "operation_position": new_position,
        })

    if len(group_ops) == 1:
        content_msg = f"Redid {op.operation_type} on {op.file_path} file. The requested operation is complete."
    else:
        files_desc = ", ".join(f"{g_op.operation_type} on {g_op.file_path}" for g_op in group_ops)
        content_msg = f"Redid {len(group_ops)} atomic operations: {files_desc}. The requested operation is complete."

    return ToolResult(
        success=True,
        content=content_msg,
        state_updates=state_updates
    )
