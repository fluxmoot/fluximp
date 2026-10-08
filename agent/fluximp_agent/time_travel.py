"""Time travel via LangGraph checkpoint history.

Provides functions for walking checkpoint chains to find and restore
earlier states:
- time_travel_to_checkpoint: reverse file operations back to a target checkpoint
- find_checkpoint_for_turn: locate a checkpoint ID for a given turn number
- create_checkpointer: create an InMemorySaver or SqliteSaver with PickleSerializer
"""

import os
import pickle
import sqlite3
from typing import Any, Optional, Tuple
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.serde.base import SerializerProtocol

from fluximp_agent.tools.file_operations import reverse_operation
from fluximp_agent.filesystem_service import FileSystemService



class SanitizedCheckpointError(Exception):
    """Placeholder exception replacing objects that could not be deserialized
    from a checkpoint.  Carries a human-readable summary so the error is
    still visible in logs / debug output."""


class PickleSerializer(SerializerProtocol):
    """Simple pickle-based serializer - fast but not human-readable."""

    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        data_ = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
        try:
            pickle.loads(data_)
        except:
            print("DEBUG pickle.loads failed in dumps_typed")
            if isinstance(obj, Exception):
                print("DEBUG trying to pickle SanitizedCheckpointError")
                data_ = pickle.dumps(SanitizedCheckpointError(str(obj)), protocol=pickle.HIGHEST_PROTOCOL)
                print("DEBUG data_ set to pickled SanitizedCheckpointError")
            else:
                print("DEBUG wasn't an exception data_ set to None")
                data_ = None
        assert isinstance(data_, bytes)
        return "pickle", data_

    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        type_, data_ = data
        if type_ == "pickle":
            try:
                return pickle.loads(data_)
            except Exception as e:
                raise Exception(f"DEBUG pickle.loads failed with {e} for data: {data_}")
        return data_

def create_checkpointer(checkpoint_db_path: Optional[Path] = None):
    """Create a checkpointer with PickleSerializer.

    Args:
        checkpoint_db_path: Optional path to SQLite database for persistent checkpoints.
                          If None, uses InMemorySaver (non-persistent).

    Returns:
        InMemorySaver or SqliteSaver instance configured with PickleSerializer
    """
    if checkpoint_db_path:
        checkpoint_db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(checkpoint_db_path), check_same_thread=False)
        memory = SqliteSaver(conn, serde=PickleSerializer())
        print(f"Using persistent checkpoints: {checkpoint_db_path}")
    else:
        memory = InMemorySaver(serde=PickleSerializer())
        print(f"Using in memory saver")
    return memory


# AgentState is a TypedDict defined in main.py - use Any to avoid circular import
AgentState = Any


def time_travel_to_checkpoint(
    memory: InMemorySaver,
    current_config: dict,
    target_checkpoint_id: str,
    filesystem_service,
    state: AgentState = None # reverse_operation takes this arg but doesn't actually do anything with it
) -> Tuple[bool, str, int]:
    """Walk checkpoints backwards from current to target and reverse operations.
    
    This function implements time travel by walking through LangGraph's checkpoint
    history using parent_config links, collecting file_operations_this_turn from
    each checkpoint, and reversing them to restore files to an earlier state.
    
    Args:
        memory: The InMemorySaver instance used by the agent
        current_config: The current checkpoint configuration dict
        target_checkpoint_id: The checkpoint ID to travel back to
        filesystem_service: The FileSystemService instance for file operations
        state: The current agent state
        
    Returns:
        Tuple of (success: bool, message: str, num_operations_reversed: int)
    """    
    operations_to_reverse = []
    checkpoint_tuple = memory.get_tuple(current_config)
    #print(f"DEBUG init checkpoint_tuple: {checkpoint_tuple}")
    steps = 0
    max_steps = 5000  # Safety limit
    found_target = False
    
    while checkpoint_tuple and steps < max_steps:
        # Extract checkpoint data
        if hasattr(checkpoint_tuple, 'checkpoint'):
            checkpoint = checkpoint_tuple.checkpoint
            parent_config = checkpoint_tuple.parent_config
        elif isinstance(checkpoint_tuple, tuple):
            # Handle tuple format
            checkpoint = checkpoint_tuple[1] if len(checkpoint_tuple) > 1 else None
            parent_config = checkpoint_tuple[2] if len(checkpoint_tuple) > 2 else None
        else:
            checkpoint = checkpoint_tuple
            print("DEBUG checkpoint.get_tuple(parent_config)")
            parent_config = checkpoint.get('parent_config') if isinstance(checkpoint, dict) else None
        
        if not checkpoint:
            raise Exception("ABORT: couldn't get checkpoint")
        
        # Check if we reached the target
        checkpoint_id = checkpoint.get('id') if isinstance(checkpoint, dict) else None
        if not checkpoint_id and target_checkpoint_id is not None:
            raise Exception("ABORT: couldn't get checkpoint_id")
        print("DEUBG time_travel_to_checkpoint checkpoint_id:", checkpoint_id)
        if checkpoint_id == target_checkpoint_id:
            print(f"DEBUG found target_checkpoint_id after {steps} steps")
            found_target = True
            break
        
        #print(checkpoint.keys())
        #print(checkpoint["channel_values"].keys())
        channel_values = checkpoint["channel_values"]
        turn_num = channel_values.get("turn_number")
        if turn_num is None:
            if target_checkpoint_id is None:
                print(f"DEBUG couldn't get turn_number from checkpoint, assuming we've gone back far enough")
                break
            else:
                turn_num = channel_values["turn_number"]
                raise Exception('second channel_values["turn_number"] did not raise as expected')
        fork_num = channel_values["fork_number"]
        ops_count = len(channel_values["file_operations_this_turn"])
        print(f"DEBUG checkpoint turn={turn_num} fork={fork_num} ops_count={ops_count}")
        operations_to_reverse.extend(reversed(channel_values["file_operations_this_turn"]))
        
        steps += 1
        
        # Move to parent checkpoint
        if parent_config:
            print("DEBUG memory.get_tuple(parent_config)", parent_config)
            checkpoint_tuple = memory.get_tuple(parent_config)
        else:
            break
    
    if target_checkpoint_id is not None and not found_target:
        raise Exception(f"ABORT: time_travel_to_checkpoint failed to find target checkpoint {target_checkpoint_id} within {max_steps} steps")
    print("DEBUG len(operations_to_reverse)", len(operations_to_reverse))

    num_reversed = 0
    print(f"DEBUG operations_to_reverse: {[(op.operation_type, op.file_path, op.timestamp) for op in operations_to_reverse]}")
    for op in operations_to_reverse:
        success, error = reverse_operation(op, state, filesystem_service)
        if not success:
            raise Exception(f"Failed to reverse {op.operation_type} on {op.file_path}: {error}")
        num_reversed += 1
        if os.path.exists(op.file_path):
            #print("DEBUG after reverse", Path(op.file_path).read_text(encoding="utf-8"))
            print("DEBUG file exists after reverse")
        else:
            print("DEBUG file doesn't exist after reverse")

    return True, f"Reversed {num_reversed} operations", num_reversed


def find_checkpoint_for_turn(memory, config, current_turn: int, target_turn: int) -> Optional[str]:
    """Walk checkpoints backwards to find one with target turn.

    Uses memory.get_tuple() and parent_config to walk the checkpoint chain,
    looking for checkpoint where channel_values["turn_number"] == target_turn.

    Args:
        memory: InMemorySaver instance
        config: Current checkpoint config
        current_turn: the turn of the current checkpoint
        target_turn: the turn of the checkpoint on the new branch

    Returns:
        Checkpoint ID if found, None otherwise
    """
    print(f"DEBUG find_checkpoint_for_turn current_turn: {current_turn}: target_turn: {target_turn}")
    if target_turn < 0:
        raise Exception("ABORT: target_turn must be at least 0")
    print("DEBUG memory.get_tuple(config)", config)
    checkpoint_tuple = memory.get_tuple(config)

    first_iteration = True

    while True:
        # Extract checkpoint data
        if hasattr(checkpoint_tuple, 'checkpoint'):
            checkpoint = checkpoint_tuple.checkpoint
            parent_config = checkpoint_tuple.parent_config
        elif isinstance(checkpoint_tuple, tuple):
            checkpoint = checkpoint_tuple[1] if len(checkpoint_tuple) > 1 else None
            parent_config = checkpoint_tuple[2] if len(checkpoint_tuple) > 2 else None
        else:
            checkpoint = checkpoint_tuple
            parent_config = checkpoint.get('parent_config') if isinstance(checkpoint, dict) else None

        if checkpoint is None:
            raise Exception(f"ABORT: couldn't find checkpoint {config} {checkpoint_tuple}")
        
        # Check if this checkpoint has the target turn_number
        channel_values = checkpoint.get("channel_values", {}) if isinstance(checkpoint, dict) else {}
        checkpoint_turn = channel_values.get("turn_number")
        checkpoint_fork = channel_values.get("fork_number")
        print(f"DEBUG checkpoint fork_number: {checkpoint_fork}, turn_number: {checkpoint_turn}, looking for: {target_turn}")

        if first_iteration:
            first_iteration = False
            if checkpoint_turn != current_turn:
                raise Exception(f"ABORT: first checkpoint_turn {checkpoint_turn} does not match current_turn {current_turn}")

        if checkpoint_turn == target_turn - 1:
            return checkpoint.get('id')

        if not parent_config:
            print(f"DEBUG parent config is falsy")
            return None

        print("DEBUG memory.get_tuple(parent_config)", parent_config)
        checkpoint_tuple = memory.get_tuple(parent_config)
