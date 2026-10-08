#!/usr/bin/env python3
"""
LangGraph Runner - Encapsulates LangGraph-specific logic for agent execution.

Handles stream execution, checkpoint recovery, time travel, and
checkpoint verification. Events are written directly to the agent
protocol during stream processing so the coordinator never sees
LangGraph implementation details like chunks.

The runner wraps create_agent() from main.py (deferred import to avoid cycles).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from fluximp_protocol.user_agent.base import RequestLifecycleHooks, MaxRequestsExceeded, AgentProtocol
from fluximp_agent.tools_node import Services
from fluximp_agent.time_travel import find_checkpoint_for_turn, time_travel_to_checkpoint


@dataclass
class RecoveryResult:
    """Result from checkpoint recovery."""
    config: dict
    fork: int
    turn: int
    is_mid_turn: bool


@dataclass
class TimeTravelResult:
    """Result from time travel execution."""
    config: dict
    fork: int
    turn: int


class LangGraphRunner:
    """Encapsulates LangGraph-specific agent execution logic.
    
    Provides stream advancement so the coordinator never sees LangGraph
    implementation details like chunks or node names. Events are written
    directly to the agent protocol during stream processing.
    """
    
    def __init__(self, filesystem_service, agent_config, process_service, registry,
                 agent_protocol: AgentProtocol,
                 max_requests: int = 5, max_transient_retries: int = 8,
                 checkpoint_db_path: Optional[Path] = None,
                 state_service = None):
        self.filesystem_service = filesystem_service
        self.agent_config = agent_config
        self.process_service = process_service
        self.registry = registry
        self._agent_protocol = agent_protocol
        self.max_requests = max_requests
        self.max_transient_retries = max_transient_retries
        self.checkpoint_db_path = checkpoint_db_path
        self._state_service = state_service
        
        self.skip_llm_next = [False]
        self._has_pending_tool_calls_recovery = False
        self._request_count = 0
        self._work_completion_summary_sent = False
        self._services: Services = {"registry": registry}
        self._stream = None
        self._stream_exhausted = False
        self._agent = None
        self._memory = None
        self._thread_id = None
        self._last_response_metadata: Optional[dict] = None
        
        # Fork handlers list - callables invoked when a fork occurs
        self.fork_handlers: List[callable] = []
    
    def create(self, lifecycle_hooks: RequestLifecycleHooks = None):
        """Create the LangGraph agent and checkpointer.
        
        Args:
            lifecycle_hooks: Optional hooks for request lifecycle events
            
        Returns:
            Tuple of (compiled_graph, memory_checkpointer)
        """
        # Deferred import to avoid circular dependency
        from fluximp_agent.main import create_agent
        from fluximp_agent.fs_agent_node import AgentNode
        
        # thread_id_ref is a mutable list so agent_node can access the current thread_id
        # It gets populated by fresh_start() or recover_state()
        self._thread_id_ref = [self._thread_id]
        
        # Register agent_protocol in services dict for tools_node to write events directly
        self._services['agent_protocol'] = self._agent_protocol
        
        agent, memory, agent_node = create_agent(
            filesystem_service=self.filesystem_service,
            agent_config=self.agent_config,
            services=self._services,
            max_requests=self.max_requests,
            max_transient_retries=self.max_transient_retries,
            lifecycle_hooks=lifecycle_hooks,
            model_client=None,
            checkpoint_db_path=self.checkpoint_db_path,
            skip_llm_next=self.skip_llm_next,
            thread_id_ref=self._thread_id_ref
        )
        self._agent = agent
        self._memory = memory
        
        # Register agent_node's on_fork handler
        self.fork_handlers.append(agent_node.on_fork)
        print(f"DEBUG: Registered on_fork handler from create_agent")
        
        return agent, memory
    
    # --- Fresh Start ---
    
    def fresh_start(self) -> Tuple[dict, str]:
        """Start a fresh conversation with a new thread_id.
        
        Returns:
            Tuple of (config, thread_id)
        """
        thread_id = str(uuid.uuid4())
        self._thread_id = thread_id
        # Update the mutable reference so agent_node can access current thread_id
        if hasattr(self, '_thread_id_ref') and self._thread_id_ref is not None:
            self._thread_id_ref[0] = thread_id
        config = {"configurable": {"thread_id": thread_id}}
        return config, thread_id
    
    # --- Checkpoint Recovery ---
    
    def recover_state(self, memory, state_service) -> RecoveryResult:
        """Recover conversation state from checkpoints.
        
        Queries the saver for the most recent checkpoint and determines
        fork/turn position and whether this is a mid-turn recovery.
        
        Args:
            memory: The checkpointer instance
            state_service: StateService for updating state reference
            
        Returns:
            RecoveryResult with config, fork, turn, and is_mid_turn flag
        """
        # Query saver for thread_id from most recent checkpoint
        checkpoints = list(memory.list(None, limit=1))
        if not checkpoints:
            raise Exception("No checkpoints found in saver")
        
        thread_id = checkpoints[0].config["configurable"]["thread_id"]
        self._thread_id = thread_id
        # Update the mutable reference so agent_node can access current thread_id
        if hasattr(self, '_thread_id_ref') and self._thread_id_ref is not None:
            self._thread_id_ref[0] = thread_id
        print(f"  Recovered thread_id from saver: {thread_id}")
        
        # Always use the latest checkpoint from saver for recovery
        checkpoint_id = checkpoints[0].config["configurable"].get("checkpoint_id")
        print(f"  Using latest checkpoint: {checkpoint_id}")
        
        if checkpoint_id is None:
            print(f"  checkpoint_id is None")
            config = {"configurable": {"thread_id": thread_id}}
        else:
            config = {"configurable": {"thread_id": thread_id, "checkpoint_id": checkpoint_id}}
        
        checkpoint_tuple = memory.get_tuple(config)
        if hasattr(checkpoint_tuple, 'checkpoint'):
            checkpoint_data = checkpoint_tuple.checkpoint
        elif isinstance(checkpoint_tuple, tuple) and len(checkpoint_tuple) > 1:
            checkpoint_data = checkpoint_tuple[1]
        else:
            checkpoint_data = checkpoint_tuple
        
        if not checkpoint_data:
            raise Exception("No checkpoint_data to recover")
        
        channel_values = checkpoint_data["channel_values"]
        checkpoint_turn = channel_values["turn_number"]
        checkpoint_fork = channel_values["fork_number"]
        print(f"  Checkpoint contains: fork={checkpoint_fork}, turn={checkpoint_turn}")
        
        # Initialize StateService from checkpoint
        state_service.update_state_reference(channel_values)
        
        return RecoveryResult(
            config=config,
            fork=checkpoint_fork,
            turn=checkpoint_turn,
            is_mid_turn=False  # Caller determines this based on recovered_state
        )
    
    # --- Time Travel ---
    
    def execute_time_travel(self, memory, config, thread_id: str,
                            current_fork: int, current_turn: int,
                            file_turn: int,
                            filesystem_service, state_service) -> TimeTravelResult:
        """Execute time travel to a checkpoint from a previous turn.
        
        Finds the target checkpoint, restores filesystem state, updates config,
        and syncs StateService.
        
        Args:
            memory: The checkpointer instance
            config: Current config dict
            thread_id: Current thread ID
            current_fork: Current fork number
            current_turn: Current turn number
            file_turn: The turn to travel back to (from the input filename)
            filesystem_service: FileSystemService for filesystem restoration
            state_service: StateService for state sync
            
        Returns:
            TimeTravelResult with updated config, fork, and turn
        """
        # Find the checkpoint by walking backwards through parent_config chain
        target_checkpoint_id = find_checkpoint_for_turn(memory, config, current_turn - 1, file_turn)
        
        if target_checkpoint_id:
            print(f"  Found checkpoint: {target_checkpoint_id}")
            if file_turn == 0:
                raise Exception("ABORT: file turn is 0 no checkpoint expected")
        elif file_turn == 0:
            print(f"  File turn is 0 and no checkpoint found as expected")
        else:
            raise Exception("ABORT: could not find checkpoint and file not turn 0")
        
        # Restore filesystem state via time_travel_to_checkpoint
        success, message, num_reversed = time_travel_to_checkpoint(
            memory=memory,
            current_config=config,
            target_checkpoint_id=target_checkpoint_id,
            filesystem_service=filesystem_service,
        )
        print(f"  Time travel: {message}")
        
        if not success:
            raise Exception("ABORT: time travel failed")
        
        # Update config to branch from the target checkpoint
        if target_checkpoint_id:
            new_config = {"configurable": {"thread_id": thread_id, "checkpoint_id": target_checkpoint_id}}
        else:
            # When going back to turn 0, omit checkpoint_id to start fresh
            new_config = {"configurable": {"thread_id": thread_id}}
        print(f"  Updated config to branch from checkpoint: {target_checkpoint_id}")
        
        # Sync StateService to new checkpoint state
        if state_service is not None:
            checkpoint_tuple = memory.get_tuple(new_config)
            if checkpoint_tuple and hasattr(checkpoint_tuple, 'checkpoint'):
                checkpoint_data = checkpoint_tuple.checkpoint
            elif isinstance(checkpoint_tuple, tuple) and len(checkpoint_tuple) > 1:
                checkpoint_data = checkpoint_tuple[1]
            else:
                checkpoint_data = checkpoint_tuple
            if checkpoint_data and isinstance(checkpoint_data, dict):
                channel_values = checkpoint_data.get("channel_values", {})
                state_service.update_state_reference(channel_values)
                print(f"  Synced StateService to checkpoint state")
        
        # Invoke fork handlers after time travel
        print(f"DEBUG execute_time_travel: invoking {len(self.fork_handlers)} fork handlers")
        for handler in self.fork_handlers:
            handler()
        
        return TimeTravelResult(config=new_config, fork=None, turn=file_turn)
    
    # --- Mid-turn Recovery Detection ---
    
    def check_mid_turn_recovery(self, memory, config, turn: int,
                                is_forking: bool,
                                is_first_turn_after_recovery: bool,
                                recovered_position) -> bool:
        """Check if this is a mid-turn recovery situation.
        
        A mid-turn recovery occurs when the checkpoint already contains
        messages for the current turn (e.g., the AI responded with tool calls
        but the tool results were never written to the output file).
        
        Args:
            memory: The checkpointer instance
            config: Current config dict
            turn: Current turn number
            is_forking: Whether we are forking (time traveling)
            is_first_turn_after_recovery: Whether this is the first turn after recovery
            recovered_position: The recovered ForkTurn from ForkTurnState.recovery_position(), or None
            
        Returns:
            True if this is a mid-turn recovery
        """
        # When forking, we NEVER do mid-turn recovery - always start fresh
        if is_forking:
            return False
        if not is_first_turn_after_recovery:
            return False
        if not recovered_position:
            return False
        
        # Get checkpoint to check if it already contains messages for this turn
        checkpoint_tuple = memory.get_tuple(config)
        if hasattr(checkpoint_tuple, 'checkpoint'):
            checkpoint_data = checkpoint_tuple.checkpoint
        elif isinstance(checkpoint_tuple, tuple) and len(checkpoint_tuple) > 1:
            checkpoint_data = checkpoint_tuple[1]
        else:
            checkpoint_data = checkpoint_tuple
        
        if checkpoint_data and isinstance(checkpoint_data, dict):
            channel_values = checkpoint_data["channel_values"]
            checkpoint_turn = channel_values["turn_number"]
            messages = channel_values["messages"]
            # Mid-turn recovery: checkpoint has messages for current turn
            if checkpoint_turn == turn and len(messages) > 0:
                print(f"  Mid-turn recovery detected: checkpoint has {len(messages)} messages")
                
                # Check for pending tool calls in the checkpoint (per-tool-call pipeline)
                pending_tool_calls = channel_values.get('pending_tool_calls', [])
                if pending_tool_calls:
                    print(f"  Checkpoint has {len(pending_tool_calls)} pending tool calls - will resume execute_tool_call")
                    self._has_pending_tool_calls_recovery = True
                    return True
                
                # Only skip LLM if the last message is an AIMessage (i.e., the turn was
                # interrupted before the tools node could execute the pending tool calls).
                # If the last message is a ToolMessage, the tools already ran and we need
                # the LLM to process the results â so skip_llm_next should stay False.
                if isinstance(messages[-1], AIMessage):
                    self.skip_llm_next[0] = True
                    print(f"  Last message is AIMessage - will skip LLM to finish tool processing")
                else:
                    print(f"  Last message is {type(messages[-1]).__name__} - LLM will process tool results")
                return True
        
        return False
    
    # --- Stream Execution ---
    
    def start_stream(self, user_input: str, turn: int, fork: int,
                     config: dict, is_mid_turn_recovery: bool = False):
        """Start the agent stream for a turn.
        
        Args:
            user_input: The user's input message
            turn: Current turn number
            fork: Current fork number
            config: LangGraph config dict
            is_mid_turn_recovery: Whether this is a mid-turn recovery
        """
        self._stream_exhausted = False
        self._last_response_metadata = None
        self._work_completion_summary_sent = False
        
        if is_mid_turn_recovery:
            if getattr(self, '_has_pending_tool_calls_recovery', False):
                # Resume from checkpoint with pending tool calls - use stream(None)
                # to let LangGraph continue from the last checkpoint (execute_tool_call)
                # instead of starting a new graph invocation from the agent node.
                self._has_pending_tool_calls_recovery = False
                self._stream = self._agent.stream(
                    None,
                    config=config,
                    stream_mode='updates'
                )
            else:
                # skip_llm_next was already set (or not) by check_mid_turn_recovery
                # based on whether the last checkpoint message is an AIMessage
                # Resume from checkpoint - don't pass messages to preserve checkpoint state
                self._stream = self._agent.stream(
                    {'file_operations_this_turn': [],
                     'turn_number': turn,
                     'fork_number': fork},
                    config=config,
                    stream_mode='updates'
                )
        else:
            # Normal flow with new user message
            # Only include SystemMessage placeholder on first turn (turn 0)
            # On subsequent turns, just add the HumanMessage
            if turn == 0:
                new_messages = [SystemMessage(content=""), HumanMessage(content=user_input)]
            else:
                new_messages = [HumanMessage(content=user_input)]
            self._stream = self._agent.stream(
                {"messages": new_messages,
                 "file_operations_this_turn": [],
                 "turn_number": turn,
                 "fork_number": fork},
                config=config,
                stream_mode="updates"
            )
    
    def advance_stream(self) -> bool:
        """Advance the stream, writing events directly to the protocol.
        
        Internally consumes chunks until one produces output or the stream
        is exhausted. Empty chunks (e.g., agent node skip during mid-turn
        recovery) are silently consumed so the coordinator only sees
        meaningful progress and cancel checking is safe.
        
        Returns True if a chunk with output was processed, False if the stream is exhausted.
        """
        while not self._stream_exhausted:
            try:
                chunk = next(self._stream)
            except StopIteration:
                self._stream_exhausted = True
                return False
            
            self._process_chunk(chunk)
            
            # Check if this chunk had any substantive content that was written
            # to the protocol. Empty chunks (e.g., agent node skip during
            # mid-turn recovery) are silently consumed.
            has_content = any(
                bool(state_update.get("messages")) or 
                bool(state_update.get("send_message_outputs")) or
                bool(state_update.get("conditional_report_outputs"))
                for state_update in chunk.values()
            )
            if has_content:
                return True
        return False
    
    @property
    def stream_complete(self) -> bool:
        """Whether the stream has been fully consumed."""
        return self._stream_exhausted
    
    @property
    def request_count(self) -> int:
        """Current request count across the session."""
        return self._request_count
    
    @property
    def work_completion_summary_sent(self) -> bool:
        """Whether a work_completion_summary was sent during the current turn."""
        return self._work_completion_summary_sent
    
    
    # --- Internal: chunk processing ---
    
    def _process_chunk(self, chunk: dict):
        """Convert a LangGraph chunk dict into protocol writes and update internal state.
        
        A chunk is a dict with node name as key and state update as value.
        This method extracts state updates (request_count)
        and writes events directly to the agent protocol.
        """
        for node_name, state_update in chunk.items():
            # Track state (not emitted as events)
            if "request_count" in state_update:
                self._request_count = state_update["request_count"]
            
            # Capture metadata for this chunk
            chunk_metadata = None
            if "response_metadata" in state_update:
                chunk_metadata = state_update["response_metadata"]
                # Save for use as fallback on subsequent non-AIMessage events
                self._last_response_metadata = chunk_metadata
            
            # Normalize messages -> direct protocol writes
            for msg in state_update["messages"]:
                if isinstance(msg, AIMessage):
                    # Extract reasoning content from the LLM response
                    reasoning = msg.additional_kwargs.get("reasoning_content")
                    if not reasoning:
                        reasoning = msg.additional_kwargs.get("reasoning")

                    # Write response event immediately
                    print("DEBUG write response via agent_protocol")
                    self._agent_protocol.write_response(
                        msg.content,
                        reasoning_content=reasoning,
                        metadata=chunk_metadata or {}
                    )

                elif isinstance(msg, ToolMessage):
                    # Detect submit_report tool calls from conditional agents
                    # and write the report as a CONDITIONAL_REPORT message.
                    if getattr(msg, 'name', None) == "submit_report":
                        self._agent_protocol.write_conditional_report(msg.content or "")
                    # Already written by tools_node - skip
                    pass
                elif isinstance(msg, HumanMessage):
                    # Tool result messages already written by tools_node - skip
                    pass
            
            # send_message outputs
            if "send_message_outputs" in state_update:
                for msg in state_update["send_message_outputs"]:
                    tags = msg.get("tags", [])
                    # Detect work_completion_summary and write diff.patch BEFORE
                    # the output event so it's available when the GUI processes it
                    if "work_completion_summary" in tags:
                        self._work_completion_summary_sent = True
                        if self._state_service is not None:
                            diff_content = self._state_service.create_aggregate_diff()
                            diff_path = self._agent_protocol.io_path / "diff.patch"
                            diff_path.write_text(diff_content, encoding='utf-8')
                    self._agent_protocol.write_model_message_to_user(
                        msg["content"], tags=tags, format=msg["format"]
                    )
            
            # conditional_report outputs
            if "conditional_report_outputs" in state_update:
                for report in state_update["conditional_report_outputs"]:
                    self._agent_protocol.write_conditional_report(report)
            
    
    # --- Checkpoint Verification ---
    
    def verify_checkpoint(self, memory, thread_id: str,
                          fork: int, turn: int) -> Tuple[str, dict]:
        """Verify the latest checkpoint matches expected fork/turn.
        
        Args:
            memory: The checkpointer instance
            thread_id: Current thread ID
            fork: Expected fork number
            turn: Expected turn number
            
        Returns:
            Tuple of (checkpoint_id, updated_config)
            
        Raises:
            Exception: If fork or turn number mismatch
        """
        checkpoint_tuple = memory.get_tuple({"configurable": {"thread_id": thread_id}})
        if hasattr(checkpoint_tuple, 'checkpoint'):
            latest_checkpoint = checkpoint_tuple.checkpoint
        elif isinstance(checkpoint_tuple, tuple):
            latest_checkpoint = checkpoint_tuple[1]
        else:
            latest_checkpoint = checkpoint_tuple
        
        channel_values = latest_checkpoint["channel_values"]
        if channel_values["fork_number"] != fork or channel_values["turn_number"] != turn:
            raise Exception(
                f'ABORT: fork and/or turn number mismatch '
                f'{channel_values["fork_number"]} != {fork} or '
                f'{channel_values["turn_number"]} != {turn}'
            )
        
        config = {"configurable": {"thread_id": thread_id, "checkpoint_id": latest_checkpoint['id']}}
        return latest_checkpoint['id'], config
