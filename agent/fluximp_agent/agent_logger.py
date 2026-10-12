import sys
import json
import time
from datetime import datetime, timezone

from fluximp_agent.agent_config import AgentConfig
from fluximp_agent.fs_agent_state import AgentState


class AgentLogger:
    """Logger for agent interactions with full visibility."""

    def __init__(self, agent_config: AgentConfig, verbose: bool = True):
        self.agent_config = agent_config
        self.verbose = verbose
        self.start_time = None

    def start_session(self):
        """Start a new logging session."""
        self.start_time = time.time()
        self._log_header("AGENT SESSION STARTED")
        self._log(f"Timestamp: {datetime.now(timezone.utc).isoformat()}")
        self._log(f"Model: {self.agent_config.model_config.model}")
        self._log(f"API Endpoint: {self.agent_config.model_config.base_url}")
        self._log("")

    def log_system_prompt(self, prompt: str):
        """Log the system prompt being sent to the agent."""
        self._log_header("SYSTEM PROMPT")
        self._log(prompt)
        self._log("")

    def log_user_message(self, content: str):
        """Log a user message."""
        self._log_header("USER INPUT")
        self._log(content)
        self._log("")

    def log_assistant_message(self, response, metadata: dict):
        """Log an assistant message."""
        self._log_header("ASSISTANT RESPONSE")
        self._log(f"Metadata: {json.dumps(metadata, indent=2)}")
        self._log("")
        if hasattr(response, 'content') and response.content:
            self._log("response.content:")
            self._log(response.content)
        if hasattr(response, 'tool_calls') and response.tool_calls:
            self._log("")
            self._log("response.tool_calls:")
            for tool_call in response.tool_calls:
                self._log(tool_call)
        if hasattr(response, 'invalid_tool_calls') and response.invalid_tool_calls:
            self._log("")
            self._log("response.invalid_tool_calls:")
            for invalid_tool_call in response.invalid_tool_calls:
                self._log(invalid_tool_call)
        if hasattr(response, 'content_blocks'):
            # content_blocks maybe a property that generates the list every time its called
            content_blocks = response.content_blocks
            if content_blocks:
                self._log("")
                self._log("response.content_blocks:")
                for content_block in content_blocks:
                    self._log(content_block)
        if hasattr(response, 'additional_kwargs') and response.additional_kwargs:
            self._log("")
            self._log("response.additional_kwargs:")
            for k, v in response.additional_kwargs.items():
                self._log(f"{k}: {v}")
            if hasattr(response.additional_kwargs, 'reasoning_content') and response.additional_kwargs.reasoning_content:
                self._log("")
                self._log("response.additional_kwargs.reasoning_content:")
                self._log(response.additional_kwargs.reasoning_content)
        if hasattr(response, 'response_metadata') and response.response_metadata:
            self._log("")
            self._log("response.response_metadata:")
            for k, v in response.response_metadata.items():
                if k == "token_usage":
                    for k2, v2 in v.items():
                        if k2 == 'prompt_tokens_details' and v2 is not None:
                            self._log(" token_usage.prompt_tokens_details:")
                            for k3, v3 in v2.items():
                                self._log(f"  {k3}: {v3}")
                        elif k2 == 'completion_tokens_details' and v2 is not None:
                            self._log(" token_usage.completion_tokens_details:")
                            for k3, v3 in v2.items():
                                self._log(f"  {k3}: {v3}")
                        else:
                            self._log(f" token_usage.{k2}: {v2}")
                else:
                    self._log(f"{k}: {v}")
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            self._log("")
            self._log(f"response.usage_metadata: {response.usage_metadata}")
        self._log("")
        self._log(f"dir response: {dir(response)}")        
        self._log("")

    def log_tool_call(self, tool_name: str, args: dict, result: str):
        """Log a tool call and its result."""
        self._log_header(f"TOOL CALL: {tool_name}")
        self._log(f"Arguments: {json.dumps(args, indent=2)}")
        self._log("-" * 40)
        self._log("RESULT:")
        self._log(result[:2000] + "..." if len(result) > 2000 else result)
        self._log("")

    def log_messages(self, messages: list):
        self._log(f"Message count: {len(messages)}")
        # Inline per-message API content byte counts. The values are stashed on
        # each message by calculate_message_api_bytes (invoked from
        # AgentNode._build_api_messages on the final transformed messages, so
        # each message is serialized exactly once per request). Messages without
        # a stashed count (e.g., from one-shot agents) keep the plain line format
        # and are excluded from the running sum.
        running_bytes = 0
        has_api_byte_counts = False
        for i, msg in enumerate(messages):
            role = getattr(msg, 'type', 'unknown')
            content_blocks = msg.content_blocks
            api_content_bytes = getattr(msg, 'additional_kwargs', {}).get('api_content_bytes')
            if api_content_bytes is None:
                self._log(f"  [{i}] {role}: {len(content_blocks)}")
            else:
                has_api_byte_counts = True
                running_bytes += api_content_bytes
                self._log(f"  [{i}] {role}: {len(content_blocks)} blocks, {api_content_bytes:,} api bytes (running: {running_bytes:,})")
            for block in content_blocks:
                self._log(f"    {str(block)[:150]}")
        if has_api_byte_counts:
            self._log(f"Total API content bytes: {running_bytes:,}")

    def log_file_operations(self, file_operations: list):
        for file_operation in file_operations:
            self._log(f"  {file_operation.operation_type} {file_operation.file_path} {file_operation.timestamp} has_patch:{bool(file_operation.patch)} has_compressed_before:{bool(file_operation.compressed_before)} has_compressed_after:{bool(file_operation.compressed_after)} operation_position:{bool(file_operation.operation_position)} atomic_group_id:{bool(file_operation.atomic_group_id)})")

    def log_state(self, state):
        """Log the current agent state.
        
        Gracefully handles both full AgentState and minimal state types
        (e.g., SummarizerState) by using .get() for all field access.
        Fields not present in the state are simply skipped.
        """
        self._log_header("AGENT STATE")
        # Disable this for now; the message used in the request are identical is
        # most cases currently, so this is redundant at present.  We'll need to
        # manage the request messages for token efficiency though, so we'll want
        # add this back where they deviate more.
        #self.log_messages(state.get('messages', []))

        file_operations = state.get('file_operations')
        if file_operations:
            self._log("\nfile_operations:")
            self.log_file_operations(file_operations)
            self._log("")
        operation_position = state.get('operation_position')
        if operation_position is not None:
            self._log(f"operation_position: {operation_position}")            
        last_scrutinized_position = state.get('last_scrutinized_position')
        if last_scrutinized_position is not None:
            self._log(f"last_scrutinized_position: {last_scrutinized_position}")

        file_operations_this_turn = state.get('file_operations_this_turn')
        if file_operations_this_turn:
            self._log("\nfile_operations_this_turn:")
            self.log_file_operations(file_operations_this_turn)
            self._log("")

        turn_number = state.get('turn_number')
        if turn_number is not None:
            self._log(f"turn_number: {turn_number}")
        fork_number = state.get('fork_number')
        if fork_number is not None:
            self._log(f"fork_number: {fork_number}")

        request_count = state.get('request_count')
        if request_count is not None:
            self._log(f"request_count: {request_count}")

        consecutive_no_tool_count = state.get('consecutive_no_tool_count')
        if consecutive_no_tool_count is not None:
            self._log(f"consecutive_no_tool_count: {consecutive_no_tool_count}")
        consecutive_max_reasoning_count = state.get('consecutive_max_reasoning_count')
        if consecutive_max_reasoning_count is not None:
            self._log(f"consecutive_max_reasoning_count: {consecutive_max_reasoning_count}")
        self._log("")

    def log_graph_info(self, graph):
        """Log information about the LangGraph structure."""
        self._log_header("LANGGRAPH WORKFLOW INFO")
        self._log(f"Graph type: {type(graph).__name__}")
        if hasattr(graph, 'nodes'):
            self._log(f"Nodes: {list(graph.nodes.keys())}")
        if hasattr(graph, 'edges'):
            self._log(f"Edges: {graph.edges}")
        self._log("")

    def end_session(self):
        """End the logging session."""
        duration = time.time() - self.start_time if self.start_time else 0
        self._log_header("AGENT SESSION ENDED")
        self._log(f"Duration: {duration:.2f}s")
        self._log("")

    def _log_header(self, title: str):
        """Log a header."""
        line = "=" * 60
        if self.verbose:
            print(f"\n{line}")
            print(f"  {title}")
            print(f"{line}")

    def _log(self, message: str):
        """Log a message."""
        if self.verbose:
            print(message)

    def flush(self):
        sys.stdout.flush()
