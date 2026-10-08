"""Agent state definition and utilities.

Provides the AgentState TypedDict, field validation, and state update building.
These are shared between main.py and the tool node implementations.
"""

import operator
from functools import reduce
from typing import Annotated, Any, List, Optional

from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


def union_deleted_files(existing: frozenset, new: frozenset) -> frozenset:
    """Reducer for deleted_file_paths - only grows, never shrinks.
    
    Unions new paths into the existing set so that deletions are never
    forgotten even after undo/redo operations.
    """
    if not existing and not new:
        return frozenset()
    return existing | new


def replace_file_operations(existing: list, new_operations: list) -> list:
    """Simple replacement reducer - just return the new value.

    The tools_node is responsible for computing the complete final list.
    """
    return new_operations


class StateConfig:
    """Configuration for a state type, encapsulating field names, defaults, and list fields.
    
    Used to parameterize state validation and update building so that different
    agent types (e.g., main agent vs summarizer) can have different state schemas.
    """
    
    def __init__(self, fields: frozenset, defaults: dict, list_fields: frozenset, reset_fields: frozenset = None):
        self.fields = fields
        self.defaults = defaults
        self.list_fields = list_fields
        self.reset_fields = reset_fields or frozenset()
    
    def validate(self, result: dict, node_name: str) -> dict:
        """Validate that all state fields are present in the result.

        Raises AssertionError if any fields are missing or unexpected fields are present.
        This ensures we catch state field issues immediately rather than silently
        failing during checkpoint recovery.
        """
        result_keys = frozenset(result.keys())
        missing = self.fields - result_keys
        unexpected = result_keys - self.fields

        errors = []
        if missing:
            errors.append(f"Missing fields: {sorted(missing)}")
        if unexpected:
            errors.append(f"Unexpected fields: {sorted(unexpected)}")

        if errors:
            raise AssertionError(f"{node_name} state validation failed: {'; '.join(errors)}")

        return result
    
    def build_full_update(self, state: dict, updates: dict, node_name: str) -> dict:
        """Build a complete state update dict with all fields.

        Takes the current state and a dict of field updates, merges them,
        ensures all required fields are present, and validates the result.
        
        Updates for fields not in this config are silently ignored.

        Args:
            state: The current state dict
            updates: Dict of fields that are being modified
            node_name: Name of the node (for error messages)

        Returns:
            Complete state dict with all fields from this config
        """
        result = {}
        for field in self.fields:
            if field in updates:
                result[field] = updates[field]
            elif field in self.reset_fields:
                result[field] = self.defaults[field]
            else:
                result[field] = state.get(field, self.defaults[field])

        return self.validate(result, node_name)


class SummaryTransformer:
    """Transformer that replaces a block of summarized messages with a single AI summary.
    
    The summary is placed in an AIMessage with a prominent header indicating
    the range of messages it replaces.
    """
    
    def __init__(self, summary: str, start_index: int, end_index: int, message_index: int):
        """Initialize the summary transformer.
        
        Args:
            summary: The summary text
            start_index: Start index of the summarized block (inclusive)
            end_index: End index of the summarized block (inclusive)
            message_index: Index where the summary message will be placed
        """
        self.summary = summary
        self.start_index = start_index
        self.end_index = end_index
        self.message_index = message_index
        self._message = None
    
    def __call__(self, msg):
        """Return an AIMessage containing the summary with a prominent header.
        
        Args:
            msg: The original message at message_index (ignored)
            
        Returns:
            AIMessage with the summary and prominent header
        """
        from langchain_core.messages import AIMessage as _AIMessage
        
        if self._message is None:
            label = f"SUMMARY OF MESSAGES {self.start_index}\u2013{self.end_index}"
            border_top = "\u2550" * 64
            border_bottom = "\u2500" * 64
            content = f"{border_top}\n  {label}\n{border_top}\n{self.summary}\n{border_bottom}"
            self._message = _AIMessage(content=content)
        return self._message


class AgentState(TypedDict):
    """State for the agent."""
    messages: Annotated[list, add_messages]

    # FOR UNDO/REDO: Linear history on current branch
    file_operations: Annotated[list, replace_file_operations]  # List of FileOperation records
    operation_position: int  # Current position in file_operations (-1 = at start)

    # FOR CUMULATIVE DIFF STABILITY: Track when the diff actually changes
    cumulative_diff_last_op_pos: int  # operation_position when diff was last computed (-1 = initial)
    cumulative_diff_last_content: str  # Last rendered diff content for change detection ("" = initial)
    cumulative_diff_last_changed_ts: float  # Timestamp when diff last changed (0.0 = initial)
    cumulative_diff_creation_ts: float  # Timestamp when diff was first created (0.0 = initial)

    # FOR SCRUTINATOR: Track what has been scrutinized
    last_scrutinized_position: int  # Position of last scrutiny, -1 if never

    # FOR FILE VIEW WINDOWS: Persist view windows across checkpoints
    file_view_windows: dict  # file_path -> FileViewEntry dataclass

    # FOR SEARCH/LIST VIEW: Persist active search and file listing queries
    search_list_view: dict  # query_id -> SearchListViewEntry dataclass

    # FOR TIME TRAVEL: File operations performed at this checkpoint
    file_operations_this_turn: list  # List of FileOperation records for this checkpoint

    # Track file paths that had add_file_view called this turn (for close_view protection)
    file_views_added_this_turn: list  # List of normalized file paths

    # Turn/fork tracking for time travel
    turn_number: int  # Current turn number (increments at each HumanMessage)
    fork_number: int  # Current fork number (for debugging time travel)

    # Request limiting
    request_count: int  # Number of API requests made

    # Invalid tool call tracking - stores recent invalid tool call strings
    invalid_tool_calls_history: list  # List of recent invalid tool call strings

    # Track which tools were executed to know when to end the turn
    last_tools_executed: list  # List of tool names executed in the batch
    last_tool_success: bool  # Whether the last tool execution succeeded

    # Track last valid tool calls to detect duplicate calls
    last_valid_calls: Optional[list]  # List of (tool_name, args) tuples from last valid calls

    # Store send_message outputs for display in output files without sending to model
    send_message_outputs: Annotated[list, operator.add]  # List of dicts from send_message calls, each dict has content, tags, and format

    # Store conditional_report outputs for display in output files
    conditional_report_outputs: Annotated[list, operator.add]  # List of report strings from submit_report calls


    # Response metadata from API calls (for file_io_mode to write to output files)
    response_metadata: Optional[dict]  # Metadata from the last API response

    # Tool call omission transformers - pre-computed in tools_node, used in agent_node
    tool_call_omission_transformers: dict  # Dict[int, ToolCallOmissionTransformer] keyed by message index
    pending_tool_call_omission_transformer: Optional[Any]  # Just-computed transformer awaiting one-turn delay

    # Track consecutive responses with no tool attempt at all (for progressive escalation)
    consecutive_no_tool_count: int  # Number of consecutive responses without any tool usage attempt
    # Track consecutive max-reasoning-only responses (for escalation when truncated by token limit)
    consecutive_max_reasoning_count: int  # Number of consecutive responses that hit max_completion_tokens with only reasoning

    # Summarization tracking: durable record of summary ranges (survives checkpoint restore)
    summarized_ranges: Annotated[list, replace_file_operations]  # List of [start_index, end_index, summary_text] entries

    # Criterion 3 byte tracking: total content bytes of the last API request,
    # the actual view content bytes and cumulative diff bytes within it (both
    # subtracted from the total because neither is ever summarized), and the
    # state message count the request was built from. Stored in agent state so
    # a fork/time travel restores the value that belongs to the rewound
    # timeline (no node-instance field to reset, no request delay after a
    # fork), and so Criterion 3 can verify the data is not from a longer
    # (abandoned) message list.
    last_api_content_bytes: Optional[int]
    last_api_request_message_count: Optional[int]
    last_api_view_bytes: Optional[int]
    last_api_diff_bytes: Optional[int]

    # Context injection manager state (survives checkpoint restore)

    # FOR CREATE_FILE EXPERIMENT: Track files deleted during session
    deleted_file_paths: Annotated[frozenset, union_deleted_files]  # Paths of files deleted during this session
    context_injection_state: Optional[Any]  # Serialized ContextInjectionManager state

    # Per-tool-call pipeline state
    pending_tool_calls: list  # [{tool_name, args, tool_call_id}, ...] for per-tool-call pipeline
    tool_execution_context: Optional[Any]  # Execution context dict for per-tool-call pipeline


# All field names in AgentState - used for validation
AGENT_STATE_FIELDS = frozenset({
    "messages",
    "file_operations",
    "operation_position",
    "cumulative_diff_last_op_pos",
    "cumulative_diff_last_content",
    "cumulative_diff_last_changed_ts",
    "cumulative_diff_creation_ts",
    "last_scrutinized_position",
    "file_view_windows",
    "search_list_view",
    "file_operations_this_turn",
    "file_views_added_this_turn",
    "turn_number",
    "fork_number",
    "request_count",
    "invalid_tool_calls_history",
    "last_tools_executed",
    "last_tool_success",
    "last_valid_calls",
    "send_message_outputs",
    "conditional_report_outputs",
    "response_metadata",
    "consecutive_no_tool_count",
    "consecutive_max_reasoning_count",
    "tool_call_omission_transformers",
    "pending_tool_call_omission_transformer",
    "summarized_ranges",
    "last_api_content_bytes",
    "last_api_request_message_count",
    "last_api_view_bytes",
    "last_api_diff_bytes",
    "context_injection_state",
    "deleted_file_paths",
    "pending_tool_calls",
    "tool_execution_context",
})


AGENT_STATE_CONFIG = StateConfig(
    fields=AGENT_STATE_FIELDS,
    defaults={
        "messages": [],
        "file_operations": [],
        "operation_position": -1,
        "cumulative_diff_last_op_pos": -1,
        "cumulative_diff_last_content": "",
        "cumulative_diff_last_changed_ts": 0.0,
        "cumulative_diff_creation_ts": 0.0,
        "last_scrutinized_position": -1,
        "file_view_windows": {},
        "search_list_view": {},
        "file_operations_this_turn": [],
        "file_views_added_this_turn": [],
        "turn_number": 0,
        "fork_number": 0,
        "request_count": 0,
        "invalid_tool_calls_history": [],
        "last_tools_executed": [],
        "last_tool_success": False,
        "last_valid_calls": None,
        "send_message_outputs": [],
        "conditional_report_outputs": [],
        "response_metadata": None,
        "consecutive_no_tool_count": 0,
        "consecutive_max_reasoning_count": 0,
        "tool_call_omission_transformers": {},
        "pending_tool_call_omission_transformer": None,
        "summarized_ranges": [],
        "last_api_content_bytes": None,
        "last_api_request_message_count": None,
        "last_api_view_bytes": None,
        "last_api_diff_bytes": None,
        "context_injection_state": None,
        "deleted_file_paths": frozenset(),
        "pending_tool_calls": [],
        "tool_execution_context": None,
    },
    list_fields={"messages", "file_operations_this_turn", "file_views_added_this_turn", "send_message_outputs", "conditional_report_outputs", "deleted_file_paths"},
    reset_fields={"send_message_outputs", "conditional_report_outputs"},
)

