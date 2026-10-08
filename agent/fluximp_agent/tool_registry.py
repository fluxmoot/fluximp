"""Tool registry for dynamic tool management.

Provides ToolRegistry class and create_tool_registry factory.
"""

import uuid
import json
from typing import Any, Dict, List, Optional, Set, Tuple
from pathlib import Path

from langchain_core.messages import HumanMessage, ToolMessage

from fluximp_agent.tool_func_parser import parse_tool_call, PartialMatch
from fluximp_agent.agent_config import AgentConfig, ToolDef
from fluximp_agent.tools.common import ToolError
from fluximp_agent.tools.tool_definitions import ToolResult
from fluximp_agent.tools.filesystem_read import ripgrep_search, read_file, find_files
from fluximp_agent.tools.filesystem_modify import delete_file_tool, write_file_tool, create_file_tool
from fluximp_agent.tools.edit_file_lines import edit_file_lines_tool
from fluximp_agent.tools.apply_patch import apply_patch_tool
from fluximp_agent.tools.run import run_python_script_tool, run_tests_tool
from fluximp_agent.tools.file_view import add_file_view, close_file_view, close_view
from fluximp_agent.tools.user import send_message
from fluximp_agent.tools.undo_redo import _execute_undo, _execute_redo
from fluximp_agent.filesystem_service import FileSystemService
from fluximp_agent.process_service import ProcessService
from fluximp_protocol.user_agent.base import ToolPermissionService
from fluximp_agent.state_service import StateService
from fluximp_agent.agent_service import AgentService
from fluximp_agent.experiments import ExperimentsService

from fluximp_agent.fs_agent_state import AgentState


class ToolRegistry:
    """Registry for managing dynamic tool sets.

    This class provides a clean mapping from tool names to their
    implementations, supporting dynamic registration
    and unregistration of tools with full metadata for both
    text-based and API protocol tool usage.
    """

    def __init__(self, filesystem_service: FileSystemService, agent_config: AgentConfig, process_service: ProcessService = None, state_service: StateService = None, agent_service: AgentService = None, experiments_service: ExperimentsService = None, **extra_services):
        self._tools: Dict[str, ToolDef] = {}
        self._filesystem_service = filesystem_service
        self._agent_config = agent_config
        self._process_service = process_service
        self._state_service = state_service
        self._agent_service = agent_service
        self._experiments_service = experiments_service
        self._extra_services = extra_services

        self._permission_service = None  # Set per-turn via set_permission_service()
        self._all_available_tools: Dict[str, ToolDef] = {}

    def set_permission_service(self, permission_service: ToolPermissionService):
        """Set the permission service for the current turn.

        Args:
            permission_service: ToolPermissionService instance (typically the FileIOAgentProtocol)
        """
        self._permission_service = permission_service

    def _generate_example(self, tool_def: ToolDef) -> str:
        """Generate a formatted example call for a tool."""
        from fluximp_agent.tools.tool_definitions import ARG_EXAMPLES

        all_args = tool_def.args
        if not all_args:
            return f"{tool_def.name}()"

        args = []
        for arg in all_args:
            value = ARG_EXAMPLES.get(arg.name, "value")
            if isinstance(value, list):
                args.append(f'{arg.name}={value}')
            elif isinstance(value, str):
                args.append(f'{arg.name}="{value}"')
            else:
                args.append(f'{arg.name}={value}')

        return f"{tool_def.name}({', '.join(args)})"

    def register_tool(self, tool_def: ToolDef):
        """Register a tool with full metadata.

        Args:
            tool_def: ToolDef namedtuple with complete tool metadata
        """
        self._tools[tool_def.name] = tool_def

    def unregister(self, name: str):
        """Remove a tool from the registry."""
        self._tools.pop(name, None)

    def get_tool(self, name: str) -> Optional[ToolDef]:
        """Get tool definition by name."""
        return self._tools.get(name)

    def get_example(self, tool_name: str) -> Optional[str]:
        """Get pre-generated example for a specific tool."""
        tool_def = self._tools.get(tool_name)
        if tool_def:
            return self._generate_example(tool_def)
        return None

    def get_all_examples(self) -> str:
        """Get all tool examples as a formatted string."""
        lines = ["Valid tool call examples:"]
        for tool_name in sorted(self._tools.keys()):
            example = self.get_example(tool_name)
            if example:
                lines.append(f"  {example}")
        return "\n".join(lines)

    def get_openai_schemas(self) -> List[Dict]:
        """Generate OpenAI function schemas for all registered tools.

        Returns:
            List of function schema dicts suitable for bind_tools()
        """
        schemas = []
        for tool_def in self._tools.values():
            schema = {
                "type": "function",
                "function": {
                    "name": tool_def.name,
                    "description": tool_def.description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            arg.name: {
                                "type": arg.arg_type,
                                "description": arg.description
                            }
                            for arg in tool_def.args
                        },
                        "required": [arg.name for arg in tool_def.args if arg.required]
                    }
                }
            }
            schemas.append(schema)
        print(schemas)
        return schemas

    def parse_tool_calls(self, content: str) -> Tuple[List[Dict], List[Tuple[str, Dict, Dict[str, Tuple[int, int]], Tuple[int, int]]], List[PartialMatch]]:
        """Extract all tool calls from AI message content using the unified parser.

        Uses the single consolidated parse_tool_call function for all tools.

        Args:
            content: AI message content to parse

        Returns:
            Tuple of (errors, valid_calls, partial_matches):
            - errors: List of error dicts with 'tool_name', 'error', 'raw_text' for fatal parsing errors
            - valid_calls: List of (tool_name, args_dict, arg_spans, call_span) tuples for fully parsed calls
              where arg_spans maps arg_name -> (start, end) in original content,
              and call_span is (start, end) of the entire tool call in original content.
            - partial_matches: List of PartialMatch objects for diagnostics

        Note: If errors is truthy, valid_calls will be empty. Errors indicate fatal
        parsing issues (e.g., unescaped quotes) that prevent reliable parsing.
        """
        # Pass self._tools directly (Dict[str, ToolDef]) to parse_tool_call
        # The parser will convert to simplified dict internally for backward compatibility
        return parse_tool_call(content, self._tools)

    def _coerce_arg_types(self, tool_def: ToolDef, args: Dict):
        """Validate and coerce argument types based on ArgDef metadata.
        
        Ensures all arguments match their declared arg_type, performing safe
        coercions (e.g., strâint) and raising ToolError for invalid types
        (e.g., dict for a string parameter).
        
        Args:
            tool_def: ToolDef with argument type declarations
            args: Mutable args dict to coerce in-place
            
        Raises:
            ToolError: If an argument cannot be coerced to the declared type
        """
        for arg_def in tool_def.args:
            if arg_def.name not in args:
                continue
            value = args[arg_def.name]
            if value is None:
                continue
            
            # Collection types that should never be coerced to primitives
            is_collection = isinstance(value, (dict, list, set, tuple))
            
            if arg_def.arg_type == "string":
                if isinstance(value, str):
                    continue
                elif is_collection:
                    raise ToolError(
                        f"Argument '{arg_def.name}' expects type 'string' but received "
                        f"{type(value).__name__} with value: {repr(value)[:200]}"
                    )
                else:
                    # Primitive-to-string coercion (int, float, bool)
                    args[arg_def.name] = str(value)
                    
            elif arg_def.arg_type == "integer":
                if isinstance(value, int) and not isinstance(value, bool):
                    continue
                elif is_collection:
                    raise ToolError(
                        f"Argument '{arg_def.name}' expects type 'integer' but received "
                        f"{type(value).__name__} with value: {repr(value)[:200]}"
                    )
                elif isinstance(value, bool):
                    args[arg_def.name] = int(value)
                elif isinstance(value, float):
                    args[arg_def.name] = int(value)
                elif isinstance(value, str):
                    try:
                        args[arg_def.name] = int(value)
                    except (ValueError, TypeError):
                        raise ToolError(
                            f"Argument '{arg_def.name}' expects type 'integer' but received "
                            f"string that could not be converted: {repr(value)[:200]}"
                        )
                else:
                    raise ToolError(
                        f"Argument '{arg_def.name}' expects type 'integer' but received "
                        f"{type(value).__name__} with value: {repr(value)[:200]}"
                    )
                    
            elif arg_def.arg_type == "boolean":
                if isinstance(value, bool):
                    continue
                elif is_collection:
                    raise ToolError(
                        f"Argument '{arg_def.name}' expects type 'boolean' but received "
                        f"{type(value).__name__} with value: {repr(value)[:200]}"
                    )
                elif isinstance(value, str):
                    if value.lower() in ("true", "1", "yes"):
                        args[arg_def.name] = True
                    elif value.lower() in ("false", "0", "no"):
                        args[arg_def.name] = False
                    else:
                        raise ToolError(
                            f"Argument '{arg_def.name}' expects type 'boolean' but received "
                            f"string that could not be converted: {repr(value)[:200]}"
                        )
                elif isinstance(value, int):
                    args[arg_def.name] = value != 0
                else:
                    raise ToolError(
                        f"Argument '{arg_def.name}' expects type 'boolean' but received "
                        f"{type(value).__name__} with value: {repr(value)[:200]}"
                    )
                    
            elif arg_def.arg_type == "array":
                if isinstance(value, list):
                    continue
                elif isinstance(value, (dict, set, tuple)):
                    raise ToolError(
                        f"Argument '{arg_def.name}' expects type 'array' but received "
                        f"{type(value).__name__} with value: {repr(value)[:200]}"
                    )
                elif isinstance(value, str):
                    try:
                        parsed = json.loads(value)
                        if isinstance(parsed, list):
                            args[arg_def.name] = parsed
                        else:
                            raise ToolError(
                                f"Argument '{arg_def.name}' expects type 'array' but JSON string "
                                f"parsed to {type(parsed).__name__}: {repr(value)[:200]}"
                            )
                    except json.JSONDecodeError:
                        raise ToolError(
                            f"Argument '{arg_def.name}' expects type 'array' but received "
                            f"string that could not be parsed as JSON: {repr(value)[:200]}"
                        )
                else:
                    # Single non-collection value â wrap in list
                    args[arg_def.name] = [value]

    def execute(self, name: str, args: Dict, state: AgentState, tool_call_id: str) -> Dict:
        """Execute a tool by name.

        Args:
            name: Tool name to execute
            args: Parsed arguments dict
            state: Current agent state
            tool_call_id: Optional tool call ID for API protocol (bind_tools)

        Returns:
            State update dict with messages and optional state changes
        """
        if name not in self._tools:
            tc_id = tool_call_id if tool_call_id else str(uuid.uuid4())
            return {"messages": [ToolMessage(tool_call_id=tc_id, content=f"Unknown tool: {name}",
                                              execution_successful=False,
                                              additional_kwargs={'human_prefix': 'Tool error:\n'})]}

        # Ensure filesystem service is in the proper state (sets OS CWD)
        self._filesystem_service.ensure_state()

        tool_def = self._tools[name]
        func = tool_def.func

        self._coerce_arg_types(tool_def, args)

        # Prepare tool kwargs with filesystem_service
        tool_kwargs = {"filesystem_service": self._filesystem_service}

        if self._process_service is not None:
            tool_kwargs["process_service"] = self._process_service

        if self._permission_service is not None:
            tool_kwargs["permission_service"] = self._permission_service

        if self._state_service is not None:
            tool_kwargs["state_service"] = self._state_service

        if self._agent_service is not None:
            tool_kwargs["agent_service"] = self._agent_service

        if self._experiments_service is not None:
            tool_kwargs["experiments_service"] = self._experiments_service

        # Inject any additional services (e.g., search provider services)
        for svc_name, svc in self._extra_services.items():
            if svc is not None:
                tool_kwargs[svc_name] = svc

        # undo and redo tools need state as a positional argument
        if name in ("undo", "redo"):
            result = func(state, **args, **tool_kwargs)
        else:
            result = func(**args, **tool_kwargs)

        if not result.success:
            # The only caller of execute expects tool errors to be raised as an exception
            # if the tool returns a ToolResult where success isn't true raise the exception
            # now.
            raise ToolError(result.content)

        if name == "submit_security_report":
            # submit_security_report returns ToolResult - on success store report in scrutinator_report_outputs
            if isinstance(result, ToolResult) and result.success:
                return {"messages": [], "scrutinator_report_outputs": [result.content]}

        if name == "submit_summary":
            # submit_summary returns ToolResult - on success store summary in summary_outputs
            if isinstance(result, ToolResult) and result.success:
                return {"messages": [], "summary_outputs": [result.content]}

        if name == "submit_report":
            # submit_report returns ToolResult - on success store report in conditional_report_outputs
            if isinstance(result, ToolResult) and result.success:
                return {"messages": [], "conditional_report_outputs": [result.content]}

        # Handle ToolResult return type - extract success, content, and state_updates
        if isinstance(result, ToolResult):
            state_update = dict(result.state_updates) if result.state_updates else {}
            content = result.content
            execution_successful = result.success

            if name == "send_message":
                # Merge tags and format from state_updates into send_message_outputs dict entries
                tags = state_update.get("send_message_tags", [])
                format = state_update.get("send_message_format", "plain_text")
                state_update["send_message_outputs"] = [{"content": content, "tags": tags, "format": format}]
                # Remove the individual fields so they don't pollute the state
                state_update.pop("send_message_tags", None)
                state_update.pop("send_message_format", None)
                content = "Send message completed successfully"

            tc_id = tool_call_id if tool_call_id else str(uuid.uuid4())
            state_update["messages"] = [ToolMessage(tool_call_id=tc_id, content=content,
                                                      execution_successful=execution_successful,
                                                      additional_kwargs={'human_prefix': 'Tool result:\n'})]

            return state_update

        # Should not reach here - all tools should return ToolResult
        raise ValueError(f"Tool '{name}' returned {type(result).__name__} instead of ToolResult.")

    def get_langchain_tools(self) -> List:
        """Get tools formatted for LangChain bind_tools().

        Returns:
            List of langchain_core.tools.StructuredTool objects
        """
        from langchain_core.tools import StructuredTool

        tools = []
        for tool_def in self._tools.values():
            # Build the JSON schema for parameters
            parameters = {
                "type": "object",
                "properties": {
                    arg.name: {
                        "type": arg.arg_type,
                        "description": arg.description
                    }
                    for arg in tool_def.args
                },
                "required": [arg.name for arg in tool_def.args if arg.required]
            }

            tool = StructuredTool(
                name=tool_def.name,
                description=tool_def.description,
                func=tool_def.func,
                args_schema=parameters
            )
            tools.append(tool)

        return tools

    @property
    def available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return list(self._tools.keys())

    @property
    def supported_tool_names(self) -> Set[str]:
        """Get the set of all tool names supported by the harness.

        This includes tools that are available to this agent as well as tools
        that are supported by the harness but not assigned to this agent.
        """
        return set(self._all_available_tools.keys())

    def parse_tool_calls_with_supported_tools(self, content: str) -> Tuple[List[Dict], List[Tuple[str, Dict, Dict[str, Tuple[int, int]], Tuple[int, int]]], List[PartialMatch]]:
        """Parse tool calls from content using all harness-supported tools.

        Unlike ``parse_tool_calls``, this uses the full set of supported tools
        so that callers can detect tool names the harness recognizes but which
        are not available to the current agent.

        Args:
            content: AI message content to parse

        Returns:
            Tuple of (errors, valid_calls, partial_matches) with the same shape
            as ``parse_tool_calls``.
        """
        return parse_tool_call(content, self._all_available_tools)

    @property
    def state_modifying_tools(self) -> Set[str]:
        """Get set of state-modifying tool names."""
        return {name for name, tool_def in self._tools.items() if tool_def.is_state_modifying}

# ============================================================================
# Tool Registry Factory
# ============================================================================
def create_tool_registry(filesystem_service: FileSystemService, agent_config: AgentConfig, process_service: ProcessService = None, state_service: StateService = None, agent_service: AgentService = None, experiments_service: ExperimentsService = None, **extra_services) -> ToolRegistry:
    """Create a ToolRegistry with configurable tools.

    Args:
        filesystem_service: The FileSystemService instance to inject into tool calls
        agent_config: The AgentConfig instance containing tool_names and configuration
        process_service: Optional ProcessService instance for script execution
        state_service: Optional StateService instance for accessing current state
        agent_service: Optional AgentService instance for running sub-agents
        experiments_service: Optional ExperimentsService for checking experiment status

    Returns:
        ToolRegistry with the requested tools registered

    Raises:
        ValueError: If any requested tool name is not available
    """
    registry = ToolRegistry(filesystem_service, agent_config, process_service, state_service, agent_service, experiments_service, **extra_services)

    # Import tool definitions from shared module
    from fluximp_agent.tools.tool_definitions import get_tool_definitions
    AVAILABLE_TOOLS = dict(get_tool_definitions(experiments_service))

    # Attach func references for tools implemented in tool_functions.py
    AVAILABLE_TOOLS["ripgrep_search"] = AVAILABLE_TOOLS["ripgrep_search"]._replace(func=ripgrep_search)
    AVAILABLE_TOOLS["read_file"] = AVAILABLE_TOOLS["read_file"]._replace(func=read_file)
    AVAILABLE_TOOLS["find_files"] = AVAILABLE_TOOLS["find_files"]._replace(func=find_files)
    AVAILABLE_TOOLS["send_message"] = AVAILABLE_TOOLS["send_message"]._replace(func=send_message)
    AVAILABLE_TOOLS["delete_file"] = AVAILABLE_TOOLS["delete_file"]._replace(func=delete_file_tool)
    AVAILABLE_TOOLS["write_file"] = AVAILABLE_TOOLS["write_file"]._replace(func=write_file_tool)
    
    # Conditionally assign create_file based on experiment
    if experiments_service and experiments_service.is_enabled("create_file"):
        AVAILABLE_TOOLS["create_file"] = AVAILABLE_TOOLS["create_file"]._replace(func=create_file_tool)
    
    # Conditionally assign edit_file_lines based on experiment
    if experiments_service and experiments_service.is_enabled("edit_line_validation_first_line_whitespace"):
        from fluximp_agent.tools.edit_file_lines import edit_file_lines_tool_validated_first_line_whitespace
        AVAILABLE_TOOLS["edit_file_lines"] = AVAILABLE_TOOLS["edit_file_lines"]._replace(func=edit_file_lines_tool_validated_first_line_whitespace)
    elif experiments_service and experiments_service.is_enabled("edit_line_validation"):
        from fluximp_agent.tools.edit_file_lines import edit_file_lines_tool_validated
        AVAILABLE_TOOLS["edit_file_lines"] = AVAILABLE_TOOLS["edit_file_lines"]._replace(func=edit_file_lines_tool_validated)
    else:
        AVAILABLE_TOOLS["edit_file_lines"] = AVAILABLE_TOOLS["edit_file_lines"]._replace(func=edit_file_lines_tool)
    
    # Conditionally assign insert_lines based on experiment
    if experiments_service and experiments_service.is_enabled("insert_line_validation_first_line_whitespace"):
        from fluximp_agent.tools.insert_lines_tool import insert_lines_tool_validated_first_line_whitespace
        AVAILABLE_TOOLS["insert_lines"] = AVAILABLE_TOOLS["insert_lines"]._replace(func=insert_lines_tool_validated_first_line_whitespace)
    elif experiments_service and experiments_service.is_enabled("insert_line_validation"):
        from fluximp_agent.tools.insert_lines_tool import insert_lines_tool_validated
        AVAILABLE_TOOLS["insert_lines"] = AVAILABLE_TOOLS["insert_lines"]._replace(func=insert_lines_tool_validated)
    elif experiments_service and experiments_service.is_enabled("insert_lines"):
        from fluximp_agent.tools.insert_lines_tool import insert_lines_tool
        AVAILABLE_TOOLS["insert_lines"] = AVAILABLE_TOOLS["insert_lines"]._replace(func=insert_lines_tool)
    
    # Conditionally assign move_lines based on experiment
    if experiments_service and experiments_service.is_enabled("move_line_validation"):
        from fluximp_agent.tools.move_lines_tool import move_lines_tool_validated
        AVAILABLE_TOOLS["move_lines"] = AVAILABLE_TOOLS["move_lines"]._replace(func=move_lines_tool_validated)
    elif experiments_service and experiments_service.is_enabled("move_lines"):
        from fluximp_agent.tools.move_lines_tool import move_lines_tool
        AVAILABLE_TOOLS["move_lines"] = AVAILABLE_TOOLS["move_lines"]._replace(func=move_lines_tool)
    
    # Conditionally assign adjust_indent based on experiment
    if experiments_service and experiments_service.is_enabled("adjust_indent_validation"):
        from fluximp_agent.tools.adjust_indent_tool import adjust_indent_tool_validated
        AVAILABLE_TOOLS["adjust_indent"] = AVAILABLE_TOOLS["adjust_indent"]._replace(func=adjust_indent_tool_validated)
    elif experiments_service and experiments_service.is_enabled("adjust_indent"):
        from fluximp_agent.tools.adjust_indent_tool import adjust_indent_tool
        AVAILABLE_TOOLS["adjust_indent"] = AVAILABLE_TOOLS["adjust_indent"]._replace(func=adjust_indent_tool)
    
    
    # Conditionally assign adjust_indent_exact based on experiment
    if experiments_service and experiments_service.is_enabled("adjust_indent_exact_validation"):
        from fluximp_agent.tools.adjust_indent_tool import adjust_indent_exact_tool_validated
        AVAILABLE_TOOLS["adjust_indent_exact"] = AVAILABLE_TOOLS["adjust_indent_exact"]._replace(func=adjust_indent_exact_tool_validated)
    elif experiments_service and experiments_service.is_enabled("adjust_indent_exact"):
        from fluximp_agent.tools.adjust_indent_tool import adjust_indent_exact_tool
        AVAILABLE_TOOLS["adjust_indent_exact"] = AVAILABLE_TOOLS["adjust_indent_exact"]._replace(func=adjust_indent_exact_tool)
    AVAILABLE_TOOLS["apply_patch"] = AVAILABLE_TOOLS["apply_patch"]._replace(func=apply_patch_tool)
    AVAILABLE_TOOLS["run_python_script"] = AVAILABLE_TOOLS["run_python_script"]._replace(func=run_python_script_tool)
    AVAILABLE_TOOLS["run_tests"] = AVAILABLE_TOOLS["run_tests"]._replace(func=run_tests_tool)
    AVAILABLE_TOOLS["add_file_view"] = AVAILABLE_TOOLS["add_file_view"]._replace(func=add_file_view)
    # Conditionally assign close tools based on close_views experiment
    if "close_file_view" in AVAILABLE_TOOLS:
        AVAILABLE_TOOLS["close_file_view"] = AVAILABLE_TOOLS["close_file_view"]._replace(func=close_file_view)
    if "close_view" in AVAILABLE_TOOLS:
        AVAILABLE_TOOLS["close_view"] = AVAILABLE_TOOLS["close_view"]._replace(func=close_view)

    # Conditionally assign tavily_search based on experiment
    if experiments_service and experiments_service.is_enabled("tavily_search"):
        from fluximp_agent.tools.search import tavily_search_tool
        AVAILABLE_TOOLS["tavily_search"] = AVAILABLE_TOOLS["tavily_search"]._replace(func=tavily_search_tool)

    # Conditionally assign tavily_extract based on experiment
    if experiments_service and experiments_service.is_enabled("tavily_extract"):
        from fluximp_agent.tools.search import tavily_extract_tool
        AVAILABLE_TOOLS["tavily_extract"] = AVAILABLE_TOOLS["tavily_extract"]._replace(func=tavily_extract_tool)


    # Conditionally assign exa_search based on experiment
    if experiments_service and experiments_service.is_enabled("exa_search"):
        from fluximp_agent.tools.search import exa_search_tool
        AVAILABLE_TOOLS["exa_search"] = AVAILABLE_TOOLS["exa_search"]._replace(func=exa_search_tool)

    # Conditionally assign tinyfish_fetch based on experiment
    if experiments_service and experiments_service.is_enabled("tinyfish_fetch"):
        from fluximp_agent.tools.search import tinyfish_fetch_tool
        AVAILABLE_TOOLS["tinyfish_fetch"] = AVAILABLE_TOOLS["tinyfish_fetch"]._replace(func=tinyfish_fetch_tool)

    # Conditionally assign exa_contents based on experiment
    if experiments_service and experiments_service.is_enabled("exa_contents"):
        from fluximp_agent.tools.search import exa_contents_tool
        AVAILABLE_TOOLS["exa_contents"] = AVAILABLE_TOOLS["exa_contents"]._replace(func=exa_contents_tool)

    # Conditionally assign tinyfish_search based on experiment
    if experiments_service and experiments_service.is_enabled("tinyfish_search"):
        from fluximp_agent.tools.search import tinyfish_search_tool
        AVAILABLE_TOOLS["tinyfish_search"] = AVAILABLE_TOOLS["tinyfish_search"]._replace(func=tinyfish_search_tool)

    # Conditionally assign linkup_search based on experiment
    if experiments_service and experiments_service.is_enabled("linkup_search"):
        from fluximp_agent.tools.search import linkup_search_tool
        AVAILABLE_TOOLS["linkup_search"] = AVAILABLE_TOOLS["linkup_search"]._replace(func=linkup_search_tool)

    # Conditionally assign linkup_fetch based on experiment
    if experiments_service and experiments_service.is_enabled("linkup_fetch"):
        from fluximp_agent.tools.search import linkup_fetch_tool
        AVAILABLE_TOOLS["linkup_fetch"] = AVAILABLE_TOOLS["linkup_fetch"]._replace(func=linkup_fetch_tool)

    # Conditionally assign parallel_search based on experiment
    if experiments_service and experiments_service.is_enabled("parallel_search"):
        from fluximp_agent.tools.search import parallel_search_tool
        AVAILABLE_TOOLS["parallel_search"] = AVAILABLE_TOOLS["parallel_search"]._replace(func=parallel_search_tool)

    # Conditionally assign parallel_extract based on experiment
    if experiments_service and experiments_service.is_enabled("parallel_extract"):
        from fluximp_agent.tools.search import parallel_extract_tool
        AVAILABLE_TOOLS["parallel_extract"] = AVAILABLE_TOOLS["parallel_extract"]._replace(func=parallel_extract_tool)

    # Conditionally assign perplexity_search based on experiment
    if experiments_service and experiments_service.is_enabled("perplexity_search"):
        from fluximp_agent.tools.search import perplexity_search_tool
        AVAILABLE_TOOLS["perplexity_search"] = AVAILABLE_TOOLS["perplexity_search"]._replace(func=perplexity_search_tool)

    # undo and redo - only assign func if the tool was defined (gated by experiment)
    if "undo" in AVAILABLE_TOOLS:
        AVAILABLE_TOOLS["undo"] = AVAILABLE_TOOLS["undo"]._replace(func=_execute_undo)
    if "redo" in AVAILABLE_TOOLS:
        AVAILABLE_TOOLS["redo"] = AVAILABLE_TOOLS["redo"]._replace(func=_execute_redo)

    # Get agent-specific tool definitions from the agent config
    # This allows agents like ScrutinatorMacaque to define their own tools
    agent_specific_tools = {}
    if hasattr(agent_config, 'get_tool_defintions'):
        for tool_def in agent_config.get_tool_defintions():
            agent_specific_tools[tool_def.name] = tool_def

    # Merge hardcoded tools with agent-specific tools
    # Agent-specific tools take precedence (can override defaults)
    all_available_tools = {**AVAILABLE_TOOLS, **agent_specific_tools}

    # Store the full set of supported tools so callers can detect attempts to
    # use real harness tools that are not registered for this agent.
    registry._all_available_tools = all_available_tools

    # Fail fast on invalid tool names
    requested = set(agent_config.tool_names)
    available = set(all_available_tools.keys())
    invalid = requested - available
    if invalid:
        raise ValueError(
            f"Invalid tool names in agent_config: {sorted(invalid)}. "
            f"Available tools: {sorted(available)}"
        )

    # Register only the requested tools
    for name in agent_config.tool_names:
        registry.register_tool(all_available_tools[name])

    return registry
