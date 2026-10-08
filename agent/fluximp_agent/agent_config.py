from typing import List, Callable, NamedTuple, Any, Optional

from fluximp_agent.model_config import ModelConfig

class AgentConfig:
    """Configuration for the coding agent.

    This class holds configuration that is consistent for the entire graph,
    outside of the AgentState. This includes model-specific settings for
    tool message handling and API protocol usage.
    """

    def __init__(self, model_config: ModelConfig, tool_names: List[str], use_tool_messages: bool = True, use_api_protocol_for_tool_functions: bool = False, use_tool_message_for_no_tool_error: bool = True, shared_tool_names: List[str] = None, max_tool_calls_per_response: Optional[int] = 20):
        """Initialize configuration.

        Args:
            model_config: ModelConfig instance containing model/API settings.
            tool_names: List of tool names to include in the registry. This defines
                        the capabilities available to the agent.
            use_tool_messages: Whether to use ToolMessage for tool results (True)
                              or format as HumanMessage (False).
                              Defaults to True. Set to False for models that
                              do not support tool messages.
            use_api_protocol_for_tool_functions: Whether to use the provider native API
                              tooling support (e.g., bind_tools) instead of text-based
                              tool parsing. Defaults to False. Set to True for models
                              that work better with native API protocol.
            use_tool_message_for_no_tool_error: Whether to use ToolMessage for
                              "no tool call" error responses. Defaults to True.
            shared_tool_names: Optional list of tool names shared across all agents.
                              When provided (and the two_part_system_prompt
                              experiment is active), tool instructions are split:
                              shared tools go in the first system message, and
                              agent-specific tools go in a follow-on system message.
            max_tool_calls_per_response: Maximum number of tool calls to process from
                              a single LLM response. Additional calls are skipped and
                              the agent is informed. Defaults to 20. Set to None to
                              disable truncation.
        """
        self.model_config = model_config
        self.tool_names = tool_names
        self.use_tool_messages = use_tool_messages
        self.use_api_protocol_for_tool_functions = use_api_protocol_for_tool_functions
        self.use_tool_message_for_no_tool_error = use_tool_message_for_no_tool_error
        self.include_arg_details_in_system_prompt = True
        self.shared_tool_names = shared_tool_names
        self.max_tool_calls_per_response = max_tool_calls_per_response
        # Populated during generate_system_prompt when two_part_system_prompt experiment is active
        self.agent_specific_system_content = None
        print("DEBUG self.use_api_protocol_for_tool_functions:", self.use_api_protocol_for_tool_functions)
        print("DEBUG self.use_tool_messages:", self.use_tool_messages)
        print("DEBUG self.use_tool_message_for_no_tool_error:", self.use_tool_message_for_no_tool_error)

    def generate_system_prompt(self, registry: "ToolRegistry", **kwargs) -> str:
        """Generate the system prompt dynamically based on tool registry and protocol.

        Args:
            registry: ToolRegistry containing all registered tools

        Returns:
            System prompt string
        """
        # API protocol: minimal instructions (tool definitions via bind_tools)
        if self.use_api_protocol_for_tool_functions:
            lines = [
                self.get_system_prompt_preamble(**kwargs),
                "",
                "Your responses must all include at least one valid tool function usage and an error will always be returned to when you fail to include a valid tool function usage.",
            ]
            return "\n".join(lines)

        # Text-based protocol: full format instructions and tool listings
        experiments_service = kwargs.get('experiments_service')
        two_part = experiments_service and experiments_service.is_enabled('two_part_system_prompt')

        if two_part:
            # Two-part: split content across shared and agent-specific messages
            shared_lines = [self._get_shared_preamble(**kwargs)]
            shared_lines.extend(self._get_format_instructions_lines())

            # Add tool instructions to shared message
            shared_tool_names = self.shared_tool_names if self.shared_tool_names else self.tool_names
            shared_instructions = self._generate_tool_instructions_for_names(registry, shared_tool_names)
            if shared_instructions:
                shared_lines.append("TOOL FUNCTIONS AVAILABLE FOR YOUR USE")
                shared_lines.append("")
                shared_lines.append(shared_instructions)

            shared_prompt = "\n".join(shared_lines)

            # Agent-specific system content (second message)
            agent_lines = ["DYNAMIC PORTION OF THE SYSTEM PROMPT\n"]

            # Agent-specific preamble
            agent_preamble = self._get_agent_specific_preamble(**kwargs)
            if agent_preamble:
                agent_lines.append(agent_preamble)

            # Agent-specific tool instructions (only when tools are split between messages)
            if self.shared_tool_names:
                agent_specific_tools = [t for t in self.tool_names if t not in self.shared_tool_names]
                if agent_specific_tools:
                    agent_tool_instructions = self._generate_tool_instructions_for_names(registry, agent_specific_tools)
                    if agent_tool_instructions:
                        agent_lines.append("")
                        agent_lines.append("ADDITIONAL TOOL FUNCTIONS AVAILABLE FOR YOUR USE")
                        agent_lines.append("")
                        agent_lines.append(agent_tool_instructions)

            # Tool use examples
            example_lines = self._get_tool_use_example_lines()
            if example_lines:
                agent_lines.append("")
                agent_lines.extend(example_lines)

            agent_content = "\n".join(agent_lines)
            self.agent_specific_system_content = agent_content if agent_content else None

            return shared_prompt
        else:
            # Single message: all content in one system prompt
            lines = [self.get_system_prompt_preamble(**kwargs)]
            lines.extend(self._get_format_instructions_lines())

            # Tool listings
            tool_instructions = self._generate_tool_instructions_for_names(registry, self.tool_names)
            if tool_instructions:
                lines.append("TOOL FUNCTIONS AVAILABLE FOR YOUR USE")
                lines.append("")
                lines.append(tool_instructions)

            # Tool use examples
            example_lines = self._get_tool_use_example_lines()
            if example_lines:
                lines.extend(example_lines)

            return "\n".join(lines)


    def _get_shared_preamble(self, **kwargs) -> str:
        """Return the shared preamble text for two_part_system_prompt experiment.

        Override in subclasses that need shared preamble content.
        Default implementation returns an empty string.
        """
        return ""

    def _get_agent_specific_preamble(self, **kwargs) -> str:
        """Return the agent-specific preamble text for two_part_system_prompt experiment.

        Override in subclasses to provide agent-specific instructions.
        Default implementation returns an empty string.
        """
        return ""


    def _get_format_instructions_lines(self) -> list:
        """Return lines describing the tool function usage format for text-based protocol."""
        return [
            "",
            "TOOL FUNCTION USAGE FORMAT",
            "",
            "The content text of your responses must all be tool function usages in the following format:",
            "- the name of the tool function followed by open and close parenthesis",
            "- one argument can be provided between the parenthesis as a name-value pair delimited the equals sign",
            "- multiple arguments can be provided as comma seperated nave-value pairs with each pair still delimited by the equal sign",
            "",
            "For string arguments you may provide double-quoted or single-quoted single-line strings.  You may provide multiline strings in triple quotes using either double quote or single quote.  You may embed types of quotes other than the one delimiting the string, but you must escape quotes of exactly the same type as used to delimit the string.  When triple quotes are used this means you must escape triple quotes within the string, but you can embed one or two consecutive characters of the same type of quote type.  Since backslash is used for escapes you have to use double backslash when you want an actual backslash included in the string after it is parsed (e.g. Windows file paths).",
            "",
            "The format just specified is the only valid format for using tools. Do not use XML, JSON, or similar formats as this will result in an invalid tool call response.",
            "",
            "This format is Python-like, but it is not Python.  The format is purely declartive.  You can only use the tools provded and you are not able to create your own tools.",
            "",
            # The below was added specifically for GLM 5.2 and its rather odd.  The first uses of it on siliconflow.com had everything seeming very similar to GLM 5.1 but then it all went to shit.  Switched to trying
            # it on chutes.ai and sometimes ran into some similar issues as what I saw when it went off the rails on siliconflow.com
            # Adding the first line helped some, but it still had issues.
            # "All of the content text included in your response will be parsed for tool uses, so avoid merely thinking about tool uses in your content text and only think about tool uses in your reasoning text.",
            # "",
            # "You will not see the results of tool uses until you complete your response.  The next request to you will include the results of any tool uses you are now writing.  Each paring of a request to you and your response is an internal transaction that will not include any new external content.",
            # "",
        ]

    def _get_tool_use_example_lines(self) -> list:
        """Return lines showing tool use examples, if available."""
        if hasattr(self, 'get_system_prompt_tool_use_example_lines'):
            example_lines = self.get_system_prompt_tool_use_example_lines()
            assert example_lines
            return example_lines
        return []

    def _generate_tool_instructions_for_names(self, registry, tool_names):
        """Generate text-based tool instructions for a subset of tools.

        Args:
            registry: ToolRegistry containing all registered tools
            tool_names: List of tool names to include (order preserved)

        Returns:
            Formatted tool instructions string, or empty string if no tools found
        """
        if not tool_names:
            return ""

        # Get tool defs for the requested names, preserving order
        tool_defs = []
        seen = set()
        for name in tool_names:
            if name in registry._tools and name not in seen:
                tool_defs.append(registry._tools[name])
                seen.add(name)

        if not tool_defs:
            return ""

        # Group by category (preserving order within each category)
        tools_by_category = {}
        category_order = []
        for tool_def in tool_defs:
            cat = tool_def.category
            if cat not in tools_by_category:
                tools_by_category[cat] = []
                category_order.append(cat)
            tools_by_category[cat].append(tool_def)

        lines = []
        tool_number = 0

        category_labels = {
            "read_only": "Read-only filesystem tool functions available to you:",
            "filesystem": "Filesystem modifying tool functions available to you:",
            "communication": "User communication tool functions available to you:",
            "process": "Process execution tool functions available to you:",
        }

        for category in category_order:
            cat_tools = tools_by_category[category]
            if not cat_tools:
                continue

            label = category_labels.get(category, f"{category.title()} tool functions available to you:")
            lines.append(label)
            for tool_def in cat_tools:
                tool_number += 1
                args_str = ", ".join([
                    arg.name if arg.required else f"{arg.name}={arg.default if arg.default is not None else 'None'}"
                    for arg in tool_def.args
                ])
                if self.include_arg_details_in_system_prompt:
                    lines.append(f"\n{tool_number}. {tool_def.name}({args_str}): {tool_def.description}")
                    for arg in tool_def.args:
                        lines.append(f"{arg.name}: {arg.arg_type} - {arg.description}")
                else:
                    lines.append(f"{tool_number}. {tool_def.name}({args_str}): {tool_def.description}")
            lines.append("")
        return "\n".join(lines)

# ============================================================================
# Tool Metadata Definitions
# ============================================================================
class ArgDef(NamedTuple):
    """Definition of a tool argument."""
    name: str
    arg_type: str  # "string", "integer", "boolean", "array"
    description: str
    required: bool = True
    default: Any = None
    tags: frozenset = frozenset()  # Semantic role tags, e.g. frozenset({"target"}), frozenset({"position", "content"})


class ToolDef(NamedTuple):
    """Definition of a tool including metadata for both text-based and API protocol usage."""
    name: str
    description: str
    category: str  # "read_only", "filesystem", "communication"
    args: List[ArgDef]
    func: Callable
    is_state_modifying: bool = False
