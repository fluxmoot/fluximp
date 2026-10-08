# Conditional agent infrastructure for Fluximp.
#
# Provides agent config classes and entry points for conditional agents:
# - ConditionalFilesystemAgent: For readonly_filesystem tool_set (uses filesystem agent infrastructure)
# - ConditionalReportOnlyAgent: For report_only tool_set (uses one-shot infrastructure)
# - submit_report: Generic tool for submitting conditional agent reports

import json
import operator
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Annotated, Dict, List, Optional

from fluximp_agent.model_config import ModelConfig, get_model_capabilities
from fluximp_agent.agent_config import AgentConfig, ArgDef, ToolDef
from fluximp_agent.tools.tool_definitions import ToolResult
from fluximp_agent.oneshot_agent import OneShotState

from fluximp_agent.fs_agent import FileSystemAgent
from fluximp_agent.fs_agents import _READONLY_TOOLS


# ============================================================================
# Conditional Agent Config Info (loaded from JSON, shared with GUI)
# ============================================================================

@dataclass
class ConditionalAgentConfigInfo:
    """Configuration for a conditional agent, loaded from a JSON file.
    
    This is the single source of truth for conditional agent configuration,
    shared between the backend (conditional_agent.py) and the GUI
    (workflow_config.py).
    """
    name: str
    prompt: str = ""                                        # System prompt / agent description
    tool_set: str = "report_only"                           # "report_only" or "readonly_filesystem"
    context_sources: List[str] = None                       # Ordered list of context inputs
    retrigger: bool = False                                 # True = re-run until empty output
    max_retriggers: int = 3                                 # Max retrigger cycles
    max_requests: int = 5                                  # Max API requests per run
    depends_on: Optional[str] = None                        # Name of prerequisite agent
    config_path: Optional[Path] = None                      # Path to the config file

    def __post_init__(self):
        if self.context_sources is None:
            self.context_sources = ["user_and_assistant_messages", "diff"]


def load_conditional_config(path: Path) -> ConditionalAgentConfigInfo:
    """Load a ConditionalAgentConfigInfo from a JSON file.
    
    Handles two JSON formats:
    1. Flat format (direct field keys): {"name": "...", "prompt": "...", ...}
    2. GUI format (wrapped in conditional_agents): {"conditional_agents": {"agent_name": {...}}}
    
    Unknown keys (e.g., legacy trigger_events, dependencies) are silently ignored.
    """
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Handle GUI format (wrapped in conditional_agents dict)
    if "conditional_agents" in data:
        agents = data["conditional_agents"]
        if len(agents) != 1:
            raise ValueError(f"Expected exactly one conditional agent in config file, got {len(agents)}")
        agent_name = next(iter(agents))
        agent_data = agents[agent_name]
        agent_data["name"] = agent_name
        data = agent_data
    # Filter out unknown keys (e.g., legacy trigger_events, dependencies)
    valid_fields = {f.name for f in fields(ConditionalAgentConfigInfo)}
    filtered = {k: v for k, v in data.items() if k in valid_fields}
    config = ConditionalAgentConfigInfo(**filtered)
    config.config_path = path
    return config


# ============================================================================
# submit_report tool function
# ============================================================================

def submit_report(report: str, **kwargs) -> ToolResult:
    """Submit a report. Empty string means nothing to report."""
    if not report.strip():
        return ToolResult(success=True, content="")
    return ToolResult(success=True, content=report)


# ============================================================================
# ConditionalFilesystemAgent (readonly_filesystem tool_set)
# ============================================================================

class ConditionalFilesystemAgent(FileSystemAgent):
    """Agent config for readonly_filesystem conditional agents.

    Inherits from FileSystemAgent to get view management, context injection,
    summarization, and tool use examples. Overrides preamble and examples
    to replace send_message with submit_report and remove agent switching note.
    """

    _SUBMIT_REPORT_EXAMPLE = [
        "submit_report usage example:",
        'submit_report(report="I found 3 instances of localhost in the files.")',
    ]

    _SUBMIT_REPORT_PREAMBLE = (
        "Free form text will not be displayed to the user.  The submit_report "
        "tool allows you to submit your report.  Using submit_report will end "
        "the conversation, so the submit_report tool must not be used to report "
        "incremental/partial progress."
    )

    def __init__(self, conditional_config, model_config, tool_names_to_drop=None):
        self.conditional_config = conditional_config
        base_tool_names = [t for t in _READONLY_TOOLS if t != "send_message"]
        base_tool_names.append("submit_report")
        super().__init__(
            agent_type=f"conditional_{conditional_config.name}",
            model_config=model_config,
            base_tool_names=base_tool_names,
            agent_description=conditional_config.prompt,
            tool_names_to_drop=tool_names_to_drop,
        )

    def _get_experiment_reminder_parts(self, flags: Dict[str, bool]) -> List[str]:
        """Build reminder text parts based on experiment flags.

        Takes the flags dict from _get_experiment_flags, returns a list of
        reminder strings.
        """
        parts = []
        # Currently there are no experiments that impact the readonly tools
        # available to conditional agents
        return parts

    def _get_shared_preamble(self, **kwargs):
        original = super()._get_shared_preamble()
        marker = "Free form text will not be displayed to the user."
        idx = original.find(marker)
        if idx >= 0:
            return original[:idx] + self._SUBMIT_REPORT_PREAMBLE
        return original.replace("send_message", "submit_report")

    def _get_category_derived_preamble(self):
        parts = []
        categories = self._get_tool_categories()
        # Always use no mutation, as the filesystem tools available to conditional
        # agents are all readonly.
        parts.append(self._NO_MUTATION_PREAMBLE)
        has_filesystem = "filesystem" in categories
        if has_filesystem:
            parts.append(
                "You have a limited number of response before you will be considered " 
                "to have failed in your task and not be called again. If you need to "
                "examine files you should definitely do so, but you need to be efficient"
                "and open the files you need to examine in bulk as soon as you identify "
                "them.  You will most likely fail if you only open one file per response "
                "repeatedly."
                )
        return "\n\n".join(parts)

    def get_system_prompt_tool_use_example_lines(self):
        lines = ["TOOL FUNCTION USAGE EXAMPLES", ""]
        available = set(self.tool_names)
        preferred = ["submit_report" if t == "send_message" else t
                     for t in self._PREFERRED_EXAMPLE_ORDER]
        for tool_name in preferred:
            if tool_name in available:
                if tool_name == "submit_report":
                    lines.extend(self._SUBMIT_REPORT_EXAMPLE)
                    lines.append("")
                elif tool_name in self._TOOL_EXAMPLES:
                    lines.extend(self._TOOL_EXAMPLES[tool_name])
                    lines.append("")
        return lines

    def get_context_chunks(self, state, **kwargs):
        chunks, state_update = super().get_context_chunks(state, **kwargs)
        from fluximp_agent.context_injection_manager import ContextChunk
        for i, chunk in enumerate(chunks):
            if chunk.section_id == 'reminders':
                modified_content = chunk.content.replace("send_message", "submit_report")
                modified_content = modified_content.replace(
                    "you must use the submit_report tool to communicate with the user",
                    "you must use the submit_report tool to submit your report")
                modified_content = modified_content.replace(
                    "Using submit_report will block you from taking further actions until the user responds",
                    "Using submit_report will end the conversation")
                modified_content = modified_content.replace(
                    "so do not use submit_report to tell the user what action your going to take as it will prevent you from taking the action",
                    "so do not use submit_report to report incremental/partial progress")
                chunks[i] = ContextChunk(
                    section_id=chunk.section_id,
                    content=modified_content,
                    stability_key=chunk.stability_key,
                    always_last=chunk.always_last,
                )
                break
        return chunks, state_update

    def get_tool_defintions(self):
        return [
            ToolDef(
                name="submit_report",
                description="Submit a report. Empty string means nothing to report. Once you use this tool, the conversation ends.",
                category="communication",
                args=[
                    ArgDef(
                        name="report",
                        arg_type="string",
                        description="The report content. Empty string means nothing to report.",
                        tags=frozenset({'content'})
                    )
                ],
                func=submit_report,
                is_state_modifying=False
            )
        ]


# ============================================================================
# ConditionalReportOnlyAgent (report_only tool_set)
# ============================================================================

class ConditionalReportOnlyAgent(AgentConfig):
    """Agent config for report_only conditional agents.

    Uses one-shot infrastructure with just the submit_report tool.
    """

    agent_type = "conditional_report_only"

    def __init__(self, conditional_config, model_config):
        self.conditional_config = conditional_config
        caps = get_model_capabilities(model_config)
        tool_names = ["submit_report"]
        super().__init__(
            model_config=model_config,
            tool_names=tool_names,
            use_tool_messages=caps.able_to_receive_tool_messages_at_all,
            use_api_protocol_for_tool_functions=not caps.able_to_use_python_like_protocol_for_tool_function,
            use_tool_message_for_no_tool_error=caps.able_to_recieve_tool_messages_for_no_tool_use_errors,
        )

    def get_tool_defintions(self):
        return [
            ToolDef(
                name="submit_report",
                description="Submit a report. Empty string means nothing to report. Once you use this tool, the conversation ends.",
                category="communication",
                args=[
                    ArgDef(
                        name="report",
                        arg_type="string",
                        description="The report content. Empty string means nothing to report.",
                        tags=frozenset({'content'})
                    )
                ],
                func=submit_report,
                is_state_modifying=False
            )
        ]

    def get_system_prompt_preamble(self, **kwargs):
        return self.conditional_config.prompt

    def get_system_prompt_tool_use_example_lines(self):
        return [
            "TOOL FUNCTION USAGE EXAMPLES",
            "",
            'submit_report(report="I found 3 instances of localhost in the files.")',
            "",
        ]


# ============================================================================
# ConditionalReportOnlyState
# ============================================================================

class ConditionalReportOnlyState(OneShotState):
    """State for the report_only conditional agent.

    Extends OneShotState with conditional_report_outputs field.
    """
    conditional_report_outputs: Annotated[list, operator.add]


# ============================================================================
# Entry point for report_only conditional agents
# ============================================================================

def run_conditional_report_only_file_io(filesystem_service, model_config,
                                         conditional_config, io_dir,
                                         max_requests=5, max_transient_retries=8,
                                         input_timeout=120.0, permission_timeout=120.0):
    """Run a report_only conditional agent in file-based I/O mode.

    Delegates to run_oneshot_file_io with conditional agent configuration.
    """
    from fluximp_agent.oneshot_agent import run_oneshot_file_io
    run_oneshot_file_io(
        filesystem_service=filesystem_service,
        model_config=model_config,
        io_dir=io_dir,
        agent_config=ConditionalReportOnlyAgent(conditional_config, model_config),
        agent_name='CONDITIONAL',
        required_tool_name='submit_report',
        state_class=ConditionalReportOnlyState,
        extra_output_keys=['conditional_report_outputs'],
        payload_type='conditional_report',
        max_requests=max_requests,
        max_transient_retries=max_transient_retries,
        input_timeout=input_timeout,
        permission_timeout=permission_timeout,
    )
