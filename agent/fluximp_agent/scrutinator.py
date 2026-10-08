import re
import operator
from dataclasses import dataclass, field
from typing import Annotated, List
from datetime import datetime, timezone

from fluximp_agent.model_config import ModelConfig, get_model_capabilities
from fluximp_agent.agent_config import AgentConfig, ArgDef, ToolDef
from fluximp_agent.tools.tool_definitions import ToolResult
from fluximp_agent.oneshot_agent import OneShotState


@dataclass
class ScrutinatorFinding:
    """A single security/dangerous operation finding from code diff analysis."""
    location: str|None  # Line number or range (e.g., "45", "15-25", "1,5,10")
    filepath: str|None  # Path to the file
    description: str|None  # Description of the issue


@dataclass
class ScrutinatorReport:
    """Report of security issues found in a code diff."""
    findings: List[ScrutinatorFinding] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

def _parse_finding(line: str) -> ScrutinatorFinding:
    """Parse a single finding from pipe-delimited format.
    
    Format: location|filepath|description
    
    Uses mechanical field detection since LLMs may provide fields in any order:
    - location: contains only digits, hyphens, and commas
    - filepath: contains path separators (/ or \) or file extensions
    - description: everything else (contains spaces, letters)
    """
    parts = line.split('|')
    if len(parts) < 3:
        # Not enough parts - assume order and fill in missing
        while len(parts) < 3:
            parts.append('')
    
    # Patterns for identification
    line_pattern = re.compile(r'^[\d,\-]+$')
    path_indicators = ['/', '\\', '.']
    
    location = filepath = description = None
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
            
        # Identify line info: only digits, hyphens, commas
        if line_pattern.match(part) and not location:
            location = part
        # Identify filepath: has path separators or extension
        elif any(c in part for c in path_indicators) and not filepath:
            filepath = part
        # Everything else is description
        elif not description:
            description = part
        else:
            # Append to description if we already have one
            description += ' ' + part
    
    return ScrutinatorFinding(
        location=location,
        filepath=filepath,
        description=description
    )


def submit_security_report(findings: str, **kwargs) -> ToolResult:
    # Validate format: each non-empty line must have at least 2 pipe separators
    lines = [line.strip() for line in findings.strip().split('\n') if line.strip()]
    
    for line in lines:
        if line.count('|') < 2:
            print("DEBUG submit_security_report received invalid format")
            return ToolResult(success=False, content=f"ERROR: Invalid format. Each finding must be in format: location|filepath|description\n\nInvalid line: {line[:100]}")
    
    parsed_findings = []
    for line in lines:
        parsed_findings.append(_parse_finding(line))
    
    # Format the report in human-readable format
    if not parsed_findings:
        return ToolResult(success=True, content="")
    
    count_issues = len(parsed_findings)
    report_lines = [f"Scrutinator Report: {count_issues} issue{'s' if count_issues > 1 else ''}"]
    
    for i, finding in enumerate(parsed_findings, 1):
        report_lines.append(f"\nIssue #: {i}")
        if finding.description:
            report_lines.append(f"Description: {finding.description}")
        if finding.filepath:
            report_lines.append(f"File: {finding.filepath}")
        if finding.location:
            report_lines.append(f"Lines: {finding.location}")
    
    return ToolResult(success=True, content="\n".join(report_lines))


class ScrutinatorMacaque(AgentConfig):
    """Communication only agent for inspecting code diffs for malicious and/or
    destructive impacts before the modified code is executed.
    """

    agent_type = "scrutinator"

    def __init__(self, model_config: ModelConfig):
        """Args:
            model_config: ModelConfig instance containing model/API settings.
        """
        caps = get_model_capabilities(model_config)
        tool_names = [
            "submit_security_report",
        ]
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
                name="submit_security_report",
                description="Submit your report of security issues found in the code diff. Once you use this tool, the conversation ends.",
                category="communication",
                args=[
                    ArgDef(
                        name="findings",
                        arg_type="string",
                        description="""It is required to use CSV format with the pipe character as the delimiter.

The CSV headers are:
- location: Line number or range (e.g., "45", "120-145", "1,5,10") but definitely not a code snippet
- filepath: Path to the file
- description: Description of the security issue

You do not need to include a header row, but you do need to have your fields ordered as location|filepath|description in each row.

Valid example:
45|src/sql.py|SQL injection via unsanitized user input
120-145|src/helper.py|Data exfiltration to external URL
200,205,210|src/main.py|Obfuscated code detected""",
                        tags=frozenset({'content'})
                    )
                ],
                func=submit_security_report,
                is_state_modifying=False
            )
        ]

    def get_system_prompt_tool_use_example_lines(self) -> list:
        return [
            "TOOL FUNCTION USAGE EXAMPLES",
            "",
            'submit_security_report(findings="""',
            '   2025|foo.py|Attempt to unmount drive',
            '   120-145|bar.py|code generation',
            '   """)',
        ]

    def get_system_prompt_preamble(self, **kwargs) -> str:
        return """Your job is to analyze a single code diff to find any dangerous operations that could harm the developer's local machine, any online resources accessible from it, cause a data breach.

The code diff has recent changes that will be included in the next execution of a process for a project on the developer's local machine.

Focus on these types of issues:
- Dynamic code generation
- File reading, deleting, or modification
- Directory deleting or renaming
- Creating filesystem names with directory escapes
- Searching the local system
- Windows registry usage of any kind
- Database schema changes that remove columns, drop tables, or delete all rows from a table
  - NoSQL changes that have an equivalent effect
- Resource exhaustion patterns
- Sending local data over the network
- Privilege escalation attempts
- Starting a subprocess or executing a shell command
- Signal handlers or systemic hooks
- Obfuscated or encoded code
- Unexpected binary data or hex-encoded payloads

Keep in mind this is about the developer's local machine.  Do not raise issue that are only a risk in production.  For example:
- SQL injection from user input (the only user is the developer)
- Unautheticated network requests that simply get data from the remote system (i.e. HTTP GET that isn't using HTTP headers or queries to upload data)

Also don't raise issues that are about local machine resource exhaustion.  The developer can simply kill the process if it runs too long or uses too much CPU or RAM.

If the diff has no issues you must use submit_security_report with no args (or at least an empty string for the findings arg).

If the diff has issues you must include all issues you found in one usage of the submit_security_report tool function.  Once you use submit_security_report the conversation ends.

The description you provide for each issue must clearly state what the risk is.  If there is no risk, there is no issue.

Another agent will investigate your findings. Your job is only to raise risks, not to do a comprehensive systemic security audit.
"""


# ---------------------------------------------------------------------------
# Scrutinator state
# ---------------------------------------------------------------------------

class ScrutinatorState(OneShotState):
    """Minimal state for the scrutinator agent.

    Extends OneShotState with scrutinator-specific output fields.
    """
    scrutinator_report_outputs: Annotated[list, operator.add]


def run_scrutinator_file_io(filesystem_service, model_config, io_dir,
                            max_requests=5, max_transient_retries=8,
                            input_timeout=120.0, permission_timeout=120.0):
    """Run the scrutinator agent in file-based I/O mode (subprocess entry point).

    Delegates to run_oneshot_file_io with scrutinator-specific configuration.
    See oneshot_agent.run_oneshot_file_io for full documentation.
    """
    from fluximp_agent.oneshot_agent import run_oneshot_file_io
    run_oneshot_file_io(
        filesystem_service=filesystem_service,
        model_config=model_config,
        io_dir=io_dir,
        agent_config=ScrutinatorMacaque(model_config),
        agent_name='SCRUTINATOR',
        required_tool_name='submit_security_report',
        state_class=ScrutinatorState,
        extra_output_keys=['scrutinator_report_outputs'],
        payload_type='scrutinator_report',
        max_requests=max_requests,
        max_transient_retries=max_transient_retries,
        input_timeout=input_timeout,
        permission_timeout=permission_timeout,
    )
