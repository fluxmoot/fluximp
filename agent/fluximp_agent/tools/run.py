from typing import List, Optional

from fluximp_agent.tools.tool_definitions import ToolResult
from fluximp_agent.tools.common import ToolError


def run_python_script_tool(script_path: str, args: List[str] = [], timeout: float = None, reason: str = "", **kwargs) -> ToolResult:
    """Execute a Python script and return its results.
    
    Executes a Python script using the ProcessService, redirecting stdout/stderr
    to a file in the .fluximp directory. Returns the exit code, output path, and
    last 10 lines of output.
    
    Before execution, the ProcessService runs scrutinator analysis on the current
    file changes. If security issues are found, they are included in the permission
    request to the user.
    
    Args:
        script_path: Path to the Python script (relative to working directory)
        args: List of command-line arguments to pass to the script
        timeout: Timeout in seconds (default: 10 seconds)
        reason: Required explanation for why the script needs to run
    
    Returns:
        ToolResult - with success status, content message, and state_updates (including scrutiny_complete)
    """
    process_service = kwargs.get('process_service')
    permission_service = kwargs.get('permission_service')
    state_service = kwargs.get('state_service')
    agent_service = kwargs.get('agent_service')

    if process_service is None:
        return ToolResult(success=False, content="Error: ProcessService not available")

    if permission_service is None:
        return ToolResult(success=False, content="Error: PermissionService not available")

    try:
        result = process_service.run_python_script(
            script_path, args, timeout, reason,
            permission_service=permission_service,
            state_service=state_service,
            agent_service=agent_service
        )
    except ToolError as e:
        return ToolResult(success=False, content=f"Error: {str(e)}")

    # Format output for the agent
    output_content = "\n".join(result.output_last_lines) if result.output_last_lines else "(empty output)"

    content = (
        f"Script execution complete.\n"
        f"Exit code: {result.exit_code}\n"
        f"Output file: {result.output_path}\n"
        f"Last 10 lines of output:\n{output_content}"
    )

    # Build scrutiny complete update to mark files as reviewed
    state_updates = state_service.build_scrutiny_complete()

    return ToolResult(success=True, content=content, state_updates=state_updates)



def run_tests_tool(reason: str, test_identity: Optional[str] = None, **kwargs) -> ToolResult:
    """Execute test runners and return results.
    
    Runs test runners using the ProcessService, sorted by cost (cheapest first).
    Stops on first failure and returns diagnostic information.
    
    Before execution, the ProcessService runs scrutinator analysis on the current
    file changes. If security issues are found, they are included in the permission
    request to the user.
    
    Args:
        reason: Required explanation for why tests need to run
        test_identity: Optional specific test to run (runs all if None)
    
    Returns:
        ToolResult - with success status, content message, and state_updates (including scrutiny_complete)
    """
    process_service = kwargs.get('process_service')
    permission_service = kwargs.get('permission_service')
    state_service = kwargs.get('state_service')
    agent_service = kwargs.get('agent_service')
    
    if process_service is None:
        return ToolResult(success=False, content="Error: ProcessService not available")
    
    if permission_service is None:
        return ToolResult(success=False, content="Error: PermissionService not available")
    
    try:
        result = process_service.run_tests(
            reason,
            test_identity=test_identity,
            permission_service=permission_service,
            state_service=state_service,
            agent_service=agent_service
        )
    except ToolError as e:
        return ToolResult(success=False, content=f"Error: {str(e)}")
        
    # Build state updates
    state_updates = state_service.build_scrutiny_complete()
    
    if result.success:
        return ToolResult(success=True, content="All tests passed", state_updates=state_updates)
    else:
        # Format diagnostic information for the model
        error_parts = ["Test failures:"]
        for diag in result.diagnostics:
            error_parts.append(f"\nTest identity: {diag.identity}")
            error_parts.append(f"  Message: {diag.message}")
            if diag.annotated_paths:
                error_parts.append(f"  Annotated paths:")
                for annotated_path in diag.annotated_paths:                    
                    error_parts.append(f"    Filesystem path: {annotated_path.path}")
                    error_parts.append(f"      Description: {annotated_path.description}")
                    if annotated_path.line_start is not None:
                        error_parts.append(f"      Start line: {annotated_path.line_start}")
                    if annotated_path.line_end is not None:
                        error_parts.append(f"      End line: {annotated_path.line_end}")
            if diag.additional_info:
                error_parts.append(f"  Additional info:")
                for key, value in diag.additional_info.items():
                    error_parts.append(f"  {key}: {value}")
            if diag.traceback:
                error_parts.append("Traceback:")
                error_parts.append(diag.traceback)
            error_parts.append(f"Test relevant block from redirected console output:")
            error_parts.append(diag.test_output if diag.test_output is not None else "")

        return ToolResult(success=False, content="\n".join(error_parts), state_updates=state_updates)
