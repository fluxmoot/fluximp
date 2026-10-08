#!/usr/bin/env python3
"""
ProcessService - Encapsulates Python script execution and output management.

This module provides a service class that executes Python scripts,
redirects their output to files in the .fluximp directory, and returns
execution results.
"""

import ast
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass
from fluximp_agent.tools.common import ToolError
from fluximp_protocol.user_agent.base import ToolPermissionService


@dataclass
class ProcessResult:
    """Result of a Python script execution.
    
    Attributes:
        exit_code: The process exit code (0 for success)
        output_path: Path to the output file containing stdout/stderr
        output_last_lines: Last 10 lines of the output file
        success: True if exit_code == 0
    """
    exit_code: int
    output_path: str
    output_last_lines: List[str]
    
    @property
    def success(self) -> bool:
        return self.exit_code == 0


@dataclass
class TestsRunResult:
    """Result of running tests.
    
    Attributes:
        success: True if all tests passed
        output_path: Path to the output file containing test output (optional)
        output_content: Test output content as string (optional)
        diagnostics: List of diagnostic information for failed tests
        runners_executed: List of runner names that were executed
    """
    success: bool
    output_path: Optional[str]
    output_content: Optional[str]
    diagnostics: List["TestDiagnostics"]
    runners_executed: List[str]


class ProcessService:
    """Service for executing Python scripts and managing their output.
    
    This class delegates path validation and directory management to
    FileSystemService, ensuring consistent security and state management.
    """
    
    def __init__(self, filesystem_service, plugin_manager: "TestRunnerPluginManager" = None, require_permission_for_new_scripts: bool = False):
        """Initialize the process service.
        
        Args:
            filesystem_service: FileSystemService instance for path operations
            plugin_manager: TestRunnerPluginManager for test runner plugins
            require_permission_for_new_scripts: If True, require permission for newly created scripts
        """
        self._filesystem_service = filesystem_service
        self._plugin_manager = plugin_manager
        self._require_permission_for_new_scripts = require_permission_for_new_scripts
        self._python_executable = sys.executable
    
    def run_python_script(
        self,
        script_path: str,
        args: List[str] = [],
        timeout: Optional[float] = None,
        reason: str = "",
        permission_service: ToolPermissionService = None,
        state_service=None,
        agent_service=None
    ) -> ProcessResult:
        """Execute a Python script and redirect its output.
        
        Args:
            script_path: Path to the Python script (relative to working dir)
            args: List of command-line arguments to pass to the script
            timeout: Optional timeout in seconds (default: 10 seconds)
            reason: Required explanation for why the script needs to run
            permission_service: Required ToolPermissionService instance for permission handling
            state_service: Optional StateService for getting file changes
            agent_service: Optional AgentService for running scrutinator
        
        Returns:
            ProcessResult with execution results
            
        Raises:
            ValueError: If script_path is invalid, not a .py file, or reason is empty
            FileNotFoundError: If script does not exist
            PermissionDeniedError: If permission is denied by user
        """
        print("DEBUG ProcessService.run_python_script")
        
        # Validate reason is provided
        if not reason:
            raise ToolError("'reason' parameter is required for run_python_script")
        
        # Validate permission_service is provided (required)
        if permission_service is None:
            raise ToolError("'permission_service' parameter is required for run_python_script")
        
        # Default timeout is 10 seconds
        if timeout is None:
            timeout = 10.0
        
        # Check if there's already a pending permission request (resume scenario)
        # If so, skip scrutinator to avoid duplicate analysis
        pending_permission = permission_service.has_pending_request("run_python_script")

        requires_permission = pending_permission
        
        # Request permission (permission_service is required)
        # Include scrutinator report in request if there are findings
        request_data = {
            "script_path": script_path,
            "args": args,
            "timeout": timeout,
            "reason": reason
        }
        
        # Run scrutinator analysis if services are available and no pending request
        print(f"DEBUG ProcessService: state_service={state_service}, agent_service={agent_service}")
        scrutinator_report = ""
        if not requires_permission:            
            # Get modified files and validate syntax before calling scrutinator
            modified_files = state_service.get_modified_files()
            print(f"DEBUG ProcessService: modified_files={modified_files}")
            
            syntax_errors = []
            for file_path in modified_files:
                if not file_path.endswith('.py'):
                    continue
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        ast.parse(f.read())
                except SyntaxError as e:
                    syntax_errors.append(f"{file_path}: {e.msg} (line {e.lineno})")
                except UnicodeDecodeError as e:
                    syntax_errors.append(f"{file_path}: Encoding error - {e}")
                except Exception as e:
                    syntax_errors.append(f"{file_path}: {type(e).__name__} - {e}")
            
            if syntax_errors:
                raise ToolError("Syntax validation failed:\n" + "\n".join(syntax_errors))
            else:
                script_is_new = state_service.is_newly_created_file(script_path)
                print(f"DEBUG ProcessService: script_is_new={script_is_new}")

                if script_is_new:
                    requires_permission = True
                    print(f"DEBUG ProcessService: script is newly created, skipping scrutinator")
                else:
                    diff = state_service.create_scrutiny_diff()
                    print(f"DEBUG ProcessService: diff length={len(diff) if diff else 0}")
                    if diff:
                        print(f"DEBUG ProcessService: Running scrutinator...")
                        scrutinator_report = agent_service.run_scrutinator(diff)
                        print(f"DEBUG ProcessService: scrutinator_report length={len(scrutinator_report)}, content preview: {scrutinator_report[:200] if scrutinator_report else 'EMPTY'}")
                        if scrutinator_report.strip():
                            requires_permission = True
                            if scrutinator_report.startswith("Error:"):
                                request_data["security_analysis_error"] = scrutinator_report
                            else:
                                request_data["security_analysis"] = scrutinator_report
        else:
            print(f"DEBUG ProcessService: Found pending permission request, skipping scrutinator")
        validated_path, error = self._filesystem_service.validate_path(script_path)
        if error:
            raise ToolError(f"Invalid script path: {error}")
        
        # Restrict to .py files only
        if validated_path.suffix.lower() != '.py':
            raise ToolError(f"Only .py files can be executed. Got: {validated_path.suffix}")
        
        # Check file exists
        if not validated_path.exists():
            raise ToolError(f"Script not found: {validated_path}")
        
        # Add scrutinator report or error to permission request
        # This ensures user knows if security analysis failed or found issues
        if scrutinator_report:
            if scrutinator_report.startswith("Error:"):
                # Include error message so user knows scrutinator couldn't run
                request_data["security_analysis_error"] = scrutinator_report
            elif scrutinator_report.strip() and "No security issues found" not in scrutinator_report:
                # Include actual security findings
                request_data["security_analysis"] = scrutinator_report
        
        if requires_permission:
            permission_result = permission_service.request_permission(
                tool_name="run_python_script",
                request_data=request_data
            )

            if not permission_result.approved:
                raise ToolError(
                    f"Permission denied for script execution. User explanation: {permission_result.user_explanation}"
                )
        else:
            print(f"DEBUG ProcessService: permission not required")

        # Generate unique output filename with nanosecond timestamp
        timestamp_ns = time.time_ns()
        output_filename = f"python_output_{timestamp_ns}.txt"
        
        # Get output path from FileSystemService
        output_path, error = self._filesystem_service.get_fluximp_output_path(output_filename)
        if error:
            raise ValueError(f"Invalid output path: {error}")
        
        # Build command: python <script> [args...]
        cmd = [self._python_executable, str(validated_path)] + args
        
        # Execute with output redirection to file.
        # The child writes bytes using its own stdio encoding, so the output
        # file may not be valid UTF-8. It is read back leniently via
        # _read_last_lines / get_all_views (errors='replace'), matching the
        # defensive pattern used by the read tools.
        try:
            with open(output_path, 'w', encoding='utf-8') as output_file:
                process = subprocess.Popen(
                    cmd,
                    stdout=output_file,
                    stderr=output_file,
                    cwd=str(self._filesystem_service.working_dir)
                )
                
                try:
                    exit_code = process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                    exit_code = -1
                
        except Exception as e:
            raise ToolError(f"Script execution failed: {str(e)}")
        
        # Read last 10 lines
        last_lines = self._read_last_lines(output_path, 10)
        
        return ProcessResult(
            exit_code=exit_code,
            output_path=str(output_path),
            output_last_lines=last_lines
        )
    
    def run_tests(
        self,
        reason: str,
        test_identity: Optional[str] = None,
        permission_service: ToolPermissionService = None,
        state_service=None,
        agent_service=None
    ) -> TestsRunResult:
        """Execute test runners and return results.
        
        Runs test runners in order of cost (cheapest first), stopping on first failure.
        
        Args:
            reason: Required explanation for why tests need to run
            test_identity: Optional specific test to run (runs all if None)
            permission_service: Required ToolPermissionService instance for permission handling
            state_service: Optional StateService for getting file changes
            agent_service: Optional AgentService for running scrutinator
        
        Returns:
            TestsRunResult with success status and diagnostics for failures
            
        Raises:
            ToolError: If permission is denied or execution fails
        """
        from fluximp_agent_sdk.test_runner import RunResult, TestDiagnostics

        if self._plugin_manager is None:
            raise ToolError("TestRunnerPluginManager not configured for this ProcessService")
        
        # Validate reason is provided
        if not reason:
            raise ToolError("'reason' parameter is required for run_tests")
        
        # Validate permission_service is provided (required)
        if permission_service is None:
            raise ToolError("'permission_service' parameter is required for run_tests")
        
        # Check if there's already a pending permission request (resume scenario)
        pending_permission = permission_service.has_pending_request("run_tests")

        # LLMs like to ignore the tool instructions and guess at test identies, so
        # make sure its valid before parsing and running the scrutinator.
        if test_identity:
            # Use get_plugin_for_identity to find the correct plugin
            target_plugin, single_test_identity = self._plugin_manager.get_plugin_for_identity(test_identity)
            if target_plugin is None:
                raise ToolError(
                    f"No plugin found for test identity '{test_identity}'. "
                    f"Expected format: 'plugin_name:test_name'. "
                    f"Available plugins: {[p.get_plugin_name() for p in self._plugin_manager.plugins]}"
                )
            
            try:
                runner = target_plugin.get_single_test_runner(single_test_identity)
            except Exception as e:
                raise ToolError(str(e))

        requires_permission = pending_permission

        # Request permission if needed
        request_data = {
            "reason": reason,
            "test_identity": test_identity,
        }
        
        # Run scrutinator analysis if services are available and no pending request
        scrutinator_report = ""
        if not requires_permission:
            # Get modified files and validate syntax before calling scrutinator
            modified_files = state_service.get_modified_files()
            
            syntax_errors = []
            for file_path in modified_files:
                if not file_path.endswith('.py'):
                    continue
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        ast.parse(f.read())
                except SyntaxError as e:
                    syntax_errors.append(f"{file_path}: {e.msg} (line {e.lineno})")
                except UnicodeDecodeError as e:
                    syntax_errors.append(f"{file_path}: Encoding error - {e}")
                except Exception as e:
                    syntax_errors.append(f"{file_path}: {type(e).__name__} - {e}")
            
            if syntax_errors:
                raise ToolError("Syntax validation failed:\n" + "\n".join(syntax_errors))
            else:
                # Check for new plugins
                new_plugins = self._check_for_new_plugins(state_service)
                request_data["new_plugins"] = new_plugins
                print(f"DEBUG ProcessService: new_plugins={new_plugins}")
                if new_plugins:
                    requires_permission = True
                else:
                    diff = state_service.create_scrutiny_diff()
                    if diff:
                        scrutinator_report = agent_service.run_scrutinator(diff)
                        if scrutinator_report.strip():
                            requires_permission = True
                            if scrutinator_report.startswith("Error:"):
                                request_data["security_analysis_error"] = scrutinator_report
                            else:
                                request_data["security_analysis"] = scrutinator_report
       
        if requires_permission:
            permission_result = permission_service.request_permission(
                tool_name="run_tests",
                request_data=request_data
            )

            if not permission_result.approved:
                raise ToolError(
                    f"Permission denied for test execution. User explanation: {permission_result.user_explanation}"
                )

        runner_and_plugins = self._plugin_manager.get_runners_with_priority_filtering()
        
        if not runner_and_plugins:
            print("DEBUG run_tests no test runners")
            raise ToolError(f"No test runners were found")
        
        # Generate output file for test results
        timestamp_ns = time.time_ns()
        output_filename = f"test_output_{timestamp_ns}.txt"
        output_path, error = self._filesystem_service.get_fluximp_output_path(output_filename)
        if error:
            raise ToolError(f"Invalid output path: {error}")
        
        # Run tests sequentially, cheapest first
        all_diagnostics = []
        runners_executed = []
        exit_code = 0  # Default to success
        
        if test_identity:
            runners_list = [runner]
            
            # Execute the runner
            exit_code, runner_output_path = self._execute_runner(runner, output_path)
            runners_executed.append(runner)
            
            if exit_code != 0:
                # Test failed - extract diagnostics
                run_result = RunResult(
                    process=None,  # Process already completed
                    output_path=runner_output_path,
                    output_content=None,
                    test_identity=test_identity,
                    runner=runner
                )
                diagnostics = target_plugin.extract_diagnostics(run_result)
                for diagnostic in diagnostics:
                    diagnostic.identity = ":".join((target_plugin.get_plugin_name(), diagnostic.identity))
                all_diagnostics.extend(diagnostics)

        if not test_identity or exit_code == 0:
            print(f"DEBUG Running all tests with {len(runner_and_plugins)} runners(s)...")
            # Run all tests from all plugins
            for runner, plugin in runner_and_plugins:
                # Execute the runner
                exit_code, runner_output_path = self._execute_runner(runner, output_path)
                runners_executed.append(runner)
                
                if exit_code != 0:
                    # Test failed - extract diagnostics
                    run_result = RunResult(
                        process=None,  # Process already completed
                        output_path=runner_output_path,
                        output_content=None,
                        test_identity=test_identity,
                        runner=runner
                    )
                    diagnostics = plugin.extract_diagnostics(run_result)
                    for diagnostic in diagnostics:
                        diagnostic.identity = ":".join((plugin.get_plugin_name(), diagnostic.identity))
                    all_diagnostics.extend(diagnostics)
                    
                    # Stop on first failure
                    break
        
        return TestsRunResult(
            success=len(all_diagnostics) == 0,
            output_path=str(output_path),
            output_content=None,
            diagnostics=all_diagnostics,
            runners_executed=[str(r) for r in runners_executed]
        )
    
    def _execute_runner(self, runner: "TestRunner", base_output_path: str) -> tuple:
        """Execute a test runner and return exit code and output path.
        
        Args:
            runner: TestRunner configuration
            base_output_path: Base path for output files
            
        Returns:
            Tuple of (exit_code, output_path)
        """
        # Generate unique output filename for this runner
        timestamp_ns = time.time_ns()
        output_filename = f"runner_output_{timestamp_ns}.txt"
        output_path, error = self._filesystem_service.get_fluximp_output_path(output_filename)
        if error:
            raise ToolError(f"Invalid output path: {error}")
        
        # Build command
        cmd = [runner.executable] + runner.args

        print("DEBUG _execute_runner", cmd)
        
        # Execute with output redirection.
        # The child writes bytes using its own stdio encoding, so the output
        # file may not be valid UTF-8. It is read back leniently via
        # _read_last_lines / get_all_views (errors='replace'), matching the
        # defensive pattern used by the read tools.
        try:
            with open(output_path, 'w', encoding='utf-8') as output_file:
                process = subprocess.Popen(
                    cmd,
                    stdout=output_file,
                    stderr=output_file,
                    cwd=str(self._filesystem_service.working_dir)
                )
                
                exit_code = process.wait()
                
        except Exception as e:
            raise ToolError(f"Test runner execution failed: {str(e)}")
        
        return exit_code, str(output_path)
    
    def _check_for_new_plugins(self, state_service) -> List[str]:
        """Check if any new plugin files have been added.
        
        Args:
            state_service: StateService for getting file changes
            
        Returns:
            List of new plugin file paths (only newly created files, not modified ones)
        """
        plugin_dirs = self._plugin_manager.plugin_directories
        modified_files = state_service.get_modified_files()
        
        new_plugins = []
        for file_path in modified_files:
            print("DEBUG _check_for_new_plugins", file_path)
            if not state_service.is_newly_created_file(file_path):
                print("DEBUG _check_for_new_plugins not newly created")
                continue
            if not file_path.endswith('.py'):
                print("DEBUG _check_for_new_plugins doesn't end with .py")
                continue
            resolved_file = (self._filesystem_service.working_dir / file_path).resolve()
            print("DEBUG _check_for_new_plugins resolved_dir", resolved_file)
            for plugin_dir in plugin_dirs:
                resolved_dir = Path(plugin_dir).resolve()
                print("DEBUG _check_for_new_plugins resolved_dir", resolved_dir)
                if resolved_file.is_relative_to(resolved_dir):
                    new_plugins.append(file_path)
        
        return new_plugins
    
    def _read_last_lines(self, path: Path, count: int = 10) -> List[str]:
        """Read the last N lines from a file.
        
        Args:
            path: Path to the file
            count: Number of lines to read
            
        Returns:
            List of the last N lines (or fewer if file is smaller)
        """
        if not path.exists():
            return []
        
        try:
            # errors='replace' as defense-in-depth: if a redirected output file
            # somehow still contains non-UTF-8 bytes, show replacement chars
            # rather than returning "empty output" to the model.
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
                return [line.rstrip('\n') for line in lines[-count:]]
        except Exception:
            return []
