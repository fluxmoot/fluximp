#!/usr/bin/env python3
"""
Test runner plugin base class and data structures.

This module defines the interface that all test runner plugins must implement.
Plugins should import from this module: `from fluximp_agent_sdk.test_runner import ...`
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional
import subprocess


@dataclass
class TestRunner:
    """Configuration for a test runner command.
    
    Attributes:
        executable: Path to the executable to run
        args: List of command-line arguments
        cost: Relative cost of running this runner (lower is cheaper)
              Set to None when running a single test (cost irrelevant)
        priority: Priority level for filtering (default 1)
                  Priority 1 and lower: always run
                  Priority 2 and higher: weighted by inverse priority, 20% randomly selected
                  Set to None when running a single test (priority irrelevant)
    """
    executable: str
    args: List[str]
    cost: Optional[float]
    priority: Optional[int] = 1

    def __str__(self):
        return self.executable+" "+" ".join(self.args)


@dataclass
class RunResult:
    """Result of a test runner execution.
    
    Attributes:
        process: The subprocess.Popen instance that was executed
        output_path: Path to the file containing redirected stdout/stderr (optional)
        output_content: Test output content as string (optional)
        test_identity: If running a specific test, its identity string; None for all tests
        runner: The TestRunner that was used
    """
    process: subprocess.Popen
    output_path: Optional[str]
    output_content: Optional[str]
    test_identity: Optional[str]
    runner: TestRunner


@dataclass
class AnnotatedPath:
    """A path with an associated description.
    
    Attributes:
        path: The file or directory path
        description: A description of what this path represents
        line_start: Optional starting line number (1-indexed)
        line_end: Optional ending line number (1-indexed)
    """
    path: str
    description: str
    line_start: Optional[int] = None
    line_end: Optional[int] = None


@dataclass
class TestDiagnostics:
    """Diagnostic information from a failed test.
    
    Attributes:
        identity: Test identity string
        message: Error message describing the failure
        annotated_paths: Optional list of paths with descriptions where the issue is located
        additional_info: Optional dict of additional diagnostic fields
        traceback: Optional full traceback string from the test output
        test_output: Optional test output block specific to the failed test
    """
    identity: str
    message: str
    annotated_paths: Optional[List[AnnotatedPath]] = None
    additional_info: Optional[Dict] = None
    traceback: Optional[str] = None
    test_output: Optional[str] = None


class TestRunnerPlugin(ABC):
    """Abstract base class for test runner plugins.
    
    Each plugin adapts a specific test runner to the common interface
    used by the run_tests tool. Plugins handle:
    - Providing the command to run tests
    - Generating commands for specific tests
    - Extracting diagnostic information from failures
    
    This is "documentation as code" - the plugin encapsulates knowledge
    about how to work with its test runner, preventing the need for
    the model to understand each runner's specifics.
    """
    
    def __init__(self, working_dir: Optional[str] = None, read_root_dir: Optional[str] = None):
        """Initialize the test runner plugin.
        
        Args:
            working_dir: The working directory for file operations, as specified
                        by the --working-dir CLI argument. Plugins can use this
                        to locate project files relative to the project root.
            read_root_dir: The root directory for read/query operations, as specified
                          by the --read-root-dir CLI argument. May be an ancestor of
                          working_dir, granting read access to a broader tree.
        """
        self._working_dir = working_dir
        self._read_root_dir = read_root_dir
    
    @abstractmethod
    def get_runners(self) -> List[TestRunner]:
        """Get the test runner configurations.
        
        Returns:
            List of TestRunner with executable, args, cost, and priority for running tests.
            Existing plugins should return a 1-element list.
        """
        pass
    
    @abstractmethod
    def get_single_test_runner(self, test_identity: str) -> TestRunner:
        """Get a runner for a specific test.
        
        Args:
            test_identity: Identity string of the test to run. Should be in format
                          "plugin_name:test_name" where plugin_name matches the
                          plugin's get_plugin_name() method.
            
        Returns:
            TestRunner configured to run just that test.
            The cost field should be None since it's irrelevant for single tests.
            
        Raises:
            ValueError: If the test_identity does not belong to this plugin
                       (e.g., has a different plugin prefix)
        """
        pass
    
    @abstractmethod
    def extract_diagnostics(self, result: RunResult) -> List[TestDiagnostics]:
        """Extract diagnostic information from a failed test run.
        
        Parses the output file and any other artifacts to produce
        actionable diagnostic information for the model.
        
        Args:
            result: The RunResult from executing the test runner
            
        Returns:
            List of TestDiagnostics, one for each failed test.
            Empty list if no failures found or if parsing fails.
        """
        pass
    
    def get_plugin_name(self) -> str:
        """Get the unique name of this plugin.
        
        This name is used as the prefix in test identities (e.g., "pytest:test_foo.py::test_bar").
        
        Returns:
            Unique plugin name string
        """
        return self.__class__.__name__
