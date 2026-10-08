#!/usr/bin/env python3
"""
TestRunnerPluginManager - Manages test runner plugins.

Dynamically loads plugins from a directory and provides access to them.
This is part of the harness (not the SDK or plugins).
"""

import importlib.util
import random
import numpy
import numpy.random
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Type

try:
    from fluximp_agent_sdk.test_runner import TestRunnerPlugin, TestRunner
    SDK_AVAILABLE = True
except ImportError:
    TestRunnerPlugin = None
    TestRunner = None
    SDK_AVAILABLE = False
from fluximp_agent.tools.common import ToolError


class TestRunnerPluginManager:
    """Manager for test runner plugins.
    
    Loads all .py files from a plugin directory and discovers
    TestRunnerPlugin subclasses. Provides access to loaded plugins
    and tracks plugin directories for permission checking.
    """
    
    def __init__(self, plugin_directory: str, working_dir: Optional[str] = None, read_root_dir: Optional[str] = None):
        """Initialize the plugin manager.
        
        Args:
            plugin_directory: Path to directory containing plugin .py files
            working_dir: The working directory from --working-dir CLI arg
            read_root_dir: The read root directory from --read-root-dir CLI arg
        """
        self._plugin_directory = Path(plugin_directory)
        self._working_dir = working_dir
        self._read_root_dir = read_root_dir
        self._plugins: List[TestRunnerPlugin] = []
    
    def _load_plugins(self) -> None:
        """Load all plugins from the plugin directory."""
        if not SDK_AVAILABLE:
            raise ImportError(
                "fluximp-agent-sdk is not installed; test runner plugins "
                "cannot be loaded."
            )
        if not self._plugin_directory.exists():
            print(f"DEBUG TestRunnerPluginManager: Plugin directory does not exist: {self._plugin_directory}")
            return
        
        plugin_library_files = []
        plugin_files = []
        for f in self._plugin_directory.iterdir():
            if f.is_file() and f.suffix == '.py' and f.name != '__init__.py':
                if str(f.stem)[0] == '_':
                    plugin_library_files.append(f)
                else: 
                    plugin_files.append(f)
        print(f"DEBUG TestRunnerPluginManager: Found {len(plugin_files)} potential plugin files")

        self._plugins = []
        
        for plugin_library_file in plugin_library_files:
            module_name = plugin_library_file.stem
            spec = importlib.util.spec_from_file_location(module_name, plugin_library_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            sys.modules[module_name] = module            

        for plugin_file in plugin_files:
            self._load_plugin_file(plugin_file)
    
    def _load_plugin_file(self, plugin_file: Path) -> None:
        """Load a single plugin file and find TestRunnerPlugin subclasses.
        
        Args:
            plugin_file: Path to the .py file to load
        """
        module_name = f"test_runner_plugin_{plugin_file.stem}"
        
        try:
            # Load the module from file
            spec = importlib.util.spec_from_file_location(module_name, plugin_file)
            if spec is None or spec.loader is None:
                print(f"DEBUG TestRunnerPluginManager: Could not load spec for {plugin_file}")
                return
            
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            # Find all TestRunnerPlugin subclasses in the module
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type) and
                    issubclass(attr, TestRunnerPlugin) and
                    attr is not TestRunnerPlugin and
                    getattr(attr, '__module__', None) == module_name
                ):
                    # Instantiate the plugin
                    plugin_instance = attr(working_dir=self._working_dir, read_root_dir=self._read_root_dir)
                    self._plugins.append(plugin_instance)
                    print(f"DEBUG TestRunnerPluginManager: Loaded plugin: {attr.__name__}")
                    
        except Exception as e:
            raise ToolError(f"TestRunnerPluginManager: Error loading plugin {plugin_file}: {e}")
    
    @property
    def plugins(self) -> List[TestRunnerPlugin]:
        """Get all loaded plugins."""
        self._load_plugins()
        return self._plugins
    
    @property
    def plugin_directories(self) -> List[str]:
        """Get list of directories containing plugins.
        
        Used for checking if new plugins have been added.
        """
        return [str(self._plugin_directory)]
    
    def get_runners_with_priority_filtering(self, max_cost: float | None = 1.5) -> List[Tuple[TestRunnerPlugin, TestRunner]]:
        """Get runners from all plugins with priority-based filtering applied.
        
        Priority 1 and lower: always run
        Priority 2 and higher: weighted by inverse priority, 20% randomly selected (minimum 1)
        
        Returns:
            List of (plugin, runner) tuples that should be executed
        """
        import random
        
        # Separate runners by numerical priority (logical priority is inverted)
        low_priority_runners = []  # priority <= 1
        high_priority_runners = []  # priority >= 2
        
        # Collect all runners from all plugins
        for plugin in self.plugins:
            try:
                runners = plugin.get_runners()
            except Exception as e:
                print(f"DEBUG TestRunnerPluginManager: Error getting runners from plugin: {e}")
            plugin_low_priority_runners = []  # priority <= 1
            plugin_high_priority_runners = []  # priority >= 2            
            priorities_present = set()
            for runner in runners:
                if max_cost is not None:
                    cost = runner.cost if runner.cost is not None else float('inf')
                    if cost > max_cost:
                        continue
                priority = runner.priority if runner.priority is not None else 1
                priorities_present.add(priority)
                if priority <= 1:
                    plugin_low_priority_runners.append((plugin, runner))
                else:
                    plugin_high_priority_runners.append((plugin, runner))
            if priorities_present == {2}:
                # Pick one randomly and discard the rest
                plugin_low_priority_runners.append(random.choice(plugin_high_priority_runners))
                plugin_high_priority_runners.clear()
            low_priority_runners.extend(plugin_low_priority_runners)
            high_priority_runners.extend(plugin_high_priority_runners)
        
        # Apply filtering to high priority runners
        selected_high_priority = []
        if high_priority_runners:
            # Calculate weights as inverse of priority
            weights = []
            for plugin, runner in high_priority_runners:
                priority = runner.priority if runner.priority is not None else 2
                weights.append(1.0 / priority)

            sum_weights = sum(weights)            

            weights = [w/sum_weights for w in weights]
            
            # Calculate number to select (20% minimum of 1)
            num_to_select = max(1, int(len(high_priority_runners) * 0.2))

            selected_indices = numpy.random.choice(numpy.arange(0, len(high_priority_runners)), num_to_select, False, weights)

            for i in selected_indices:
                selected_high_priority.append(high_priority_runners[i])
        
        # Combine low priority (all) with selected high priority
        result_with_costs = []
        for plugin, runner in low_priority_runners + selected_high_priority:
            cost = runner.cost if runner.cost is not None else float('inf')
            priority = runner.priority if runner.priority is not None else 1            
            result_with_costs.append((cost, priority, plugin, runner))
        
        # Sort by cost (cheapest and numerically lowest priority first )
        result_with_costs.sort(key=lambda x: x[0])
        return [(r[-1], r[-2]) for r in result_with_costs]
    
    def get_plugin_for_identity(self, test_identity: str) -> Optional[TestRunnerPlugin]:
        """Find the plugin that can handle a given test identity.
        
        Parses the plugin prefix from the test identity (format: "plugin_name:test_name")
        and routes to the appropriate plugin.
        
        Args:
            test_identity: The test identity string in format "plugin_name:test_name"
            
        Returns:
            The plugin that can handle the identity, or None if no plugin matches
        """
        if not test_identity or ':' not in test_identity:
            return None, None
        
        # Extract plugin name from identity (format: "plugin_name:test_name")
        plugin_name, test_name = test_identity.split(':', 1)
        
        # Find the plugin with matching name
        for plugin in self._plugins:
            try:
                if plugin.get_plugin_name() == plugin_name:
                    return plugin, test_name
            except Exception as e:
                print(f"DEBUG TestRunnerPluginManager: Error getting plugin name: {e}")
                continue
        
        return None, None
