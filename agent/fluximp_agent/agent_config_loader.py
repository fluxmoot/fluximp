"""Agent configuration loader for data/config files.

Supports loading agent definitions from JSON config files to override
existing agent types or add new ones.  Config files are processed in
order.  If the same agent type name appears in more than one config
file, a ``ValueError`` is raised â this is almost certainly a mistake
in the calling code, so the loader fails fast rather than silently
picking a winner.

Config file format (JSON)::

    {
        "agents": {
            "<agent_type>": {
                "tool_names": ["tool1", "tool2", ...],
                "agent_description": "System prompt description string"
            },
            ...
        }
    }

- ``tool_names``: full list of base tool names for the agent (before
  any experiment-driven drops).  Must match names registered in
  ``tool_registry``.
- ``agent_description``: the system-prompt description text embedded
  after the category-derived preamble.  Newline characters should be
  expressed as ``\n`` in the JSON string.

When a config file specifies an agent type that already exists in
``AGENT_DEFINITIONS`` (i.e. a built-in), the config file's definition
replaces it entirely.  When it specifies a new agent type, that type
is added.
"""

import json
from pathlib import Path
from typing import Dict, Any, List


def load_agent_config(config_path: str) -> Dict[str, Any]:
    """Load a single agent config file.

    Args:
        config_path: Path to the JSON config file.

    Returns:
        Dict with the config file contents.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        json.JSONDecodeError: If the file is not valid JSON.
        ValueError: If the config structure is invalid.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Agent config file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        config = json.load(f)

    _validate_agent_config(config, config_path)
    return config


def _validate_agent_config(config: Dict[str, Any], config_path: str):
    """Validate the structure of an agent config.

    Args:
        config: Parsed JSON config dict.
        config_path: Path for error messages.

    Raises:
        ValueError: If the config structure is invalid.
    """
    if not isinstance(config, dict):
        raise ValueError(
            f"Agent config {config_path}: top level must be a JSON object"
        )

    if "agents" not in config:
        raise ValueError(
            f"Agent config {config_path}: missing required 'agents' key"
        )

    agents = config["agents"]
    if not isinstance(agents, dict):
        raise ValueError(
            f"Agent config {config_path}: 'agents' must be a JSON object"
        )

    for agent_name, agent_def in agents.items():
        if not isinstance(agent_def, dict):
            raise ValueError(
                f"Agent config {config_path}: agent '{agent_name}' must be a JSON object"
            )

        if "tool_names" not in agent_def:
            raise ValueError(
                f"Agent config {config_path}: agent '{agent_name}' missing required 'tool_names'"
            )

        if "agent_description" not in agent_def:
            raise ValueError(
                f"Agent config {config_path}: agent '{agent_name}' missing required 'agent_description'"
            )

        if not isinstance(agent_def["tool_names"], list):
            raise ValueError(
                f"Agent config {config_path}: agent '{agent_name}' tool_names must be a list"
            )

        if not isinstance(agent_def["agent_description"], str):
            raise ValueError(
                f"Agent config {config_path}: agent '{agent_name}' agent_description must be a string"
            )

        if not agent_def["agent_description"].strip():
            raise ValueError(
                f"Agent config {config_path}: agent '{agent_name}' agent_description must not be empty"
            )

        for i, tool_name in enumerate(agent_def["tool_names"]):
            if not isinstance(tool_name, str):
                raise ValueError(
                    f"Agent config {config_path}: agent '{agent_name}' tool_names[{i}] must be a string"
                )


def merge_agent_configs(
    base_definitions: Dict[str, Dict], config_paths: List[str]
) -> Dict[str, Dict]:
    """Merge agent configs from files into base definitions.

    Processes config files in order.  A config file may override a
    built-in agent type or add a new one, but if the same agent type
    name appears in more than one config file a ``ValueError`` is
    raised â the caller should not be passing conflicting definitions.

    The *base_definitions* dict is modified in-place and also returned
    for convenience.

    Args:
        base_definitions: The base AGENT_DEFINITIONS dict to start from.
        config_paths: List of paths to JSON config files.

    Returns:
        Updated definitions dict (same object as *base_definitions*).

    Raises:
        FileNotFoundError: If any config file doesn't exist.
        json.JSONDecodeError: If any config file is not valid JSON.
        ValueError: If any config structure is invalid, or if the same
            agent type name appears in more than one config file.
    """
    seen_in_configs: Dict[str, str] = {}  # agent_name -> config_path

    for config_path in config_paths:
        config = load_agent_config(config_path)
        for agent_name, agent_def in config["agents"].items():
            if agent_name in seen_in_configs:
                raise ValueError(
                    f"Agent type '{agent_name}' is defined in multiple config files: "
                    f"'{seen_in_configs[agent_name]}' and '{config_path}'"
                )
            seen_in_configs[agent_name] = config_path
            base_definitions[agent_name] = {
                "tool_names": list(agent_def["tool_names"]),
                "agent_description": agent_def["agent_description"],
            }

    return base_definitions
