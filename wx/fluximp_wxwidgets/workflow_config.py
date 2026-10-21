"""Workflow and agent configuration management for FluxImp GUI.

Manages workflow configs and agent configs stored in the ~/.fluximp directory.
Stock workflows and agents are defined in code (fs_agents.py) and have no
config files.  User modifications to stock items create config files that
override the stock definitions.  "Restore" operations delete those config
files.

Agent config files use the --agent-config JSON format so they can be passed
directly to main.py.
"""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

from fluximp_agent import fs_agents
from fluximp_agent.conditional_agent import ConditionalAgentConfigInfo


def _config_filename(name: str) -> str:
    """Convert a name to a config filename: lowercase, spaces -> underscores."""
    return name.lower().replace(" ", "_") + ".json"


@dataclass
class EventSequenceGroup:
    """A named group of conditional agents for an event, scoped to specific agent types."""
    name: str
    agent_types: List[str]  # empty = all types
    sequence: List[str]     # ordered conditional agent names


def _parse_event_groups(event_data) -> List[EventSequenceGroup]:
    """Parse event data from JSON, handling both old and new formats.

    Old format: ["agent1", "agent2"] (flat list of strings)
    New format: [{"name": "...", "agent_types": [...], "sequence": [...]}]
    """
    if not isinstance(event_data, list):
        return []
    if not event_data:
        return []
    # Check if old format (list of strings)
    if isinstance(event_data[0], str):
        return [EventSequenceGroup(name="Default", agent_types=[], sequence=list(event_data))]
    # New format (list of dicts)
    groups = []
    for item in event_data:
        groups.append(EventSequenceGroup(
            name=item.get("name", "Unnamed"),
            agent_types=list(item.get("agent_types", [])),
            sequence=list(item.get("sequence", [])),
        ))
    return groups


@dataclass
class WorkflowConfig:
    """A workflow configuration (stock or user-defined)."""
    name: str
    agents: List[str]  # list of agent type names
    events: Dict[str, List[EventSequenceGroup]]  # event key -> list of sequence groups
    is_stock: bool
    config_path: Optional[Path] = None  # None for stock workflow (no config file)

    def is_modified_stock(self) -> bool:
        """True if this is a stock workflow that has been modified (has a config file)."""
        return self.is_stock and self.config_path is not None


@dataclass
class AgentConfigInfo:
    """An agent configuration (stock or user-defined)."""
    name: str
    tool_names: List[str]
    agent_description: str
    is_stock: bool
    config_path: Optional[Path] = None  # None for unmodified stock agents

    def is_overridden(self) -> bool:
        """True if this stock agent has been overridden by a config file."""
        return self.is_stock and self.config_path is not None




# Stock workflows: canonical name -> ordered list of agent type names.
# The order determines the default agent of a new conversation (the first
# agent in the list) and the display order of the workflow's agents.
# "Coding" is the primary/default stock workflow.
STOCK_WORKFLOWS: Dict[str, List[str]] = {
    "Coding": ["coder", "consultant", "developer", "engineer", "researcher"],
    "Web Research": ["hound", "scribe"],
}

# The primary stock workflow (used as the fallback when a workspace has no
# default workflow set).
STOCK_WORKFLOW_NAME = "Coding"

# Back-compat alias: the agent list of the primary stock workflow.
STOCK_WORKFLOW_AGENTS = list(STOCK_WORKFLOWS[STOCK_WORKFLOW_NAME])

# Legacy name of the primary stock workflow.  Workflows persisted before the
# rename (a conversation's ".config.json" "workflow_name", a workspace's
# "default_workflow", and the override file "dev_workflow.json") still use it.
LEGACY_STOCK_WORKFLOW_NAME = "Dev Workflow"
LEGACY_STOCK_WORKFLOW_FILENAME = _config_filename(LEGACY_STOCK_WORKFLOW_NAME)

# Runtime name aliases: legacy name -> canonical name.
LEGACY_WORKFLOW_ALIASES: Dict[str, str] = {
    LEGACY_STOCK_WORKFLOW_NAME: STOCK_WORKFLOW_NAME,
}


def canonical_workflow_name(name: str) -> str:
    """Map a legacy workflow name to its current canonical name."""
    return LEGACY_WORKFLOW_ALIASES.get(name, name)


def _reserved_workflow_filenames() -> set:
    """Config filenames owned by the stock workflows (incl. the legacy file).

    These files must never be loaded as (or overwritten by) user workflows.
    """
    reserved = {_config_filename(n) for n in STOCK_WORKFLOWS}
    reserved.add(LEGACY_STOCK_WORKFLOW_FILENAME)
    return reserved


class LegacyWorkflowConflictError(Exception):
    """Both the legacy and the current primary stock-workflow config files exist.

    Raised by :func:`migrate_legacy_stock_workflow`.  The GUI surfaces it in a
    pre-main-frame dialog and exits.
    """

    def __init__(self, legacy_path: Path, current_path: Path):
        self.legacy_path = legacy_path
        self.current_path = current_path
        super().__init__(
            f"Both '{legacy_path}' and '{current_path}' exist.  Compare the two "
            f"files, set '{current_path}' the way you want it, then delete "
            f"'{legacy_path}'."
        )


def migrate_legacy_stock_workflow(workflows_dir: Path) -> Optional[Path]:
    """Rename the legacy primary stock-workflow override file to its new name.

    Renames ``dev_workflow.json`` -> ``coding.json``, but only when
    ``coding.json`` does not already exist.  If both exist the ambiguity must
    be resolved by hand, so :class:`LegacyWorkflowConflictError` is raised
    instead.

    Safe to call on every startup: it is a no-op once migrated.

    Returns:
        The new config path if a rename happened, otherwise ``None``.

    Raises:
        LegacyWorkflowConflictError: If both files exist.
    """
    legacy_path = workflows_dir / LEGACY_STOCK_WORKFLOW_FILENAME
    current_path = workflows_dir / _config_filename(STOCK_WORKFLOW_NAME)
    if not legacy_path.exists():
        return None
    if current_path.exists():
        raise LegacyWorkflowConflictError(legacy_path, current_path)
    legacy_path.rename(current_path)
    return current_path

# Ordered tool groups for the agent editing UI (group title, tool names).
# Order: read-only, file editing, execution, web.
TOOL_GROUPS = [
    ("Read-only Tools",             list(fs_agents._READONLY_TOOLS)),
    ("File Editing Tools",          list(fs_agents._FILE_EDIT_TOOLS)),
    ("Execution Tools",             list(fs_agents._EXEC_TOOLS)),
    ("Web Search/Extraction Tools", list(fs_agents._WEB_TOOLS)),
]

ALL_TOOL_NAMES = sorted(set(t for _, tools in TOOL_GROUPS for t in tools))

class WorkflowConfigManager:
    """Manages workflow and agent configuration files."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.workflows_dir = base_dir / "workflows"
        self.agents_dir = base_dir / "agents"
        self.conditional_agents_dir = base_dir / "conditional_agents"
        self.workflows_dir.mkdir(parents=True, exist_ok=True)
        self.agents_dir.mkdir(parents=True, exist_ok=True)
        self.conditional_agents_dir.mkdir(parents=True, exist_ok=True)
        self._workflows: Dict[str, WorkflowConfig] = {}
        self._agents: Dict[str, AgentConfigInfo] = {}
        self._conditional_agents: Dict[str, ConditionalAgentConfigInfo] = {}
        self._load()

    def reload(self):
        """Reload all configs from disk, refreshing in-memory state."""
        self._workflows.clear()
        self._agents.clear()
        self._conditional_agents.clear()
        self._load()

    def _load(self):
        """Load all workflow and agent configs from disk, merging with stock definitions."""
        # Load stock agents first
        for name, defn in fs_agents.AGENT_DEFINITIONS.items():
            config_path = self.agents_dir / _config_filename(name)
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                agents_data = data.get("agents", data)
                agent_data = agents_data.get(name, defn)
                self._agents[name] = AgentConfigInfo(
                    name=name,
                    tool_names=list(agent_data.get("tool_names", defn["tool_names"])),
                    agent_description=agent_data.get("agent_description", defn["agent_description"]),
                    is_stock=True,
                    config_path=config_path,
                )
            else:
                self._agents[name] = AgentConfigInfo(
                    name=name,
                    tool_names=list(defn["tool_names"]),
                    agent_description=defn["agent_description"],
                    is_stock=True,
                    config_path=None,
                )

        # Load user agents (non-stock agent config files)
        for config_file in self.agents_dir.iterdir():
            if not config_file.is_file() or not config_file.suffix == '.json':
                continue
            with open(config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for agent_name, agent_data in data.get("agents", data).items():
                if agent_name in self._agents:
                    continue  # Already handled (stock agent override)
                self._agents[agent_name] = AgentConfigInfo(
                    name=agent_name,
                    tool_names=list(agent_data.get("tool_names", [])),
                    agent_description=agent_data.get("agent_description", ""),
                    is_stock=False,
                    config_path=config_file,
                )

        # Load stock workflows (Coding, Web Research, ...).  For a stock
        # workflow the available agents are always the current stock agents
        # from fs_agents; the persisted "agents" list is only a record of
        # which agents have user overrides/configs to load from the agents
        # directory.  It must not determine what stock agents appear in the
        # workflow or in the GUI dropdown.
        for stock_name, stock_agent_list in STOCK_WORKFLOWS.items():
            stock_wf_path = self.workflows_dir / _config_filename(stock_name)
            if stock_wf_path.exists():
                with open(stock_wf_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                raw_events = data.get("events", {"work_completion_summary": []})
                events = {k: _parse_event_groups(v) for k, v in raw_events.items()}
                self._workflows[stock_name] = WorkflowConfig(
                    name=stock_name,
                    agents=list(stock_agent_list),
                    events=events,
                    is_stock=True,
                    config_path=stock_wf_path,
                )
            else:
                self._workflows[stock_name] = WorkflowConfig(
                    name=stock_name,
                    agents=list(stock_agent_list),
                    events={"work_completion_summary": []},
                    is_stock=True,
                    config_path=None,
                )

        # Load user workflows.  Config filenames owned by the stock workflows
        # (and the legacy primary stock-workflow file) must never load as user
        # workflows.
        reserved_filenames = _reserved_workflow_filenames()
        for config_file in self.workflows_dir.iterdir():
            if not config_file.is_file() or not config_file.suffix == '.json':
                continue
            if config_file.name in reserved_filenames:
                continue
            with open(config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            wf_name = data.get("name", config_file.stem.replace("_", " ").title())
            raw_events = data.get("events", {"work_completion_summary": []})
            events = {k: _parse_event_groups(v) for k, v in raw_events.items()}
            self._workflows[wf_name] = WorkflowConfig(
                name=wf_name,
                agents=list(data.get("agents", [])),
                events=events,
                is_stock=False,
                config_path=config_file,
            )

        # Load conditional agents
        for config_file in self.conditional_agents_dir.iterdir():
            if not config_file.is_file() or not config_file.suffix == '.json':
                continue
            with open(config_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for agent_name, agent_data in data.get("conditional_agents", {config_file.stem.replace("_", " "): data}).items():
                self._conditional_agents[agent_name] = ConditionalAgentConfigInfo(
                    name=agent_name,
                    prompt=agent_data.get("prompt", ""),
                    tool_set=agent_data.get("tool_set", "report_only"),
                    context_sources=agent_data.get("context_sources", ["user_and_assistant_messages", "diff"]),
                    retrigger=agent_data.get("retrigger", False),
                    max_retriggers=agent_data.get("max_retriggers", 3),
                    max_requests=agent_data.get("max_requests", 5),
                    depends_on=agent_data.get("depends_on"),
                    config_path=config_file,
                )

    # --- Workflow operations ---

    def get_workflows(self) -> List[WorkflowConfig]:
        """Get all workflows sorted by name (stock first)."""
        workflows = list(self._workflows.values())
        stock = [w for w in workflows if w.is_stock]
        user = sorted([w for w in workflows if not w.is_stock], key=lambda w: w.name)
        return stock + user

    def get_workflow(self, name: str) -> Optional[WorkflowConfig]:
        """Get a workflow by name (legacy names are canonicalised)."""
        return self._workflows.get(canonical_workflow_name(name))

    def get_workflow_agent_names(self, workflow_name: str) -> List[str]:
        """Get the list of agent names for a specific workflow."""
        workflow = self._workflows.get(canonical_workflow_name(workflow_name))
        if workflow:
            return list(workflow.agents)
        return list(STOCK_WORKFLOW_AGENTS)

    def get_workflow_agent_config_paths(self, workflow_name: str) -> List[Path]:
        """Get config file paths for all custom/overridden agents in a workflow."""
        workflow = self._workflows.get(canonical_workflow_name(workflow_name))
        if workflow is None:
            return []
        paths = []
        for agent_name in workflow.agents:
            agent = self._agents.get(agent_name)
            if agent and agent.config_path:
                paths.append(agent.config_path)
        return paths

    def get_conversation_agent_configs(self, workflow_name: str, agent_type: str) -> List[Path]:
        """Get all agent config paths needed for a conversation.

        Returns config paths for all custom/overridden agents in the workflow.
        The agent_type must be one of the workflow's agents; if it is not in
        the workflow, its config is NOT included (this indicates a bug in the
        GUI that should be prevented upstream by ensuring the Agent dropdown
        only contains agents defined for the workflow).
        """
        return list(self.get_workflow_agent_config_paths(workflow_name))

    def is_stock_workflow(self, name: str) -> bool:
        """Check if a workflow is one of the stock workflows."""
        return canonical_workflow_name(name) in STOCK_WORKFLOWS

    def create_workflow(self, name: str) -> WorkflowConfig:
        """Create a new user workflow with no agents."""
        if name in self._workflows:
            raise ValueError(f"Workflow '{name}' already exists")
        if _config_filename(name) in _reserved_workflow_filenames():
            raise ValueError(
                f"Workflow name '{name}' is reserved: its config file name "
                f"('{_config_filename(name)}') collides with a stock workflow "
                f"config file."
            )
        config_path = self.workflows_dir / _config_filename(name)
        data = {"name": name, "agents": [], "events": {"work_completion_summary": []}}
        config_path.write_text(json.dumps(data, indent=2), encoding='utf-8')
        workflow = WorkflowConfig(
            name=name, agents=[], events={"work_completion_summary": []},
            is_stock=False, config_path=config_path,
        )
        self._workflows[name] = workflow
        return workflow

    def delete_workflow(self, name: str) -> bool:
        """Delete a user workflow. Cannot delete stock workflow."""
        if self.is_stock_workflow(name):
            raise ValueError("Cannot delete stock workflow")
        workflow = self._workflows.get(name)
        if workflow is None:
            return False
        if workflow.config_path and workflow.config_path.exists():
            workflow.config_path.unlink()
        del self._workflows[name]
        return True

    def restore_stock_workflow(self, name: str = STOCK_WORKFLOW_NAME) -> WorkflowConfig:
        """Restore a stock workflow to its default agent list."""
        name = canonical_workflow_name(name)
        if name not in STOCK_WORKFLOWS:
            raise ValueError(f"'{name}' is not a stock workflow")
        workflow = self._workflows.get(name)
        if workflow and workflow.config_path and workflow.config_path.exists():
            workflow.config_path.unlink()
        # Defensive: also remove a lingering legacy primary-workflow file.
        if name == STOCK_WORKFLOW_NAME:
            legacy_path = self.workflows_dir / LEGACY_STOCK_WORKFLOW_FILENAME
            if legacy_path.exists():
                legacy_path.unlink()
        self._workflows[name] = WorkflowConfig(
            name=name,
            agents=list(STOCK_WORKFLOWS[name]),
            events={"work_completion_summary": []},
            is_stock=True,
            config_path=None,
        )
        return self._workflows[name]

    def add_agent_to_workflow(self, workflow_name: str, agent_name: str) -> bool:
        """Add an agent to a workflow's agent list."""
        workflow = self._workflows.get(workflow_name)
        if workflow is None:
            return False
        if agent_name in workflow.agents:
            return False
        workflow.agents.append(agent_name)
        self._save_workflow_config(workflow)
        return True

    def remove_agent_from_workflow(self, workflow_name: str, agent_name: str) -> bool:
        """Remove an agent from a workflow's agent list. Does not delete the agent config file."""
        workflow = self._workflows.get(workflow_name)
        if workflow is None:
            return False
        if agent_name not in workflow.agents:
            return False
        workflow.agents.remove(agent_name)
        self._save_workflow_config(workflow)
        return True

    def _save_workflow_config(self, workflow: WorkflowConfig):
        """Save a workflow config to disk."""
        config_path = self.workflows_dir / _config_filename(workflow.name)
        events_json = {}
        for event_key, groups in workflow.events.items():
            events_json[event_key] = [
                {"name": g.name, "agent_types": g.agent_types, "sequence": g.sequence}
                for g in groups
            ]
        data = {"name": workflow.name, "agents": workflow.agents, "events": events_json}
        config_path.write_text(json.dumps(data, indent=2), encoding='utf-8')
        workflow.config_path = config_path

    # --- Agent operations ---

    def get_agents(self) -> List[AgentConfigInfo]:
        """Get all agents sorted by name (stock first)."""
        agents = list(self._agents.values())
        stock = sorted([a for a in agents if a.is_stock], key=lambda a: a.name)
        user = sorted([a for a in agents if not a.is_stock], key=lambda a: a.name)
        return stock + user

    def get_agent(self, name: str) -> Optional[AgentConfigInfo]:
        """Get an agent by name."""
        return self._agents.get(name)

    def is_stock_agent(self, name: str) -> bool:
        """Check if an agent is one of the stock agents."""
        return name in fs_agents.AGENT_DEFINITIONS

    def is_agent_overridden(self, name: str) -> bool:
        """Check if a stock agent has been overridden by a config file."""
        agent = self._agents.get(name)
        if agent and agent.is_stock:
            return agent.is_overridden()
        return False

    def create_agent(self, name: str) -> AgentConfigInfo:
        """Create a new user agent with empty tool list and description."""
        if name in self._agents:
            raise ValueError(f"Agent '{name}' already exists")
        config_path = self.agents_dir / _config_filename(name)
        data = {"agents": {name: {"tool_names": [], "agent_description": ""}}}
        config_path.write_text(json.dumps(data, indent=2), encoding='utf-8')
        agent = AgentConfigInfo(
            name=name, tool_names=[], agent_description="",
            is_stock=False, config_path=config_path,
        )
        self._agents[name] = agent
        return agent

    def save_agent(self, name: str, tool_names: List[str],
                   agent_description: str) -> AgentConfigInfo:
        """Save an agent's configuration to disk."""
        agent = self._agents.get(name)
        if agent is None:
            raise ValueError(f"Agent '{name}' not found")
        agent.tool_names = list(tool_names)
        agent.agent_description = agent_description
        config_path = self.agents_dir / _config_filename(name)
        data = {"agents": {name: {"tool_names": tool_names, "agent_description": agent_description}}}
        config_path.write_text(json.dumps(data, indent=2), encoding='utf-8')
        agent.config_path = config_path
        return agent

    def restore_stock_agent(self, name: str) -> AgentConfigInfo:
        """Restore a stock agent by deleting its overriding config file."""
        if not self.is_stock_agent(name):
            raise ValueError(f"'{name}' is not a stock agent")
        agent = self._agents.get(name)
        if agent and agent.config_path and agent.config_path.exists():
            agent.config_path.unlink()
        defn = fs_agents.AGENT_DEFINITIONS[name]
        self._agents[name] = AgentConfigInfo(
            name=name,
            tool_names=list(defn["tool_names"]),
            agent_description=defn["agent_description"],
            is_stock=True,
            config_path=None,
        )
        return self._agents[name]

    def get_agent_config_paths(self) -> List[Path]:
        """Get all agent config file paths for passing to main.py via --agent-config."""
        paths = []
        for config_file in self.agents_dir.iterdir():
            if config_file.is_file() and config_file.suffix == '.json':
                paths.append(config_file)
        return sorted(paths)

    def get_tool_groups(self) -> List[Tuple[str, List[str]]]:
        """Get ordered (group title, tool names) pairs for the agent editing UI.

        Order: read-only, file editing, execution, web.  Tool names within
        each group are sorted for display.
        """
        return [(title, sorted(tools)) for title, tools in TOOL_GROUPS]

    def get_all_tool_names(self) -> List[str]:
        """Get all known tool names for the agent editing UI."""
        return list(ALL_TOOL_NAMES)

    def get_agent_config_path(self, agent_name: str) -> Optional[Path]:
        """Get the config file path for an agent, or None if it has no config file (unmodified stock agent)."""
        agent = self._agents.get(agent_name)
        if agent and agent.config_path:
            return agent.config_path
        return None

    # --- Conditional agent operations ---

    def get_conditional_agents(self) -> List[ConditionalAgentConfigInfo]:
        """Get all conditional agents."""
        return list(self._conditional_agents.values())

    def get_conditional_agent(self, name: str) -> Optional[ConditionalAgentConfigInfo]:
        """Get a conditional agent by name."""
        return self._conditional_agents.get(name)

    def conditional_agent_exists(self, name: str) -> bool:
        """Check if a conditional agent exists (in memory or on disk)."""
        return name in self._conditional_agents

    def save_conditional_agent(self, config: ConditionalAgentConfigInfo):
        """Save a conditional agent config to disk."""
        config_path = self.conditional_agents_dir / _config_filename(config.name)
        data = {"conditional_agents": {config.name: {
            "prompt": config.prompt,
            "tool_set": config.tool_set,
            "context_sources": config.context_sources,
            "retrigger": config.retrigger,
            "max_retriggers": config.max_retriggers,
            "max_requests": config.max_requests,
            "depends_on": config.depends_on,
        }}}
        config_path.write_text(json.dumps(data, indent=2), encoding='utf-8')
        config.config_path = config_path
        self._conditional_agents[config.name] = config

    def delete_conditional_agent(self, name: str) -> bool:
        """Delete a conditional agent config file."""
        agent = self._conditional_agents.get(name)
        if agent is None:
            return False
        if agent.config_path and agent.config_path.exists():
            agent.config_path.unlink()
        del self._conditional_agents[name]
        return True

    def rename_conditional_agent(self, old_name: str, new_name: str) -> bool:
        """Rename a conditional agent. Updates config file and all workflow event references."""
        if old_name not in self._conditional_agents:
            return False
        if new_name in self._conditional_agents and old_name != new_name:
            raise ValueError(f"Conditional agent '{new_name}' already exists")
        agent = self._conditional_agents[old_name]
        # Delete old config file
        if agent.config_path and agent.config_path.exists():
            agent.config_path.unlink()
        # Update name and save new config
        agent.name = new_name
        self.save_conditional_agent(agent)
        # Update references in _conditional_agents dict
        if old_name != new_name:
            del self._conditional_agents[old_name]
            self._conditional_agents[new_name] = agent
        # Update references in workflow events
        for workflow in self._workflows.values():
            for event_key, groups in workflow.events.items():
                for group in groups:
                    for i, name in enumerate(group.sequence):
                        if name == old_name:
                            group.sequence[i] = new_name
            self._save_workflow_config(workflow)
        return True

    # --- Event sequence operations ---

    def get_event_sequence(self, workflow_name: str, event_key: str) -> List[str]:
        """Get the combined ordered list of conditional agent names for an event (all groups)."""
        workflow = self._workflows.get(workflow_name)
        if workflow is None:
            return []
        groups = workflow.events.get(event_key, [])
        result = []
        for group in groups:
            result.extend(group.sequence)
        return result

    def get_event_sequence_groups(self, workflow_name: str, event_key: str) -> List[EventSequenceGroup]:
        """Get all sequence groups for an event."""
        workflow = self._workflows.get(workflow_name)
        if workflow is None:
            return []
        return list(workflow.events.get(event_key, []))

    def get_event_sequence_group(self, workflow_name: str, event_key: str,
                                  group_name: str) -> Optional[EventSequenceGroup]:
        """Get a specific sequence group by name."""
        groups = self.get_event_sequence_groups(workflow_name, event_key)
        for group in groups:
            if group.name == group_name:
                return group
        return None

    def add_event_sequence_group(self, workflow_name: str, event_key: str,
                                  group_name: str,
                                  agent_types: Optional[List[str]] = None) -> EventSequenceGroup:
        """Add a new sequence group to an event."""
        workflow = self._workflows.get(workflow_name)
        if workflow is None:
            raise ValueError(f"Workflow '{workflow_name}' not found")
        if event_key not in workflow.events:
            raise ValueError(f"Event '{event_key}' not found in workflow '{workflow_name}'")
        # Check for duplicate group name
        for existing in workflow.events[event_key]:
            if existing.name == group_name:
                raise ValueError(f"Group '{group_name}' already exists in event '{event_key}'")
        group = EventSequenceGroup(
            name=group_name,
            agent_types=list(agent_types) if agent_types else [],
            sequence=[],
        )
        workflow.events[event_key].append(group)
        self._save_workflow_config(workflow)
        return group

    def remove_event_sequence_group(self, workflow_name: str, event_key: str,
                                     group_name: str) -> bool:
        """Remove a sequence group from an event."""
        workflow = self._workflows.get(workflow_name)
        if workflow is None:
            return False
        groups = workflow.events.get(event_key, [])
        for i, group in enumerate(groups):
            if group.name == group_name:
                groups.pop(i)
                self._save_workflow_config(workflow)
                return True
        return False

    def get_event_sequence_for_agent_type(self, workflow_name: str, event_key: str,
                                           agent_type: str) -> List[str]:
        """Get the conditional agent sequence for an event scoped to a specific agent type.

        Returns the sequence from the first group whose agent_types list contains
        agent_type, or whose agent_types list is empty (matches all types).
        If no matching group is found, returns an empty list.
        """
        groups = self.get_event_sequence_groups(workflow_name, event_key)
        for group in groups:
            if not group.agent_types or agent_type in group.agent_types:
                return list(group.sequence)
        return []

    def remove_event(self, workflow_name: str, event_key: str) -> bool:
        """Remove an event and its sequence from a workflow config.
        Does NOT delete the conditional agent config files."""
        workflow = self._workflows.get(workflow_name)
        if workflow is None:
            return False
        if event_key not in workflow.events:
            return False
        del workflow.events[event_key]
        self._save_workflow_config(workflow)
        return True

