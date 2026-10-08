"""Workspace management for Fluximp agents."""

from fluximp_agent import fs_agents
import hashlib
import json
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List
from datetime import datetime, timezone

from fluximp_agent.model_config import create_provider, get_default_provider_name
from fluximp_wxwidgets.global_state import get_global_state
from fluximp_wxwidgets.conversation_manager import Conversation
from fluximp_wxwidgets.workflow_config import STOCK_WORKFLOW_NAME


@dataclass
class ConversationStub:
    """Lightweight conversation metadata - loaded at startup without parsing I/O files."""
    conversation_id: str
    conversation_dir: Path
    start_time: datetime
    workspace: 'Workspace'
    workflow_name: str = "Dev Workflow"


class Workspace:
    """Represents a workspace (working directory) with multiple conversations."""
    
    def __init__(self, workspace_dir: Path, workspace_id: str, read_root_dir: Optional[Path] = None, write_file: Optional[Path] = None):
        self.workspace_dir = workspace_dir
        self.workspace_id = workspace_id
        self.config_file = workspace_dir / "config.json"
        self.workflows_dir = workspace_dir / "workflows"
        
        # Load config
        config = json.loads(self.config_file.read_text())
        self.path = Path(config['path']).resolve()
        self.name = config.get('name', self.path.name)
        self.default_workflow: Optional[str] = config.get('default_workflow')
        # Raw test-runner plugin dir, stored exactly as entered in the
        # Settings tab (None when unset) and processed at agent launch
        # time by resolve_plugin_dir() - which is also what the tab's
        # Check and Save uses, so checked == what gets passed to agents.
        self.test_runner_plugin_dir: Optional[str] = config.get('test_runner_plugin_dir')
        self.read_root_dir = read_root_dir
        self.write_file = write_file

        
        # Store ConversationStub objects (lightweight), not full Conversations
        self.conversations: Dict[str, ConversationStub] = {}

        # Scan for existing conversations when loading an existing workspace
        self._scan_workflows()

    def _scan_workflows(self):
        """Scan workflows directory for existing conversations - creates stubs only.

        Also scans legacy 'conversations' directory for backward compatibility.
        """
        # Check global deferred deletions to skip conversations marked for deletion
        global_state = get_global_state(self.workspace_dir.parent.parent)
        deferred_deletions = global_state.get_deferred_deletions()

        # Scan new workflows directory
        if self.workflows_dir.exists():
            for conv_dir in self.workflows_dir.iterdir():
                if not conv_dir.is_dir():
                    continue

                conversation_id = conv_dir.name

                # Skip if this conversation is marked for deferred deletion
                if str(conv_dir.absolute()) in deferred_deletions:
                    continue

                # Check for io directory to validate it's a conversation
                io_dir = conv_dir / "io"
                if not io_dir.exists():
                    continue

                # Get timestamp from directory modification time (fast, no file parsing)
                try:
                    start_time = datetime.fromtimestamp(conv_dir.stat().st_mtime, tz=timezone.utc)
                except (OSError, ValueError):
                    start_time = datetime.now(timezone.utc)

                # Read workflow_name from config if available
                workflow_name = "Dev Workflow"
                config_path = conv_dir / ".config.json"
                if config_path.exists():
                    try:
                        with open(config_path, 'r', encoding='utf-8') as f:
                            cfg = json.load(f)
                            workflow_name = cfg.get('workflow_name', 'Dev Workflow')
                    except (json.JSONDecodeError, IOError):
                        pass

                # Create lightweight stub - no file parsing yet!
                self.conversations[conversation_id] = ConversationStub(
                    conversation_id=conversation_id,
                    conversation_dir=conv_dir,
                    start_time=start_time,
                    workspace=self,
                    workflow_name=workflow_name
                )

        # Backward compat: also scan legacy conversations directory
        legacy_dir = self.workspace_dir / "conversations"
        if legacy_dir.exists():
            for conv_dir in legacy_dir.iterdir():
                if not conv_dir.is_dir():
                    continue

                conversation_id = conv_dir.name

                # Skip if already loaded from workflows dir
                if conversation_id in self.conversations:
                    continue

                # Skip if this conversation is marked for deferred deletion
                if str(conv_dir.absolute()) in deferred_deletions:
                    continue

                # Check for io directory to validate it's a conversation
                io_dir = conv_dir / "io"
                if not io_dir.exists():
                    continue

                # Get timestamp from directory modification time (fast, no file parsing)
                try:
                    start_time = datetime.fromtimestamp(conv_dir.stat().st_mtime, tz=timezone.utc)
                except (OSError, ValueError):
                    start_time = datetime.now(timezone.utc)

                # Read workflow_name from config if available
                workflow_name = "Dev Workflow"
                config_path = conv_dir / ".config.json"
                if config_path.exists():
                    try:
                        with open(config_path, 'r', encoding='utf-8') as f:
                            cfg = json.load(f)
                            workflow_name = cfg.get('workflow_name', 'Dev Workflow')
                    except (json.JSONDecodeError, IOError):
                        pass

                # Create lightweight stub - no file parsing yet!
                self.conversations[conversation_id] = ConversationStub(
                    conversation_id=conversation_id,
                    conversation_dir=conv_dir,
                    start_time=start_time,
                    workspace=self,
                    workflow_name=workflow_name
                )

    def build_shell_conversation(self, conversation_id: str) -> Conversation:
        """Build a parse-free shell Conversation for an existing conversation.

        Synchronous and cheap: constructs the Conversation object and loads
        its saved config, but performs no I/O stream parsing (Phase 2 reload
        streams the backlog through the file watcher instead). The tab is
        created from this shell and history loading is deferred until after
        the page is added (main_frame defers ``load_existing``).
        """
        if conversation_id not in self.conversations:
            raise KeyError(conversation_id)
        stub = self.conversations[conversation_id]
        conversation_dir = stub.conversation_dir

        conversation = Conversation(
            conversation_dir=conversation_dir,
            working_dir=self.path,
            read_root_dir=self.read_root_dir,
            write_file=self.write_file,
            process=None,  # No active process for loaded conversations
            conversation_id=conversation_id,
            start_time=stub.start_time,
            workspace=self,
        )

        # Load saved configuration (provider/model/agent_type from .config.json)
        conversation._load_config()

        return conversation

    @classmethod
    def create(cls, base_workspaces_dir: Path, path: Path, name: Optional[str] = None) -> "Workspace":
        """Create a new workspace at the specified path.

        Returns the created Workspace instance.
        """
        resolved_path = path.resolve()
        workspace_name = name or resolved_path.name

        # Generate workspace_id from path hash
        workspace_id = hashlib.sha256(str(resolved_path).encode()).hexdigest()[:16]

        # Create workspace directory
        workspace_dir = base_workspaces_dir / workspace_id
        workspace_dir.mkdir(parents=True, exist_ok=True)

        # Create workflows subdirectory
        workflows_dir = workspace_dir / "workflows"
        workflows_dir.mkdir(parents=True, exist_ok=True)

        # Write config.json
        config = {
            'path': str(resolved_path),
            'name': workspace_name
        }
        config_file = workspace_dir / "config.json"
        config_file.write_text(json.dumps(config, indent=2))

        return cls(workspace_dir, workspace_id)

    def get_default_workflow(self) -> str:
        """Workflow created by the tree's 'New <workflow>' context-menu item."""
        return self.default_workflow or STOCK_WORKFLOW_NAME

    def set_default_workflow(self, workflow_name: Optional[str]) -> None:
        """Set the workspace's default workflow (persisted in config.json)."""
        self.default_workflow = workflow_name
        self._save_workspace_config()

    def get_test_runner_plugin_dir(self) -> Optional[str]:
        """The workspace's raw stored test-runner plugin dir (None if unset)."""
        return self.test_runner_plugin_dir

    def set_test_runner_plugin_dir(self, value: Optional[str]) -> None:
        """Set the workspace's test-runner plugin dir (persisted in config.json).

        Stores exactly what the user entered ('' / whitespace-only is
        stored as None); the raw string is processed at agent launch time
        by resolve_plugin_dir().
        """
        self.test_runner_plugin_dir = value if (value is None or value.strip()) else None
        self._save_workspace_config()

    def resolve_plugin_dir(self, raw: Optional[str] = None) -> Optional[Path]:
        """Process a raw plugin-dir string exactly as it is passed to the agent.

        THE single processing implementation: the Settings tab's Check and
        Save calls it with the text box's content, and Conversation.
        start_agent_process calls it with ``raw=None`` to process the
        stored workspace value, so what was checked is exactly what gets
        passed.

        Blank / whitespace-only / None -> None (which omits
        --test-runner-plugin-dir); otherwise expanduser + resolve to an
        absolute path.
        """
        if raw is None:
            raw = self.test_runner_plugin_dir
        if raw is None or not raw.strip():
            return None
        return Path(raw).expanduser().resolve()

    def _save_workspace_config(self) -> None:
        """Save config.json, merging into existing keys (preserves unknown keys)."""
        try:
            config = json.loads(self.config_file.read_text())
        except (OSError, json.JSONDecodeError):
            config = {'path': str(self.path), 'name': self.name}
        config['default_workflow'] = self.default_workflow
        config['test_runner_plugin_dir'] = self.test_runner_plugin_dir
        self.config_file.write_text(json.dumps(config, indent=2))

    def create_conversation(self, workflow_name: str = "Dev Workflow",
                           agent_type: str = "developer",
                           max_requests: int = 100000) -> ConversationStub:
        """Create a new conversation in this workspace."""
        # Generate unique ID based on timestamp
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        prefix = workflow_name.lower().replace(" ", "_")
        conversation_id = f"{prefix}_{timestamp}"

        conversation_dir = self.workflows_dir / conversation_id
        io_dir = conversation_dir / "io"
        log_dir = conversation_dir / "log"
        conversation_dir.mkdir(parents=True, exist_ok=True)
        io_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Get default provider and model
        provider = get_default_provider_name()
        provider_instance = create_provider(provider)
        model = provider_instance.get_default_model()

        # Create conversation without starting process
        # Process will be started on first send (like existing conversations)
        conversation = Conversation(
            conversation_dir=conversation_dir,
            working_dir=self.path,
            read_root_dir=self.read_root_dir,
            write_file=self.write_file,
            process=None,  # No process for new conversations - started on first send
            conversation_id=conversation_id,
            start_time=datetime.now(timezone.utc),
            agent_type=agent_type,
            workspace=self
        )

        # Set initial provider/model config
        conversation.current_provider = provider
        conversation.current_model = model
        conversation.provider_models[provider] = model
        conversation._save_config()

        # Store as a lightweight stub; the shell Conversation is built by
        # build_shell_conversation when the tab is opened (Phase 2).
        stub = ConversationStub(
            conversation_id=conversation_id,
            conversation_dir=conversation_dir,
            start_time=conversation.start_time,
            workspace=self,
            workflow_name=workflow_name,
        )
        self.conversations[conversation_id] = stub

        return stub


    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and its files.

        Always removes the conversation from the GUI immediately.
        If deletion fails (e.g., file lock), marks for deferred deletion.
        """
        if conversation_id not in self.conversations:
            return False

        stub = self.conversations[conversation_id]


        # Always remove from GUI immediately
        conv_dir = stub.conversation_dir
        del self.conversations[conversation_id]

        # Attempt to delete conversation directory
        if conv_dir.exists():
            try:
                shutil.rmtree(conv_dir)
                # Success - also ensure it's not in deferred set
                global_state = get_global_state(self.workspace_dir.parent.parent)
                global_state.remove_deferred_deletion(conv_dir)
            except (PermissionError, OSError):
                # File locked or other error - mark for deferred deletion
                global_state = get_global_state(self.workspace_dir.parent.parent)
                global_state.add_deferred_deletion(conv_dir)

        return True

    def get_conversation_ids(self) -> List[str]:
        """Get list of conversation IDs sorted by creation time."""
        return sorted(
            self.conversations.keys(),
            key=lambda cid: self.conversations[cid].start_time
        )

    def delete(self):
        """Delete this workspace and all its data.

        Conversations marked for deferred deletion remain tracked globally.
        Successfully deleted conversations are removed from deferred set.
        """
        global_state = get_global_state(self.workspace_dir.parent.parent)

        # Process each conversation
        for conversation_id in list(self.conversations.keys()):
            stub = self.conversations[conversation_id]


            conv_dir = stub.conversation_dir
            del self.conversations[conversation_id]

            # Attempt to delete conversation directory
            if conv_dir.exists():
                try:
                    shutil.rmtree(conv_dir)
                    # Success - remove from deferred set if present
                    global_state.remove_deferred_deletion(conv_dir)
                except (PermissionError, OSError):
                    # File locked - ensure it's in deferred set for retry
                    global_state.add_deferred_deletion(conv_dir)

        # Attempt to delete the entire workspace directory
        if self.workspace_dir.exists():
            try:
                shutil.rmtree(self.workspace_dir)
            except (PermissionError, OSError):
                # Workspace directory may still contain locked files
                # Those files are already in deferred_deletions from above
                pass


class WorkspaceManager:
    """Manages multiple workspaces and their conversations."""

    def __init__(self, base_dir: Optional[Path] = None):
        if base_dir is None:
            base_dir = Path.home() / ".fluximp"
        self.base_dir = base_dir
        self.workspaces_dir = base_dir / "workspaces"
        self.workspaces_dir.mkdir(parents=True, exist_ok=True)

        self.workspaces: Dict[str, Workspace] = {}

        # Process any deferred deletions from previous sessions
        global_state = get_global_state(base_dir)
        global_state.process_deferred_deletions()

        self._scan_workspaces()

    def _scan_workspaces(self):
        """Scan workspaces directory for all workspaces."""
        if not self.workspaces_dir.exists():
            return

        for workspace_dir in self.workspaces_dir.iterdir():
            if not workspace_dir.is_dir():
                continue

            workspace_id = workspace_dir.name
            config_file = workspace_dir / "config.json"

            if not config_file.exists():
                continue

            try:
                workspace = Workspace(workspace_dir, workspace_id)
                self.workspaces[workspace_id] = workspace
            except (json.JSONDecodeError, KeyError):
                continue

    def create_workspace(self, path: Path, name: Optional[str] = None) -> Workspace:
        """Create a new workspace."""
        workspace = Workspace.create(self.workspaces_dir, path, name)
        self.workspaces[workspace.workspace_id] = workspace
        return workspace

    def remove_workspace(self, workspace_id: str) -> bool:
        """Remove a workspace (does not delete files)."""
        if workspace_id in self.workspaces:
            del self.workspaces[workspace_id]
            return True
        return False

    def delete_workspace(self, workspace_id: str) -> bool:
        """Delete a workspace and all its conversations."""
        if workspace_id not in self.workspaces:
            return False

        workspace = self.workspaces[workspace_id]

        # Let the workspace handle all deletion logic (conversations + directory)
        workspace.delete()

        del self.workspaces[workspace_id]
        return True

    def get_workspaces(self) -> List[Workspace]:
        """Get list of workspaces."""
        return list(self.workspaces.values())

    def get_workspace(self, workspace_id: str) -> Optional[Workspace]:
        """Get a workspace by ID."""
        return self.workspaces.get(workspace_id)

    def get_workspace_by_path(self, path: Path) -> Optional[Workspace]:
        """Get a workspace by path."""
        resolved_path = path.resolve()
        target_id = hashlib.sha256(str(resolved_path).encode()).hexdigest()[:16]
        return self.workspaces.get(target_id)
