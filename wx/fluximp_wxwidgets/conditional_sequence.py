# Conditional sequence orchestration for FluxImp.

import json
import pickle
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from fluximp_agent.conditional_agent import ConditionalAgentConfigInfo


@dataclass
class ConditionalSequenceState:
    event_key: str
    current_step_index: int = 0
    agent_outputs: Dict[str, str] = field(default_factory=dict)
    retrigger_counts: Dict[str, int] = field(default_factory=dict)
    status: str = "idle"


@dataclass
class ConditionalSequenceStateV2:
    """Versioned sequence state carrying the launched run's identity (§7.7).

    Adds ``active_run_id``/``active_io_dir`` — the run minted at the most
    recent launch — so a reopen can tie the persisted state to its run dir
    and stream evidence. Legacy ``ConditionalSequenceState`` pickles keep
    loading: ``load_state`` migrates them in memory via ``from_legacy`` and
    the next write persists as V2.
    """
    event_key: str
    current_step_index: int = 0
    agent_outputs: Dict[str, str] = field(default_factory=dict)
    retrigger_counts: Dict[str, int] = field(default_factory=dict)
    status: str = "idle"
    active_run_id: str = ""
    active_io_dir: Optional[str] = None

    @classmethod
    def from_legacy(cls, legacy: 'ConditionalSequenceState') -> 'ConditionalSequenceStateV2':
        """Migrate a legacy persisted state to V2 in memory (§7.7).

        A legacy state has no run identity; it simply arrives with
        ``active_run_id``/``active_io_dir`` unset, exactly like a V2 state
        that has never been launched under the new writer.
        """
        return cls(
            event_key=legacy.event_key,
            current_step_index=legacy.current_step_index,
            agent_outputs=dict(legacy.agent_outputs),
            retrigger_counts=dict(legacy.retrigger_counts),
            status=legacy.status,
        )


@dataclass
class SequenceCallbacks:
    """Callbacks the sequence engine uses to poke the GUI adapter (§9).

    ``subprocess_launched`` receives the minted ``run_id`` so the tab keys its
    ``_conditional_subprocesses``/``_conditional_io_dirs`` dicts by run identity
    (§7.7). ``pre_populate_user_input`` was deleted (§7.5): the engine no
    longer writes the input box — the message path is the sole owner.
    """
    show_resume_button: Callable[[str], None] = lambda event_key: None
    on_sequence_completed: Callable[[str], None] = lambda event_key: None
    subprocess_launched: Callable[[subprocess.Popen, str, Path, str], None] = (
        lambda proc, agent_name, io_dir, run_id: None)


def _state_dir(conversation_dir: Path) -> Path:
    d = conversation_dir / "conditional_agents"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _state_path(conversation_dir: Path, event_key: str) -> Path:
    safe_key = event_key.replace("/", "_").replace("\\", "_")
    return _state_dir(conversation_dir) / f"{safe_key}.pkl"


def save_state(conversation_dir: Path, state: ConditionalSequenceStateV2):
    path = _state_path(conversation_dir, state.event_key)
    with open(path, 'wb') as f:
        pickle.dump(state, f)


def _unpickle_state(path: Path) -> ConditionalSequenceStateV2:
    """Unpickle one state file through the §7.7 load dispatch.

    A ``ConditionalSequenceStateV2`` is returned as-is; a legacy
    ``ConditionalSequenceState`` migrates in memory via
    ``ConditionalSequenceStateV2.from_legacy``; any other value raises —
    the §7.1 step-0 probe turns that into ``non_resumable`` (§7.8).
    """
    with open(path, 'rb') as f:
        state = pickle.load(f)
    if isinstance(state, ConditionalSequenceStateV2):
        return state
    if isinstance(state, ConditionalSequenceState):
        return ConditionalSequenceStateV2.from_legacy(state)
    raise ValueError(
        f"unrecognized sequence-state pickle in {path}: {type(state).__name__}"
    )


def load_state(conversation_dir: Path, event_key: str) -> Optional[ConditionalSequenceStateV2]:
    path = _state_path(conversation_dir, event_key)
    if not path.exists():
        return None
    return _unpickle_state(path)


def load_all_states(conversation_dir: Path) -> Dict[str, ConditionalSequenceStateV2]:
    """Load every conditional sequence state in a conversation (§7.1 step 0).

    Globs ``conditional_agents/*.pkl`` and runs each file through the §7.7
    load dispatch, keyed by each state's own ``state.event_key``. Any file
    that fails to unpickle, or unpickles to a value the dispatch does not
    recognize, raises — the step-0 probe flags the workflow non-resumable
    (§7.8).
    """
    state_dir = conversation_dir / "conditional_agents"
    if not state_dir.exists():
        return {}
    states: Dict[str, ConditionalSequenceStateV2] = {}
    for path in sorted(state_dir.glob("*.pkl")):
        v2 = _unpickle_state(path)
        states[v2.event_key] = v2
    return states


def delete_state(conversation_dir: Path, event_key: str):
    path = _state_path(conversation_dir, event_key)
    if path.exists():
        path.unlink()

# ============================================================================
# Diff Patch Writing
# ============================================================================

def write_diff_patch(conversation_dir: Path, diff_content: str):
    """Write a diff patch file to the conversation's io directory.

    Called before advancing a sequence so that build_context can read
    the diff when 'diff' is in the agent's context_sources.
    """
    io_dir = conversation_dir / "io"
    io_dir.mkdir(parents=True, exist_ok=True)
    patch_path = io_dir / "diff.patch"
    patch_path.write_text(diff_content, encoding='utf-8')


# ============================================================================
# Context Construction
# ============================================================================

def build_context(conversation_messages: List[dict], conversation_dir: Path,
                  agent_config: ConditionalAgentConfigInfo,
                  state: ConditionalSequenceStateV2) -> str:
    """Build the context text for a conditional agent from its context_sources.

    Args:
        conversation_messages: List of dicts with 'role' and 'content' keys.
            Roles are 'user' or 'assistant'.
        conversation_dir: The conversation directory (diff.patch is at
            conversation_dir/io/diff.patch).
        agent_config: The conditional agent's configuration.
        state: The current sequence state (contains prior agent outputs).

    Returns:
        Combined context text with sections for each context source.
    """
    sections = []

    for source in agent_config.context_sources:
        if source == "user_and_assistant_messages":
            text = _format_conversation_messages(conversation_messages)
            if text:
                sections.append(text)
        elif source == "diff":
            text = _read_diff_patch(conversation_dir)
            sections.append(text)
        elif source == "prior_agent_message":
            text = _get_prior_agent_message(agent_config, state)
            if text:
                sections.append(text)

    return "\n\n".join(sections)


def build_conversation_context_messages(messages: list,
                                        fork_turn_state=None) -> list:
    """Filter conversation messages to those shown in the GUI conversation view.

    Only includes user messages and assistant messages that were sent to the
    user via send_message (MODEL_MESSAGE_TO_USER).  Excludes API responses
    (reasoning/tool-call text that accompanied tool calls), tool messages,
    permission requests, conditional agent reports, conditional agent messages,
    and system agent messages, as these do not appear as conversational
    messages in the GUI.

    When ``fork_turn_state`` is provided, fork-shadowed messages are excluded
    too, so the conditional agent sees exactly the conversation the user
    sees (the §5.8 positional predicate): a message is dropped when
    ``fork_turn_state.is_visible(msg.fork, msg.turn, view_fork)`` is False
    for ``view_fork = fork_turn_state.latest_fork``.  Messages with
    ``fork > latest_fork`` cannot legitimately appear (the IN fact precedes
    the message, §5.10) and ``is_visible`` raises for them — the fork guard
    keeps them, the same defensive posture the controller's §5.8 sweep
    uses.  ``None`` (the default) skips the fork dimension.

    Args:
        messages: List of message objects with *role*, *content*, and
            *metadata* attributes (e.g. ``ConversationMessage`` instances).
            Fork filtering additionally reads *fork* and *turn* (always
            present on ``ConversationMessage``).
        fork_turn_state: Optional ``ForkTurnState``; the conversation tab
            passes ``conversation.fork_turn_state`` so a conditional agent's
            context matches the tab's visible set.

    Returns:
        List of ``{'role': str, 'content': str}`` dicts.
    """
    result = []
    for msg in messages:
        role = getattr(msg, 'role', '')
        content = getattr(msg, 'content', '')
        metadata = getattr(msg, 'metadata', {})
        if fork_turn_state is not None:
            view_fork = fork_turn_state.latest_fork
            msg_fork = getattr(msg, 'fork', 0)
            if (msg_fork <= view_fork
                    and not fork_turn_state.is_visible(
                        msg_fork, getattr(msg, 'turn', 0), view_fork)):
                continue
        if role == 'user':
            result.append({'role': 'user', 'content': content})
        elif role == 'assistant':
            if metadata.get('is_api_response'):
                continue
            if metadata.get('is_conditional_report'):
                continue
            if metadata.get('is_conditional'):
                continue
            result.append({'role': 'assistant', 'content': content})
    return result


def _format_conversation_messages(messages: List[dict]) -> str:
    """Format conversation messages with [User]/[Assistant] headers."""
    lines = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            lines.append(f"[User]\n{content}")
        elif role == "assistant":
            if isinstance(content, str) and content.strip():
                lines.append(f"[Assistant]\n{content}")
    return "\n\n".join(lines)


def _read_diff_patch(conversation_dir: Path) -> str:
    """Read the diff patch file from the conversation's io directory."""
    patch_path = conversation_dir / "io" / "diff.patch"
    if patch_path.exists():
        content = patch_path.read_text(encoding='utf-8').strip()
        if content:
            return f"[Diff of changes since last work_completion_summary]\n{content}"
    return "[No diff available]"


def _get_prior_agent_message(agent_config: ConditionalAgentConfigInfo,
                              state: ConditionalSequenceStateV2) -> str:
    """Get the output from the agent that this agent depends on."""
    if not agent_config.depends_on:
        return ""

    output = state.agent_outputs.get(agent_config.depends_on, "")
    if output:
        return f"[From: {agent_config.depends_on}]\n{output}"
    return ""


# ============================================================================
# System Prompt Augmentation
# ============================================================================

_CONVERSATION_FORMAT_DESCRIPTION = (
    "\n\n---\n\n"
    "The user message below contains a conversation history between a user and an AI assistant, "
    "formatted with [User] and [Assistant] headers. Each section represents one message in the "
    "conversation. Use this context to understand what work has been done and what the current "
    "state of the project is."
)


def augment_prompt(base_prompt: str, context_sources: List[str]) -> str:
    """Augment a conditional agent system prompt based on its context sources."""
    if "user_and_assistant_messages" in context_sources:
        return base_prompt + _CONVERSATION_FORMAT_DESCRIPTION
    return base_prompt


# ============================================================================
# Orchestration Functions
# ============================================================================

def on_primary_agent_work_completion_summary(
    conversation_dir: Path,
    working_dir: Path,
    event_key: str,
    agent_list: List[ConditionalAgentConfigInfo],
    conversation_messages: List[dict],
    callbacks: SequenceCallbacks = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
):
    """Handle a work_completion_summary event from the primary agent.

    Called by the GUI when the primary agent sends a message tagged
    work_completion_summary. The diff.patch file is already written by
    the agent process (in langgraph_runner._process_chunk) before this
    function is called, so build_context can read it when needed.
    """
    if callbacks is None:
        callbacks = SequenceCallbacks()

    # Load or create state
    state = load_state(conversation_dir, event_key)
    if state is None:
        state = ConditionalSequenceStateV2(event_key=event_key)

    if state.status == "suspended":
        return  # User must press Resume button

    if state.status == "waiting_for_user":
        # The user sent a message, primary agent responded with work_completion_summary.
        # This is the retrigger signal.
        _handle_retrigger(conversation_dir, working_dir, event_key, state, agent_list,
                          conversation_messages, callbacks,
                          provider=provider, model=model)
        return

    if state.status in ("idle", "completed"):
        if state.status == "completed":
            state.current_step_index = 0
            state.agent_outputs.clear()
            state.retrigger_counts.clear()
        state.status = "running"
        save_state(conversation_dir, state)
        advance_sequence(conversation_dir, working_dir, event_key, state, agent_list,
                        conversation_messages, callbacks,
                        provider=provider, model=model)
        return

    # status == "running" - already running, ignore


def _handle_retrigger(
    conversation_dir: Path,
    working_dir: Path,
    event_key: str,
    state: ConditionalSequenceStateV2,
    agent_list: List[ConditionalAgentConfigInfo],
    conversation_messages: List[dict],
    callbacks: SequenceCallbacks,
    provider: Optional[str] = None,
    model: Optional[str] = None,
):
    """Handle a retrigger: the user responded and the primary agent produced
    work_completion_summary again. Re-run the current conditional agent
    with updated context.
    """
    if state.current_step_index >= len(agent_list):
        state.status = "completed"
        save_state(conversation_dir, state)
        callbacks.on_sequence_completed(event_key)
        return

    agent_config = agent_list[state.current_step_index]

    if agent_config.retrigger:
        # Re-run the same agent with updated context
        state.status = "running"
        save_state(conversation_dir, state)
        _launch_current_agent(conversation_dir, working_dir, state, agent_list,
                              conversation_messages, callbacks,
                              provider=provider, model=model)
    else:
        # No retrigger - advance to next step
        state.current_step_index += 1
        advance_sequence(conversation_dir, working_dir, event_key, state, agent_list,
                        conversation_messages, callbacks,
                        provider=provider, model=model)


def advance_sequence(
    conversation_dir: Path,
    working_dir: Path,
    event_key: str,
    state: ConditionalSequenceStateV2,
    agent_list: List[ConditionalAgentConfigInfo],
    conversation_messages: List[dict],
    callbacks: SequenceCallbacks = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
):
    """Advance the sequence to the next step, launching a conditional agent.

    Marks sequence completed if all steps are done.
    """
    if callbacks is None:
        callbacks = SequenceCallbacks()

    if state.current_step_index >= len(agent_list):
        state.status = "completed"
        save_state(conversation_dir, state)
        callbacks.on_sequence_completed(event_key)
        return

    agent_config = agent_list[state.current_step_index]

    # Build context and augmented prompt
    context_text = build_context(conversation_messages, conversation_dir,
                                 agent_config, state)
    augmented_prompt = augment_prompt(agent_config.prompt, agent_config.context_sources)

    state.status = "running"
    save_state(conversation_dir, state)

    # Launch the conditional agent subprocess
    _launch_agent_subprocess(conversation_dir, working_dir, agent_config,
                             augmented_prompt, context_text,
                             state=state,
                             callbacks=callbacks,
                             provider=provider, model=model)


def _launch_current_agent(
    conversation_dir: Path,
    working_dir: Path,
    state: ConditionalSequenceStateV2,
    agent_list: List[ConditionalAgentConfigInfo],
    conversation_messages: List[dict],
    callbacks: SequenceCallbacks = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
):
    """Build context and launch the current agent in the sequence."""
    agent_config = agent_list[state.current_step_index]
    context_text = build_context(conversation_messages, conversation_dir,
                                 agent_config, state)
    augmented_prompt = augment_prompt(agent_config.prompt, agent_config.context_sources)

    _launch_agent_subprocess(conversation_dir, working_dir, agent_config,
                             augmented_prompt, context_text,
                             state=state,
                             callbacks=callbacks,
                             provider=provider, model=model)


def on_conditional_agent_complete(
    conversation_dir: Path,
    working_dir: Path,
    event_key: str,
    agent_name: str,
    output: str,
    agent_list: List[ConditionalAgentConfigInfo],
    conversation_messages: List[dict],
    callbacks: SequenceCallbacks = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
):
    """Handle completion of a conditional agent subprocess.

    Called by the GUI when a conditional agent finishes and produces output.

    Args:
        conversation_dir: Path to the conversation working directory.
        working_dir: Path to the codebase directory.
        event_key: The event key for this sequence.
        agent_name: Name of the completed agent.
        output: The agent report output (empty string = nothing to report).
        agent_list: Ordered list of conditional agent configs for this event.
        conversation_messages: Conversation messages for context construction.
        callbacks: Optional callbacks for GUI operations.
    """
    if callbacks is None:
        callbacks = SequenceCallbacks()

    state = load_state(conversation_dir, event_key)
    if state is None:
        return  # No active sequence

    # Store the output
    state.agent_outputs[agent_name] = output

    # Get current agent config
    if state.current_step_index >= len(agent_list):
        state.status = "completed"
        save_state(conversation_dir, state)
        callbacks.on_sequence_completed(event_key)
        return

    agent_config = agent_list[state.current_step_index]

    if not output.strip():
        # Empty output - nothing to report, advance to next step immediately
        state.current_step_index += 1
        save_state(conversation_dir, state)
        advance_sequence(conversation_dir, working_dir, event_key, state,
                         agent_list, conversation_messages, callbacks,
                         provider=provider, model=model)
        return

    # Non-empty output. §7.5: the engine keeps state transitions only — the
    # `pre_populate_user_input` callback was deleted; the message path's
    # `_on_message` check is the sole box writer.
    if agent_config.retrigger:
        state.retrigger_counts[agent_name] = state.retrigger_counts.get(agent_name, 0) + 1

        if state.retrigger_counts[agent_name] <= agent_config.max_retriggers:
            # Still within retrigger limit - wait for user input
            state.status = "waiting_for_user"
            save_state(conversation_dir, state)
        else:
            # Exceeded max retriggers - suspend
            state.status = "suspended"
            save_state(conversation_dir, state)
            callbacks.show_resume_button(event_key)
    else:
        # No retrigger - show output and wait for user to continue
        state.status = "waiting_for_user"
        save_state(conversation_dir, state)


def on_resume_button_pressed(
    conversation_dir: Path,
    working_dir: Path,
    event_key: str,
    agent_list: List[ConditionalAgentConfigInfo],
    conversation_messages: List[dict],
    callbacks: SequenceCallbacks = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
):
    """Handle the Resume button press for a suspended sequence.

    Resets the retrigger count for the current agent and re-launches it.
    This is the same action as _handle_retrigger when retrigger is within limits.
    """
    if callbacks is None:
        callbacks = SequenceCallbacks()

    state = load_state(conversation_dir, event_key)
    if state is None:
        return
    if state.status == "suspended":
        # Reset retrigger counts for all agents
        for key in state.retrigger_counts:
            state.retrigger_counts[key] = 0
    elif state.status != "running":
        # Park-on-stop (§7.7): Resume relaunches a Stop-parked `running`
        # state; any other status is not resumable.
        return

    # Re-launch the current agent
    state.status = "running"
    save_state(conversation_dir, state)
    _launch_current_agent(conversation_dir, working_dir, state, agent_list,
                          conversation_messages, callbacks,
                          provider=provider, model=model)


def apply_offline_terminal_transition(
    conversation_dir: Path,
    event_key: str,
    agent_name: str,
    output: str,
    agent_list: List[ConditionalAgentConfigInfo],
):
    """Apply the §7.7 offline terminal transition (never launching).

    Mirrors ``on_conditional_agent_complete``'s branches but never launches
    and never fires a box callback: a report (non-empty output) stores the
    output and bumps the retrigger count (landing ``waiting_for_user``, or
    ``suspended`` past ``max_retriggers``); an empty report or an error
    marker advances ``current_step_index`` once with no retrigger bump
    (past the last step, complete the status bookkeeping), landing
    ``waiting_for_user``.
    """
    state = load_state(conversation_dir, event_key)
    if state is None:
        return

    state.agent_outputs[agent_name] = output

    if state.current_step_index >= len(agent_list):
        state.status = "completed"
        save_state(conversation_dir, state)
        return

    agent_config = agent_list[state.current_step_index]

    if not output.strip():
        # Empty report / error marker: step-advance once, no bump, no launch.
        state.current_step_index += 1
        if state.current_step_index >= len(agent_list):
            state.status = "completed"
        else:
            state.status = "waiting_for_user"
        save_state(conversation_dir, state)
        return

    # Report / success marker: retrigger-path bookkeeping, never launching.
    if agent_config.retrigger:
        state.retrigger_counts[agent_name] = state.retrigger_counts.get(agent_name, 0) + 1
        if state.retrigger_counts[agent_name] <= agent_config.max_retriggers:
            state.status = "waiting_for_user"
        else:
            state.status = "suspended"
    else:
        state.status = "waiting_for_user"
    save_state(conversation_dir, state)


def reset_sequence(conversation_dir: Path, event_key: str):
    """Delete the sequence state, allowing it to start fresh on the next event."""
    delete_state(conversation_dir, event_key)


# ============================================================================
# Subprocess Launching
# ============================================================================

def _launch_agent_subprocess(
    conversation_dir: Path,
    working_dir: Path,
    agent_config: ConditionalAgentConfigInfo,
    augmented_prompt: str,
    context_text: str,
    state: Optional[ConditionalSequenceStateV2] = None,
    callbacks: SequenceCallbacks = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> subprocess.Popen:
    """Launch a conditional agent as a subprocess.

    Creates an I/O subdirectory, writes the augmented config and input,
    and launches main.py with the appropriate arguments.

    Args:
        conversation_dir: Path to the conversation metadata directory
            (e.g. ~/.fluximp/workspaces/.../workflows/<id>/).
        working_dir: Path to the codebase directory (e.g. C:\code\myproject).
            Passed as --working-dir to the subprocess.
        agent_config: The conditional agent's configuration.
        augmented_prompt: The agent's system prompt (possibly augmented).
        context_text: The context text to send as the first user message.
        callbacks: Optional callbacks for GUI operations.
    """
    from fluximp_protocol.user_agent.file_io.user import FileIOUserProtocol
    from fluximp_agent.process_utils import build_agent_command, launch_agent_process

    # Ensure conversation_dir is absolute so all derived paths are absolute.
    # If conversation_dir is relative, .resolve() would use CWD which may be wrong.
    conversation_dir = conversation_dir.resolve()

    # Create I/O subdirectory with UUID suffix (like scrutinator/summarizer).
    # Normalize agent name: lowercase, spaces -> underscores (matches config filename convention).
    normalized_name = agent_config.name.lower().replace(" ", "_")
    run_id = uuid.uuid4().hex[:8]
    io_dir = conversation_dir / "io" / f"conditional_{normalized_name}_{run_id}"
    io_dir.mkdir(parents=True, exist_ok=True)

    # §7.7: persist the launched run's identity on the sequence state so a
    # reopen can tie the persisted state to its run dir / stream evidence.
    if state is not None:
        state.active_run_id = run_id
        state.active_io_dir = str(io_dir)
        save_state(conversation_dir, state)

    # Write augmented config to a temp file in the I/O directory
    config_data = {
        "conditional_agents": {
            agent_config.name: {
                "prompt": augmented_prompt,
                "tool_set": agent_config.tool_set,
                "context_sources": agent_config.context_sources,
                "retrigger": agent_config.retrigger,
                "max_retriggers": agent_config.max_retriggers,
                "max_requests": agent_config.max_requests,
                "depends_on": agent_config.depends_on,
            }
        }
    }
    config_path = io_dir / "agent_config.json"
    config_path.write_text(json.dumps(config_data, indent=2), encoding='utf-8')

    # Write the context text as the initial user message using the file I/O
    # protocol wire format (0.0.in.txt).  This is the same pattern used by
    # scrutinator/summarizer â see AgentService._run_oneshot_agent().
    protocol = FileIOUserProtocol(io_dir)
    protocol.send_message(context_text, 0, 0)

    # Build the CLI command using the shared process_utils helper.
    # --agent-type conditional routes to the conditional agent entry point.
    # --conditional-config tells it which agent config to load.
    cmd = build_agent_command(
        agent_type='conditional',
        io_dir=io_dir,
        working_dir=working_dir,
        provider=provider,
        model=model,
        extra_args=['--conditional-config', str(config_path.resolve())],
    )

    # Launch with stdout/stderr redirected to agent_stdout.log (same pattern
    # as the primary agent â see Conversation.start_agent_process()).
    stdout_log_path = io_dir / "agent_stdout.log"
    proc = launch_agent_process(
        cmd,
        stdout=open(stdout_log_path, 'a', encoding='utf-8'),
        stderr=subprocess.STDOUT,
    )

    # Notify GUI that subprocess was launched (include io_dir for output reading
    # and the minted run_id so the tab keys its dicts by run identity, §7.7).
    if callbacks is not None:
        callbacks.subprocess_launched(proc, agent_config.name, io_dir, run_id)

    return proc
