#!/usr/bin/env python3
"""
Coding Agent using LangGraph with checkpointing and tool nodes.
This agent uses LangGraph's built-in checkpointing for time travel
while maintaining application-level undo/redo for file operations.

Tool functions are managed through a dynamic ToolRegistry for flexibility.
"""

from __future__ import annotations

# Force UTF-8 encoding for stdout/stderr to handle Unicode characters
import sys

import logging

import os
import time
import argparse
from typing import Optional
from pathlib import Path

from fluximp_agent.agent_logger import AgentLogger
from langgraph.graph import StateGraph, START, END
from fluximp_agent.time_travel import create_checkpointer
from fluximp_agent.checkpoint_markers import write_checkpoint_marker

from fluximp_agent.model_config import get_model_config, create_provider, get_default_provider_name, get_provider_names, get_model_capabilities
from fluximp_agent.agent_config import AgentConfig

# Import tool functions from the tool_funcs package
from fluximp_agent.filesystem_service import FileSystemService
from fluximp_agent.process_service import ProcessService
from fluximp_agent.state_service import StateService
from fluximp_agent.agent_service import AgentService
from fluximp_agent.experiments import ExperimentsService
from fluximp_agent.fs_agent_state import AgentState

from fluximp_protocol.user_agent.base import RequestLifecycleHooks, MaxRequestsExceeded, TurnCancelled
from fluximp_protocol.user_agent.file_io.agent import FileIOAgentProtocol
from fluximp_protocol.user_agent.file_io.wire_format import scan_io_facts, expected_terminal_marker_for_agent
from fluximp_protocol.user_agent.fork_turn_state import IOFactKind, IOTurnStatus
from fluximp_agent.langgraph_runner import LangGraphRunner

# ============================================================================
# Tool Registry, Tools Node, and Should Continue
# Imported from tool_registry, tools_node
# ============================================================================
from fluximp_agent.tool_registry import create_tool_registry
from fluximp_agent.tools_node import route_after_parse, route_after_execute
from fluximp_agent.fs_tool_nodes import create_should_continue
from fluximp_agent.fs_tool_nodes import create_parse_tool_calls_node, create_execute_tool_call_node

# ============================================================================
# Agent Node
# ============================================================================
from fluximp_agent.fs_agent_node import AgentNode

# ============================================================================



# ============================================================================
# Agent Creation
# ============================================================================
def create_agent(filesystem_service: FileSystemService, agent_config: AgentConfig, services: dict = None, max_requests: int = 5, max_transient_retries: int = 8, lifecycle_hooks: Optional[RequestLifecycleHooks] = None, model_client = None, checkpoint_db_path: Optional[Path] = None, skip_llm_next: list = None, thread_id_ref: list = None):
    """Create a LangGraph-based coding agent with checkpointing.

    Args:
        filesystem_service: The FileSystemService instance for file operations
        agent_config: The AgentConfig instance for configuration
        services: Optional Services dict (created from registry if None)
        max_requests: Maximum number of API requests allowed per session
        max_transient_retries: Maximum retries for transient 429/infrastructure errors
        model_client: Optional pre-configured model client (creates one if None)
        checkpoint_db_path: Optional path to SQLite database for persistent checkpoints.
                          If None, uses InMemorySaver (non-persistent).

    Returns:
        Tuple of (compiled_graph, memory_checkpointer, agent_node) where:
        - compiled_graph: The compiled LangGraph state graph
        - memory_checkpointer: The checkpointer (InMemorySaver or SqliteSaver)
        - agent_node: The AgentNode instance (for fork notification etc.)
    """
    registry = services["registry"]

    # Create the agent node (encapsulates LLM interaction, retry, context injection)
    agent_node = AgentNode(
        filesystem_service=filesystem_service,
        agent_config=agent_config,
        registry=registry,
        max_requests=max_requests,
        max_transient_retries=max_transient_retries,
        lifecycle_hooks=lifecycle_hooks,
        model_client=model_client,
        skip_llm_next=skip_llm_next,
        thread_id_ref=thread_id_ref,
    )

    # Create per-tool-call pipeline nodes and should_continue
    parse_node = create_parse_tool_calls_node(services, agent_config=agent_config)
    execute_node = create_execute_tool_call_node(services)
    should_continue_func = create_should_continue(services)

    # Build the state graph with checkpointing
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("parse_tool_calls", parse_node)
    workflow.add_node("execute_tool_call", execute_node)

    # Add edges with conditional routing
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue_func,
        {
            "continue": "parse_tool_calls",
            "end": END
        }
    )
    workflow.add_conditional_edges(
        "parse_tool_calls",
        route_after_parse,
        {
            "execute": "execute_tool_call",
            "agent": "agent"
        }
    )
    workflow.add_conditional_edges(
        "execute_tool_call",
        route_after_execute,
        {
            "continue": "execute_tool_call",
            "agent": "agent",
            "end": END
        }
    )

    # Create checkpointer using utility function
    memory = create_checkpointer(checkpoint_db_path)

    compiled = workflow.compile(checkpointer=memory)

    # Log graph info
    agent_node.logger.log_graph_info(workflow)

    return compiled, memory, agent_node


# ============================================================================
# Helper Functions
# ============================================================================


# ============================================================================
# File-based I/O Mode
# ============================================================================

def file_io_mode(filesystem_service: FileSystemService, agent_config: AgentConfig, io_dir: str, experiment_service: ExperimentsService, max_requests: int = 5, max_transient_retries: int = 8, input_timeout: float = 120.0, permission_timeout: float = 120.0, persist_checkpoints: bool = False, single_turn: bool = False, args=None, model_config=None):
    """Run the agent in file-based I/O mode for testing.

    Reads input from {fork}.{turn}.in.txt files and writes output to {fork}.{turn}.out.txt files.
    Both fork and turn start at 0 and increase monotonically.
    Fork increments when time traveling creates a new conversation branch.

    Time travel is triggered when a client writes an input file with an incremented fork number.
    For example, after 0.0.in.txt and 0.1.in.txt, writing 1.1.in.txt signals time travel:
    - Fork 1 > current fork 0, so we time travel
    - Turn 1 means we restore to checkpoint after turn 0 (before turn 1)
    - Then continue with the new human message on fork 1

    Args:
        filesystem_service: The FileSystemService instance for file operations
        agent_config: The AgentConfig instance for configuration
        io_dir: Directory containing the conversation files
        max_requests: Maximum number of API requests allowed
        max_transient_retries: Maximum retries for transient 429/infrastructure errors
        input_timeout: Timeout in seconds for waiting on input files (default: 120)
        permission_timeout: Timeout in seconds for waiting on permission responses (default: 120)
        persist_checkpoints: Whether to persist checkpoints to SQLite (default: False)
    """
    io_path = Path(io_dir).resolve()
    if not io_path.exists():
        print(f"Error: I/O directory does not exist: {io_path}")
        return

    # Create logger with agent_config
    logger = AgentLogger(agent_config, verbose=True)
    logger.start_session()

    try:
        # Create the FileIOAgentProtocol (owns handles, wire format, request headers)
        protocol = FileIOAgentProtocol(
            io_path=io_path,
            agent_type=agent_config.agent_type,
            provider=agent_config.model_config.provider.name,
            requested_model=agent_config.model_config.model,
            input_timeout=input_timeout,
            permission_timeout=permission_timeout
        )

        if args.test_runner_plugin_dir:
            from fluximp_agent.test_runner_plugin_manager import TestRunnerPluginManager
            plugin_manager = TestRunnerPluginManager(args.test_runner_plugin_dir, working_dir=args.working_dir, read_root_dir=args.read_root_dir)
        else:
            plugin_manager = None

        # Create the process service for Python script execution
        process_service = ProcessService(filesystem_service, plugin_manager=plugin_manager, require_permission_for_new_scripts=args.always_require_permission_for_new_scripts)

        # Create state service for accessing file operations state
        state_service = StateService(filesystem_service=filesystem_service)

        # Create agent service for running sub-agents (io_path is the conversation directory)
        agent_service = AgentService(io_path, primary_protocol=protocol, model_config=model_config)

        # Create search services (only for providers with API keys in keyring)
        search_services = {}
        try:
            from fluximp_agent.search_service_tavily import TavilySearchService
            search_services["tavily_search_service"] = TavilySearchService(
                filesystem_service=filesystem_service,
                state_service=state_service,
                experiments_service=experiment_service,
                io_path=io_path,
            )
        except RuntimeError as e:
            print(f"Search service not available: {e}")


        try:
            from fluximp_agent.search_service_exa import ExaSearchService
            search_services["exa_search_service"] = ExaSearchService(
                filesystem_service=filesystem_service,
                state_service=state_service,
                experiments_service=experiment_service,
                io_path=io_path,
            )
        except RuntimeError as e:
            print(f"Exa search service not available: {e}")

        try:
            from fluximp_agent.search_service_tinyfish import TinyFishSearchService
            search_services["tinyfish_fetch_service"] = TinyFishSearchService(
                filesystem_service=filesystem_service,
                state_service=state_service,
                experiments_service=experiment_service,
                io_path=io_path,
            )
        except RuntimeError as e:
            print(f"TinyFish fetch service not available: {e}")



        try:
            from fluximp_agent.search_service_linkup import LinkupSearchService
            search_services["linkup_search_service"] = LinkupSearchService(
                filesystem_service=filesystem_service,
                state_service=state_service,
                experiments_service=experiment_service,
                io_path=io_path,
            )
        except RuntimeError as e:
            print(f"Linkup search service not available: {e}")

        try:
            from fluximp_agent.search_service_parallel import ParallelSearchService
            search_services["parallel_search_service"] = ParallelSearchService(
                filesystem_service=filesystem_service,
                state_service=state_service,
                experiments_service=experiment_service,
                io_path=io_path,
            )
        except RuntimeError as e:
            print(f"Parallel search service not available: {e}")

        try:
            from fluximp_agent.search_service_perplexity import PerplexitySearchService
            search_services["perplexity_search_service"] = PerplexitySearchService(
                filesystem_service=filesystem_service,
                state_service=state_service,
                experiments_service=experiment_service,
                io_path=io_path,
            )
        except RuntimeError as e:
            print(f"Perplexity search service not available: {e}")


        # Web search/fetch are optional: when an experiment is enabled but the
        # backing service could not be created (e.g. no API key in keyring),
        # drop the corresponding tools instead of failing the whole run.
        search_service_tool_names = {
            "tavily_search_service": ["tavily_search", "tavily_extract"],
            "exa_search_service": ["exa_search", "exa_contents"],
            "tinyfish_fetch_service": ["tinyfish_fetch", "tinyfish_search"],
            "linkup_search_service": ["linkup_search", "linkup_fetch"],
            "parallel_search_service": ["parallel_search", "parallel_extract"],
            "perplexity_search_service": ["perplexity_search"],
        }
        for service_key, tool_names in search_service_tool_names.items():
            if service_key in search_services:
                continue
            for tool_name in tool_names:
                if tool_name in agent_config.tool_names:
                    agent_config.tool_names.remove(tool_name)
                    print(
                        f"Search/fetch tool '{tool_name}' disabled: "
                        f"{service_key} is not available (no API key in keyring)."
                    )
        # Create registry first so we can set permission handles per turn
        registry = create_tool_registry(filesystem_service, agent_config, process_service, state_service, agent_service, experiment_service, **search_services)

        # Determine checkpoint database path if persistence is enabled
        checkpoint_db_path = None
        if persist_checkpoints:
            checkpoints_dir = io_path / ".checkpoints"
            checkpoint_db_path = checkpoints_dir / "checkpoints.db"
        
        # Create the LangGraph runner (owns stream execution, checkpointing, time travel)
        runner = LangGraphRunner(
            filesystem_service=filesystem_service,
            agent_config=agent_config,
            process_service=process_service,
            registry=registry,
            agent_protocol=protocol,
            max_requests=max_requests,
            max_transient_retries=max_transient_retries,
            checkpoint_db_path=checkpoint_db_path,
            state_service=state_service,
        )
        agent, memory = runner.create(lifecycle_hooks=protocol)
        print(f"Agent Type: {agent_config.agent_type}")
        print(f"I/O directory: {io_path}")
        print(f"Max requests: {max_requests}")
        print(f"Test runner plugin directories: {plugin_manager.plugin_directories if plugin_manager else None}")
        print()

        # Build the immutable fork/turn state from the whole I/O directory.
        state = scan_io_facts(io_path,
                              expected_terminal_marker_for_agent(agent_config.agent_type))
        recovered_position = state.recovery_position()

        if recovered_position is not None:
            # Resume from existing conversation
            file_fork, file_turn = recovered_position.fork, recovered_position.turn
            print(f"Recovered conversation state: fork={file_fork}, turn={file_turn}")
            print(f"  (Resuming from previous run - will skip already-processed turns)")
            print()

            if not persist_checkpoints:
                raise Exception("Can't recover state when not persisting checkpoints")
            
            # Recover state from checkpoints
            result = runner.recover_state(memory, state_service)
            config = result.config
            checkpoint_fork = result.fork
            checkpoint_turn = result.turn
            thread_id = runner._thread_id

            # Classify the startup cursor against the recovered position. This
            # preserves today's raise/check order (fork-ahead first, then the
            # mid-turn/normal classification, and only the residual same-fork
            # behind state raises) -- see the ordered rules in section 2.
            start = state.compute_start_position(checkpoint_fork, checkpoint_turn)
            fork = start.fork
            turn = start.turn
            is_mid_turn_recovery = start.is_mid_turn_recovery

            # A CANCELLED turn is terminal (recovery_position() includes it), but
            # a checkpoint that matches it means the agent was interrupted
            # mid-turn (it wrote cancel_ack and exited). Resume that turn for
            # mid-turn recovery instead of skipping to the next turn.
            recovered_out = next(
                (f for f in state.facts
                 if f.kind is IOFactKind.OUT
                 and f.fork == recovered_position.fork
                 and f.turn == recovered_position.turn),
                None,
            )
            if (checkpoint_fork == recovered_position.fork
                    and checkpoint_turn == recovered_position.turn
                    and recovered_out is not None
                    and recovered_out.status is IOTurnStatus.CANCELLED):
                fork = checkpoint_fork
                turn = checkpoint_turn
                is_mid_turn_recovery = True

            print(f"  Start position: fork {fork}, turn {turn} (mid-turn={is_mid_turn_recovery})")
            is_fresh_start = False
        else:
            # Fresh start
            config, thread_id = runner.fresh_start()
            fork = 0
            turn = 0
            is_mid_turn_recovery = False
            print(f"Starting new conversation: thread_id={thread_id}")
            print()
            is_fresh_start = True

            # Initialize StateService for fresh start with empty state
            fresh_state = {
                "file_operations": [],
                "operation_position": -1,
                "cumulative_diff_last_op_pos": -1,
                "cumulative_diff_last_content": "",
                "cumulative_diff_last_changed_ts": 0.0,
                "cumulative_diff_creation_ts": 0.0,
                "last_scrutinized_position": -1,
                "file_view_windows": {},
            }
            state_service.update_state_reference(fresh_state)

        # Validate end files before entering processing loop
        if protocol.validate_end_files(state, fork, turn, is_fresh_start):
            return

        # Track if this is the first turn after recovery (for output_path optimization)
        is_first_turn_after_recovery = recovered_position is not None

        # Pre-populate seen files from the recovered facts
        protocol.pre_populate_seen_files(state)

        # ====================================================================
        # COORDINATOR: Main event loop
        # ====================================================================

        while True:
            # === COORDINATOR OWNS THE INPUT POLLING LOOP ===
            input_wait_start = time.time()
            while True:
                result = protocol.check_for_new_input(fork, turn)
                if result and (result.input_file or result.end_file):
                    break
                time.sleep(0.05)
                if time.time() - input_wait_start > input_timeout:
                    print(f"Timeout: No input file received within {input_timeout}s")
                    return

            # Check for end signal
            if result.end_file is not None and result.end_file.exists():
                print(f"End signal found at fork {fork}, turn {turn}. Stopping.")
                break

            # Track whether we're forking (starting a new branch)
            is_forking = result.fork > fork

            # Time travel if needed
            if is_forking:
                print(f"\n{'='*60}")
                print(f"TIME TRAVEL DETECTED: Fork {fork} -> {result.fork}, continuing at turn {result.turn}")
                print(f"{'='*60}")

                # When abandoning mid-turn recovery by forking, the latest checkpoint
                # is at checkpoint_turn (mid-turn). Pass turn + 1 as current_turn so
                # that current_turn - 1 = checkpoint_turn matches the latest checkpoint.
                effective_current_turn = turn + 1 if is_mid_turn_recovery else turn
                tt = runner.execute_time_travel(
                    memory=memory, config=config, thread_id=thread_id,
                    current_fork=fork, current_turn=effective_current_turn,
                    file_turn=result.turn,
                    filesystem_service=filesystem_service,
                    state_service=state_service
                )
                config = tt.config
                fork = result.fork
                turn = result.turn

            # If not forking but result has a newer turn, the user abandoned the
            # current turn (e.g., stopped mid-turn then sent a new message).
            # Update turn and skip mid-turn recovery.
            if not is_forking and result.turn > turn:
                print(f"  Abandoning mid-turn recovery for turn {turn}, starting turn {result.turn}")
                turn = result.turn
                recovered_position = None
                is_first_turn_after_recovery = False
            # Mark this input file as seen so we don't process it again
            protocol.mark_seen(result.input_file)

            # Open file handles for the current turn (protocol owns them)
            protocol.open_handles(result.input_file, result.output_file)

            # Check if this is a mid-turn recovery
            is_mid_turn_recovery = runner.check_mid_turn_recovery(
                memory=memory, config=config, turn=turn,
                is_forking=is_forking,
                is_first_turn_after_recovery=is_first_turn_after_recovery,
                recovered_position=recovered_position
            )

            # Tell the protocol whether we are in a mid-turn recovery so
            # request_permission() only replays prior grants (prior approval)
            # during an actual recovery, never for a fresh in-turn re-request.
            protocol.set_mid_turn_recovery(is_mid_turn_recovery)

            # Set permission service on the registry for this turn
            # The protocol implements ToolPermissionService, so pass it directly
            registry.set_permission_service(protocol)
            is_first_turn_after_recovery = False  # Clear flag after first turn

            msg_type, user_input = protocol.read_user_message()

            assert msg_type == 'user_message'

            # Drain stale permission responses from the input stream before
            # stream processing begins. During mid-turn recovery, the input file
            # is re-opened from position 0, so tool_permission_response messages
            # from the original run are still present. Each acknowledged request
            # in the output file corresponds to one stale response to drain.
            if is_mid_turn_recovery:
                protocol.drain_stale_permission_responses()
                protocol.drain_stale_cancel_turn()

            print(f"\n{'='*60}")
            print(f"FORK {fork}, TURN {turn}")
            print(f"{'='*60}")
            print(f"Input: {user_input}")

            # Log user message
            logger.log_user_message(user_input)

            # Start the LangGraph stream for this turn
            runner.start_stream(user_input, turn, fork, config,
                               is_mid_turn_recovery=is_mid_turn_recovery)

            # === COORDINATOR OWNS THE TURN LOOP ===
            cancel = False
            while True:
                # Check cancel (non-blocking, file-I/O-specific)
                # Skip during mid-turn recovery -- the input stream may contain a
                # tool_permission_response that request_permission() must consume
                # as part of recovery. Once recovery completes, cancel checking
                # is safe because the stream position will be past the response.
                if not is_mid_turn_recovery and protocol.check_cancel_turn():
                    protocol.write_cancel_ack()
                    cancel = True
                    break

                # Advance stream by one chunk (events are written directly
                # to the protocol inside advance_stream)
                if not runner.advance_stream():
                    # stream complete
                    # If we're recovering mid-turn that is done as well since
                    # we're done with this turn.
                    is_mid_turn_recovery = False
                    protocol.set_mid_turn_recovery(False)
                    break  

                # Mid-turn recovery is complete after the first chunk with content.
                # Stale permission responses were already drained before
                # stream processing began (after read_user_message), so
                # cancel checking is safe.
                is_mid_turn_recovery = False
                protocol.set_mid_turn_recovery(False)

            if not cancel:
                # Verify checkpoint and write marker file (before closing handles!)
                checkpoint_id, config = runner.verify_checkpoint(memory, thread_id, fork, turn)
                if persist_checkpoints:
                    write_checkpoint_marker(checkpoints_dir, fork, turn, checkpoint_id)

                # Note: diff.patch is written by langgraph_runner._process_chunk() when
                # work_completion_summary is detected, making it available immediately
                # for conditional agents during the same turn.

                print(f"Output written to: {result.output_file}")
                print(f"  State: request_count={runner.request_count}/{max_requests}")

                turn += 1

                # For conditional agents: exit after processing a single turn.
                # The conditional agent calls submit_report and the conversation
                # ends â there is no second user input.
                if single_turn:
                    break

            # Close file handles for this turn
            protocol.close_handles()

            if cancel:
                break

        print(f"\n{'='*60}")
        print(f"File I/O mode completed. Processed {turn} turns.")
        print(f"{'='*60}")

    except MaxRequestsExceeded as e:
        # Write marker to output stream for test runner to detect
        protocol.write_max_requests_reached(str(e))
        print(f"Max requests reached: {e}")

    except TurnCancelled as e:
        print(f"Turn cancelled: {e}")
        try:
            protocol.close_handles()
        except Exception:
            pass

    except Exception as e:
        print(f"Error in file I/O mode: {e}")
        import traceback
        traceback.print_exc()
        print("DEBUG after traceback.print_exc()")

        # Append error information to output file for debugging
        try:
            if protocol._output_file:
                import io
                tb_buffer = io.StringIO()
                traceback.print_exc(file=tb_buffer)

                error_content = f"\nERROR:\n{'-' * 40}\n{str(e)}\n\nTRACEBACK:\n{tb_buffer.getvalue()}"
                print(error_content)
        except Exception as write_error:
            print(f"Failed to write error to output file: {write_error}")

        sys.stdout.flush()
        sys.exit(1)

    logger.end_session()


# ============================================================================
# Entry Point
# ============================================================================
def main() -> int:

    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    from fluximp_agent import fs_agents

    provider_names = get_provider_names()
    default_provider = get_default_provider_name()

    builtin_agent_types = ", ".join(sorted(fs_agents.AGENT_DEFINITIONS.keys()))
    parser = argparse.ArgumentParser(description="Fluximp Coding Agent POC (LangGraph)")
    parser.add_argument("--agent-type", type=str, default="coder",
                        help=f"Set the agent type (default: coder).  Built-in types: {builtin_agent_types}.  Additional types may be available via --agent-config.")
    parser.add_argument("--agent-config", type=str, action="append", default=[],
                        help="Path to a JSON agent config file.  May be specified multiple times.  If the same agent type appears in more than one config file the the process will fail.  Config definitions also replace built-in agent types by the same rule.")
    parser.add_argument("--provider", type=str, default=default_provider, choices=provider_names,
                        help=f"Model provider to use (default: {default_provider})")
    parser.add_argument("--model", type=str,
                        help="Model name to use. If not specified, uses the provider's default model")
    parser.add_argument("--working-dir", type=str, default=".",
                        help="Set the working directory for file operations (default: current directory)")
    parser.add_argument("--read-root-dir", type=str, default=None,
                        help="Set the root directory for read/query operations. Must be an ancestor of or equal to --working-dir. (default: same as --working-dir)")
    parser.add_argument("--write-file", type=str, default=None,
                        help="Restrict write access to a single file. Must be within --working-dir. (default: None, write access to full working directory)")
    parser.add_argument("--io-dir", type=str, required=True,
                        help="Directory for conversation files ({fork}.{turn}.in.txt and {fork}.{turn}.out.txt)")
    parser.add_argument("--max-requests", type=int, default=5,
                        help="Maximum number of API requests allowed per session (default: 5)")
    parser.add_argument("--max-transient-api-error-retries", type=int, default=9,
                        help="Maximum retries for transient API errors like 429 (default: 9)")
    parser.add_argument("--input-timeout", type=float, default=120.0,
                        help="Timeout in seconds for waiting on input files in file I/O mode (default: 120)")
    parser.add_argument("--permission-timeout", type=float, default=120.0,
                        help="Timeout in seconds for waiting on permission responses from the user (default: 120)")
    parser.add_argument("--persist-checkpoints", action="store_true", default=False,
                        help="Persist checkpoints to SQLite database for conversation resumption (default: False)")
    parser.add_argument("--always-require-permission-for-new-scripts", action="store_true", default=False,
                        help="Require permission when running scripts created by the assistant (default: False)")
    parser.add_argument("--test-runner-plugin-dir", type=str, default=None,
                        help="Directory containing test runner plugins. If not provided, run_tests tool will be disabled.")
    parser.add_argument("--enable-experiment", type=str, action="append", default=[],
                        help="Enable an additional experiment. Can be specified multiple times.")
    parser.add_argument("--disable-experiment", type=str, action="append", default=[],
                        help="Disable an experiment. Can be specified multiple times.")
    parser.add_argument("--conditional-config", type=str, default=None,
                        help="Path to a JSON file containing ConditionalAgentConfigInfo. When present, runs in conditional agent mode.")

    parser.add_argument("--conversation-difficulty", type=int, default=None,
                        help="Conversation difficulty level (0+). When set, the provider selects a model appropriate for the difficulty. Overrides the default model but not an explicit --model arg.")
    args = parser.parse_args()

    # Load agent config files (if any) and validate agent_type
    from fluximp_agent import fs_agents
    if args.agent_config:
        from fluximp_agent.agent_config_loader import merge_agent_configs
        merge_agent_configs(fs_agents.AGENT_DEFINITIONS, args.agent_config)

    # Validate agent_type against known types (built-in + config-defined)
    _BUILTIN_SPECIAL_TYPES = {"scrutinator", "summarizer", "conditional"}
    known_types = set(fs_agents.AGENT_DEFINITIONS.keys()) | _BUILTIN_SPECIAL_TYPES
    if args.agent_type not in known_types:
        parser.error(
            f"Invalid agent type '{args.agent_type}'. "
            f"Available types: {sorted(known_types)}"
        )

    # Hoist the SDK-availability check: with a plugin dir given but the SDK
    # absent, disable run_tests exactly as when no plugin dir was provided.
    if args.test_runner_plugin_dir is not None:
        from fluximp_agent.test_runner_plugin_manager import SDK_AVAILABLE
        if not SDK_AVAILABLE:
            print("Error: fluximp-agent-sdk is not installed; the run_tests "
                  "tool is disabled. Install fluximp-agent-sdk to enable it.")
            args.test_runner_plugin_dir = None
    try:
        # Create the filesystem service with the working directory and optional read root directory
        filesystem_service = FileSystemService(working_dir=args.working_dir, read_root_dir=args.read_root_dir, write_file=args.write_file)
        print(f"Working directory set to: {filesystem_service.working_dir}")
        if filesystem_service.read_root_dir != filesystem_service.working_dir:
            print(f"Read root directory set to: {filesystem_service.read_root_dir}")
        if filesystem_service.write_file is not None:
            print(f"Write access restricted to: {filesystem_service.write_file}")

        # Create agent_config with ModelConfig (model-specific settings)
        provider = create_provider(args.provider)
        # Resolve model: explicit --model takes precedence, then difficulty-based, then default
        if args.model is not None:
            model_name = args.model
        elif args.conversation_difficulty is not None:
            model_name = provider.get_model_for_difficulty(args.conversation_difficulty)
        else:
            model_name = None  # get_model_config will use provider.get_default_model()
        model_config = get_model_config(provider=provider, model=model_name)
        assert args.model == args.model
        model_capabilities = get_model_capabilities(model_config)
        
        enabled_experiments = {'file_view_windows', 'edit_line_validation', 'insert_line_validation', 'move_line_validation', 'search_list_view', 'line_content_multiline_fix', 'tavily_search', 'tavily_extract', 'exa_search', 'tinyfish_fetch', 'exa_contents', 'tinyfish_search', 'parallel_search', 'parallel_extract', 'perplexity_search'}
        if not model_capabilities.able_to_use_python_like_protocol_for_tool_function:
            # Both Minimmax-M2.5 and Minimmax-M2.7 seems to have trouble with levels and would give number of spaces instead
            #enabled_experiments.add('adjust_indent_validation')
            enabled_experiments.add('adjust_indent_exact_validation')
            # And GLM5.1 had issues with leading whitespace using atlascloud.com with the openai compatiable native JSON API
            enabled_experiments.add('edit_line_validation_first_line_whitespace')            
            enabled_experiments.add('insert_line_validation_first_line_whitespace')            

        # With this enabled mimo-v2.5-pro failed to provides its review after 9.6M prompt tokens on a workflow at
        # had been successful on most prior attempts.  On the second try it provided its review after 5.4M prompt
        # tokens but failed to complete the edits for its first change after 4.6M prompt tokens (it was having a
        # lot of trouble making tool calls).
        # Other models didn't have any obvious trouble
        # enabled_experiments.add('close_views')        

        model_lower = model_config.model.lower()
        allow_more_than_one_system_message = True
        if "minimax-m2" in model_lower:
            # While this technically allowed performace seemed to degrade significantly (at least on minimax.io didn't try others)
            allow_more_than_one_system_message = False
        elif model_config.provider.name == "siliconflow.com" and model_config.model in (
                # Limited to one system prompt
                "Qwen/Qwen3.6-35B-A3B",
                "Qwen/Qwen3.6-27B",
                "Qwen/Qwen3.5-397B-A17B",
            ):
                allow_more_than_one_system_message = False

        # Lenia 2D benchmarks didn't find any model & provider combos where
        # this worked well for the native JSON API, so just skip it for now        
        if not model_capabilities.able_to_use_python_like_protocol_for_tool_function:
            allow_more_than_one_system_message = False

        if allow_more_than_one_system_message:
            enabled_experiments.add('system_prompt_context_injection')

        enabled_experiments.update(args.enable_experiment)
        enabled_experiments.difference_update(args.disable_experiment)
        enabled_experiments.add('create_file')

        # Enable two_part_system_prompt when the Python-like protocol is in use.
        # In standard JSON API mode, tools are placed in the context before the
        # system prompt, so changing agents would break the entire prefix cache and
        # there is no benefit. With the Python-like protocol the tools are embedded
        # in the system prompt content itself, so splitting the prompt preserves
        # the prefix cache across agent type changes.
        if model_capabilities.able_to_use_python_like_protocol_for_tool_function:
            enabled_experiments.add('two_part_system_prompt')

        print("DEBUG enabled_experiments:", enabled_experiments)
        experiments_service = ExperimentsService(enabled_experiments=frozenset(enabled_experiments))
        
        tool_names_to_drop = []

        # When create_file experiment is enabled, drop write_file to prevent abuse
        if experiments_service.is_enabled('create_file'):
            tool_names_to_drop.append('write_file')

        # TODO re-enable this after getting sorted so that create_aggregate_diff can work with
        # patches.  For the new file_view_windows experiment the view windows will need to auto-added to.
        tool_names_to_drop.append('apply_patch')

        if experiments_service.is_enabled("file_view_windows"):
            tool_names_to_drop.append('read_file')
        else:
            tool_names_to_drop.append('add_file_view')
            tool_names_to_drop.append('close_file_view')
        
        # When search_list_view experiment is active, swap close_file_view for close_view
        if experiments_service.is_enabled("search_list_view"):
            if 'close_file_view' not in tool_names_to_drop:
                tool_names_to_drop.append('close_file_view')
            if 'close_view' in tool_names_to_drop:
                tool_names_to_drop.remove('close_view')
        
        # close_views experiment gates the close_file_view and close_view tool definitions
        if not experiments_service.is_enabled('close_views'):
            if 'close_file_view' not in tool_names_to_drop:
                tool_names_to_drop.append('close_file_view')
            if 'close_view' not in tool_names_to_drop:
                tool_names_to_drop.append('close_view')
        
        if not (experiments_service.is_enabled('insert_lines') or experiments_service.is_enabled('insert_line_validation')):
            tool_names_to_drop.append('insert_lines')
        
        if not (experiments_service.is_enabled('move_lines') or experiments_service.is_enabled('move_line_validation')):
            tool_names_to_drop.append('move_lines')
        
        if not (experiments_service.is_enabled('adjust_indent') or experiments_service.is_enabled('adjust_indent_validation')):
            tool_names_to_drop.append('adjust_indent')
        
        if not (experiments_service.is_enabled('adjust_indent_exact') or experiments_service.is_enabled('adjust_indent_exact_validation')):
            tool_names_to_drop.append('adjust_indent_exact')
        
        if not experiments_service.is_enabled('tavily_search'):
            tool_names_to_drop.append('tavily_search')

        if not experiments_service.is_enabled('tavily_extract'):
            tool_names_to_drop.append('tavily_extract')


        if not experiments_service.is_enabled('exa_search'):
            tool_names_to_drop.append('exa_search')

        if not experiments_service.is_enabled('tinyfish_fetch'):
            tool_names_to_drop.append('tinyfish_fetch')

        if not experiments_service.is_enabled('exa_contents'):
            tool_names_to_drop.append('exa_contents')

        if not experiments_service.is_enabled('tinyfish_search'):
            tool_names_to_drop.append('tinyfish_search')


        if not experiments_service.is_enabled('linkup_search'):
            tool_names_to_drop.append('linkup_search')

        if not experiments_service.is_enabled('linkup_fetch'):
            tool_names_to_drop.append('linkup_fetch')

        if not experiments_service.is_enabled('parallel_search'):
            tool_names_to_drop.append('parallel_search')

        if not experiments_service.is_enabled('parallel_extract'):
            tool_names_to_drop.append('parallel_extract')

        if not experiments_service.is_enabled('perplexity_search'):
            tool_names_to_drop.append('perplexity_search')

        # redo depends on undo - can't have redo without undo
        if experiments_service.is_enabled('redo') and not experiments_service.is_enabled('undo'):
            raise AssertionError("The 'redo' experiment requires the 'undo' experiment to also be enabled.")

        if not experiments_service.is_enabled('undo'):
            tool_names_to_drop.append('undo')

        if not experiments_service.is_enabled('redo'):
            tool_names_to_drop.append('redo')

        if args.test_runner_plugin_dir is None:
            tool_names_to_drop.append('run_tests')

        if args.agent_type == 'scrutinator':
            from fluximp_agent import scrutinator
            scrutinator.run_scrutinator_file_io(
                filesystem_service, model_config, args.io_dir,
                max_requests=args.max_requests,
                max_transient_retries=args.max_transient_api_error_retries,
                input_timeout=args.input_timeout,
                permission_timeout=args.permission_timeout
            )
            # Flush stdout before successful exit
            sys.stdout.flush()
            sys.exit(0)

        if args.agent_type == 'summarizer':
            from fluximp_agent import summarizer
            summarizer.run_summarizer_file_io(
                filesystem_service, model_config, args.io_dir,
                max_requests=args.max_requests,
                max_transient_retries=args.max_transient_api_error_retries,
                input_timeout=args.input_timeout,
                permission_timeout=args.permission_timeout
            )
            # Flush stdout before successful exit
            sys.stdout.flush()
            sys.exit(0)

        if args.agent_type == 'conditional':
            from fluximp_agent.conditional_agent import load_conditional_config, ConditionalFilesystemAgent, run_conditional_report_only_file_io
            conditional_config_path = Path(args.conditional_config)
            if not conditional_config_path.is_absolute():
                conditional_config_path = Path(args.working_dir) / conditional_config_path
            conditional_config = load_conditional_config(conditional_config_path)
            if conditional_config.tool_set == 'readonly_filesystem':
                agent_config = ConditionalFilesystemAgent(conditional_config, model_config, tool_names_to_drop=tool_names_to_drop)
                file_io_mode(filesystem_service, agent_config, args.io_dir, experiments_service, max_requests=conditional_config.max_requests, max_transient_retries=args.max_transient_api_error_retries, input_timeout=args.input_timeout, permission_timeout=args.permission_timeout, persist_checkpoints=args.persist_checkpoints, single_turn=True, args=args, model_config=model_config)
            elif conditional_config.tool_set == 'report_only':
                run_conditional_report_only_file_io(filesystem_service, model_config, conditional_config, args.io_dir, max_requests=conditional_config.max_requests, max_transient_retries=args.max_transient_api_error_retries, input_timeout=args.input_timeout, permission_timeout=args.permission_timeout)
            # Flush stdout before successful exit
            sys.stdout.flush()
            sys.exit(0)

        agent_config = fs_agents.create_filesystem_agent(
            args.agent_type, model_config,
            tool_names_to_drop=tool_names_to_drop
        )

        file_io_mode(filesystem_service, agent_config, args.io_dir, experiments_service, max_requests=args.max_requests, max_transient_retries=args.max_transient_api_error_retries, input_timeout=args.input_timeout, permission_timeout=args.permission_timeout, persist_checkpoints=args.persist_checkpoints, args=args, model_config=model_config)

        # Flush stdout before successful exit to ensure all output is captured
        sys.stdout.flush()
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.exit(1)

    return 0
if __name__ == "__main__":
    raise SystemExit(main())
