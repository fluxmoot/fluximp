from typing import Dict, Any, Optional, Tuple, List

from fluximp_agent.model_config import ModelConfig, get_model_capabilities
from fluximp_agent.agent_config import AgentConfig
from fluximp_agent.tools.common import create_aggregate_diff
from fluximp_agent.context_injection_manager import ContextChunk


class FileSystemAgent(AgentConfig):
    """Base class for agents that interact with the filesystem.

    Agents are defined purely by their tool list and optional description string.
    All behavior (preamble text, tool use examples) is derived from the tool
    categories present in the tool list, plus any agent_description string.
    """

    # --- Category-derived preamble constants ---

    _NO_MUTATION_PREAMBLE = (
        "Your available tools do not allow you to create or modify any files "
        "or run anything, so you should never try to do so.  You will fail "
        "every attempt to mutate state so never try.  If the user asks you "
        "to do these things, do not attempt to comply, as you cannot comply.  "
        "Instead inform the user about the capabilities of other agents and "
        "the limits of your own capabilities."
    )

    _FILESYSTEM_ARG_NOTE = (
        "Text spans that describe omitted string arguments for tools (e.g. "
        "<N characters omitted as ...>) may be inserted by this harness "
        "in replacement of the actual arguments you provided to save context. "
        "A string argument replacement will only occur after your original "
        "argument was returned to you once.  If you see an omission span "
        "in your very last message its there because you yourself generated it."
        "\n\nWhen using multiple tools to edit, insert, or move lines in the "
        "same file you must list the tool uses in reverse order to avoid a "
        "preceding tool use changing line numbers for following tool uses."
    )

    _FILESYSTEM_DIFF_NOTE = (
        "Once you create or modify any file, you will be provided the current, "
        "cumulative diff of the file changes you have made in this session."
    )

    _NO_FILESYSTEM_DIFF_NOTE = (
        "You will be provided the current, cumulative diff of file changes "
        "made in this session by any agent."
    )

    _AGENT_SWITCHING_NOTE = (
        "The user may change agents within this conversation.  Only the tools "
        "in the system prompt are available to you.  While you may see successful "
        "uses of tools in this conversation which are not available to you, those "
        "tool uses were made by other AI agents."
    )

    # --- Tool use example data ---

    _TOOL_EXAMPLES = {
        "undo": ["undo usage example:", "undo()"],
        "send_message": [
            "send_message usage example:",
            'send_message(tags=["work_completion_summary"], content="I\'ve created the requested file for you.", format="plain_text")',
        ],
        "ripgrep_search": [
            "ripgrep_search usage example:",
            "ripgrep_search(patterns=['foo', 'bar'], globs=['*.txt'], paths=['funky/monkey'], max_count=50)",
        ],
        "find_files": [
            "find_files usage example:",
            "find_files(globs=['*.txt'], max_count=200)",
        ],
    }

    _PREFERRED_EXAMPLE_ORDER = ["undo", "send_message", "ripgrep_search", "find_files"]

    def __init__(self, agent_type, model_config, base_tool_names,
                 agent_description, tool_names_to_drop=None, shared_tool_names=None,
                 max_tool_calls_per_response=20):
        """Initialize FileSystemAgent with tool set derived from base_tool_names.

        Args:
            agent_type: String identifying the agent type (e.g., "coder", "developer", "consultant").
            model_config: ModelConfig instance containing model/API settings.
            base_tool_names: List of tool names defining this agent's capabilities.
            tool_names_to_drop: Optional list of tool names to drop from the default set.
            shared_tool_names: Optional list of tool names shared across all agents
                (used by two_part_system_prompt experiment to split tool instructions).
            agent_description: String describing this agent's role and capabilities.
                Included in the system prompt preamble after category-derived text. Required.
            max_tool_calls_per_response: Maximum number of tool calls to process from
                a single LLM response for this agent type.
        """
        self.agent_type = agent_type
        self.agent_description = agent_description
        caps = get_model_capabilities(model_config)
        tool_names = self.apply_tool_drops(base_tool_names, tool_names_to_drop)

        super().__init__(
            model_config=model_config,
            tool_names=tool_names,
            use_tool_messages=caps.able_to_receive_tool_messages_at_all,
            use_api_protocol_for_tool_functions=not caps.able_to_use_python_like_protocol_for_tool_function,
            use_tool_message_for_no_tool_error=caps.able_to_recieve_tool_messages_for_no_tool_use_errors,
            shared_tool_names=shared_tool_names,
            max_tool_calls_per_response=max_tool_calls_per_response,
        )

    @staticmethod
    def apply_tool_drops(base_tool_names, tool_names_to_drop=None):
        """Return the list of tool names after applying drops."""
        tool_names = list(base_tool_names)
        if tool_names_to_drop:
            for tn in tool_names_to_drop:
                if tn in tool_names:
                    tool_names.remove(tn)
        return tool_names

    @staticmethod
    def _get_experiment_flags(experiments_service) -> Dict[str, bool]:
        """Extract experiment flags from the experiments service.

        Returns a dict with keys: move_line, adjust_indent, adjust_indent_exact.
        """
        return {
            "move_line": experiments_service and (experiments_service.is_enabled("move_line_validation") or experiments_service.is_enabled("move_lines")),
            "adjust_indent": experiments_service and (experiments_service.is_enabled("adjust_indent") or experiments_service.is_enabled("adjust_indent_validation")),
            "adjust_indent_exact": experiments_service and (experiments_service.is_enabled("adjust_indent_exact") or experiments_service.is_enabled("adjust_indent_exact_validation")),
        }

    def _get_experiment_reminder_parts(self, flags: Dict[str, bool]) -> List[str]:
        """Build reminder text parts based on experiment flags.

        Takes the flags dict from _get_experiment_flags, returns a list of
        reminder strings.
        """
        parts = []
        if flags.get("move_line"):
            parts.append("You must use the move_lines tool to move blocks of code around a file and between files when refactoring.  Do not try to precisely transcribe blocks as the args of write_file and edit_file_lines tools (move_lines will create the destination file if it doesn't exist).  Failure to do so will cause your context to rot and you know that's very, very bad.")
        if flags.get("adjust_indent"):
            parts.append("If you're having trouble getting indentation correct using edit_file_lines then utlize the adjust_indent tool to correct the indentation.  Some backends (i.e. your side of the internet) having issues with leading spaces.")
        if flags.get("adjust_indent_exact"):
            parts.append("If you're having trouble getting indentation correct using edit_file_lines then utlize the adjust_indent_exact tool to set the exact leading whitespace on lines.")
        return parts

    def _get_tool_categories(self):
        """Derive tool categories from the agent's tool list using tool definitions."""
        if hasattr(self, '_tool_categories_cache'):
            return self._tool_categories_cache
        from fluximp_agent.tools.tool_definitions import get_tool_definitions
        all_defs = get_tool_definitions(None)
        categories = set()
        for name in self.tool_names:
            if name in all_defs:
                categories.add(all_defs[name].category)
            else:
                # Conditional tools (create_file, insert_lines, move_lines, etc.)
                # all belong to the "filesystem" category
                categories.add("filesystem")
        self._tool_categories_cache = categories
        return categories

    def _get_category_derived_preamble(self):
        """Assemble preamble text based on which tool categories are present."""
        parts = []
        categories = self._get_tool_categories()
        has_filesystem = "filesystem" in categories
        has_process = "process" in categories

        if not has_filesystem and not has_process:
            parts.append(self._NO_MUTATION_PREAMBLE)

        if has_filesystem:
            parts.append(self._FILESYSTEM_ARG_NOTE)
            parts.append(self._FILESYSTEM_DIFF_NOTE)
        else:
            parts.append(self._NO_FILESYSTEM_DIFF_NOTE)

        parts.append(self._AGENT_SWITCHING_NOTE)
        return "\n\n".join(parts)

    _SHARED_PREAMBLE = """You can and should use multiple tools with each response, up to a maximum of {max_tool_calls_per_response} tool uses per response, so there is no value in counting historical tool uses.  Any tool uses beyond the first {max_tool_calls_per_response} in a single response will not be processed and you will be informed how many were skipped (you can try them again in your next response).  The first {max_tool_calls_per_response} tool uses in a single response will be executed in English reading order (right to left, top to bottom) and execution will halt on the first failure.

Free form text will not be displayed to the user.  The send_message tool allows you to communicate with the user.  Using send_message will block you from taking further actions until the user responds, so the send_message tool must not be used to report incremental/partial progress."""

    def _get_shared_preamble(self, **kwargs) -> str:
        """Return the shared preamble text.

        This is used when the two_part_system_prompt experiment is active.
        The returned text is identical for all agent types that share the same
        max_tool_calls_per_response value, preserving the API provider's prefix
        cache when the limit is uniform.
        """
        return self._SHARED_PREAMBLE.format(max_tool_calls_per_response=self.max_tool_calls_per_response)

    def get_system_prompt_preamble(self, **kwargs) -> str:
        flags = self._get_experiment_flags(kwargs.get("experiments_service"))
        parts = [self._get_shared_preamble()]
        parts.append(self._get_category_derived_preamble())
        parts.append(self.agent_description)
        parts.extend(self._get_experiment_reminder_parts(flags))
        return "\n\n".join(parts)

    def _get_agent_specific_preamble(self, **kwargs) -> str:
        flags = self._get_experiment_flags(kwargs.get("experiments_service"))
        parts = []
        parts.append(self._get_category_derived_preamble())
        parts.append(self.agent_description)
        parts.extend(self._get_experiment_reminder_parts(flags))
        return "\n\n".join(parts)

    def get_system_prompt_tool_use_example_lines(self) -> list:
        """Return tool use examples based on available tools."""
        lines = ["TOOL FUNCTION USAGE EXAMPLES", ""]
        available = set(self.tool_names)
        for tool_name in self._PREFERRED_EXAMPLE_ORDER:
            if tool_name in available and tool_name in self._TOOL_EXAMPLES:
                lines.extend(self._TOOL_EXAMPLES[tool_name])
                lines.append("")
        return lines

    def get_context_chunks(self, state, **kwargs) -> Tuple[List[ContextChunk], Optional[Dict[str, Any]]]:
        """Return context as a list of chunks sorted by stability for prefix caching.

        Each chunk has a section_id and stability_key that allows the
        ContextInjectionManager to determine what changed between requests
        and split the injection across multiple tool messages for optimal
        prefix caching.

        Returns:
            Tuple of (chunks_list, state_update_dict) where chunks_list is
            sorted by stability_key (most stable first) and state_update_dict
            contains file_view_windows cleanup if needed
        """
        import time as _time
        
        experiments_service = kwargs.get("experiments_service")
        filesystem_service = kwargs["filesystem_service"]
        
        chunks = []
        
        # 1. System Prompt Reminders (most stable - only changes when experiments change)
        reminders_parts = []
        # When context is injected into system messages (not tool messages),
        # the SYSTEM PROMPT REMINDERS header is redundant since the message
        # role already conveys that. Only include it when appending to tool messages.
        system_prompt_injection = experiments_service and experiments_service.is_enabled('system_prompt_context_injection')
        if not system_prompt_injection:
            reminders_parts.append('\n\nSYSTEM PROMPT REMINDERS:')
        reminders_parts.append('\nWhen you need to communicate with the user (i.e. ask a question, provide a summary of the completed tasks) you must use the send_message tool to communicate with the user. Free form text will not be displayed to the user.  Using send_message will block you from taking further actions until the user responds, so do not use send_message to tell the user what action your going to take as it will prevent you from taking the action.')
        move_line_experiment = experiments_service and (experiments_service.is_enabled("move_line_validation") or experiments_service.is_enabled("move_lines"))
        if move_line_experiment:
            reminders_parts.append("\nYou must use the move_lines tool to move blocks of code around a file and between files when refactoring.  Do not try to precisely transcribe blocks as the args of write_file and edit_file_lines tools (move_lines will create the destination file if it doesn't exist).  Failure to do so will cause your context to rot and you know that's very, very bad.")
        adjust_indent_experiment = experiments_service and (experiments_service.is_enabled("adjust_indent") or experiments_service.is_enabled("adjust_indent_validation"))
        if adjust_indent_experiment:
            reminders_parts.append("\nIf you're having trouble getting indentation correct using edit_file_lines then utlize the adjust_indent tool to correct the indentation.  Some backends (i.e. your side of the internet) having issues with leading spaces.")
        adjust_indent_exact_experiment = experiments_service and (experiments_service.is_enabled("adjust_indent_exact") or experiments_service.is_enabled("adjust_indent_exact_validation"))
        if adjust_indent_exact_experiment:
            reminders_parts.append("\nIf you're having trouble getting indentation correct using edit_file_lines then utlize the adjust_indent_exact tool to set the exact leading whitespace on lines.")
        
        reminders_content = "".join(reminders_parts)
        chunks.append(ContextChunk(
            section_id='reminders',
            content=reminders_content,
            stability_key=(0.0, 0.0, 'reminders'),
            always_last=True,
        ))
        
        # 2. File Views & Query Views (stability varies per view)
        state_update = None
        file_view_windows_experiment = experiments_service and experiments_service.is_enabled("file_view_windows")
        
        if file_view_windows_experiment:
            state_service = kwargs["state_service"]
            fraction_used, files_removed, interleaved_views, state_update = \
                state_service.get_all_views(filesystem_service, experiments_service, include_stability_keys=True)
            
            if interleaved_views:
                search_list_view_experiment = experiments_service and experiments_service.is_enabled("search_list_view")
                
                # Group file windows by section_id (same file = same chunk)
                file_chunks = {}  # section_id -> (stability_key, parts)
                search_chunks = {}  # section_id -> (stability_key, parts)
                
                # Track order of first appearance for each section_id
                section_order = []
                
                for view in interleaved_views:
                    if view[0] == "file":
                        _, file_path, start_line, end_line, content, total_lines, section_id, stability_key = view
                        if section_id not in file_chunks:
                            file_chunks[section_id] = (stability_key, [])
                            section_order.append(section_id)
                        if search_list_view_experiment:
                            entry = state_service.file_view_windows[file_path]
                            from fluximp_agent.state_service import FileViewEntry
                            if isinstance(entry, FileViewEntry):
                                view_id = entry.id
                            elif isinstance(entry, dict):
                                view_id = entry["id"]
                            else:
                                assert False
                            if start_line == 1 and end_line == total_lines:
                                file_chunks[section_id][1].append(
                                    f"\n\n--- {file_path} --- entire file ({total_lines} lines) view_id: {view_id} ---\n"
                                )
                            else:
                                file_chunks[section_id][1].append(
                                    f"\n\n--- {file_path} --- lines: {start_line}-{end_line} of {total_lines} view_id: {view_id} ---\n"
                                )
                        else:
                            if start_line == 1 and end_line == total_lines:
                                file_chunks[section_id][1].append(
                                    f"\n--- {file_path} --- entire file ({total_lines} lines) ---\n"
                                )
                            else:
                                file_chunks[section_id][1].append(
                                    f"\n--- {file_path} --- lines {start_line}-{end_line} of {total_lines} ---\n"
                                )
                        file_chunks[section_id][1].append(content)
                        file_chunks[section_id][1].append("")
                    elif view[0] == "search":
                        _, header, content, section_id, stability_key = view
                        if section_id not in search_chunks:
                            search_chunks[section_id] = (stability_key, [])
                            section_order.append(section_id)
                        search_chunks[section_id][1].append(
                            f"\n--- {header} ---\n"
                        )
                        search_chunks[section_id][1].append(content)
                        search_chunks[section_id][1].append("")
                
                # Create chunks in the order they appear (already sorted by stability)
                for section_id in section_order:
                    if section_id in file_chunks:
                        stability_key, parts = file_chunks[section_id]
                    elif section_id in search_chunks:
                        stability_key, parts = search_chunks[section_id]
                    else:
                        continue
                    
                    chunk_content = "\n".join(parts)
                    chunks.append(ContextChunk(
                        section_id=section_id,
                        content=chunk_content,
                        stability_key=stability_key,
                    ))
                
                # View capacity/eviction info (changes when views are added/removed)
                # Only show capacity when >= 80% full (not worth the prefix cache cost otherwise)
                # AND only when the agent can close views (close_views experiment enabled);
                # otherwise the percentage is noise the LLM needlessly worries about
                capacity_parts = []
                close_views_enabled = experiments_service and experiments_service.is_enabled("close_views")
                if close_views_enabled and fraction_used >= 0.80:
                    capacity_parts.append(
                        f"\nView capacity used:  {fraction_used*100.0:.1f}%\n"
                    )
                if files_removed:
                    capacity_parts.append(
                        f"Open views exceeded capacity; {len(files_removed)} files automatically closed:\n"
                    )
                    capacity_parts.extend(files_removed)
                    capacity_parts.append("Use more targeted ranges if you are going to immediately add windows views for any of the files just auto-closed.")
                
                # Only create capacity chunk if there is something to show
                if capacity_parts:
                    # Find max last_changed_ts from view chunks for capacity stability
                    max_view_changed_ts = 0.0
                    for chunk in chunks:
                        if chunk.section_id.startswith("view:") or chunk.section_id.startswith("search:"):
                            max_view_changed_ts = max(max_view_changed_ts, chunk.stability_key[0])
                    
                    chunks.append(ContextChunk(
                        section_id="view_capacity",
                        content="\n".join(capacity_parts),
                        stability_key=(max_view_changed_ts, _time.time(), 'view_capacity'),
                    ))
            else:
                print("DEBUG get_context_chunks no file_views")
        
        # 3. Cumulative Diff (stability depends on when it last changed)
        diff_state_update = {}
        if "operation_position" in state:
            file_ops = state["file_operations"]
            op_pos = state["operation_position"]
            
            # Read tracking fields from state
            last_op_pos = state.get("cumulative_diff_last_op_pos", -1)
            last_content = state.get("cumulative_diff_last_content", "")
            last_changed_ts = state.get("cumulative_diff_last_changed_ts", 0.0)
            creation_ts = state.get("cumulative_diff_creation_ts", 0.0)
            
            # Use operation_position as a cheap gate: if it hasn't changed,
            # skip the expensive diff recomputation entirely
            if op_pos == last_op_pos:
                # No file operations since last time - reuse previous content
                if last_content:
                    chunks.append(ContextChunk(
                        section_id="cumulative_diff",
                        content=last_content,
                        stability_key=(last_changed_ts, creation_ts, 'cumulative_diff'),
                    ))
            else:
                # operation_position changed - recompute the diff
                file_states = {}
                for op in file_ops[:op_pos + 1] if op_pos >= 0 else []:
                    if op.file_path is None:
                        continue
                    is_deleted = op.operation_type == "delete_file"
                    if op.file_path not in file_states:
                        file_states[op.file_path] = (op.compressed_before, is_deleted)
                    else:
                        original, _ = file_states[op.file_path]
                        file_states[op.file_path] = (original, is_deleted)
                
                created_files = []
                deleted_files = []
                modified_file_paths = set()
                
                for filepath, (compressed_before, was_deleted) in file_states.items():
                    if was_deleted:
                        if compressed_before is None:
                            # File was created and then deleted during this session -
                            # net effect is no change, so skip it entirely
                            continue
                        deleted_files.append(filepath)
                    else:
                        if compressed_before is None:
                            created_files.append(filepath)
                        else:
                            modified_file_paths.add(filepath)
                
                change_parts = []
                
                if modified_file_paths:
                    filtered_ops = [op for op in file_ops[:op_pos + 1]
                                    if op.file_path in modified_file_paths or op.file_path is None]
                    diff = create_aggregate_diff(
                        filtered_ops,
                        len(filtered_ops) - 1,
                        filesystem_service
                    )
                    if diff:
                        change_parts.append("```diff\n" + diff + "\n```")
                
                if created_files:
                    change_parts.append(
                        "Newly created files (diffs omitted for brevity):\n"
                        + "\n".join(created_files)
                    )
                
                if deleted_files:
                    change_parts.append(
                        "Deleted files (diffs omitted for brevity):\n"
                        + "\n".join(deleted_files)
                    )
                
                diff_content = ""
                if change_parts:
                    diff_content = (
                        "\n\n**Current, cumulative file changes you have made this session**\n\n"
                        + "\n\n".join(change_parts) + "\n"
                    )
                
                # Compare actual content to detect changes (not just operation_position)
                if diff_content != last_content:
                    last_changed_ts = _time.time()
                    if creation_ts == 0.0:
                        creation_ts = last_changed_ts
                
                # Always update state when op_pos changed (to track last_op_pos and content)
                diff_state_update = {
                    "cumulative_diff_last_op_pos": op_pos,
                    "cumulative_diff_last_content": diff_content,
                    "cumulative_diff_last_changed_ts": last_changed_ts,
                    "cumulative_diff_creation_ts": creation_ts,
                }
                
                # Create chunk if there's content
                if change_parts:
                    chunks.append(ContextChunk(
                        section_id="cumulative_diff",
                        content=diff_content,
                        stability_key=(last_changed_ts, creation_ts, 'cumulative_diff'),
                    ))
        
        # Agent-specific system content (when two_part_system_prompt experiment is active)
        if self.agent_specific_system_content is not None:
            chunks.append(ContextChunk(
                section_id="agent_specific",
                content="\n\n" + self.agent_specific_system_content,
                stability_key=(0.0, 0.0, "agent_specific", self.agent_type),
            ))

        # Sort chunks by stability_key (most stable first)
        chunks.sort(key=lambda c: c.stability_key)
        
        # Merge diff tracking fields into state_update
        if diff_state_update:
            if state_update is None:
                state_update = diff_state_update
            else:
                state_update.update(diff_state_update)
        
        return chunks, (state_update if state_update else None)
