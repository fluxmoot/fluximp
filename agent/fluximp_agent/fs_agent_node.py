"""Filesystem agent node for the LangGraph coding agent.

Contains the AgentNode class which is specific to filesystem agents (consultant, coder, developer)
and encapsulates LLM interaction logic including transformer-based message handling, summarization,
context injection, and response processing.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Callable, List, Tuple

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from fluximp_agent.agent_node import BaseAgentNode, SystemPromptTransformer, DropTransformer, ContextInjectionTransformer, ContextInjectionSystemTransformer, AgentSpecificSystemTransformer, AgentSpecificToolTransformer, calculate_message_api_bytes
from fluximp_agent.fs_agent_state import AgentState, StateConfig, AGENT_STATE_CONFIG, SummaryTransformer
from fluximp_agent.tool_call_omission import ToolCallOmissionTransformer
from fluximp_agent.context_injection_manager import ContextInjectionManager, ContextChunk
from fluximp_protocol.user_agent.file_io.defs import parse_utc_timestamp
from fluximp_protocol.user_agent.base import RequestLifecycleHooks

# --- Summarization Criterion 3 thresholds (byte-size trigger) ---
# Trigger summarization when the effective API request byte size (the last
# request's total content bytes minus the actual view content bytes and
# cumulative diff bytes, since both are out of scope for summarization)
# exceeds this threshold.
SUMMARIZATION_TOTAL_BYTES_THRESHOLD = 120000

# Tail summarization must always leave at least this many unsummarized AI
# messages in the tail (the consecutive messages after the last human message).
SUMMARIZATION_TAIL_MIN_REMAINING_AI = 10

# Tail summarization must summarize at least this many messages (summarizing
# a single message is pointless).
SUMMARIZATION_TAIL_MIN_SUMMARIZE_AI = 3

# Tags whose presence on a string-typed tool arg marks the tool as
# "content-carrying": its substantive content is preserved elsewhere (file views
# / cumulative diff for file tools, or user-facing text for send_message), so a
# failed call of such a tool can be safely pruned from history by
# _scan_for_drop_opportunities. Non-content tools (run_python_script,
# run_tests, ...) may surface diagnostics or user redirects in their results
# that the model must retain, so their failed calls must NOT be dropped.
_CONTENT_ARG_TAGS = frozenset({'message', 'content'})

class AgentNode(BaseAgentNode):
    """LLM-interaction node for the LangGraph agent.
    
    Processes messages and generates responses with retry logic for transient errors.
    Designed to be used as a node in a LangGraph StateGraph.
    
    Uses a transformers list to allow flexible message transformation:
    - Each entry corresponds to a message in state["messages"]
    - If None, use original message; otherwise use transformed result
    - Can be used to inject context, drop messages, or transform content
    """
    
    def __init__(self, filesystem_service, agent_config, registry,
                 max_requests: int = 5, max_transient_retries: int = 8,
                 lifecycle_hooks: Optional[RequestLifecycleHooks] = None,
                 model_client=None, skip_llm_next: list = None,
                 thread_id_ref: list = None,
                 state_config: StateConfig = None):
        super().__init__(
            filesystem_service=filesystem_service,
            agent_config=agent_config,
            registry=registry,
            max_requests=max_requests,
            max_transient_retries=max_transient_retries,
            model_client=model_client,
        )
        self.lifecycle_hooks = lifecycle_hooks
        self.skip_llm_next = skip_llm_next
        self.thread_id_ref = thread_id_ref
        self.state_config = state_config or AGENT_STATE_CONFIG
        
        # Transformers list - parallel to state["messages"]
        # Each entry is either None (pass through) or a callable that transforms the message
        self.transformers: List[Optional[Callable]] = []
        
        # References for state tracking
        self._current_state = None
        self._pending_context_state_update = None
        self._pending_context_injection_state = None
        self._skip_context_injection_restore = False
        self._fork_occurred = False
        self._pending_summarized_ranges = None
        self._pending_tool_call_omission_transformers_update = None
        
        
        # Create shared transformer instances
        self._system_prompt_transformer = SystemPromptTransformer(self.system_prompt)
        
        # Context injection manager for prefix-cache-aware context splitting
        self.context_injection_manager = ContextInjectionManager()
        # Note: ToolCallOmissionTransformer instances are stored in state as tool_call_omission_transformers
        
        # Track which positions already have drop transformers scanned
        self._drop_scanned_positions = set()
        
        # Track which AI message position has the tool call omission transformer
        self._tool_call_omission_positions = set()
        
    
    def _attach_tool_call_omission_transformer(self, state: AgentState):
        """Attach all active ToolCallOmissionTransformers to their target AI messages.
        
        Reads tool_call_omission_transformers from state (a dict of all active
        omission transformers keyed by message index) and attaches each one
        to its target position in the transformers list.
        
        Unlike the previous single-slot approach, this does NOT clear old positions.
        All historical omission transformers remain active until their positions
        are overwritten by SummaryTransformer/DropTransformer (during summarization
        or failed_tool_prune) or pruned (after a fork).
        
        The one-turn delay is handled by the pending_tool_call_omission_transformer
        field: new transformers are stored there and only promoted to the dict
        in the parse/execute nodes on the next turn.
        """
        # Get the dict of active omission transformers from state
        active_transformers = state.get("tool_call_omission_transformers", {})
        if not active_transformers:
            return
        
        messages = state["messages"]
        
        # Attach all active transformers (no clearing of old positions)
        for msg_index, omission_transformer in active_transformers.items():
            # Validate the index is still in range and points to an AI message
            if msg_index < 0 or msg_index >= len(messages) or not isinstance(messages[msg_index], AIMessage):
                continue
            
            # Skip if position already has a SummaryTransformer or DropTransformer
            if msg_index < len(self.transformers) and isinstance(self.transformers[msg_index], (SummaryTransformer, DropTransformer)):
                continue
            
            # Make sure we have enough slots
            while len(self.transformers) <= msg_index:
                self.transformers.append(None)
            
            # Attach the pre-computed transformer from state
            self.transformers[msg_index] = omission_transformer
            self._tool_call_omission_positions.add(msg_index)
            print(f"DEBUG _attach_tool_call_omission_transformer: at index {msg_index}")
    
    def _check_and_run_summarization(self, state: AgentState):
        """Check if summarization criteria are met and run the summarizer if so.
        
        Criteria:
        1) Unsummarized AI messages between a human message and a successful
           send_message, with >=10 unsummarized AI messages after the sandwiched block.
        2) >=25 unsummarized AI messages after a human message with no successful
           send_message in between -> summarize first 15, retain last 10.
        
        3) Effective API request byte size: the previous request's total
           content bytes (persisted in state as last_api_content_bytes) minus
           the actual view content bytes (last_api_view_bytes) and cumulative
           diff bytes (last_api_diff_bytes) â both out of scope for
           summarization â exceeds SUMMARIZATION_TOTAL_BYTES_THRESHOLD.
           Priority: the oldest unsummarized sandwich block if any unsummarized
           sandwich blocks remain; otherwise tail summarization (the
           consecutive messages after the last human message), summarizing the
           oldest tail AI messages so at least
           SUMMARIZATION_TAIL_MIN_REMAINING_AI remain unsummarized, requiring
           at least SUMMARIZATION_TAIL_MIN_SUMMARIZE_AI messages to be
           summarized.

        Only one summary per invocation, even if all criteria match.
        """
        from fluximp_agent.summarizer import format_messages_for_summarization
        
        messages = state["messages"]
        if not messages:
            print("DEBUG summarization: no messages, skipping")
            return
        
        print(f"DEBUG summarization: checking {len(messages)} messages")
        
        # Build set of indices already covered by existing summaries (from durable state)
        summarized_ranges = [(r[0], r[1]) for r in state.get("summarized_ranges", [])]
        summary_map = {r[0]: (r[0], r[1]) for r in state.get("summarized_ranges", [])}
        
        print(f"DEBUG summarization: existing summarized ranges: {summarized_ranges}")
        
        def is_summarized(idx):
            for s, e in summarized_ranges:
                if s <= idx <= e:
                    return True
            return False
        
        # Find AI messages with successful send_message
        # Use stashed valid tool calls from additional_kwargs (set in tools_node)
        send_message_ai_indices = set()
        print("DEBUG summarization: checking additional_kwargs for fluximp_valid_tool_calls")
        
        for i, msg in enumerate(messages):
            if not isinstance(msg, AIMessage):
                continue
            
            # Primary: use stashed valid tool calls from additional_kwargs
            valid_calls = getattr(msg, 'additional_kwargs', {}).get('fluximp_valid_tool_calls', None)
            
            if valid_calls is not None:
                print(f"  [{i}] found fluximp_valid_tool_calls with {len(valid_calls)} calls: {[n for n, a in valid_calls]}")
                for tool_name, args in valid_calls:
                    if tool_name == 'send_message':
                        # Verify the tool result was successful
                        for j in range(i + 1, len(messages)):
                            next_msg = messages[j]
                            if isinstance(next_msg, AIMessage):
                                break
                            is_tool_result = isinstance(next_msg, ToolMessage)
                            if is_tool_result:
                                is_successful = getattr(next_msg, 'execution_successful', True)
                                print(f"  [{i}] send_message tool result at [{j}], success={is_successful}")
                                if is_successful:
                                    send_message_ai_indices.add(i)
                                break
        
        print(f"DEBUG summarization: send_message AI indices: {sorted(send_message_ai_indices)}")
        
        # Find the last AI message index
        last_ai_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], AIMessage):
                last_ai_idx = i
                break
        
        print(f"DEBUG summarization: last AI message index: {last_ai_idx}")
        
        # Collect unsummarized AI message indices
        unsummarized_ai = []
        for i, msg in enumerate(messages):
            if isinstance(msg, AIMessage) and i not in send_message_ai_indices and not is_summarized(i):
                unsummarized_ai.append(i)
        
        print(f"DEBUG summarization: unsummarized AI count: {len(unsummarized_ai)}, indices: {unsummarized_ai}")
        
        # --- Criterion 1: sandwiched block between human msg and send_message ---
        # Iterate through send_message AI indices and look backward for the
        # corresponding human message, rather than iterating through all human
        # messages and scanning forward.
        print("DEBUG summarization: checking Criterion 1 (sandwiched block)")
        for sm_idx in sorted(send_message_ai_indices):
            # Look backward from this send_message for the nearest real human message
            sandwich_start = None
            for j in range(sm_idx - 1, -1, -1):
                msg_j = messages[j]
                if isinstance(msg_j, HumanMessage):
                    if not is_summarized(j):
                        sandwich_start = j
                    break
                # Stop if we hit a send_message AI message (another sandwich boundary)
                if isinstance(msg_j, AIMessage) and j in send_message_ai_indices:
                    break
            
            print(f"  Criterion 1: send_message at index {sm_idx}, sandwich_start={sandwich_start}")
            
            if sandwich_start is None:
                print(f"  Criterion 1: no human message found before send_message at index {sm_idx}, skipping")
                continue
            
            # Find unsummarized AI messages between human msg and send_message
            sandwich_ai = [j for j in unsummarized_ai if sandwich_start < j < sm_idx]
            sandwich_ai_len = len(sandwich_ai)
            print(f"  Criterion 1: sandwich AI count={sandwich_ai_len}, indices={sandwich_ai}")
            
            if not sandwich_ai:
                print(f"  Criterion 1: no unsummarized AI in sandwich, skipping")
                continue
            
            # Count unsummarized AI messages after the send_message (+ its tool results)
            after_idx = sm_idx + 1
            for j in range(sm_idx + 1, len(messages)):
                next_msg = messages[j]
                is_tool_result = isinstance(next_msg, ToolMessage)
                if is_tool_result:
                    after_idx = j + 1
                else:
                    break
            
            after_unsummarized_threshold = min(10, max(3, sandwich_ai_len))
            after_unsummarized = [j for j in unsummarized_ai if j >= after_idx]
            print(f"  Criterion 1: after_unsummarized count={len(after_unsummarized)} (need >={after_unsummarized_threshold}), starting after index {after_idx}")
            
            if len(after_unsummarized) >= after_unsummarized_threshold:
                # Summarize the sandwiched block (all messages from first AI to just before send_message)
                block_start = sandwich_ai[0]
                # Find the actual end of the block (including tool results before send_message)
                block_end = sm_idx - 1
                print(f"  Criterion 1: MET! Summarizing sandwiched block {block_start}\u2013{block_end}")
                self._summarize_and_install(state, block_start, block_end, messages,
                                           is_summarized, send_message_ai_indices, last_ai_idx, summary_map)
                return
            else:
                print(f"  Criterion 1: NOT MET (after_unsummarized={len(after_unsummarized)} < 10)")
                break  # Don't skip over unsummarized messages to summarize a later block
        
        print("DEBUG summarization: no Criterion 1 match found")
        
        # --- Criterion 2: >=25 unsummarized AI messages after human msg, no send_message ---
        print("DEBUG summarization: checking Criterion 2 (25+ unsummarized AI)")
        for i, msg in enumerate(messages):
            if not isinstance(msg, HumanMessage):
                continue
            if is_summarized(i):
                continue
            
            # Count unsummarized AI messages after this human message
            # until the next human message or send_message
            after_ai = []
            for j in range(i + 1, len(messages)):
                if isinstance(messages[j], HumanMessage):
                    break
                if isinstance(messages[j], AIMessage) and j in send_message_ai_indices:
                    break
                if isinstance(messages[j], AIMessage) and not is_summarized(j):
                    after_ai.append(j)
            
            print(f"  Criterion 2: human msg at index {i}, after_ai count={len(after_ai)} (need >=25)")
            
            if len(after_ai) >= 25:
                # Summarize the first 15 AI messages
                block_start_ai = after_ai[0]
                block_end_ai = after_ai[14]  # 15th (0-indexed: 0..14)
                
                # Expand block_end to include tool results after the last AI message
                block_start = block_start_ai
                block_end = block_end_ai
                for j in range(block_end_ai + 1, len(messages)):
                    if isinstance(messages[j], AIMessage):
                        break
                    is_tool_result = isinstance(messages[j], ToolMessage)
                    if is_tool_result:
                        block_end = j
                    else:
                        break
                
                print(f"  Criterion 2: MET! Summarizing block {block_start}\u2013{block_end} (first 15 of {len(after_ai)} AI msgs)")
                self._summarize_and_install(state, block_start, block_end, messages,
                                           is_summarized, send_message_ai_indices, last_ai_idx, summary_map)
                return
            else:
                print(f"  Criterion 2: NOT MET (after_ai={len(after_ai)} < 25)")
                if after_ai:
                    break  # Don't skip over unsummarized messages to summarize a later block
        
        print("DEBUG summarization: no Criterion 2 match found")
    
        # --- Criterion 3: effective API request byte size ---
        # Uses the previous request's byte totals (written to state by
        # _update_context_injections and _build_api_messages, and persisted
        # via build_full_update carry-over). View content (file views,
        # search/list views) and the cumulative diff are injected into the
        # request but are never summarized, so their actual byte counts
        # (stored in state as last_api_view_bytes and last_api_diff_bytes)
        # are subtracted from the total.
        last_total = state.get("last_api_content_bytes")
        last_view = state.get("last_api_view_bytes")
        last_diff = state.get("last_api_diff_bytes")
        last_msg_count = state.get("last_api_request_message_count")
        byte_fields = [last_total, last_view, last_diff, last_msg_count]

        if all(f is None for f in byte_fields):
            # First invocation / fresh checkpoint â no prior request recorded.
            print("DEBUG summarization: Criterion 3 skipped - no prior request bytes recorded")
        elif any(f is None for f in byte_fields):
            raise RuntimeError(
                f"Corrupted state: Criterion 3 byte fields partially set: "
                f"total={last_total}, view={last_view}, diff={last_diff}, "
                f"msg_count={last_msg_count}"
            )
        else:
            if len(messages) < last_msg_count:
                raise RuntimeError(
                    f"Stale byte-field: recorded for {last_msg_count} messages, "
                    f"current has {len(messages)} (rewind bypassing on_fork?)"
                )
            non_summarizable = last_view + last_diff
            effective_bytes = last_total - non_summarizable
            print(f"  Criterion 3: effective={effective_bytes:,} (api={last_total:,} - views={last_view:,} - diff={last_diff:,}), threshold={SUMMARIZATION_TOTAL_BYTES_THRESHOLD:,}")
            if effective_bytes > SUMMARIZATION_TOTAL_BYTES_THRESHOLD:
                # Priority A: oldest unsummarized sandwich block. Sandwich
                # blocks take priority over the tail; the tail is only
                # considered when no unsummarized sandwich block remains
                # (summarization must never skip over unsummarized messages).
                for sm_idx in sorted(send_message_ai_indices):
                    # Look backward from this send_message for the nearest real
                    # human message (same sandwich scan as Criterion 1)
                    sandwich_start = None
                    for j in range(sm_idx - 1, -1, -1):
                        msg_j = messages[j]
                        if isinstance(msg_j, HumanMessage):
                            if not is_summarized(j):
                                sandwich_start = j
                            break
                        # Stop if we hit a send_message AI message (another sandwich boundary)
                        if isinstance(msg_j, AIMessage) and j in send_message_ai_indices:
                            break

                    if sandwich_start is None:
                        print(f"  Criterion 3.A: send_message at [{sm_idx}] has no unsummarized human before it, skipping")
                        continue

                    # Find unsummarized AI messages between human msg and send_message
                    sandwich_ai = [j for j in unsummarized_ai if sandwich_start < j < sm_idx]
                    if not sandwich_ai:
                        print(f"  Criterion 3.A: no unsummarized AI in sandwich for send_message at [{sm_idx}], skipping")
                        continue

                    block_start = sandwich_ai[0]
                    block_end = sm_idx - 1
                    print(f"  Criterion 3: MET via oldest unsummarized sandwich! Summarizing block {block_start}\u2013{block_end}")
                    self._summarize_and_install(state, block_start, block_end, messages,
                                                is_summarized, send_message_ai_indices, last_ai_idx, summary_map)
                    return

                # Priority B: tail summarization - the consecutive messages after
                # the last human message. Summarize the oldest unsummarized AI
                # messages in the tail while keeping at least
                # SUMMARIZATION_TAIL_MIN_REMAINING_AI unsummarized, and requiring
                # at least SUMMARIZATION_TAIL_MIN_SUMMARIZE_AI summarized
                # (summarizing a single message is pointless).
                last_human_idx = None
                for i in range(len(messages) - 1, -1, -1):
                    if isinstance(messages[i], HumanMessage):
                        last_human_idx = i
                        break

                # unsummarized_ai already excludes send_message AI messages
                tail_ai = [j for j in unsummarized_ai if last_human_idx is not None and j > last_human_idx]
                summarize_count = len(tail_ai) - SUMMARIZATION_TAIL_MIN_REMAINING_AI
                print(f"  Criterion 3.B: tail after human [{last_human_idx}] has {len(tail_ai)} unsummarized AI, summarize_count={summarize_count} (need >= {SUMMARIZATION_TAIL_MIN_SUMMARIZE_AI})")
                if summarize_count >= SUMMARIZATION_TAIL_MIN_SUMMARIZE_AI:
                    block_start = tail_ai[0]
                    block_end_ai = tail_ai[summarize_count - 1]
                    block_end = block_end_ai

                    # Expand block_end to include tool results following the last AI in the block
                    for j in range(block_end_ai + 1, len(messages)):
                        if isinstance(messages[j], AIMessage):
                            break
                        is_tool_result = isinstance(messages[j], ToolMessage)
                        if is_tool_result:
                            block_end = j
                        else:
                            break

                    print(f"  Criterion 3: MET via tail! Summarizing {summarize_count} of {len(tail_ai)} tail AI msgs, block {block_start}\u2013{block_end}, leaving {len(tail_ai) - summarize_count} unsummarized")
                    self._summarize_and_install(state, block_start, block_end, messages,
                                                is_summarized, send_message_ai_indices, last_ai_idx, summary_map)
                    return

        print("DEBUG summarization: no Criterion 3 match found")
    
    def _summarize_and_install(self, state, start_idx, end_idx, messages,
                               is_summarized, send_message_ai_indices, last_ai_idx, summary_map):
        """Run the summarizer agent and install SummaryTransformer + DropTransformers."""
        from fluximp_agent.summarizer import format_messages_for_summarization
        
        # Build retained_indices for the full message list (for formatting context)
        retained_indices = set()
        for i, msg in enumerate(messages):
            if isinstance(msg, HumanMessage):
                retained_indices.add(i)
            elif isinstance(msg, AIMessage) and i in send_message_ai_indices:
                retained_indices.add(i)
                # Also retain tool results following this AI message
                for j in range(i + 1, len(messages)):
                    if isinstance(messages[j], AIMessage):
                        break
                    is_tool_result = isinstance(messages[j], ToolMessage)
                    if is_tool_result:
                        retained_indices.add(j)
                    else:
                        break
            elif i in summary_map:
                retained_indices.add(i)
            elif isinstance(msg, AIMessage) and i == last_ai_idx:
                retained_indices.add(i)
                # Also retain tool results following the last AI message
                for j in range(i + 1, len(messages)):
                    if isinstance(messages[j], AIMessage):
                        break
                    is_tool_result = isinstance(messages[j], ToolMessage)
                    if is_tool_result:
                        retained_indices.add(j)
                    else:
                        break
        
        # Build the set of indices being summarized in this run
        summarized_indices = set(range(start_idx, end_idx + 1))

        # Build summary_contents by pulling summary text from existing SummaryTransformers
        summary_contents = {}
        for i in summary_map:
            if i < len(self.transformers) and isinstance(self.transformers[i], SummaryTransformer):
                summary_contents[i] = self.transformers[i].summary

        # Determine where to start formatting. If we have fewer than two prior
        # summaries, keep the full history. Otherwise, start at the real human
        # message immediately before the second-most-recent summary range.
        format_start_idx = 0
        if len(summary_map) >= 2:
            sorted_summary_starts = sorted(summary_map.keys())
            second_most_recent_start = sorted_summary_starts[-2]
            for idx in range(second_most_recent_start - 1, -1, -1):
                msg = messages[idx]
                if isinstance(msg, HumanMessage):
                    format_start_idx = idx
                    break

        # Format the message window for summarization (from format_start_idx onward)
        formatted_text = format_messages_for_summarization(
            messages, retained_indices, summary_map, summarized_indices, summary_contents, format_start_idx
        )
        
        print(f"DEBUG _summarize_and_install: block {start_idx}\u2013{end_idx}, formatted_text length={len(formatted_text)}")
        
        # Run the summarizer via AgentService (subprocess)
        summary = self.registry._agent_service.run_summarizer(
            formatted_text=formatted_text,
            model_config=self.agent_config.model_config,
        )
        
        if summary is None:
            print(f"DEBUG _summarize_and_install: FAILED - summarizer returned None for block {start_idx}\u2013{end_idx}")
            return
        
        summary_preview = summary[:100].replace('\n', '\\n')
        print(f"DEBUG _summarize_and_install: summary received, length={len(summary)}, preview: {summary_preview}...")
        
        # Install SummaryTransformer at start_idx
        summary_transformer = SummaryTransformer(
            summary=summary,
            start_index=start_idx,
            end_index=end_idx,
            message_index=start_idx,
        )
        
        # Make sure we have enough transformer slots
        while len(self.transformers) <= end_idx:
            self.transformers.append(None)
        
        # Install the summary transformer at start_idx
        self.transformers[start_idx] = summary_transformer
        print(f"DEBUG _summarize_and_install: installed SummaryTransformer at index {start_idx}")
        
        # Install DropTransformers for all other positions in the summarized block
        drop_indices = list(range(start_idx + 1, end_idx + 1))
        for i in drop_indices:
            self.transformers[i] = DropTransformer(source="summarization")
        print(f"DEBUG _summarize_and_install: installed DropTransformers at indices {drop_indices}")
        
        # Reset drop scanned positions since we've changed transformers
        self._drop_scanned_positions = set()
        
        # Update durable summarized_ranges for state persistence
        current_ranges = list(state.get("summarized_ranges", []))
        current_ranges.append([start_idx, end_idx, summary])
        self._pending_summarized_ranges = current_ranges
        # Clean up omission transformer dict entries for summarized indices
        # (these positions now have SummaryTransformer/DropTransformer which take priority)
        current_omission = dict(state.get("tool_call_omission_transformers", {}))
        stale_omission_keys = [k for k in current_omission if start_idx <= k <= end_idx]
        if stale_omission_keys:
            for k in stale_omission_keys:
                del current_omission[k]
            state["tool_call_omission_transformers"] = current_omission
            self._pending_tool_call_omission_transformers_update = current_omission
            self._tool_call_omission_positions -= set(range(start_idx, end_idx + 1))
            print(f"DEBUG _summarize_and_install: removed {len(stale_omission_keys)} omission transformer entries for summarized range")
        # Reset context injection manager since summarization breaks the prefix
        self.context_injection_manager.reset()
        print(f"DEBUG _summarize_and_install: COMPLETE - summary replaces messages {start_idx}\u2013{end_idx}, transformer list length={len(self.transformers)}")
    
    def _prune_stale_transformers(self, state, messages):
        """Remove stale transformers after message list truncation (e.g., after fork).

        Preserves valid SummaryTransformers (whose range is fully within bounds)
        and their associated DropTransformers. Removes stale SummaryTransformers
        (whose end_index >= len(messages)) and their associated DropTransformers.
        Other transformers (SystemPromptTransformer, ContextInjectionTransformer,
        ToolCallOmissionTransformer) are always preserved if within bounds.
        
        Also filters state["summarized_ranges"] to remove stale entries and
        stores the filtered list as _pending_summarized_ranges for the next
        state update.
        """
        msg_count = len(messages)

        if len(self.transformers) <= msg_count:
            # Still need to filter summarized_ranges even if transformers fit
            current_ranges = state.get("summarized_ranges", [])
            filtered_ranges = [r for r in current_ranges if r[1] < msg_count]
            if len(filtered_ranges) != len(current_ranges):
                print(f"DEBUG _prune_stale_transformers: filtered summarized_ranges from {len(current_ranges)} to {len(filtered_ranges)} entries")
                state["summarized_ranges"] = filtered_ranges
                self._pending_summarized_ranges = filtered_ranges
            # Also clean up stale omission transformer dict entries
            current_omission = dict(state.get("tool_call_omission_transformers", {}))
            stale_omission_keys = [k for k in current_omission if k >= msg_count]
            if stale_omission_keys:
                for k in stale_omission_keys:
                    del current_omission[k]
                state["tool_call_omission_transformers"] = current_omission
                self._pending_tool_call_omission_transformers_update = current_omission
                print(f"DEBUG _prune_stale_transformers: removed {len(stale_omission_keys)} stale omission transformer entries")
            return  # No transformer truncation needed

        # Step 1: Truncate transformers list to message count
        truncated = self.transformers[:msg_count]

        # Step 2: Find stale SummaryTransformers whose end_index >= msg_count
        stale_ranges = []
        for i, t in enumerate(truncated):
            if isinstance(t, SummaryTransformer) and t.end_index >= msg_count:
                stale_ranges.append((t.start_index, t.end_index, i))

        if not stale_ranges:
            # Simple truncation only — no stale summaries within surviving range
            print(f"DEBUG _prune_stale_transformers: truncated from {len(self.transformers)} to {msg_count}, no stale summaries")
            self.transformers = truncated
            # Still filter summarized_ranges
            current_ranges = state.get("summarized_ranges", [])
            filtered_ranges = [r for r in current_ranges if r[1] < msg_count]
            if len(filtered_ranges) != len(current_ranges):
                print(f"DEBUG _prune_stale_transformers: filtered summarized_ranges from {len(current_ranges)} to {len(filtered_ranges)} entries")
                state["summarized_ranges"] = filtered_ranges
                self._pending_summarized_ranges = filtered_ranges
            # Clean up stale omission transformer dict entries (out of bounds)
            current_omission = dict(state.get("tool_call_omission_transformers", {}))
            stale_omission_keys = [k for k in current_omission if k >= msg_count]
            if stale_omission_keys:
                for k in stale_omission_keys:
                    del current_omission[k]
                state["tool_call_omission_transformers"] = current_omission
                self._pending_tool_call_omission_transformers_update = current_omission
                print(f"DEBUG _prune_stale_transformers: removed {len(stale_omission_keys)} stale omission transformer entries")
            return

        # Step 3: Remove stale summaries and their associated DropTransformers
        for start, end, pos in stale_ranges:
            print(f"DEBUG _prune_stale_transformers: removing stale SummaryTransformer at {pos}, range {start}–{end}")
            for i in range(start, min(end + 1, msg_count)):
                if isinstance(truncated[i], (SummaryTransformer, DropTransformer)):
                    truncated[i] = None

        self.transformers = truncated

        # Reset tracking since we changed transformers — positions within valid
        # summary ranges will be re-added by _scan_for_drop_opportunities guards
        self._drop_scanned_positions = set()
        self._tool_call_omission_positions = set()

        # Filter summarized_ranges to remove stale entries
        current_ranges = state.get("summarized_ranges", [])
        filtered_ranges = [r for r in current_ranges if r[1] < msg_count]
        if len(filtered_ranges) != len(current_ranges):
            print(f"DEBUG _prune_stale_transformers: filtered summarized_ranges from {len(current_ranges)} to {len(filtered_ranges)} entries")
            state["summarized_ranges"] = filtered_ranges
            self._pending_summarized_ranges = filtered_ranges

        # Clean up stale omission transformer dict entries (out of bounds)
        current_omission = dict(state.get("tool_call_omission_transformers", {}))
        stale_omission_keys = [k for k in current_omission if k >= msg_count]
        if stale_omission_keys:
            for k in stale_omission_keys:
                del current_omission[k]
            state["tool_call_omission_transformers"] = current_omission
            self._pending_tool_call_omission_transformers_update = current_omission
            print(f"DEBUG _prune_stale_transformers: removed {len(stale_omission_keys)} stale omission transformer entries")

        print(f"DEBUG _prune_stale_transformers: pruned {len(stale_ranges)} stale summaries, kept valid ones")

    def on_fork(self):
        """Called when a fork occurs - clears transformers and sets flag for rebuild."""
        print("DEBUG AgentNode.on_fork() - clearing transformers for fork")
        self._fork_occurred = True
        self.transformers = []
        self._drop_scanned_positions = set()
        self._pending_context_state_update = None
        self._pending_summarized_ranges = None  # fall back to durable state["summarized_ranges"] on rebuild
        self._pending_tool_call_omission_transformers_update = None
        self._pending_context_injection_state = None
        self._skip_context_injection_restore = True
        self.context_injection_manager.reset()
    
    def _maybe_skip(self, state):
        """Skip LLM invocation for mid-turn recovery."""
        if self.skip_llm_next and self.skip_llm_next[0]:
            self.skip_llm_next[0] = False
            print(f"DEBUG: skip_llm_next=True")
            return self.state_config.build_full_update(
                state,
                {
                    "messages": [],
                    "file_operations_this_turn": [],
                },
                "agent_node(skip_llm)"
            )
        return None
    
    def _prepare_for_invoke(self, state, request_count):
        """Prepare for LLM invocation: handle fork, summarization, transformers."""
        self.logger.log_state(state)
        
        # Store current state reference for context injection transformer
        self._current_state = state

        # If a fork occurred, transformers were cleared by on_fork() - just clear the flag.
        # _sync_transformers will rebuild from state (using _pending_summarized_ranges
        # to avoid stale checkpoint data).
        state_messages = state["messages"]
        print(f"DEBUG __call__ 1 len(self.transformers) {len(self.transformers)} len(state_messages) {len(state_messages)}")
        if self._fork_occurred:
            self._prune_stale_transformers(state, state_messages)
            self._fork_occurred = False        
        print(f"DEBUG __call__ 2 len(self.transformers) {len(self.transformers)} len(state_messages) {len(state_messages)}")
        assert len(self.transformers) <= len(state_messages)

        # On a cold restart (or fork), self.transformers is empty while messages
        # exist. Rebuild the persisted summaries NOW, BEFORE
        # _check_and_run_summarization runs: that routine installs a new
        # SummaryTransformer + DropTransformers, which makes the list non-empty and
        # silently masks the empty-list full-rebuild gate inside _sync_transformers.
        # If we wait, the old summarized ranges (from state["summarized_ranges"]) are
        # never reconstructed and all their raw messages leak through to the API
        # verbatim (the restart-summarization defect).
        if len(self.transformers) == 0 and len(state_messages) > 0:
            print("DEBUG _prepare_for_invoke: cold-restart rebuild before summarization")
            self._rebuild_transformers(state)

        # Check if summarization criteria are met before syncing transformers
        self._check_and_run_summarization(state)
        print(f"DEBUG __call__ 4 len(self.transformers) {len(self.transformers)} len(state_messages) {len(state_messages)}")
        
        # Reset Criterion 3 byte fields to None before computing new values.
        # _check_and_run_summarization has already read the previous request's
        # values; now clear all four so a failure between _update_context_injections
        # (which sets view/diff) and _build_api_messages (which sets total/count)
        # is detected as partially-set by the next invocation's corrupted-state check.
        state["last_api_content_bytes"] = None
        state["last_api_view_bytes"] = None
        state["last_api_diff_bytes"] = None
        state["last_api_request_message_count"] = None
        
        # Sync transformers (extend, rebuild, scan for drops, reposition context)
        self._sync_transformers(state)
        print(f"DEBUG __call__ 5 len(self.transformers) {len(self.transformers)} len(state_messages) {len(state_messages)}")
    
    def _sync_transformers(self, state: AgentState):
        """Sync transformers list with state messages.
        
        Handles:
        - Full rebuild if resuming from checkpoint (transformers empty but messages exist)
        - Extend with None entries for new messages
        - Scan for new drop-transformer opportunities
        - Reposition context injection transformer
        """
        messages = state["messages"]
        
        if self._fork_occurred:
            self._prune_stale_transformers(state, messages)
            self._fork_occurred = False
        assert len(self.transformers) <= len(messages)

        # Full rebuild case: transformer list empty but messages exist (resumed from checkpoint)
        if len(self.transformers) == 0 and len(messages) > 0:
            print("DEBUG _sync_transformers: full rebuild")
            self._rebuild_transformers(state)
            # _scan_for_drop_opportunities is deliberately not called here:
            # _prepare_for_invoke always rebuilds (line 731-733) before this
            # point, so this branch is unreachable in practice, and the scan runs
            # on every normal _sync_transformers pass below. Even if it did run,
            # a missed scan would only delay a prune by one request.
            # Attach omission transformers before computing placements so the
            # manager sees the current turn's active omission positions.
            self._attach_tool_call_omission_transformer(state)
            self._update_context_injections(state)
            return
        
        # Extend transformers with None entries for new messages
        new_message_count = len(messages) - len(self.transformers)
        if new_message_count > 0:
            print(f"DEBUG _sync_transformers: extending with {new_message_count} None entries")
            self.transformers.extend([None] * new_message_count)
        
        # Scan for new drop-transformer opportunities (only newly added positions)
        self._scan_for_drop_opportunities(state)
        
        # Attach tool call omission transformers FIRST, so _update_context_injections
        # can see this turn's active omission positions (promoted from the pending
        # transformer in the previous turn) when deciding where to place context.
        self._attach_tool_call_omission_transformer(state)

        # Reposition context injection transformer to last tool message position
        self._update_context_injections(state)
    
    def _rebuild_transformers(self, state: AgentState):
        """Full rebuild of transformers list when resuming from checkpoint or fork.
        
        Sets up:
        - SystemPromptTransformer at index 0 (ignores placeholder, returns actual system prompt)
        - SummaryTransformer + DropTransformers from state["summarized_ranges"]
        - ToolCallOmissionTransformer instances from state["tool_call_omission_transformers"]
          are attached afterwards by _attach_tool_call_omission_transformer, not here
        - None for all other messages (pass through)
        """
        messages = state["messages"]
        
        # Start fresh with system prompt transformer at index 0
        self.transformers = [self._system_prompt_transformer]
        
        # Add None for all other messages (pass through)
        self.transformers.extend([None] * (len(messages) - 1))
        
        # Reconstruct SummaryTransformer + DropTransformers from durable state
        # Prefer _pending_summarized_ranges only when it is a non-None override.
        # on_fork() sets it to None, so after a fork this falls back to the durable
        # state["summarized_ranges"] and reconstructs the existing summaries, avoiding
        # the raw-message leak that an empty list would otherwise cause.
        if self._pending_summarized_ranges is not None:
            summarized_ranges = self._pending_summarized_ranges
        else:
            summarized_ranges = state.get("summarized_ranges", [])
        for entry in summarized_ranges:
            start_idx, end_idx, summary_text = entry[0], entry[1], entry[2]
            if start_idx < len(messages) and end_idx < len(messages):
                self.transformers[start_idx] = SummaryTransformer(
                    summary=summary_text,
                    start_index=start_idx,
                    end_index=end_idx,
                    message_index=start_idx,
                )
                for i in range(start_idx + 1, end_idx + 1):
                    self.transformers[i] = DropTransformer(source="summarization_rebuild")
        
        # ToolCallOmissionTransformers are NOT attached here. The shared
        # _attach_tool_call_omission_transformer (which skips positions that
        # already hold a SummaryTransformer or DropTransformer) runs on the
        # _sync_transformers pass that follows every rebuild, so the two attach
        # paths cannot diverge.
        active_omission_transformers = state.get("tool_call_omission_transformers", {})
        self._tool_call_omission_positions = set()
        
        # Reset scanned positions
        self._drop_scanned_positions = set()
        
        # Restore context injection manager state from checkpoint (unless fork invalidated it)
        if not self._skip_context_injection_restore:
            saved_state = state.get("context_injection_state")
            if saved_state:
                self.context_injection_manager.from_state_dict(saved_state)
                print(f"DEBUG _rebuild_transformers: restored context injection state with {len(self.context_injection_manager.injection_records)} records")
        else:
            print("DEBUG _rebuild_transformers: skipping context injection restore due to fork")
        self._skip_context_injection_restore = False
        
        print(f"DEBUG _rebuild_transformers: built {len(self.transformers)} transformers with {len(summarized_ranges)} summaries and {len(active_omission_transformers)} omission transformers from state (attached by _attach_tool_call_omission_transformer)")
    
    def _is_content_carrying_tool(self, tool_name: str) -> bool:
        """True if the tool has a string-typed arg tagged 'message' or 'content'.

        Content-carrying tools (send_message, write_file, create_file,
        edit_file_lines, apply_patch, ...) have their substantive content
        preserved elsewhere (file views / cumulative diff, or user-facing text
        for send_message), so pruning a failed call from history is safe.
        Non-content tools (run_python_script, run_tests, ...) may surface
        diagnostics or user redirects in their results that the model must
        retain, so their failed calls must NOT be dropped.
        """
        tool_def = getattr(self.registry, '_tools', {}).get(tool_name)
        if not tool_def:
            return False
        return any(
            arg.arg_type == 'string' and bool(arg.tags & _CONTENT_ARG_TAGS)
            for arg in tool_def.args
        )

    def _failed_tool_prune_enabled(self) -> bool:
        """Return True when the 'failed_tool_prune' experiment is active.

        A registry without an experiments service (e.g. lightweight test
        registries) is treated as enabled so the historical always-on
        behavior is preserved.
        """
        experiments_service = getattr(self.registry, '_experiments_service', None)
        if experiments_service is None:
            return True
        return experiments_service.is_enabled('failed_tool_prune')
    
    def _scan_for_drop_opportunities(self, state: AgentState):
        """Scan for new drop-transformer opportunities.

        Prunes an (AIMessage, tool_results) batch from history when the whole
        batch failed and every call is content-carrying (see _CONTENT_ARG_TAGS
        and _is_content_carrying_tool): the substantive content of such a tool is
        preserved elsewhere (file views / cumulative diff for file tools, or the
        user-facing text for send_message), so the failed attempt can be removed
        from history once the model has seen it.

        The trigger is unified and deferral-based - it does NOT depend on a
        successful send_message anywhere in the history:

          * The AI message at index i first appears in the request after the one
            that produced it, and a later AI message can only exist after that
            request's response was received, so requiring "a later AIMessage
            exists" installs the drop on the request after the LLM's single
            verbatim view of the call. The failure (its error text) disappears
            at that same request.
          * Because the scan runs before _attach_tool_call_omission_transformer
            in _sync_transformers, the omission transformer that would otherwise
            rewrite this message's args is never attached - the drop replaces it
            - so there is a single byte transition (present -> absent) and no
            dead omission entry is left in state.

        Tool names come from additional_kwargs['fluximp_valid_tool_calls'] (what
        the parse node actually executed) when present, falling back to
        msg.tool_calls, so the rule also covers the text protocol, where the
        calls live in content/reasoning and tool_calls stays empty.

        Only scans positions that haven't been scanned yet. A position whose
        decision can still change (the batch's results do not exist yet, no later
        AI message exists yet, or dropping would leave no live tool result) is
        left unmarked so it is reconsidered on the next request.
        """
        if not self._failed_tool_prune_enabled():
            return
        
        messages = state["messages"]
        
        # Index of the newest AI message. Its batch has not been sent to the LLM
        # yet, so pruning it now would rob the model of its single verbatim view
        # of the failed call.
        last_ai_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], AIMessage):
                last_ai_idx = i
                break
        
        for ai_msg_idx, msg in enumerate(messages):
            # Skip already scanned positions
            if ai_msg_idx in self._drop_scanned_positions:
                continue
            
            if not isinstance(msg, AIMessage):
                continue
            
            # A SummaryTransformer replaces the message (dropping it would delete
            # the summary) and an existing DropTransformer means it is already
            # pruned, so both decisions are final.
            if ai_msg_idx < len(self.transformers) and isinstance(
                    self.transformers[ai_msg_idx], (SummaryTransformer, DropTransformer)):
                self._drop_scanned_positions.add(ai_msg_idx)
                continue
            
            # Tool names for this batch. The text protocol (and any provider that
            # returns the calls inside content/reasoning instead of tool_calls)
            # leaves msg.tool_calls empty, so prefer the calls the parse node
            # actually executed, recorded in
            # additional_kwargs['fluximp_valid_tool_calls'] - the same source
            # _check_and_run_summarization uses. The stash is present but empty
            # when the response had no valid calls, which correctly leaves the
            # batch non-prune-eligible.
            stashed_calls = getattr(msg, 'additional_kwargs', {}).get('fluximp_valid_tool_calls')
            if stashed_calls is not None:
                tool_names = [name for name, _args in stashed_calls]
            else:
                tool_names = [tc.get('name', '') for tc in (getattr(msg, 'tool_calls', None) or [])]
            if not tool_names:
                self._drop_scanned_positions.add(ai_msg_idx)
                continue
            
            # Find all tool results following this AIMessage
            tool_result_indices = []
            for j in range(ai_msg_idx + 1, len(messages)):
                next_msg = messages[j]
                if isinstance(next_msg, AIMessage):
                    break
                is_tool_result = isinstance(next_msg, ToolMessage)
                if is_tool_result:
                    tool_result_indices.append(j)
            
            if not tool_result_indices:
                # The results have not been produced yet (mid-turn) or the turn
                # was abandoned. Defer without marking so the position is
                # reconsidered once the results exist.
                continue
            
            # Check if ALL tool results have execution_successful=False
            all_failed = True
            for tr_idx in tool_result_indices:
                is_successful = getattr(messages[tr_idx], 'execution_successful', True)
                if is_successful:
                    all_failed = False
                    break
            
            if not all_failed:
                # The results are final, so this decision is stable.
                self._drop_scanned_positions.add(ai_msg_idx)
                continue
            
            # Only prune failed pairs for content-carrying tools.
            # Non-content tools (e.g. run_python_script, run_tests)
            # can surface diagnostics or user redirects in their
            # results that must remain in the model's view.
            # (tool_names was derived above from the stashed executed calls when
            # present, so text-protocol batches are covered too)
            all_content = all(
                self._is_content_carrying_tool(name) for name in tool_names)
            if not all_content:
                self._drop_scanned_positions.add(ai_msg_idx)
                continue
            
            # One-verbatim-send rule: the batch must already have been sent to the
            # LLM once, which is exactly when a later AI message exists.
            if last_ai_idx is None or ai_msg_idx >= last_ai_idx:
                continue
            
            # The file views / cumulative diff are injected at the last live
            # (non-dropped) tool result (_update_context_injections), so never
            # install a drop that would leave no live tool result behind.
            live_result_elsewhere = False
            for j, other in enumerate(messages):
                if not isinstance(other, ToolMessage):
                    continue
                if j in tool_result_indices:
                    continue
                if j < len(self.transformers) and isinstance(self.transformers[j], DropTransformer):
                    continue
                live_result_elsewhere = True
                break
            if not live_result_elsewhere:
                continue
            
            # Install drop transformers
            print(f"DEBUG _scan_for_drop_opportunities: installing drop transformers at {ai_msg_idx} and {tool_result_indices}")
            
            # Make sure we have enough transformer slots
            max_idx = max(ai_msg_idx, max(tool_result_indices)) + 1
            while len(self.transformers) < max_idx:
                self.transformers.append(None)
            
            # Install DropTransformer at the AIMessage position. This position may
            # already hold a ToolCallOmissionTransformer attached on an earlier
            # request; the drop takes precedence, which is what makes this a prune
            # instead of an omit (mirroring _attach_tool_call_omission_transformer,
            # which yields to a drop).
            self.transformers[ai_msg_idx] = DropTransformer(source="failed_tool_prune")
            self._drop_scanned_positions.add(ai_msg_idx)
            
            # Install DropTransformer at all tool result positions
            for tr_idx in tool_result_indices:
                self.transformers[tr_idx] = DropTransformer(source="failed_tool_prune")
                self._drop_scanned_positions.add(tr_idx)
            
            # The omission transformer for this position is now dead (the whole
            # message is dropped), so remove it from the active dict as well:
            # leaving it behind would persist it in the checkpoint and re-attach
            # it on the next rebuild.
            active_omission = state.get("tool_call_omission_transformers") or {}
            if ai_msg_idx in active_omission:
                updated_omission = dict(active_omission)
                del updated_omission[ai_msg_idx]
                state["tool_call_omission_transformers"] = updated_omission
                self._pending_tool_call_omission_transformers_update = updated_omission
                self._tool_call_omission_positions.discard(ai_msg_idx)
                print(f"DEBUG _scan_for_drop_opportunities: removed dead omission transformer entry at {ai_msg_idx}")
    
    def _update_context_injections(self, state: AgentState):
        """Update context injection transformers based on current context chunks.
        
        Uses ContextInjectionManager to determine optimal placement of context
        chunks across tool message positions for prefix caching. Creates one
        ContextInjectionTransformer per injection record with fixed text.
        """

        if not hasattr(self.agent_config, "get_context_chunks"):
            return

        messages = state["messages"]
        
        # Find the last tool result message index that is NOT being dropped.
        # A dropped tool result would leave an orphaned injection, and if it's the
        # last_tool_result_idx the ContextInjectionManager would place all new/changed
        # chunks there â which we'd then have to skip, losing that context entirely.
        last_tool_result_idx = None
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            is_tool_result = isinstance(msg, ToolMessage)
            if is_tool_result:
                # Skip positions that are being dropped
                if i < len(self.transformers) and isinstance(self.transformers[i], DropTransformer):
                    continue
                last_tool_result_idx = i
                break
        
        # Get context chunks from agent config
        tool_kwargs = {
            'filesystem_service': self.filesystem_service,
            'experiments_service': self.registry._experiments_service,
            'state_service': self.registry._state_service,
        }
        
        self.registry._state_service.update_state_reference(state)
        
        chunks, state_update = self.agent_config.get_context_chunks(state, **tool_kwargs)
        
        # Compute actual view/diff content bytes for Criterion 3. These are a
        # subset of the total API content bytes (measured in _build_api_messages
        # via calculate_message_api_bytes) and are subtracted because view and
        # diff content is never summarized.
        context_view_bytes = 0
        context_diff_bytes = 0
        for chunk in chunks:
            if chunk.section_id.startswith("view:") or chunk.section_id.startswith("search:"):
                context_view_bytes += len(chunk.content.encode('utf-8'))
            elif chunk.section_id == "cumulative_diff":
                context_diff_bytes += len(chunk.content.encode('utf-8'))
        state["last_api_view_bytes"] = context_view_bytes
        state["last_api_diff_bytes"] = context_diff_bytes
        
        if state_update:
            self._pending_context_state_update = state_update
        
        # Compute time since last API response for prefix cache TTL checking
        seconds_since_last_response = None
        response_metadata = state.get("response_metadata")
        if response_metadata and response_metadata.get("end_time"):
            try:
                last_end_time = parse_utc_timestamp(response_metadata["end_time"])
                seconds_since_last_response = (datetime.now(timezone.utc) - last_end_time).total_seconds()
            except (ValueError, TypeError):
                pass
        
        # Compute the set of positions that have DropTransformers.
        # The ContextInjectionManager needs to know which positions are dropped
        # so it doesn't assign content to them (which would be silently discarded).
        dropped_indices = set()
        for i in range(len(self.transformers)):
            if isinstance(self.transformers[i], DropTransformer):
                dropped_indices.add(i)

        # Compute the other structural-break positions the manager needs to detect
        # a prefix break it cannot see from injection content alone:
        #  - omission_indices: AI messages whose args are rewritten in place by a
        #    ToolCallOmissionTransformer (a mid-history byte change).
        #  - summary_indices: positions replaced by a SummaryTransformer.
        #  - valid_tool_indices: positions that are still live (non-dropped)
        #    ToolMessages, so the manager can detect an injection record whose
        #    target message was dropped or otherwise moved.
        omission_indices = set()
        summary_indices = set()
        for i in range(len(self.transformers)):
            if isinstance(self.transformers[i], ToolCallOmissionTransformer):
                omission_indices.add(i)
            elif isinstance(self.transformers[i], SummaryTransformer):
                summary_indices.add(i)
        valid_tool_indices = {
            i for i, m in enumerate(messages)
            if isinstance(m, ToolMessage) and i not in dropped_indices
        }

        # Use the manager to determine placements
        placements = self.context_injection_manager.get_placements(
            chunks, last_tool_result_idx, seconds_since_last_response,
            dropped_indices=dropped_indices,
            omission_indices=omission_indices,
            summary_indices=summary_indices,
            valid_tool_indices=valid_tool_indices,
        )
        
        # Check if system prompt context injection experiment is enabled
        use_system_prompt_injection = self.registry._experiments_service and self.registry._experiments_service.is_enabled(
            'system_prompt_context_injection'
        )
        
        # Clear any existing context injection transformer instances
        for i in range(len(self.transformers)):
            if isinstance(self.transformers[i], (ContextInjectionTransformer, ContextInjectionSystemTransformer, AgentSpecificSystemTransformer, AgentSpecificToolTransformer)):
                self.transformers[i] = None
        
        # Install new transformer instances at the determined positions.
        # The ContextInjectionManager has been given dropped_indices, so it should
        # never produce a placement at a dropped position. If it does, that's a bug
        # â fail fast rather than silently discarding the content.
        for msg_idx, text, is_reminders_only in placements:
            # Make sure we have enough slots
            while len(self.transformers) <= msg_idx:
                self.transformers.append(None)
            
            if isinstance(self.transformers[msg_idx], DropTransformer):
                raise AssertionError(
                    f"ContextInjectionManager produced a placement at index {msg_idx}, "
                    f"but that position has a DropTransformer. dropped_indices={dropped_indices}. "
                    f"This means the manager did not correctly avoid dropped positions."
                )
            
            if use_system_prompt_injection:
                if is_reminders_only:
                    # Skip system message injection when only reminders content
                    # â it's redundant with the system prompt
                    continue
                self.transformers[msg_idx] = ContextInjectionSystemTransformer(text_to_inject=text)
            else:
                self.transformers[msg_idx] = ContextInjectionTransformer(text_to_inject=text)
        
        print(f"DEBUG _update_context_injections: {len(placements)} injection(s) at indices {[p[0] for p in placements]}")
        
        # Save injection manager state for checkpoint persistence
        self._pending_context_injection_state = self.context_injection_manager.to_state_dict()
        
        # Handle Case A: no tool messages yet (two_part_system_prompt experiment)
        # When there are no tool messages, inject agent-specific content after
        # the first HumanMessage. The message type depends on whether
        # system_prompt_context_injection is enabled:
        #   - Enabled: use a SystemMessage (AgentSpecificSystemTransformer)
        #   - Not enabled: use a tool-like message (AgentSpecificToolTransformer)
        #     since we're in append-to-tool-messages mode
        two_part_system_prompt = (
            self.registry._experiments_service and
            self.registry._experiments_service.is_enabled('two_part_system_prompt')
        )
        if two_part_system_prompt and last_tool_result_idx is None:
            if self.agent_config.agent_specific_system_content is not None:
                # Find the first HumanMessage position (not a tool result)
                first_human_idx = None
                for i, msg in enumerate(messages):
                    if isinstance(msg, HumanMessage):
                        first_human_idx = i
                        break
                
                if first_human_idx is not None:
                    while len(self.transformers) <= first_human_idx:
                        self.transformers.append(None)
                    if use_system_prompt_injection:
                        self.transformers[first_human_idx] = AgentSpecificSystemTransformer(
                            content=self.agent_config.agent_specific_system_content
                        )
                        print(f"DEBUG _update_context_injections: agent-specific system message at HumanMessage index {first_human_idx}")
                    else:
                        self.transformers[first_human_idx] = AgentSpecificToolTransformer(
                            content=self.agent_config.agent_specific_system_content,
                        )
                        print(f"DEBUG _update_context_injections: agent-specific tool message at HumanMessage index {first_human_idx}")
    
    
    def _build_api_messages(self, state: AgentState) -> Tuple[list, set]:
        """Build API request messages using transformers.
        
        Iterates through messages and applies transformers:
        - If transformer exists and returns None, omit message
        - If transformer exists, use returned message
        - If no transformer, use original message
        
        Returns:
            Tuple of (api_messages, context_injected_indices) where
            context_injected_indices is the set of API message indices
            that had ContextInjectionTransformer applied.
        """
        messages = state["messages"]
        api_messages = []
        context_injected_indices = set()
        included_count = 0
        dropped_count = 0
        summary_count = 0
        for i, msg in enumerate(messages):
            transformer = self.transformers[i] if i < len(self.transformers) else None
            
            if transformer is not None:
                if isinstance(transformer, DropTransformer):
                    dropped_count += 1
                    print(f'DEBUG _build_api_messages: [{i}] DROPPED by DropTransformer(source={transformer.source})')
                    continue  # drop this message
                elif isinstance(transformer, SummaryTransformer):
                    summary_count += 1
                    result = transformer(msg)
                    api_messages.append(result)
                    included_count += 1
                    preview = result.content[:120].replace(chr(10), chr(92)+chr(110)) + "..." if len(result.content) > 120 else result.content.replace(chr(10), chr(92)+chr(110))
                    print(f"DEBUG _build_api_messages: [{i}] SUMMARY replaces {transformer.start_index}\u2013{transformer.end_index}: {preview}")
                    continue
                else:
                    result = transformer(msg)
                    if result is None:
                        dropped_count += 1
                        print(f'DEBUG _build_api_messages: [{i}] DROPPED by {type(transformer).__name__}')
                        continue
                    if isinstance(result, (list, tuple)):
                        api_messages.extend(result)
                        included_count += len(result)
                        if isinstance(transformer, ContextInjectionSystemTransformer):
                            # Track the system message (last in the list) for logging
                            context_injected_indices.add(len(api_messages) - 1)
                        print(f'DEBUG _build_api_messages: [{i}] transformed by {type(transformer).__name__} -> {len(result)} messages')
                    else:
                        api_messages.append(result)
                        included_count += 1
                        if isinstance(transformer, ContextInjectionTransformer):
                            context_injected_indices.add(len(api_messages) - 1)
                        print(f'DEBUG _build_api_messages: [{i}] transformed by {type(transformer).__name__}')
                    continue
            else:
                # Default: use original message
                api_messages.append(msg)
                included_count += 1
        
        # Apply ToolMessage compatibility transformer as second pass
        # This converts ToolMessage to HumanMessage for models that don't support ToolMessage,
        # after all position-specific transformers (context injection, summarization, drops) have been applied.
        api_messages = self._compatibility_transformer.transform_messages(api_messages)
        
        # Strip dangling tool calls from AI messages whose tool results are absent
        # (e.g., after a turn was abandoned before the tools executed). Needed for
        # API-protocol models that rigidly require tool-message ids to match a tool
        # use in the preceding AI message.
        api_messages = self._strip_dangling_tool_calls(api_messages, messages)
        # Calculate API content byte counts for each transformed message.
        # calculate_message_api_bytes serializes each message exactly as the
        # LLM client will (via _convert_message_to_dict_with_reasoning) and
        # stashes the per-message byte count in additional_kwargs[
        # 'api_content_bytes'] for inline consumption by AgentLogger.log_messages
        # (so each message is serialized exactly once per request).
        api_content_bytes_total = 0
        for api_msg in api_messages:
            api_content_bytes_total += calculate_message_api_bytes(api_msg)
        # Write Criterion 3 byte totals directly to state. The view/diff
        # bytes were set by _update_context_injections; write the total and
        # message count here. All four are persisted via build_full_update
        # carry-over in _process_response / _make_error_state.
        state["last_api_content_bytes"] = api_content_bytes_total
        state["last_api_request_message_count"] = len(state["messages"])
        view_bytes = state.get("last_api_view_bytes", 0)
        diff_bytes = state.get("last_api_diff_bytes", 0)
        print(f"DEBUG _build_api_messages: total={len(messages)} included={included_count} dropped={dropped_count} summaries={summary_count} api_msgs={len(api_messages)} api_content_bytes={api_content_bytes_total:,} (views={view_bytes:,} diff={diff_bytes:,})")
        return api_messages, context_injected_indices
    
    def _strip_dangling_tool_calls(self, api_messages, state_messages):
        """Remove tool calls from AI messages whose tool results are absent.

        When a turn is abandoned after the AI emitted tool calls but before the
        tools executed, no ToolMessage exists for those calls in the message list.
        For API-protocol providers that rigidly require a tool message's id to
        match a tool use in the preceding AI message, such a dangling tool call
        (a tool use with no corresponding result) causes a bad-request error.
        We strip those tool calls so no id is referenced without a corresponding
        result.

        Scoped to use_api_protocol_for_tool_functions=True with
        use_tool_message_for_no_tool_error=False -- the configuration where the
        provider is rigid about id matching. Other configs (including
        use_tool_message_for_no_tool_error=True and the text protocol) are left
        untouched.

        Note on prefix caching: stripping does NOT break the prefix cache in the
        normal flow. An AI message is only ever sent after its tool batch has
        run, so a call is never "present at first send, dangling later"; an
        abandoned turn leaves its calls dangling permanently, so the stripped
        output is identical on every request that includes the message. No
        transition => no break. (Contrast ToolCallOmissionTransformer, which is
        deliberately attached one turn AFTER the AI message was first sent
        verbatim -- that delay is what creates a real transition.)
        """
        if not (self.use_api_protocol and not self.agent_config.use_tool_message_for_no_tool_error):
            return api_messages

        # Collect tool_call_ids that have a corresponding ToolMessage in the full
        # message history (i.e. the tool call was executed and produced a result).
        tool_msg_ids = {
            msg.tool_call_id for msg in state_messages if isinstance(msg, ToolMessage)
        }

        result = []
        for msg in api_messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                kept = [tc for tc in msg.tool_calls if tc.get("id") in tool_msg_ids]
                if len(kept) != len(msg.tool_calls):
                    new_msg = msg.model_copy(deep=True)
                    new_msg.tool_calls = kept
                    result.append(new_msg)
                else:
                    result.append(msg)
            else:
                result.append(msg)
        return result

    def _log_api_request(self, messages, request_count, extra_log_context=None):
        """Log API request details with context injection information.
        
        Extends the base class logging with details about tool messages
        that have injected context.
        """
        super()._log_api_request(messages, request_count, extra_log_context)
        context_injected_indices = extra_log_context or set()

        indexes_to_log = set()
        last_idx = None
        # Always log the last tool message
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].type == 'tool':
                last_idx = i
                indexes_to_log.add(i)
                break

        if context_injected_indices:
            if last_idx is None:
                last_idx = max(context_injected_indices)
                print("last_idx was None; set to:", last_idx)

        if last_idx is None:
            print("last_idx is None; context_injected_indices:", context_injected_indices)
            # With ContextInjectionSystemTransformer, context_injected_indices may
            # point to system messages even when there are no tool messages
            pass
        else:
            # Log tool messages with injected context
            if context_injected_indices:
                indexes_to_log.update(context_injected_indices)

            indexes_to_log = list(indexes_to_log)
            indexes_to_log.sort(reverse=True)

            for idx in indexes_to_log:
                with_injected = " with injected context" if idx in context_injected_indices else ""
                is_last = "last " if idx == last_idx else ""
                msg_type = messages[idx].type  # 'tool', 'system', etc.
                self.logger._log(f"\nDEBUG {is_last}{msg_type} message [{idx}]{with_injected}:")
                self.logger._log(messages[idx].content)
                self.logger._log(f"\nDEBUG after {msg_type} message [{idx}]")

        self.logger.flush()
    
    def _make_error_state(self, state, error_content, request_count, node_name):
        """Build a complete state update for error paths, ensuring no stale fields.
        
        Unlike the success path where parse_tool_calls/execute_tool_call will set these,
        tools-related fields are preserved from the previous turn.
        """
        updates = {
            "messages": [AIMessage(content=error_content)],
            "request_count": request_count,
            "file_operations_this_turn": [],
            "response_metadata": None,
        }
        if self._pending_context_state_update:
            updates.update(self._pending_context_state_update)
            self._pending_context_state_update = None
        if self._pending_summarized_ranges is not None:
            updates["summarized_ranges"] = self._pending_summarized_ranges
            self._pending_summarized_ranges = None
        if self._pending_context_injection_state is not None:
            updates["context_injection_state"] = self._pending_context_injection_state
            self._pending_context_injection_state = None
        if self._pending_tool_call_omission_transformers_update is not None:
            updates["tool_call_omission_transformers"] = self._pending_tool_call_omission_transformers_update
            self._pending_tool_call_omission_transformers_update = None
        return self.state_config.build_full_update(state, updates, node_name)

    def _on_request_start(self, start_dt):
        """Notify lifecycle hooks of request start."""
        if self.lifecycle_hooks:
            self.lifecycle_hooks.on_request_start(start_dt)
    
    def _invoke_client(self, messages):
        """Invoke LLM with cache key for prefix caching."""
        print("DEBUG _invoke_with_retry start_dt", datetime.now(timezone.utc))
        assert self.thread_id_ref
        assert self.thread_id_ref[0]
        # Use shared cache key when two_part_system_prompt is active
        # so the prefix cache is preserved across agent type changes
        if (self.registry._experiments_service and
            self.registry._experiments_service.is_enabled('two_part_system_prompt')):
            cache_key = self.thread_id_ref[0] + '_shared'
        else:
            cache_key = self.thread_id_ref[0] + self.agent_config.agent_type
        print("DEBUG cache_key", cache_key)
        response = self.client.invoke(messages, prompt_cache_key=cache_key)
        if "model_provider" in response.response_metadata:
            # The provider isn't actually openai and maybe many different things, the specific
            # provider doesn't matter though, and an openai value with cause messages to be translated
            # for openai specifically and they are lame only have reasoning summaries instead of
            # actual the actual reasoning.
            response.response_metadata["model_provider"] = "o1 compatible"
        return response

    def _invoke_client_streaming(self, messages):
        """Invoke LLM with streaming and cache key for prefix caching."""
        print("DEBUG _invoke_client_streaming start_dt", datetime.now(timezone.utc))
        assert self.thread_id_ref
        assert self.thread_id_ref[0]
        # Use shared cache key when two_part_system_prompt is active
        # so the prefix cache is preserved across agent type changes
        if (self.registry._experiments_service and
            self.registry._experiments_service.is_enabled('two_part_system_prompt')):
            cache_key = self.thread_id_ref[0] + '_shared'
        else:
            cache_key = self.thread_id_ref[0] + self.agent_config.agent_type
        print("DEBUG cache_key", cache_key)

        accumulated = None
        reasoning_content_parts = []
        reasoning_parts = []
        last_token_usage = None
        last_usage_metadata = None

        for chunk in self.client.stream(messages, prompt_cache_key=cache_key):
            if accumulated is None:
                accumulated = chunk
            else:
                accumulated = accumulated + chunk

            # Accumulate reasoning content (strings that merge_dicts would overwrite)
            if hasattr(chunk, "additional_kwargs") and chunk.additional_kwargs:
                rc = chunk.additional_kwargs.get("reasoning_content")
                if rc:
                    reasoning_content_parts.append(rc)
                r = chunk.additional_kwargs.get("reasoning")
                if r:
                    reasoning_parts.append(r)

            # Capture token usage from response_metadata (typically only in last chunk)
            if hasattr(chunk, "response_metadata") and chunk.response_metadata:
                usage = chunk.response_metadata.get("token_usage")
                if usage:
                    last_token_usage = usage

            # Capture usage_metadata (LangChain structured token usage)
            if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                last_usage_metadata = chunk.usage_metadata

        if accumulated is not None:
            # Set accumulated reasoning content on the final chunk
            # to_message() preserves additional_kwargs, so set before conversion
            if reasoning_content_parts:
                accumulated.additional_kwargs["reasoning_content"] = "".join(reasoning_content_parts)
            if reasoning_parts:
                accumulated.additional_kwargs["reasoning"] = "".join(reasoning_parts)

            # Save response_metadata before to_message() which does NOT preserve it
            saved_response_metadata = accumulated.response_metadata or {}
            if last_token_usage is not None:
                saved_response_metadata["token_usage"] = last_token_usage

            # Ensure we return an AIMessage (not AIMessageChunk)
            if hasattr(accumulated, 'to_message'):
                accumulated = accumulated.to_message()

            # Restore response_metadata after to_message() since it's not preserved
            accumulated.response_metadata = saved_response_metadata

            # Restore usage_metadata if present
            if last_usage_metadata is not None:
                accumulated.usage_metadata = last_usage_metadata

            # Normalize token usage from usage_metadata into response_metadata["token_usage"]
            # so _build_log_metadata always has a consistent data source, regardless of
            # whether the response came from streaming or non-streaming invocation.
            if "token_usage" not in accumulated.response_metadata:
                token_usage = self._extract_token_usage(accumulated)
                if token_usage and any(v is not None for v in token_usage.values()):
                    accumulated.response_metadata["token_usage"] = token_usage

            # Apply LenientChatOpenAI post-processing if needed
            if hasattr(self.client, '_postprocess_message'):
                accumulated = self.client._postprocess_message(accumulated)
            if 'model_provider' in accumulated.response_metadata:
                accumulated.response_metadata['model_provider'] = 'o1 compatible'
        return accumulated
    
    def _make_transient_error_result(self, state, error_msg, request_count):
        """Build full state update when all transient error retries are exhausted."""
        error_content = f"INFRASTRUCTURE_ERROR: API returned 429 after {self.max_transient_retries} retries. {error_msg}"
        self.logger.log_assistant_message(error_content, {"error": "infrastructure", "retries": self.max_transient_retries})
        return self._make_error_state(state, error_content, request_count, "agent_node(transient_error)")
    
    def _process_response(self, response, request_count, start_dt, end_dt, state):
        """Extract metadata, verify model, build state update."""
        # Verify model name matches what we requested
        requested_model = self.agent_config.model_config.model
        actual_model = response.response_metadata.get('model_name', 'unknown')
        self.agent_config.model_config.provider.verify_model_response(requested_model, actual_model)
        
        # Build log metadata and response_metadata using shared helper
        metadata, response_metadata = self._build_log_metadata(response, start_dt, end_dt)
        
        self.logger.log_assistant_message(response, metadata)
        
        content = response.content
        # if not content:
        #     content = "tool use"
        #print(f"DEBUG response.content: {content}")
        
        # Build result with response metadata for file_io_mode
        result_messages = [response]
        # if self.use_api_protocol:
        #     tool_calls = getattr(response, 'tool_calls', None)
        #     if tool_calls is not None:
        #         result_messages = [AIMessage(content=content, tool_calls=tool_calls)]
        #     else:
        #         result_messages = [AIMessage(content=content)]
        # else:
        #     result_messages = [AIMessage(content=content)]
        
        updates = {
            "messages": result_messages,
            "request_count": request_count,
            "response_metadata": response_metadata,
            "file_operations_this_turn": [],
        }
        if self._pending_context_state_update:
            updates.update(self._pending_context_state_update)
            self._pending_context_state_update = None
        if self._pending_summarized_ranges is not None:
            updates["summarized_ranges"] = self._pending_summarized_ranges
            self._pending_summarized_ranges = None
        if self._pending_context_injection_state is not None:
            updates["context_injection_state"] = self._pending_context_injection_state
            self._pending_context_injection_state = None
        if self._pending_tool_call_omission_transformers_update is not None:
            updates["tool_call_omission_transformers"] = self._pending_tool_call_omission_transformers_update
            self._pending_tool_call_omission_transformers_update = None
        return self.state_config.build_full_update(
            state,
            updates,
            "agent_node(success)"
        )
