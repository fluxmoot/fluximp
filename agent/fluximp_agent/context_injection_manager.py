
"""Context Injection Manager for prefix-cache-aware context splitting.

Manages the splitting of injected context across multiple tool message positions
to maximize prefix caching across consecutive API requests.

The key insight: injected context has varying stability (some parts change rarely,
some change every request). By splitting context at historical injection points
and preserving unchanged prefixes, we get prefix cache hits for the stable parts.

Algorithm overview:
- On each request, get current ContextChunks sorted by stability (most stable first)
- Compare against previous injection records to find the first change point
- Preserve matching prefixes at their existing message positions (cache hit!)
- Move changed and new chunks to the last tool message (cache miss, but 
  stabilizes future requests)
- When an earlier injection changes, consolidate all later injections into
  the last tool message (since the prefix break cascades)

`prefix_broken` inside _compute_new_records is a *local consolidation trigger*
for injection splitting, not a general prefix-cache-break detector. It is set
when an earlier injection chunk changed, or when a structural break (drop /
omission / summary) appeared at or before a record, or when a record was
relocated. The broader break signals are supplied by the caller via
dropped_indices / omission_indices / summary_indices / valid_tool_indices.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple


@dataclass
class ContextChunk:
    section_id: str
    content: str
    stability_key: tuple  # (last_changed_ts, creation_ts, section_id)
    always_last: bool = False  # If True, always place at end of last injection


@dataclass
class InjectionRecord:
    message_index: int
    chunks: List[ContextChunk]
    combined_text: str
    ghost_text: str = ""  # Preserved always_last text for prefix caching at earlier positions
    # Positioning context captured when the record was created, used to detect
    # that the record no longer sits at the same byte prefix on a later request:
    # break_indices is the set of structural-break positions (drops/omissions/
    # summaries) at or before message_index that existed at creation time.
    break_indices: frozenset = frozenset()



class ContextInjectionManager:
    """Manages context injection splitting for prefix caching.
    
    Tracks previous injection records and determines optimal placement
    of context chunks across tool message positions to maximize prefix
    cache hits on subsequent requests.
    """
    
    def __init__(self, cache_ttl_seconds: int = 3594):
        self.cache_ttl_seconds = cache_ttl_seconds
        self.injection_records: List[InjectionRecord] = []
        self.previous_chunks: List[ContextChunk] = []
    
    def reset(self):
        """Clear all state (called on fork, rebuild, etc.)."""
        self.injection_records = []
        self.previous_chunks = []
        print("DEBUG ContextInjectionManager.reset")
    
    def to_state_dict(self) -> dict:
        """Serialize manager state for checkpoint persistence."""
        return {
            "injection_records": self.injection_records,
            "previous_chunks": self.previous_chunks,
        }
    
    def from_state_dict(self, state_dict: dict) -> None:
        """Restore manager state from checkpoint data.

        A checkpoint written by a build that predates the break_indices
        positioning context has records that cannot be trusted for prefix-break
        detection. Discard such state so the next request rebuilds it from
        scratch with correct positioning context (mirrors reset-on-fork /
        reset-on-summarize).
        """
        injection_records = state_dict.get("injection_records", [])
        if any(
            "break_indices" not in getattr(record, "__dict__", {})
            for record in injection_records
        ):
            print("DEBUG ContextInjectionManager.from_state_dict: discarding legacy state lacking positioning context")
            self.reset()
            return
        self.injection_records = injection_records
        self.previous_chunks = state_dict.get("previous_chunks", [])
    
    def prepare_injections(
        self, 
        current_chunks: List[ContextChunk],
        last_tool_msg_idx: Optional[int],
        seconds_since_last_response: Optional[float] = None,
        dropped_indices: Optional[set] = None,
        omission_indices: Optional[set] = None,
        summary_indices: Optional[set] = None,
        valid_tool_indices: Optional[set] = None,
    ) -> List[InjectionRecord]:
        """Determine where to inject context chunks for maximum prefix caching.
        
        Args:
            current_chunks: Current context chunks sorted by stability 
                (most stable first). The caller must sort these before passing.
            last_tool_msg_idx: Index of the last tool result message in the
                message list, or None if there are no tool result messages.
            seconds_since_last_response: Time elapsed since the last API
                response was received. If this exceeds cache_ttl_seconds,
                the prefix cache is considered cold and all context is
                consolidated at the last tool message.
            dropped_indices: Positions (message indices) that have a
                DropTransformer this request; content is never placed on them.
            omission_indices: AI message positions rewritten in place by a
                ToolCallOmissionTransformer this request.
            summary_indices: Positions replaced by a SummaryTransformer this
                request.
            valid_tool_indices: Positions that are still live (non-dropped)
                ToolMessages this request, or None if unknown.
        
        Returns:
            List of InjectionRecord describing where and what to inject.
            Empty list if last_tool_msg_idx is None (no tool messages).
        """
        if last_tool_msg_idx is None:
            return []
        
        if not current_chunks:
            return []
        
        # If the prefix cache has expired (TTL exceeded), reset to consolidate
        # all context at the last tool message position
        if seconds_since_last_response is not None:
            if seconds_since_last_response > self.cache_ttl_seconds:
                self.reset()
        else:
            print("DEBUG seconds_since_last_response is None")
        
        # Separate always_last chunks from regular chunks
        regular_chunks = [c for c in current_chunks if not c.always_last]
        always_last_chunks = [c for c in current_chunks if c.always_last]
        always_last_section_ids = {c.section_id for c in current_chunks if c.always_last}
        always_last_text = "".join(c.content for c in always_last_chunks)
        
        # Structural break positions for the CURRENT request. A newly appeared
        # break at or before a record means the bytes at/before that record have
        # changed since it was written: a DropTransformer removes a message
        # (shifting the API index of everything after it), a
        # ToolCallOmissionTransformer rewrites a historical AI message's args in
        # place, and a SummaryTransformer replaces a range of messages. Either
        # way the prefix cache is broken from that point on.
        current_breaks = set(dropped_indices or ())
        current_breaks |= set(omission_indices or ())
        current_breaks |= set(summary_indices or ())
        
        # First injection ever: inject everything at the last tool message
        # (regular chunks first, then always_last chunks at the end)
        if not self.injection_records:
            combined = "".join(c.content for c in regular_chunks) + always_last_text
            all_chunks = list(regular_chunks) + list(always_last_chunks)
            record = InjectionRecord(
                message_index=last_tool_msg_idx,
                chunks=all_chunks,
                combined_text=combined,
                break_indices=self._breaks_at_or_before(last_tool_msg_idx, current_breaks),
            )
            self.injection_records = [record]
            self.previous_chunks = list(current_chunks)
            return self.injection_records
        
        # Subsequent injections: compare with previous to find split point
        new_records = self._compute_new_records(
            regular_chunks, always_last_chunks, always_last_section_ids,
            always_last_text, last_tool_msg_idx, current_breaks=current_breaks,
            valid_tool_indices=valid_tool_indices,
        )
        
        self.injection_records = new_records
        self.previous_chunks = list(current_chunks)
        return new_records
    
    def _compute_new_records(
        self,
        regular_chunks: List[ContextChunk],
        always_last_chunks: List[ContextChunk],
        always_last_section_ids: set,
        always_last_text: str,
        last_tool_msg_idx: int,
        current_breaks: Optional[set] = None,
        valid_tool_indices: Optional[set] = None,
    ) -> List[InjectionRecord]:
        """Core algorithm: compare regular chunks against injection records.
        
        Walks through injection records from earliest to latest, matching
        against regular_chunks to find the first change point. Preserves
        matching prefixes for prefix caching, moves changes to the end.
        
        Always_last chunks are always placed at the end of the last injection.
        Ghost copies of always_last text are left at earlier injection positions
        (on full matches only) to preserve prefix cache hits until those
        positions' prefixes break naturally.
        
        Args:
            current_breaks: Structural-break positions for this request (the
                union of dropped, omission and summary positions). A record
                whose stored break set differs from the current set at/before
                its message_index has had its byte prefix changed.
            valid_tool_indices: Positions that are still live (non-dropped)
                ToolMessages, or None if unknown. A record targeting a position
                not in this set is no longer a valid injection site.
        """
        new_records: List[InjectionRecord] = []
        pointer = 0  # Current position in regular_chunks
        prefix_broken = False

        current_breaks = set(current_breaks or ())

        for record in self.injection_records:
            if prefix_broken:
                # Earlier change broke prefix cache for all subsequent positions.
                # Skip this record - all remaining chunks go to last tool message.
                break

            # (a) Position-invalidity: the record's target is no longer a live
            # (non-dropped) tool message, so we cannot inject there.
            if valid_tool_indices is not None and record.message_index not in valid_tool_indices:
                prefix_broken = True
                continue

            # (b) A new structural break appeared at/before this record since it
            # was created. Covers a DropTransformer at a non-injection position
            # (failed_tool_prune) and a ToolCallOmissionTransformer on a
            # historical AI message, as well as a drop landing exactly on the
            # record (the previously-detected case). A None break_indices means
            # "unknown" and skips the comparison; records that predate this field
            # are discarded in from_state_dict, so None is only a defensive
            # fallback.
            rec_break_indices = getattr(record, 'break_indices', None)
            if rec_break_indices is not None:
                breaks_now = frozenset(b for b in current_breaks if b <= record.message_index)
                if breaks_now != rec_break_indices:
                    prefix_broken = True
                    continue
            
            # Separate previous record's chunks into regular and always_last
            prev_regular = []
            prev_always_last = []
            for chunk in record.chunks:
                if chunk.section_id in always_last_section_ids:
                    prev_always_last.append(chunk)
                else:
                    prev_regular.append(chunk)
            
            # Get existing ghost_text (for backward compat with old records)
            existing_ghost = getattr(record, 'ghost_text', '')
            
            # Special case: record had only always_last chunks (no regular)
            if not prev_regular:
                # Preserve ghost_text if it exists, otherwise use prev_always_last content
                ghost = existing_ghost or "".join(c.content for c in prev_always_last)
                new_records.append(InjectionRecord(
                    message_index=record.message_index,
                    chunks=[],  # No regular chunks at this position
                    combined_text="",
                    ghost_text=ghost,
                    break_indices=self._breaks_at_or_before(record.message_index, current_breaks),
                ))
                # Don't advance pointer - we didn't consume any regular chunks
                continue
            
            # Try to match prev_regular against regular_chunks starting at pointer
            match_count = 0
            for i, prev_chunk in enumerate(prev_regular):
                if pointer + i >= len(regular_chunks):
                    break
                
                curr_chunk = regular_chunks[pointer + i]
                if (curr_chunk.section_id == prev_chunk.section_id and 
                    curr_chunk.content == prev_chunk.content):
                    match_count += 1
                else:
                    break
            
            if match_count == len(prev_regular):
                # Full match: keep this record with ghost_text for prefix caching
                # Ghost text preserves the always_last content that was previously
                # at this position, keeping the byte-level prefix intact
                ghost = existing_ghost or "".join(c.content for c in prev_always_last)
                matched_chunks = regular_chunks[pointer:pointer + match_count]
                combined = "".join(c.content for c in matched_chunks)
                new_records.append(InjectionRecord(
                    message_index=record.message_index,
                    chunks=matched_chunks,
                    combined_text=combined,
                    ghost_text=ghost,
                    break_indices=self._breaks_at_or_before(record.message_index, current_breaks),
                ))
                pointer += match_count
            elif match_count > 0:
                # Partial match: prefix is broken at this position
                # No ghost_text - it wouldn't preserve the prefix anyway
                kept_chunks = regular_chunks[pointer:pointer + match_count]
                combined = "".join(c.content for c in kept_chunks)
                new_records.append(InjectionRecord(
                    message_index=record.message_index,
                    chunks=kept_chunks,
                    combined_text=combined,
                    ghost_text="",  # No ghost - prefix already broken
                    break_indices=self._breaks_at_or_before(record.message_index, current_breaks),
                ))
                pointer += match_count
                prefix_broken = True
            else:
                # No match at all: first chunk already changed
                prefix_broken = True
        
        # Collect all remaining regular chunks + always_last chunks for the last tool message
        remaining_regular = regular_chunks[pointer:]
        
        if remaining_regular or always_last_chunks:
            # Check if there's already a record at last_tool_msg_idx
            # (from a fully-matched record during the loop above)
            existing_at_last = None
            for i, r in enumerate(new_records):
                if r.message_index == last_tool_msg_idx:
                    existing_at_last = i
                    break
            
            if existing_at_last is not None:
                # Merge into the existing record: keep its matched regular chunks,
                # append remaining regular + always_last chunks
                existing = new_records[existing_at_last]
                merged_chunks = list(existing.chunks) + list(remaining_regular) + list(always_last_chunks)
                merged_combined = existing.combined_text + "".join(c.content for c in remaining_regular) + always_last_text
                new_records[existing_at_last] = InjectionRecord(
                    message_index=last_tool_msg_idx,
                    chunks=merged_chunks,
                    combined_text=merged_combined,
                    ghost_text="",  # Real always_last text at end, not ghost
                    break_indices=self._breaks_at_or_before(last_tool_msg_idx, current_breaks),
                )
            else:
                remaining_chunks = list(remaining_regular) + list(always_last_chunks)
                combined = "".join(c.content for c in remaining_regular) + always_last_text
                new_records.append(InjectionRecord(
                    message_index=last_tool_msg_idx,
                    chunks=remaining_chunks,
                    combined_text=combined,
                    ghost_text="",  # Real always_last text, not ghost
                    break_indices=self._breaks_at_or_before(last_tool_msg_idx, current_breaks),
                ))
        
        return new_records

    def _breaks_at_or_before(self, msg_index: int, *break_sets) -> frozenset:
        """Structural-break positions at or before msg_index (drops/omissions/summaries).

        These positions change the serialized bytes at or before msg_index, so an
        injection record created when the set was smaller is no longer at the same
        byte prefix.
        """
        breaks = set()
        for s in break_sets:
            if s:
                breaks |= {b for b in s if b <= msg_index}
        return frozenset(breaks)

    def get_placements(self, current_chunks: List[ContextChunk], 
                       last_tool_msg_idx: Optional[int],
                       seconds_since_last_response: Optional[float] = None,
                       dropped_indices: Optional[set] = None,
                       omission_indices: Optional[set] = None,
                       summary_indices: Optional[set] = None,
                       valid_tool_indices: Optional[set] = None) -> List[Tuple[int, str, bool]]:
        """Convenience method: get (message_index, text, is_reminders_only) tuples for transformer placement.
        
        Args:
            current_chunks: Current context chunks sorted by stability
            last_tool_msg_idx: Index of last tool result message, or None
            seconds_since_last_response: Time since last API response, or None
            dropped_indices: Positions with a DropTransformer this request.
            omission_indices: AI positions rewritten by a
                ToolCallOmissionTransformer this request.
            summary_indices: Positions replaced by a SummaryTransformer this
                request.
            valid_tool_indices: Positions that are still live (non-dropped)
                ToolMessages this request, or None if unknown.
            
        Returns:
            List of (message_index, text_to_inject, is_reminders_only) tuples.
            is_reminders_only is True when all content in the placement comes from
            always_last chunks (e.g., system prompt reminders), meaning the text
            is redundant with the system prompt.
        """
        records = self.prepare_injections(
            current_chunks, last_tool_msg_idx, seconds_since_last_response,
            dropped_indices=dropped_indices,
            omission_indices=omission_indices,
            summary_indices=summary_indices,
            valid_tool_indices=valid_tool_indices,
        )
        always_last_ids = {c.section_id for c in current_chunks if c.always_last}
        results = []
        for r in records:
            is_reminders_only = (
                all(c.section_id in always_last_ids for c in r.chunks) and 
                bool(r.chunks or r.ghost_text)
            )
            results.append((r.message_index, r.combined_text + r.ghost_text, is_reminders_only))
        return results
