#!/usr/bin/env python3
"""StateService - Read-only view into AgentState with state update builders.

This service provides read-only access to the current state and methods that
return state update dicts. All state modifications go through LangGraph's
reducer flow via return values.
"""

import os
import time
import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple, Any
from pathlib import Path
from copy import deepcopy

from fluximp_agent.tools.common import FileOperation, create_aggregate_diff


def normalize_file_path(file_path: str, filesystem_service=None) -> str:
    """Normalize a file path for consistent use as a dict key in file_view_windows.
    
    Handles: relative vs absolute paths, case differences on case-insensitive
    filesystems, and path separator differences. Falls back to simple
    normalization (forward slashes + case normalization) if filesystem_service
    is not provided.
    
    Args:
        file_path: The file path string to normalize
        filesystem_service: Optional FileSystemService instance for path resolution
        
    Returns:
        Normalized file path string using forward slashes, relative to
        the working directory where possible.
    """
    if filesystem_service is None:
        # Simple fallback normalization without filesystem_service
        # Convert backslashes to forward slashes, collapse multiple slashes
        normalized = file_path.replace(chr(92), '/').replace('//', '/')
        if os.path.normcase('A') == 'a':
            normalized = normalized.lower()
        return normalized
    
    try:
        resolved, error = filesystem_service.validate_path(file_path, access_type='read')
        if error:
            # Try write access as fallback (e.g., for files that don't exist yet)
            resolved, error = filesystem_service.validate_path(file_path, access_type='write')
        if error:
            return file_path  # Can't normalize, return original
        # Use relative path from working directory with forward slashes
        try:
            relative = resolved.relative_to(filesystem_service.working_dir)
            normalized = relative.as_posix()
        except ValueError:
            # Can't make relative (shouldn't normally happen after validate_path)
            normalized = resolved.as_posix()
        # Normalize case on case-insensitive filesystems
        # os.path.normcase('A') == 'a' on Windows and macOS (case-insensitive)
        # os.path.normcase('A') == 'A' on Linux (case-sensitive)
        if os.path.normcase('A') == 'a':
            normalized = normalized.lower()
        return normalized
    except Exception:
        return file_path  # Fall back to original on any error


@dataclass
class FileViewRequest:
    """Request to add a file view window."""
    file_path: str
    offset: int  # 1-indexed start line
    limit: Optional[int]  # number of lines to include, None means no limit (to end of file)


@dataclass
class FileViewWindow:
    """A window into a file with line range information."""
    start_line: int  # 1-indexed
    end_line: Optional[int]  # 1-indexed, inclusive, None means to end of file
    edit_timestamp: Optional[float] = None
    add_timestamp: Optional[float] = None
    deferred_add_timestamp: Optional[float] = None  # Held back when ranges unchanged, applied on content change or eviction


@dataclass
class FileViewEntry:
    """Entry for a file in the file_view_windows dict, keyed by file path."""
    id: str
    windows: List[FileViewWindow] = field(default_factory=list)
    last_content_hash: Optional[str] = None
    last_content_changed_timestamp: Optional[float] = None


@dataclass
class SearchListViewEntry:
    """A stored search or file listing query."""
    id: str
    query_type: str  # "ripgrep_search" or "find_files"
    params: dict = field(default_factory=dict)
    timestamp: Optional[float] = None
    last_content_hash: Optional[str] = None
    last_content_changed_timestamp: Optional[float] = None


class StateService:
    """Read-only view into AgentState with state update builder methods.
    
    This class holds a reference to the current state for reading only.
    All state modifications must be done via the build_* methods which
    return state update dicts for LangGraph's reducer flow.
    """
    #MAX_FILE_VIEW_BYTES = 360000
    MAX_FILE_VIEW_BYTES = 320000
    #MAX_FILE_VIEW_BYTES = 280000
    #MAX_FILE_VIEW_BYTES = 240000
    #MAX_FILE_VIEW_BYTES = 200000
    #MAX_FILE_VIEW_BYTES = 163840  # 160KB limit for total view content (file views + search/list views)
    #MAX_FILE_VIEW_BYTES = 120000
    # Too small both GLM5.1 and Kimi K2.7 Code failed to start editing poc/sdl_vulkan_cuda_lvgl to continue
    # phase 3 of a plan to refactor from C-sytle to C++20 with RAII.
    #MAX_FILE_VIEW_BYTES = 163840/2
    
    def __init__(self, state: Optional[Dict[str, Any]] = None, filesystem_service=None):
        """Initialize the state service with a read-only view into state.
 
        Args:
            state: The AgentState dict to read from. If None, uses empty state.
            filesystem_service: Optional FileSystemService for path normalization.
        """
        self._state = state if state is not None else {}
        self._filesystem_service = filesystem_service
   
    # ========================================================================
    # Read-only accessors (for convenience)
    # ========================================================================
    
    @property
    def file_operations(self) -> List[FileOperation]:
        """Get the file operations list."""
        return self._state.get("file_operations", [])
    
    @property
    def operation_position(self) -> int:
        """Get the current operation position."""
        return self._state.get("operation_position", -1)
    
    @property
    def last_scrutinized_position(self) -> int:
        """Get the last scrutinized position."""
        return self._state.get("last_scrutinized_position", -1)
    
    @property
    def file_view_windows(self) -> Dict[str, Any]:
        """Get the file view windows dict (file_path -> FileViewEntry)."""
        return self._state.get("file_view_windows", {})
    
    @property
    def search_list_view(self) -> Dict[str, Any]:
        """Get the search/list view dict."""
        return self._state.get("search_list_view", {})
    
    @property
    def file_views_added_this_turn(self) -> List[str]:
        """Get the list of normalized file paths that had add_file_view called this turn."""
        return self._state.get("file_views_added_this_turn", [])
    
    @property
    def deleted_file_paths(self) -> frozenset:
        """Get the set of file paths deleted during this session."""
        return self._state.get("deleted_file_paths", frozenset())
    
    def normalize_path(self, file_path: str) -> str:
        """Normalize a file path for consistent comparison and keying.

        Delegates to normalize_file_path using this service's
        filesystem_service, so all path comparison sites share one
        canonicalization entry point.
        """
        return normalize_file_path(file_path, self._filesystem_service)

    def update_state_reference(self, state: Dict[str, Any]) -> None:
        """Update the internal state reference (called before each tool execution).
        
        This should be called before every tool invocation to ensure the service
        has the current state.
        
        Args:
            state: The current AgentState
        """
        print("DEBUG update_state_reference file_view_windows:", state.get('file_view_windows', {}))
        self._state = state
    
    # ========================================================================
    # State update builders (return state update dicts for LangGraph)
    # ========================================================================
    
    def build_new_file_operation(self, op: FileOperation) -> Dict[str, Any]:
        """Build state update for a new file operation.
        
        Args:
            op: The FileOperation to add
            
        Returns:
            State update dict with file_operations, file_operations_this_turn, and operation_position
        """
        return self.build_new_file_operations([op])
    
    def build_new_file_operations(self, ops: List[FileOperation]) -> Dict[str, Any]:
        """Build state update for multiple file operations that should be atomic.
        
        Used when a single tool invocation modifies multiple files (e.g., cross-file
        move_lines). All operations share the same operation position so they are
        undone and redone together as a unit.
        
        Args:
            ops: List of FileOperation instances to add atomically
            
        Returns:
            State update dict with file_operations, file_operations_this_turn, and operation_position
        """
        # Truncate redo history if we are adding forward ops while not at end of stack
        file_ops = list(self.file_operations)
        current_pos = self.operation_position
        
        if current_pos < len(file_ops) - 1:
            # Truncate redo history
            file_ops = file_ops[:current_pos + 1]
        
        # Add all operations
        for op in ops:
            file_ops.append(op)
        new_position = len(file_ops) - 1
        
        return {
            "file_operations": file_ops,
            "file_operations_this_turn": list(ops),
            "operation_position": new_position,
        }
    
    def build_undo_update(self) -> Dict[str, Any]:
        """Build state update for undo operation.
        
        Returns:
            State update dict with new operation_position, or empty if at start
        """
        current_pos = self.operation_position
        if current_pos <= -1:
            # Already at start, can't undo
            return {}
        
        new_position = current_pos - 1
        
        # Handle undo past scrutiny point
        last_scrutinized = min(self.last_scrutinized_position, new_position)
        
        return {
            "operation_position": new_position,
            "last_scrutinized_position": last_scrutinized,
        }
    
    def build_redo_update(self) -> Dict[str, Any]:
        """Build state update for redo operation.
        
        Returns:
            State update dict with new operation_position, or empty if at end
        """
        current_pos = self.operation_position
        file_ops_len = len(self.file_operations)
        
        if current_pos >= file_ops_len - 1:
            # Already at end, can't redo
            return {}
        
        new_position = current_pos + 1
        
        return {
            "operation_position": new_position,
        }
    
    def build_deleted_file_paths_update(self, file_path: str) -> Dict[str, Any]:
        """Build state update to add a file path to deleted_file_paths.
        
        Args:
            file_path: The path of the deleted file
            
        Returns:
            State update dict with deleted_file_paths
        """
        return {
            "deleted_file_paths": frozenset({file_path})
        }

    def build_scrutiny_complete(self) -> Dict[str, Any]:
        """Build state update to mark scrutiny as complete.
 
        Returns:
            State update dict with last_scrutinized_position set to current position
        """
        return {
            "last_scrutinized_position": self.operation_position,
        }

    def _generate_view_id(self, prefix: str = "fv") -> str:
        """Generate a unique view ID with the given prefix."""
        return f"{prefix}_{uuid.uuid4().hex[:6]}"
    
    def build_file_view_windows_update(
        self, 
        requests: List[FileViewRequest], 
        source: str = "add"
    ) -> Dict[str, Any]:
        """Build state update for adding file view windows.
        
        Calculates the new file_view_windows dict from current state + requests.
        
        Args:
            requests: List of FileViewRequest to add
            source: Either "add" (from add_file_view tool) or "edit" (from edit_file_lines)
            
        Returns:
            State update dict with complete new file_view_windows
        """
        if not requests:
            print("DEBUG build_file_view_windows_update empty requests")
            return {}
        
        # Start with deep copy of current windows
        current = self.file_view_windows
        if current:
            new_windows = deepcopy(current)
        else:
            print("DEBUG build_file_view_windows_update empty self.file_view_windows")
            new_windows = {}

        now = time.time()
        
        for request in requests:
            file_path = normalize_file_path(request.file_path, self._filesystem_service)
            start_line = max(request.offset, 1)
            end_line = None if request.limit is None else start_line + max(request.limit, 1) - 1
            
            # Save original window ranges for comparison (before modification)
            # Used to detect when add_file_view didn't actually change visible ranges
            original_ranges = None
            orig_window_by_range = None
            if file_path in new_windows:
                orig_entry = new_windows[file_path]
                orig_windows = self._get_entry_windows(orig_entry)
                original_ranges = sorted([(w.start_line, w.end_line) for w in orig_windows])
                orig_window_by_range = {(w.start_line, w.end_line): w for w in orig_windows}
            
            # Create new window
            new_window = FileViewWindow(
                start_line=start_line,
                end_line=end_line
            )
            
            # Set appropriate timestamp
            if source == "edit":
                new_window.edit_timestamp = now
            else:
                new_window.add_timestamp = now
            
            # Add to existing entry for this file, or create new entry
            if file_path not in new_windows:
                print(f"DEBUG build_file_view_windows_update new file: {file_path}")
                entry_id = self._generate_view_id("fv")
                new_windows[file_path] = FileViewEntry(id=entry_id, windows=[new_window])
            else:
                entry = new_windows[file_path]
                if isinstance(entry, dict):
                    # Migrate from old dict format to FileViewEntry
                    entry = FileViewEntry(
                        id=entry.get("id", self._generate_view_id("fv")),
                        windows=entry.get("windows", [])
                    )
                    new_windows[file_path] = entry
                elif isinstance(entry, list):
                    # Migrate from old list-of-tuples format to FileViewEntry
                    entry_id = self._generate_view_id("fv")
                    migrated_windows = []
                    for w in entry:
                        if isinstance(w, FileViewWindow):
                            migrated_windows.append(w)
                        elif isinstance(w, (list, tuple)):
                            migrated_windows.append(FileViewWindow(
                                start_line=w[0],
                                end_line=w[1],
                                edit_timestamp=self._convert_timestamp(w[2]) if len(w) > 2 else None,
                                add_timestamp=self._convert_timestamp(w[3]) if len(w) > 3 else None,
                            ))
                    entry = FileViewEntry(id=entry_id, windows=migrated_windows)
                    new_windows[file_path] = entry
                entry.windows.append(new_window)
            
            # Merge overlapping windows
            entry = new_windows[file_path]
            if isinstance(entry, FileViewEntry):
                entry.windows = self._merge_overlapping_windows(entry.windows)
                
                # After merge, check if ranges are identical to before
                # If so, defer the add_timestamp update (only for "add" source, not "edit")
                if original_ranges is not None and source == "add" and orig_window_by_range is not None:
                    merged_ranges = sorted([(w.start_line, w.end_line) for w in entry.windows])
                    if merged_ranges == original_ranges:
                        # Ranges unchanged â defer add_timestamp updates and carry forward deferred values
                        for w in entry.windows:
                            ow = orig_window_by_range.get((w.start_line, w.end_line))
                            if ow is not None:
                                # Defer add_timestamp if it's newer than the original
                                if w.add_timestamp is not None and (ow.add_timestamp is None or w.add_timestamp > ow.add_timestamp):
                                    w.deferred_add_timestamp = w.add_timestamp
                                    w.add_timestamp = ow.add_timestamp
                                # Carry forward any existing deferred_add_timestamp (keep the largest)
                                if ow.deferred_add_timestamp is not None:
                                    if w.deferred_add_timestamp is None or ow.deferred_add_timestamp > w.deferred_add_timestamp:
                                        w.deferred_add_timestamp = ow.deferred_add_timestamp

        return {"file_view_windows": new_windows}
    

    def build_file_view_windows_edit_adjustment(
        self,
        file_path: str,
        start_line: int,
        old_line_count: int,
        new_line_count: int,
        source: str = "edit",
        current_windows: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Build state update adjusting existing file view windows for an edit.
        
        Instead of always adding a new window, this method intelligently adjusts
        existing windows to account for line insertions/deletions, ensuring that
        content previously visible at the bottom of a window stays visible.
        
        Args:
            file_path: Path of the edited file
            start_line: 1-indexed first line affected in the original file
            old_line_count: Number of lines being replaced/removed (0 for pure insert)
            new_line_count: Number of lines being added (0 for pure deletion)
            source: Timestamp source ("edit" or "add")
            
        Returns:
            State update dict with adjusted file_view_windows
        """
        delta = new_line_count - old_line_count
        
        # Get current windows for this file (use provided override or state)
        current = current_windows if current_windows is not None else self.file_view_windows
        entry = current.get(file_path)
        
        # If no windows exist for this file, fall back to adding a +/-5 window
        if entry is None:
            view_start = max(1, start_line - 5)
            return self.build_file_view_windows_update(
                [FileViewRequest(file_path=file_path, offset=view_start, limit=max(old_line_count, new_line_count, 1) + 10)],
                source=source
            )
        
        # Extract windows
        windows = self._get_entry_windows(entry)
        
        # If any window has end_line=None, the whole file is already visible; skip update
        for w in windows:
            if w.end_line is None:
                return {}
        
        # If delta is 0 and the edit region is already covered by a window, no change needed
        if delta == 0:
            for w in windows:
                # Check if edit is fully within this window
                if w.start_line <= start_line and w.end_line >= start_line + max(old_line_count, 1) - 1:
                    return {}
                # For pure replacement with same line count (old_line_count > 0)
                if old_line_count > 0 and w.start_line <= start_line and w.end_line >= start_line + old_line_count - 1:
                    return {}
        
        # Deep copy current state
        new_windows = deepcopy(current) if current else {}
        entry = new_windows[file_path]
        
        # Migrate entry if needed
        if isinstance(entry, dict):
            entry = FileViewEntry(
                id=entry.get("id", self._generate_view_id("fv")),
                windows=entry.get("windows", [])
            )
            new_windows[file_path] = entry
        elif isinstance(entry, list):
            entry_id = self._generate_view_id("fv")
            migrated_windows = []
            for w in entry:
                if isinstance(w, FileViewWindow):
                    migrated_windows.append(w)
                elif isinstance(w, (list, tuple)):
                    migrated_windows.append(FileViewWindow(
                        start_line=w[0],
                        end_line=w[1],
                        edit_timestamp=self._convert_timestamp(w[2]) if len(w) > 2 else None,
                        add_timestamp=self._convert_timestamp(w[3]) if len(w) > 3 else None,
                    ))
            entry = FileViewEntry(id=entry_id, windows=migrated_windows)
            new_windows[file_path] = entry
        
        now = time.time()
        edit_covered = False
        windows_to_remove = []
        
        for i, w in enumerate(entry.windows):
            # Determine relationship between window and edit
            
            # The edit affects lines [start_line, start_line + old_line_count - 1]
            # For inserts (old_line_count == 0), the edit is at start_line (no lines removed)
            edit_end = start_line + max(old_line_count, 1) - 1 if old_line_count > 0 else start_line
            
            if old_line_count == 0:
                # Pure insertion: no lines removed, new lines inserted at start_line
                # Window contains the insertion point
                if w.start_line <= start_line <= w.end_line + 1:
                    # Insertion is within or adjacent to window
                    w.end_line += delta
                    w.edit_timestamp = now
                    edit_covered = True
                elif w.start_line > start_line:
                    # Window is entirely after the insertion point: shift it
                    w.start_line += delta
                    w.end_line += delta
                # else: window is entirely before insertion point, no change
            else:
                # Replacement or deletion: lines [start_line, edit_end] are removed
                if w.end_line < start_line:
                    # Window is entirely before the edit: no change
                    pass
                elif w.start_line > edit_end:
                    # Window is entirely after the edit: shift it
                    w.start_line += delta
                    w.end_line += delta
                else:
                    # Window overlaps with the edit region
                    edit_covered = True
                    w.end_line += delta
                    w.edit_timestamp = now
            
            # Check if window has collapsed
            if w.end_line is not None and w.end_line < w.start_line:
                windows_to_remove.append(i)
        
        # Remove collapsed windows (reverse order to preserve indices)
        for i in reversed(windows_to_remove):
            entry.windows.pop(i)
        
        # If edit was not covered by any window, add a new window for the edit region
        if not edit_covered:
            view_start = max(1, start_line - 5)
            view_limit = max(new_line_count, 1) + 10
            new_window = FileViewWindow(
                start_line=view_start,
                end_line=view_start + view_limit - 1
            )
            new_window.edit_timestamp = now
            entry.windows.append(new_window)
        
        # If all windows were removed, remove the file entry
        if not entry.windows:
            del new_windows[file_path]
            return {"file_view_windows": new_windows}
        
        # Merge overlapping windows (in case the new/adjusted windows overlap)
        entry.windows = self._merge_overlapping_windows(entry.windows)
        
        return {"file_view_windows": new_windows}

    def build_file_view_windows_removal(self, file_paths: List[str]) -> Dict[str, Any]:
        """Build state update for removing file view windows.
        
        Args:
            file_paths: List of file paths to remove view windows for
            
        Returns:
            State update dict with complete new file_view_windows
        """
        if not file_paths:
            return {}
        
        # Start with deep copy of current windows
        current = self.file_view_windows
        new_windows = deepcopy(current) if current else {}
        
        for file_path in file_paths:
            new_windows.pop(file_path, None)
        
        return {"file_view_windows": new_windows}
    
    def build_search_list_view_add(self, query_type: str, params: dict) -> Dict[str, Any]:
        """Build state update to add a search/list view query.
        
        Args:
            query_type: "ripgrep_search" or "find_files"
            params: Query parameters dict
            
        Returns:
            State update dict with complete new search_list_view
        """
        current = self.search_list_view
        new_queries = dict(current) if current else {}
        
        entry_id = self._generate_view_id("sq")
        now = time.time()
        new_queries[entry_id] = SearchListViewEntry(
            id=entry_id,
            query_type=query_type,
            params=params,
            timestamp=now
        )
        
        return {"search_list_view": new_queries}
    
    def build_search_list_view_remove(self, query_ids: List[str]) -> Dict[str, Any]:
        """Build state update to remove specific search/list view queries.
        
        Args:
            query_ids: List of query IDs to remove
            
        Returns:
            State update dict with complete new search_list_view
        """
        if not query_ids:
            return {}
        
        current = self.search_list_view
        new_queries = dict(current) if current else {}
        
        for qid in query_ids:
            new_queries.pop(qid, None)
        
        return {"search_list_view": new_queries}
    
    def build_view_removal(self, view_ids: List[str]) -> Dict[str, Any]:
        """Build state update to remove views by their IDs (across both file views and search/list views).
        
        Args:
            view_ids: List of view IDs to close
            
        Returns:
            State update dict with updated file_view_windows and search_list_view
        """
        if not view_ids:
            return {}
        
        view_id_set = set(view_ids)
        
        # Remove from file_view_windows (by matching FileViewEntry.id)
        current_fv = self.file_view_windows
        new_fv = deepcopy(current_fv) if current_fv else {}
        files_to_remove = []
        for file_path, entry in list(new_fv.items()):
            if isinstance(entry, dict):
                if entry.get("id") in view_id_set:
                    files_to_remove.append(file_path)
            elif isinstance(entry, FileViewEntry):
                if entry.id in view_id_set:
                    files_to_remove.append(file_path)
            elif isinstance(entry, list):
                # Old format, no ID to match
                pass
        for fp in files_to_remove:
            del new_fv[fp]
        
        # Remove from search_list_view (by key)
        current_sl = self.search_list_view
        new_sl = dict(current_sl) if current_sl else {}
        for qid in view_id_set:
            new_sl.pop(qid, None)
        
        result = {}
        if files_to_remove or new_fv != current_fv:
            result["file_view_windows"] = new_fv
        if new_sl != current_sl:
            result["search_list_view"] = new_sl
        
        return result

    def build_clear_file_view_windows(self) -> Dict[str, Any]:
        """Build state update to clear all file view windows.
        
        Returns:
            State update dict with empty file_view_windows
        """
        return {"file_view_windows": {}}
 
    # ========================================================================
    # Read-only helper methods (for getting information from state)
    # ========================================================================
    
    def create_aggregate_diff(self, context_lines: int = 3) -> str:
        """Create a unified diff aggregating all file changes up to current position.
        
        Returns:
            Unified diff string showing net changes, or empty string if no changes
        """
        return create_aggregate_diff(
            self.file_operations,
            self.operation_position,
            None,  # filesystem_service is not actually used
            context_lines=context_lines,
        )
    
    def create_scrutiny_diff(self) -> str:
        """Create a unified diff of changes since last scrutiny.
        
        Returns:
            Unified diff string showing changes since last scrutiny,
            or empty string if no new changes
        """
        last_scrutinized = self.last_scrutinized_position
        current_pos = self.operation_position
        file_ops = self.file_operations
        
        if last_scrutinized == -1:
            # Never scrutinized - use full cumulative diff
            return self.create_aggregate_diff(context_lines=20)
        
        if last_scrutinized >= current_pos:
            # Already scrutinized up to current position
            return ""
        
        # Slice operations from after last scrutiny to current position
        start_idx = last_scrutinized + 1
        recent_ops = file_ops[start_idx:current_pos + 1]
        
        if not recent_ops:
            return ""
        
        return create_aggregate_diff(recent_ops, len(recent_ops) - 1, None, context_lines=20)
    
    def get_modified_files(self) -> List[str]:
        """Get list of file paths modified since last scrutiny.
        
        Returns:
            List of unique file paths that need to be scrutinized
        """
        last_scrutinized = self.last_scrutinized_position
        current_pos = self.operation_position
        file_ops = self.file_operations
        
        if last_scrutinized == -1:
            # Never scrutinized - use all operations up to current position
            relevant_ops = file_ops[:current_pos + 1]
        elif last_scrutinized >= current_pos:
            # Already scrutinized up to current position
            return []
        else:
            # Slice operations from after last scrutiny to current position
            start_idx = last_scrutinized + 1
            relevant_ops = file_ops[start_idx:current_pos + 1]
        
        # Extract unique file paths (excluding deletes and None paths).
        # Normalize for dedupe/exclusion, but return the first raw path form
        # so callers that open(file_path) see the original string.
        by_key = {}
        for op in relevant_ops:
            if op.file_path is not None and op.operation_type != "delete_file":
                key = self.normalize_path(op.file_path)
                by_key.setdefault(key, op.file_path)
        
        # Exclude files whose last operation was a delete
        # (a file may have been modified then deleted - we shouldn't return it)
        seen = set()
        for op in reversed(relevant_ops):
            if op.file_path is not None:
                key = self.normalize_path(op.file_path)
                if key not in seen:
                    seen.add(key)
                    if op.operation_type == "delete_file":
                        by_key.pop(key, None)
        
        return list(by_key.values())
    
    def is_newly_created_file(self, file_path: str) -> bool:
        """Check if a file was newly created (not modified) since last scrutiny.
        
        Args:
            file_path: Path to check
            
        Returns:
            True if file was created during modified operations, False if it was modified
        """
        last_scrutinized = self.last_scrutinized_position
        current_pos = self.operation_position
        file_ops = self.file_operations
        
        if last_scrutinized == -1:
            relevant_ops = file_ops[:current_pos + 1]
        elif last_scrutinized >= current_pos:
            return False
        else:
            start_idx = last_scrutinized + 1
            relevant_ops = file_ops[start_idx:current_pos + 1]
        
        target = self.normalize_path(file_path)
        for op in relevant_ops:
            if op.file_path is not None and op.operation_type != "delete_file":
                if self.normalize_path(op.file_path) == target:
                    return op.compressed_before is None
        
        return False
    
    def _get_entry_windows(self, entry) -> List[FileViewWindow]:
        """Extract windows list from a file_view_windows entry, handling migration.
        
        Args:
            entry: Could be FileViewEntry, dict, or list (old format)
            
        Returns:
            List of FileViewWindow
        """
        if isinstance(entry, FileViewEntry):
            return entry.windows
        elif isinstance(entry, dict):
            return entry.get("windows", [])
        elif isinstance(entry, list):
            # Old format: list of tuples or FileViewWindow
            result = []
            for w in entry:
                if isinstance(w, FileViewWindow):
                    result.append(w)
                elif isinstance(w, (list, tuple)):
                    result.append(FileViewWindow(
                        start_line=w[0],
                        end_line=w[1],
                        edit_timestamp=self._convert_timestamp(w[2]) if len(w) > 2 else None,
                        add_timestamp=self._convert_timestamp(w[3]) if len(w) > 3 else None,
                    ))
            return result
        return []
    
    def _get_entry_id(self, entry) -> str:
        """Extract ID from a file_view_windows entry, handling migration.
        
        Args:
            entry: Could be FileViewEntry, dict, or list (old format)
            
        Returns:
            View ID string, or empty string if not available
        """
        if isinstance(entry, FileViewEntry):
            return entry.id
        elif isinstance(entry, dict):
            return entry.get("id", "")
        return ""
    
    @staticmethod
    def _convert_timestamp(ts):
        """Convert a timestamp value to float, requiring UTC values.

        Args:
            ts: Timestamp value (float, int, ISO string, timezone-aware UTC
                datetime, or None)

        Returns:
            Float timestamp, or None if input is None

        Raises:
            ValueError: if a string or datetime is not a timezone-aware UTC
                value, or if the type is unsupported
        """
        if ts is None:
            return None
        if isinstance(ts, (int, float)):
            return float(ts)
        if isinstance(ts, str):
            dt = datetime.fromisoformat(ts)
            if dt.tzinfo is None or dt.utcoffset() != timedelta(0):
                raise ValueError(
                    f"{ts!r} is not a UTC ISO-8601 timestamp "
                    "(expected a timezone-aware value with a +00:00 offset)"
                )
            return dt.timestamp()
        if isinstance(ts, datetime):
            if ts.tzinfo is None or ts.utcoffset() != timedelta(0):
                raise ValueError(
                    f"{ts!r} is not a UTC datetime "
                    "(expected a timezone-aware value with a +00:00 offset)"
                )
            return ts.timestamp()
        raise ValueError(
            f"unsupported timestamp type {type(ts).__name__!r} "
            "(expected float, int, ISO string, or a timezone-aware UTC datetime)"
        )

    def get_all_views(self, filesystem_service, experiments_service=None, include_stability_keys=False) -> Tuple[float, List[str], List[Tuple[str, Any]], Optional[Dict[str, Any]]]:
        """Get all views (file views + search/list views) with current content.

        Inclusion is prioritized by most recent agent interaction (descending original_ts),
        with active edits as tiebreaker (descending edit_ts). Output is then reordered
        by content stability (ascending last_content_changed_ts) for prefix caching.

        Args:
            filesystem_service: FileSystemService for reading file content
            experiments_service: Optional ExperimentsService for checking search_list_view
            include_stability_keys: If True, include stability_key tuple as last element
                in each view tuple for context chunk creation.

        Returns:
            Tuple of (fraction_used, files_removed, interleaved_views, state_update)
            where interleaved_views is a list of tuples ordered for prefix caching:
              ("file", file_path, start_line, end_line, content)
              ("search", header, content)
            and when include_stability_keys=True:
              ("file", file_path, start_line, end_line, content, total_lines, section_id, stability_key)
              ("search", header, content, section_id, stability_key)
            where section_id is "view:{file_path}" or "search:{query_id}"
            and stability_key is (last_content_changed_ts, original_ts, section_id)
            state_update is an optional state update dict for evicted entries
            and content-hash metadata updates.
        """
        now = time.time()
        
        # Create a working copy of file_view_windows that we can modify
        # (for applying deferred timestamps during eviction checks)
        working_fv = deepcopy(self.file_view_windows) if self.file_view_windows else {}
        any_deferred_applied = False

        while True:  # Restart loop for deferred timestamp application during eviction
            needs_restart = False
            result_views = []
            total_bytes = 0
            included_file_paths = set()
            included_query_ids = set()
            files_to_remove = set()
            queries_to_remove = set()
            file_metadata_changed = False
            search_metadata_changed = False
            deferred_applied = False

            # Phase 1: Render all content and compute content hashes
            # Each rendered entry: (last_content_changed_ts, original_ts, entry_type, key, rendered_data, byte_count, content_hash)
            rendered_entries = []

            # File view entries
            for file_path, entry in working_fv.items():
                windows = self._get_entry_windows(entry)

                # Read the file
                path, error = filesystem_service.validate_path(file_path)
                assert not error, f"Path validation error for {file_path}: {error}"

                if not path.exists():
                    print(f"DEBUG removing missing file from file view windows: {file_path}")
                    files_to_remove.add(file_path)
                    continue
                else:
                    assert path.is_file(), f"Path is not a file: {file_path}"

                # Handle non-UTF-8 / binary content defensively so a bad file
                # never crashes the whole agent run (mirrors filesystem_read.py
                # and state_service._get_search_results handling).
                try:
                    content = path.read_text(encoding='utf-8')
                except UnicodeDecodeError:
                    content = path.read_bytes().decode('utf-8', errors='replace')
                lines = content.splitlines()
                file_bytes = 0
                file_windows_content = []

                for window in windows:
                    # Adjust window to actual file bounds
                    start = max(1, window.start_line)
                    end = len(lines) if window.end_line is None else min(len(lines), window.end_line)

                    if start > len(lines):
                        print(f"DEBUG get_all_views start past end {start} > {len(lines)} for {path}")
                        continue

                    # Extract lines with line numbers
                    selected_lines = lines[start - 1:end]
                    numbered_content = "\n".join(
                        f"{i:4d} | {line}"
                        for i, line in enumerate(selected_lines, start=start)
                    )

                    window_bytes = len(numbered_content.encode('utf-8'))
                    file_windows_content.append((start, end, numbered_content))
                    file_bytes += window_bytes

                if file_windows_content:
                    # Compute hash of all rendered content for this file
                    all_content = "\n".join(c for _, _, c in file_windows_content)
                    content_hash = hashlib.md5(all_content.encode('utf-8')).hexdigest()

                    # Get stored hash from entry
                    if isinstance(entry, FileViewEntry):
                        stored_hash = entry.last_content_hash
                        stored_changed_ts = entry.last_content_changed_timestamp
                    elif isinstance(entry, dict):
                        stored_hash = entry.get('last_content_hash')
                        stored_changed_ts = entry.get('last_content_changed_timestamp')
                    else:
                        stored_hash = None
                        stored_changed_ts = None

                    # Compare and determine last_content_changed_timestamp
                    if content_hash != stored_hash:
                        last_content_changed_ts = now
                        file_metadata_changed = True
                        # Content changed â apply any deferred add_timestamps that are greater than current
                        for w in windows:
                            if w.deferred_add_timestamp is not None and w.deferred_add_timestamp > (w.add_timestamp or 0):
                                w.add_timestamp = w.deferred_add_timestamp
                                w.deferred_add_timestamp = None
                                deferred_applied = True
                    else:
                        last_content_changed_ts = stored_changed_ts if stored_changed_ts is not None else now

                    # Compute timestamps AFTER potentially applying deferred timestamps
                    edit_ts = max((w.edit_timestamp for w in windows if w.edit_timestamp is not None), default=0.0)
                    add_ts = max((w.add_timestamp for w in windows if w.add_timestamp is not None), default=0.0)
                    original_ts = max(edit_ts, add_ts)

                    rendered_entries.append((last_content_changed_ts, original_ts, edit_ts, "file", file_path, file_windows_content, file_bytes, content_hash, len(lines)))

            # Search/list view entries
            search_list_enabled = experiments_service and experiments_service.is_enabled("search_list_view")
            if search_list_enabled:
                for query_id, entry in self.search_list_view.items():
                    if isinstance(entry, SearchListViewEntry):
                        query_type = entry.query_type
                        params = entry.params
                        original_ts = entry.timestamp if entry.timestamp is not None else 0.0
                        stored_hash = entry.last_content_hash
                        stored_changed_ts = entry.last_content_changed_timestamp
                    elif isinstance(entry, dict):
                        query_type = entry.get("query_type", "")
                        params = entry.get("params", {})
                        original_ts = self._convert_timestamp(entry.get("timestamp"))
                        stored_hash = entry.get('last_content_hash')
                        stored_changed_ts = entry.get('last_content_changed_timestamp')
                    else:
                        continue

                    # Re-execute the query fresh
                    search_content, search_bytes = self._execute_search_query(
                        query_type, params, filesystem_service
                    )

                    if search_content is not None:
                        # Compute hash
                        content_hash = hashlib.md5(search_content.encode('utf-8')).hexdigest()

                        if content_hash != stored_hash:
                            last_content_changed_ts = now
                            search_metadata_changed = True
                        else:
                            last_content_changed_ts = stored_changed_ts if stored_changed_ts is not None else now

                        # Build header
                        if query_type == "ripgrep_search":
                            patterns = params.get("patterns", [])
                            paths = params.get("paths", [])
                            header = f"ripgrep_search: patterns={patterns} in {paths} --- view_id: {query_id}"
                        else:
                            paths = params.get("paths", [])
                            globs = params.get("globs", [])
                            header = f"find_files: globs={globs} in {paths} --- view_id: {query_id}"

                        rendered_entries.append((last_content_changed_ts, original_ts, 0.0, "search", query_id, (header, search_content), search_bytes, content_hash, None))

            # Phase 2: Sort for inclusion priority
            # Most recently agent-touched views included first (descending original_ts).
            # Tiebreaker: files the agent is actively editing get priority (descending edit_ts).
            # edit_ts is 0 for files the agent hasn't modified through a tool.
            rendered_entries.sort(key=lambda x: (-x[1], -x[2], x[4]))

            # Phase 3: Apply byte budget in sorted order
            # Track natural output keys for Phase 4 reordering (prefix caching)
            result_output_keys = []

            for last_changed_ts, original_ts, edit_ts, entry_type, key, rendered_data, byte_count, content_hash, total_lines in rendered_entries:
                if entry_type == "file":
                    print(f"DEBUG get_all_views file: {key} bytes: {byte_count} running_total: {total_bytes + byte_count} / {self.MAX_FILE_VIEW_BYTES}")
                    if total_bytes + byte_count > self.MAX_FILE_VIEW_BYTES and result_views:
                        # Before evicting, apply ALL deferred timestamps across ALL files â
                        # applying them may change sort order and allow a different file to be evicted
                        any_deferred_applied_this_eviction = False
                        for fp, fv_entry in working_fv.items():
                            if isinstance(fv_entry, FileViewEntry):
                                fv_windows = self._get_entry_windows(fv_entry)
                                for w in fv_windows:
                                    if w.deferred_add_timestamp is not None and w.deferred_add_timestamp > (w.add_timestamp or 0):
                                        w.add_timestamp = w.deferred_add_timestamp
                                        w.deferred_add_timestamp = None
                                        deferred_applied = True
                                        any_deferred_applied_this_eviction = True

                        if any_deferred_applied_this_eviction:
                            needs_restart = True
                            break  # Exit Phase 3 loop â restart will re-sort with updated timestamps

                        print(f"DEBUG get_all_views skipping file total bytes exceeded {total_bytes + byte_count} > {self.MAX_FILE_VIEW_BYTES} for {key}")
                        files_to_remove.add(key)
                        continue
                    section_id = f"view:{key}"
                    stability_key = (last_changed_ts, original_ts, section_id)
                    output_key = (last_changed_ts, original_ts, key)
                    for start, end, numbered_content in rendered_data:
                        if include_stability_keys:
                            result_views.append(("file", key, start, end, numbered_content, total_lines, section_id, stability_key))
                        else:
                            result_views.append(("file", key, start, end, numbered_content, total_lines))
                        result_output_keys.append(output_key)
                    total_bytes += byte_count
                    included_file_paths.add(key)

                elif entry_type == "search":
                    print(f"DEBUG get_all_views search: {key} bytes: {byte_count} running_total: {total_bytes + byte_count} / {self.MAX_FILE_VIEW_BYTES}")
                    if total_bytes + byte_count > self.MAX_FILE_VIEW_BYTES and result_views:
                        print(f"DEBUG get_all_views skipping search query total bytes exceeded {total_bytes + byte_count} > {self.MAX_FILE_VIEW_BYTES}")
                        queries_to_remove.add(key)
                        continue
                    header, search_content = rendered_data
                    section_id = f"search:{key}"
                    stability_key = (last_changed_ts, original_ts, section_id)
                    output_key = (last_changed_ts, original_ts, key)
                    if include_stability_keys:
                        result_views.append(("search", header, search_content, section_id, stability_key))
                    else:
                        result_views.append(("search", header, search_content))
                    result_output_keys.append(output_key)
                    total_bytes += byte_count
                    included_query_ids.add(key)

            # Mark any rendered entries not included due to size limit for removal
            for last_changed_ts, original_ts, edit_ts, entry_type, key, rendered_data, byte_count, content_hash, total_lines in rendered_entries:
                if entry_type == "file" and key not in included_file_paths and key not in files_to_remove:
                    files_to_remove.add(key)
                elif entry_type == "search" and key not in included_query_ids and key not in queries_to_remove:
                    queries_to_remove.add(key)

            # Track deferred application across iterations for state update
            if deferred_applied:
                any_deferred_applied = True

            # If deferred timestamps were applied during eviction, restart to re-sort
            if needs_restart:
                continue

            # Phase 4: Reorder by natural stability sort for prefix caching
            # Most-stable (oldest content change) first for cache hits.
            # Tiebreaker: older views (lower original_ts) = more likely cached.
            if result_output_keys:
                paired = list(zip(result_output_keys, result_views))
                paired.sort(key=lambda x: x[0])
                result_views = [v for _, v in paired]
            # Build state update (cleanup evictions + metadata updates)
            state_update = None
            need_file_update = bool(files_to_remove) or file_metadata_changed or any_deferred_applied
            need_search_update = bool(queries_to_remove) or search_metadata_changed

            if need_file_update or need_search_update:
                update_parts = {}

                if need_file_update:
                    # Use working_fv as the base (it may have deferred timestamps applied)
                    new_windows = working_fv
                    # Remove evicted entries
                    for fp in files_to_remove:
                        new_windows.pop(fp, None)
                    # Update metadata on remaining entries that were rendered
                    for last_changed_ts, original_ts, edit_ts, entry_type, key, rendered_data, byte_count, content_hash, _total_lines in rendered_entries:
                        if entry_type == "file" and key in new_windows:
                            e = new_windows[key]
                            if isinstance(e, FileViewEntry):
                                e.last_content_hash = content_hash
                                e.last_content_changed_timestamp = last_changed_ts
                            elif isinstance(e, dict):
                                e['last_content_hash'] = content_hash
                                e['last_content_changed_timestamp'] = last_changed_ts
                    update_parts["file_view_windows"] = new_windows

                if need_search_update:
                    new_queries = deepcopy(self.search_list_view) if self.search_list_view else {}
                    # Remove evicted entries
                    for qid in queries_to_remove:
                        new_queries.pop(qid, None)
                    # Update metadata on remaining entries that were rendered
                    for last_changed_ts, original_ts, edit_ts, entry_type, key, rendered_data, byte_count, content_hash, _total_lines in rendered_entries:
                        if entry_type == "search" and key in new_queries:
                            e = new_queries[key]
                            if isinstance(e, SearchListViewEntry):
                                e.last_content_hash = content_hash
                                e.last_content_changed_timestamp = last_changed_ts
                            elif isinstance(e, dict):
                                e['last_content_hash'] = content_hash
                                e['last_content_changed_timestamp'] = last_changed_ts
                    update_parts["search_list_view"] = new_queries

                state_update = update_parts

            break  # Exit restart loop

        print("DEBUG get_all_views total_bytes:", total_bytes)

        return total_bytes/self.MAX_FILE_VIEW_BYTES, sorted(files_to_remove), result_views, state_update
    
    def _execute_search_query(self, query_type: str, params: dict, filesystem_service) -> Tuple[Optional[str], int]:
        """Re-execute a search or find_files query and return (content, bytes).
        
        Args:
            query_type: "ripgrep_search" or "find_files"
            params: Query parameters dict
            filesystem_service: FileSystemService instance
            
        Returns:
            Tuple of (content_string, byte_count) or (None, 0) on error
        """
        try:
            if query_type == "ripgrep_search":
                import python_ripgrep
                patterns = params["patterns"]
                paths = params["paths"]
                globs = params["globs"]
                max_count = params["max_count"]
                
                results = python_ripgrep.search(
                    patterns=patterns,
                    paths=paths,
                    globs=globs,
                    max_count=max_count,
                    line_number=True,
                    heading=True
                )
                
                if not results:
                    content = "No matches found"
                else:
                    if len(results) > max_count * 10:
                        results = results[:max_count * 10]
                        results.append("... (results truncated)")
                    content = "\n".join(results)
                    try:
                        content = content.encode('utf-8', errors='replace').decode('utf-8')
                    except:
                        if len(content) > 10000:
                            content = content[:10000] + "\n... (output truncated due to encoding issues)"
                
                return content, len(content.encode('utf-8'))
            
            elif query_type == "find_files":
                import fnmatch
                from pathlib import Path as FilePath
                
                paths = params["paths"]
                globs = params["globs"]
                max_count = params["max_count"]
                
                matched_files = []
                for search_path in paths:
                    p = FilePath(search_path)
                    if p.is_file():
                        for pattern in globs:
                            if fnmatch.fnmatch(p.name, pattern):
                                matched_files.append(str(p))
                                break
                    elif p.is_dir():
                        has_recursive = any('**/' in pattern for pattern in globs)
                        if has_recursive:
                            for pattern in globs:
                                for file_path in p.rglob(pattern):
                                    if file_path.is_file():
                                        matched_files.append(str(file_path))
                                        if len(matched_files) >= max_count:
                                            break
                                if len(matched_files) >= max_count:
                                    break
                        else:
                            import os as _os
                            for root, dirs, files in _os.walk(p):
                                for filename in files:
                                    for pattern in globs:
                                        if fnmatch.fnmatch(filename, pattern):
                                            full_path = _os.path.join(root, filename)
                                            matched_files.append(full_path)
                                            break
                                    if len(matched_files) >= max_count:
                                        break
                                if len(matched_files) >= max_count:
                                    break
                    if len(matched_files) >= max_count:
                        break
                
                if not matched_files:
                    content = "No files found matching the specified patterns"
                else:
                    output_lines = [f"Found {len(matched_files)} file(s):", ""]
                    for fp in sorted(matched_files):
                        output_lines.append(fp)
                    content = "\n".join(output_lines)
                
                return content, len(content.encode('utf-8'))
        
        except Exception as e:
            print(f"DEBUG _execute_search_query error: {e}")
            return None, 0
        
        return None, 0
    
    # ========================================================================
    # Private helper methods
    # ========================================================================
    
    def _merge_overlapping_windows(self, windows: List[FileViewWindow]) -> List[FileViewWindow]:
        """Merge overlapping windows into non-overlapping ranges.
        
        Args:
            windows: List of FileViewWindow (all for the same file)
            
        Returns:
            List of merged FileViewWindow with no overlaps
        """
        if not windows:
            print(f"DEBUG _merge_overlapping_windows empty windows")
            return []
        
        # Sort by start line
        sorted_windows = sorted(windows, key=lambda w: w.start_line)
        
        merged = [sorted_windows[0]]
        
        for current in sorted_windows[1:]:
            last = merged[-1]
            # Check for overlap or adjacency (adjacent windows can be merged)
            # If last.end_line is None, it extends to end of file, so always overlaps
            if last.end_line is None or current.start_line <= last.end_line + 1:
                # Merge: extend end_line if current extends further
                # If current.end_line is None, merged window extends to end of file
                if current.end_line is None:
                    last.end_line = None
                elif last.end_line is not None and current.end_line > last.end_line:
                    last.end_line = current.end_line
                # Keep the most recent timestamps
                if current.edit_timestamp is not None and (last.edit_timestamp is None or current.edit_timestamp > last.edit_timestamp):
                    last.edit_timestamp = current.edit_timestamp
                if current.add_timestamp is not None and (last.add_timestamp is None or current.add_timestamp > last.add_timestamp):
                    last.add_timestamp = current.add_timestamp
                # Keep the most recent deferred_add_timestamp
                if current.deferred_add_timestamp is not None and (last.deferred_add_timestamp is None or current.deferred_add_timestamp > last.deferred_add_timestamp):
                    last.deferred_add_timestamp = current.deferred_add_timestamp
            else:
                merged.append(current)
        
        return merged
