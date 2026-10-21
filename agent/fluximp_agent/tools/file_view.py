from fluximp_agent.filesystem_service import check_line_mode_file_size
from fluximp_agent.tools.tool_definitions import ToolResult
from fluximp_agent.context_managment_constants import VIEW_BUDGET_DESC
from fluximp_agent.state_service import normalize_file_path


def add_file_view(file_path: str, offset: int = 1, limit: int = 2500, **kwargs) -> ToolResult:
    f"""Add a file view window to track specific lines in a file.
    
    This tool registers a window into a file that will be included in the
    context sent to the model. The content is read fresh each time context
    is generated, so changes to the file will be reflected automatically.

    File view windows are merged when they overlap and limited by total size ({VIEW_BUDGET_DESC}).
    Files are sorted by most recent activity, with the most recent files
    appearing at the bottom of the context for prominence.

    Args:
        file_path: Path to the file to view
        offset: Starting line (1-indexed, default 1)
        limit: Maximum number of lines to include (default 2500)

    Returns:
        ToolResult - with success status, content message, and state_updates dict
    """
    filesystem_service = kwargs.get('filesystem_service')
    state_service = kwargs.get('state_service')

    # Validate path (read access uses read_root_dir)
    path, error = filesystem_service.validate_path(file_path, access_type='read')
    if error:
        return ToolResult(success=False, content=error)

    if not path.exists():
        return ToolResult(success=False, content=f"Error: File '{file_path}' not found")

    if not path.is_file():
        return ToolResult(success=False, content=f"Error: '{file_path}' is not a file")

    size_error = check_line_mode_file_size(path, file_path)
    if size_error:
        return ToolResult(success=False, content=size_error)
    # Normalize the file path for consistent dict key usage
    normalized_path = normalize_file_path(file_path, filesystem_service)

    # Capture original window ranges before update (to detect no-op)
    current_windows = state_service.file_view_windows
    existing_entry = current_windows.get(normalized_path)
    original_ranges = None
    if existing_entry:
        orig_windows = state_service._get_entry_windows(existing_entry)
        original_ranges = sorted([(w.start_line, w.end_line) for w in orig_windows])

    # Build file view windows update
    from fluximp_agent.state_service import FileViewRequest
    file_view_update = state_service.build_file_view_windows_update(
        [FileViewRequest(file_path=normalized_path, offset=offset, limit=limit)],
        source="add"
    )

    # Get the view ID for the new/updated entry
    new_windows = file_view_update.get("file_view_windows", {})
    entry = new_windows.get(normalized_path)
    view_id = ""
    from fluximp_agent.state_service import FileViewEntry
    if entry:
        if isinstance(entry, FileViewEntry):
            view_id = entry.id
        elif isinstance(entry, dict):
            view_id = entry.get("id", "")

    # Check if the requested lines were already fully in view
    ranges_unchanged = False
    if original_ranges is not None and entry:
        if isinstance(entry, FileViewEntry):
            new_ranges = sorted([(w.start_line, w.end_line) for w in entry.windows])
        elif isinstance(entry, dict):
            new_ranges = sorted([(w.start_line, w.end_line) for w in entry.get("windows", [])])
        else:
            new_ranges = None
        if new_ranges is not None and new_ranges == original_ranges:
            ranges_unchanged = True

    id_msg = f" (view id: {view_id})" if view_id else ""
    if ranges_unchanged:
        content = (f"File view for '{file_path}' offset: {offset} limit: {limit}{id_msg} is already present in your context. "
                   f"The requested lines are already fully visible in an existing view window. "
                   f"You don't need to use add_file_view on blocks of lines that are already in your view "
                   f"-- what you see is already maximally fresh.")
    else:
        content = f"File view added for '{file_path}' offset: {offset} limit: {limit}{id_msg}. The requested operation is complete."

    return ToolResult(
        success=True,
        content=content,
        state_updates={**file_view_update, "file_views_added_this_turn": [normalized_path]}
    )



def close_file_view(file_paths: list, **kwargs) -> ToolResult:
    """Close file view windows for the specified files.

    Removes all view windows for the given file paths. Files not currently
    having view windows are ignored with a warning.

    Args:
        file_paths: List of file paths to close view windows for

    Returns:
        ToolResult - with success status, content message, and state_updates dict
    """
    state_service = kwargs.get('state_service')

    if not file_paths:
        return ToolResult(success=False, content="No files specified.")

    # Ensure file_paths is a list
    if isinstance(file_paths, str):
        file_paths = [file_paths]

    # Build removal update
    file_view_update = state_service.build_file_view_windows_removal(file_paths)

    # Check which files actually had view windows
    current_windows = state_service.file_view_windows
    warnings = [fp for fp in file_paths if fp not in current_windows]

    parts = ["File view windows closed for specified files. The requested operation is complete."]
    if warnings:
        parts.append(f"\nWarnings: The following files had no view windows: {', '.join(warnings)}")

    return ToolResult(
        success=True,
        content="\n".join(parts),
        state_updates=file_view_update
    )


def close_view(view_ids: list, **kwargs) -> ToolResult:
    """Close views by their IDs (shown in [id] in view headers).

    Removes file views and search/list queries matching the given view IDs.
    IDs not matching any current view are ignored with a warning.

    If a view ID corresponds to a file that had add_file_view called earlier
    in the same response, the close is ignored to prevent the LLM from
    inadvertently removing a view it just refreshed.

    Args:
        view_ids: List of view IDs to close

    Returns:
        ToolResult - with success status, content message, and state_updates dict
    """
    state_service = kwargs.get('state_service')

    if not view_ids:
        return ToolResult(success=False, content="No view IDs specified.")

    # Ensure view_ids is a list
    if isinstance(view_ids, str):
        view_ids = [view_ids]

    # Check which view IDs correspond to files that had add_file_view this turn
    added_this_turn = state_service.file_views_added_this_turn
    from fluximp_agent.state_service import FileViewEntry
    current_fv = state_service.file_view_windows
    
    ignored_ids = set()
    if added_this_turn:
        # Find view IDs for files that were added this turn
        for file_path, entry in current_fv.items():
            if file_path in added_this_turn:
                if isinstance(entry, FileViewEntry):
                    if entry.id in view_ids:
                        ignored_ids.add(entry.id)
                elif isinstance(entry, dict):
                    eid = entry.get("id", "")
                    if eid in view_ids:
                        ignored_ids.add(eid)
    
    # Only remove non-ignored view IDs
    effective_view_ids = [vid for vid in view_ids if vid not in ignored_ids]

    # Build removal update (only for non-ignored IDs)
    view_update = state_service.build_view_removal(effective_view_ids) if effective_view_ids else {}

    # Check which IDs actually matched something
    current_sl = state_service.search_list_view
    
    matched_ids = set()
    for file_path, entry in current_fv.items():
        if isinstance(entry, FileViewEntry):
            if entry.id in view_ids:
                matched_ids.add(entry.id)
        elif isinstance(entry, dict):
            eid = entry.get("id", "")
            if eid in view_ids:
                matched_ids.add(eid)
    
    for qid, entry in current_sl.items():
        if qid in view_ids:
            matched_ids.add(qid)
    
    warnings = [vid for vid in view_ids if vid not in matched_ids]

    parts = ["Views closed for specified IDs. The requested operation is complete."]
    if ignored_ids:
        parts.append(f"\nThe following view IDs were ignored due to a preceding add_file_view in the same response: {', '.join(sorted(ignored_ids))}")
    if warnings:
        parts.append(f"\nWarnings: The following view IDs were not found: {', '.join(warnings)}")

    return ToolResult(
        success=True,
        content="\n".join(parts),
        state_updates=view_update
    )
