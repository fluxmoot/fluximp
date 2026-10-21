import fnmatch
import os
from pathlib import Path
from typing import Optional

import python_ripgrep

from fluximp_agent.filesystem_service import check_line_mode_file_size
from fluximp_agent.tools.tool_definitions import ToolResult
from fluximp_agent.context_managment_constants import (
    MAX_FILE_VIEW_BYTES,
    VIEW_BUDGET_DESC,
)
from fluximp_agent.tools.common import _validate_paths_exist, _validate_paths_within_working_dir


def ripgrep_search(patterns: list, paths: Optional[list] = None, globs: Optional[list] = None, max_count: int = 50, **kwargs) -> ToolResult:
    """
    Search for patterns in code files using ripgrep.

    This is a READ-ONLY tool that searches code using ripgrep.
    It returns a ToolResult with success status and search results.

    Args:
        patterns: List of text patterns to search for
        paths: List of directories or files to search in (defaults to current directory)
        globs: List of file patterns to include (e.g., ["*.py", "*.js"])
        max_count: Maximum number of matches to return (default: 50)

    Returns:
        ToolResult - with success=True and content=search results, or success=False on error
    """
    filesystem_service = kwargs.get('filesystem_service')
    
    # Default to working directory if no paths specified
    if paths is None:
        paths = filesystem_service.get_default_paths()

    # Validate that all paths are within the read root directory FIRST
    # (before checking existence, to prevent info leakage about outside paths)
    success, error = _validate_paths_within_working_dir(paths, filesystem_service, access_type='read')
    if not success:
        return ToolResult(success=False, content=error)

    # Validate that all paths exist
    success, error = _validate_paths_exist(paths)
    if not success:
        return ToolResult(success=False, content=error)

    # Perform the search with a reasonable default max_count if not specified or invalid
    if max_count is None or max_count < 1:
        max_count = 50

    # Ensure patterns is a list (handle case where model passes a single string)
    if isinstance(patterns, str):
        patterns = [patterns]

    # Ensure globs is a list if provided (handle case where model passes a single string)
    if globs is not None and isinstance(globs, str):
        globs = [globs]

    print(f"DEBUG ripgrep_search called with: {patterns}, {paths}, {globs}, {max_count}")

    # Check if search_list_view experiment is active
    experiments_service = kwargs.get('experiments_service')
    state_service = kwargs.get('state_service')
    if experiments_service and experiments_service.is_enabled('search_list_view') and state_service:
        # Store query params instead of executing the search inline
        search_update = state_service.build_search_list_view_add(
            query_type='ripgrep_search',
            params={'patterns': patterns, 'paths': paths, 'globs': globs, 'max_count': max_count}
        )
        return ToolResult(
            success=True,
            content=f'Search query added to view (searching for {len(patterns)} pattern(s) in {len(paths)} path(s)). Results will appear in the search/list view section.',
            state_updates=search_update
        )

    # Perform the search
    try:
        results = python_ripgrep.search(
            patterns=patterns,
            paths=paths,
            globs=globs,
            max_count=max_count,
            line_number=True,  # Include line numbers
            heading=True       # Group matches by file
        )
    except ValueError as e:
        msg = str(e)
        if msg.startswith('regex parse error'):
            return ToolResult(success=False, content=msg)
        elif 'multiline' in msg:
            return ToolResult(success=False, content="Multiline regex not support")
        raise

    if not results:
        return ToolResult(success=True, content="No matches found")

    # Limit the number of results to prevent overwhelming output
    # Even though we set max_count, we'll add an additional safeguard
    if len(results) > max_count * 10:  # Allow some buffer for context lines
        results = results[:max_count * 10]
        results.append("... (results truncated)")

    # Join the results into a single string
    output = "\n".join(results)

    # Handle potential encoding issues by replacing problematic characters
    # This is a workaround for encoding issues in some environments
    try:
        # Try to encode and decode to handle any encoding issues
        output = output.encode('utf-8', errors='replace').decode('utf-8')
    except:
        # If there are still issues, truncate the output to prevent problems
        if len(output) > 10000:
            output = output[:10000] + "\n... (output truncated due to encoding issues)"

    return ToolResult(success=True, content=output)


def read_file(file_path: str, offset: int = 1, limit: int = 2500, line_numbers: bool = True, **kwargs) -> ToolResult:
    """
    Read file contents with optional line numbering.

    This is a READ-ONLY tool that reads file contents.
    It returns a ToolResult with success status and file contents.

    Args:
        file_path: Path to the file to read
        offset: Starting line (1-indexed, default 1)
        limit: Maximum number of lines to read (default 2500)
        line_numbers: Whether to prefix lines with line numbers (default True)

    Returns:
        ToolResult - with success=True and content=file contents, or success=False on error
    """
    filesystem_service = kwargs.get('filesystem_service')
    
    # Validate path (read access uses read_root_dir)
    path, error = filesystem_service.validate_path(file_path, access_type='read')
    if error:
        return ToolResult(success=False, content=error)
    
    try:
        if not path.exists():
            return ToolResult(success=False, content=f"Error: File '{file_path}' not found at {path}")

        if not path.is_file():
            return ToolResult(success=False, content=f"Error: '{file_path}' is not a file")

        size_error = check_line_mode_file_size(path, file_path)
        if size_error:
            return ToolResult(success=False, content=size_error)
        # Read file content
        content = path.read_text(encoding='utf-8')
        lines = content.splitlines()

        # Validate offset (1-indexed)
        if offset < 1:
            offset = 1

        # Calculate end line
        start_idx = offset - 1  # Convert to 0-indexed
        end_idx = start_idx + limit

        # Handle case where offset is beyond file length
        if start_idx >= len(lines):
            return ToolResult(success=False, content=f"Error: Offset {offset} is beyond file length ({len(lines)} lines)")

        # Slice lines
        selected_lines = lines[start_idx:end_idx]

        # Format output with optional line numbers
        result_lines = []
        for i, line in enumerate(selected_lines, start=offset):
            if line_numbers:
                result_lines.append(f"{i:4d} | {line}")
            else:
                result_lines.append(line)

        # Add truncation notice if file was truncated
        if end_idx < len(lines):
            result_lines.append(f"\n... ({len(lines) - end_idx} more lines)")

        # Bound rendered output to the view content byte budget: a file with
        # very long lines can exceed the context budget even when under the
        # line-mode size limit, so truncate the rendered text here too.
        max_bytes = MAX_FILE_VIEW_BYTES
        output = "\n".join(result_lines)
        if len(output.encode('utf-8')) > max_bytes:
            truncated_lines = []
            used_bytes = 0
            for line in result_lines:
                line_bytes = len(line.encode('utf-8')) + 1  # +1 for joining newline
                if used_bytes + line_bytes > max_bytes:
                    break
                truncated_lines.append(line)
                used_bytes += line_bytes
            truncated_lines.append(f"... (output truncated: exceeds view content budget of {VIEW_BUDGET_DESC})")
            output = "\n".join(truncated_lines)

        return ToolResult(success=True, content=output)

    except Exception as e:
        return ToolResult(success=False, content=f"Error reading file '{file_path}': {e}")

def find_files(paths: Optional[list] = None, globs: Optional[list] = None, max_count: int = 100, **kwargs) -> ToolResult:
    """
    Find files matching glob patterns.

    This is a READ-ONLY tool that finds files matching glob patterns.
    It returns a ToolResult with success status and matching files list.

    Args:
        paths: List of directories to search (defaults to current directory ["."])
        globs: List of glob patterns to match (defaults to ["*"] for all files)
        max_count: Maximum number of files to return (default: 100)

    Returns:
        ToolResult - with success=True and content=list of matching files, or success=False on error
    """
    filesystem_service = kwargs.get('filesystem_service')
    
    # Default values
    if paths is None:
        paths = filesystem_service.get_default_paths()
    if globs is None:
        globs = ["*"]
    if max_count is None:
        max_count = 100

    # Ensure paths is a list if provided (handle case where model passes a single string)
    if isinstance(paths, str):
        paths = [paths]

    # Ensure globs is a list if provided (handle case where model passes a single string)
    if isinstance(globs, str):
        globs = [globs]

    # Validate that all paths are within the read root directory FIRST
    # (before checking existence, to prevent info leakage about outside paths)
    success, error = _validate_paths_within_working_dir(paths, filesystem_service, access_type='read')
    if not success:
        return ToolResult(success=False, content=error)

    # Validate that all paths exist
    success, error = _validate_paths_exist(paths)
    if not success:
        return ToolResult(success=False, content=error)

    print(f"DEBUG find_files called with: {paths}, {globs}, {max_count}")

    # Check if search_list_view experiment is active
    experiments_service = kwargs.get('experiments_service')
    state_service = kwargs.get('state_service')
    if experiments_service and experiments_service.is_enabled('search_list_view') and state_service:
        # Store query params instead of executing the search inline
        search_update = state_service.build_search_list_view_add(
            query_type='find_files',
            params={'paths': paths, 'globs': globs, 'max_count': max_count}
        )
        return ToolResult(
            success=True,
            content=f'File listing query added to view (searching in {len(paths)} path(s) for {len(globs)} pattern(s)). Results will appear in the search/list view section.',
            state_updates=search_update
        )

    matched_files = []

    for search_path in paths:
        path = Path(search_path)
        
        if path.is_file():
            # If path is a file, check if it matches any glob
            for pattern in globs:
                if fnmatch.fnmatch(path.name, pattern):
                    matched_files.append(str(path))
                    break
        elif path.is_dir():
            # Check for recursive glob patterns (containing **/)
            has_recursive = any('**/' in pattern for pattern in globs)
            
            if has_recursive:
                # Use rglob for recursive patterns
                for pattern in globs:
                    # Pass pattern directly to rglob - it handles **/ natively
                    # e.g., "**/read_test.txt" matches read_test.txt at any depth
                    # e.g., "**/src/*.py" matches *.py in any src/ directory at any depth
                    for file_path in path.rglob(pattern):
                        if file_path.is_file():
                            matched_files.append(str(file_path))
                            if len(matched_files) >= max_count:
                                break
                    if len(matched_files) >= max_count:
                        break
            else:
                # Walk directory tree for non-recursive patterns
                for root, dirs, files in os.walk(path):
                    for filename in files:
                        for pattern in globs:
                            if fnmatch.fnmatch(filename, pattern):
                                full_path = os.path.join(root, filename)
                                matched_files.append(full_path)
                                break
                        
                        # Check max_count limit
                        if len(matched_files) >= max_count:
                            break
                    
                    if len(matched_files) >= max_count:
                        break
        
        if len(matched_files) >= max_count:
            break

    if not matched_files:
        return ToolResult(success=True, content="No files found matching the specified patterns")

    # Format output
    output_lines = []
    output_lines.append(f"Found {len(matched_files)} file(s):")
    output_lines.append("")
    
    for file_path in sorted(matched_files):
        output_lines.append(file_path)
    
    return ToolResult(success=True, content="\n".join(output_lines))
