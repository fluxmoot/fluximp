import zlib
import difflib
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
from pathlib import Path


def compress(data: str) -> bytes:
    """Compress string data using zlib."""
    return zlib.compress(data.encode('utf-8'))


def decompress(data: bytes) -> str:
    """Decompress data using zlib."""
    return zlib.decompress(data).decode('utf-8')


# ============================================================================
# Path Validation Utilities
# ============================================================================
def lines_match(line_idx: int, expected_content: str, lines_list: list) -> bool:
    """Compare whitespace-stripped content of a line at line_idx with expected_content.
    
    Strips all whitespace for comparison so that indentation differences are ignored.
    Also normalizes single quotes (') to double quotes (") to handle LLM quote-style
    inconsistencies.
    
    Args:
        line_idx: 0-indexed line number to check
        expected_content: The expected content to match against
        lines_list: List of line strings to look up line_idx in
        
    Returns:
        True if the whitespace-stripped, quote-normalized content matches, False otherwise
        (including when line_idx is out of range).
    """
    if line_idx < 0 or line_idx >= len(lines_list):
        return False
    actual = "".join(lines_list[line_idx].split()).replace("'", '"')
    expected = "".join(expected_content.split()).replace("'", '"')
    return actual == expected

def _validate_paths_exist(paths: list) -> Tuple[bool, Optional[str]]:
    """
    Validate that all paths in the list exist.

    Args:
        paths: List of path strings to validate

    Returns:
        Tuple of (success, error_message). If success is False, error_message
        contains details about which path(s) don't exist.
    """
    nonexistent_paths = []
    for path_str in paths:
        path = Path(path_str)
        if not path.exists():
            nonexistent_paths.append(path_str)

    if nonexistent_paths:
        if len(nonexistent_paths) == 1:
            return (False, f"Error: Path does not exist: '{nonexistent_paths[0]}'")
        else:
            paths_str = "', '".join(nonexistent_paths)
            return (False, f"Error: Paths do not exist: '{paths_str}'")

    return (True, None)


def _validate_paths_within_working_dir(paths: list, filesystem_service, access_type: str = 'write') -> Tuple[bool, Optional[str]]:
    """
    Validate that all paths are within the allowed directory.

    Args:
        paths: List of path strings to validate
        filesystem_service: The FileSystemService instance for path validation
        access_type: 'write' to validate against working_dir (default),
                    'read' to validate against read_root_dir

    Returns:
        Tuple of (success, error_message). If success is False, error_message
        contains details about which path(s) are outside the allowed directory.
    """
    if access_type == 'read':
        root_dir = filesystem_service.read_root_dir
        dir_label = "the read root directory"
    else:
        root_dir = filesystem_service.working_dir
        dir_label = "the working directory"
    
    outside_paths = []
    for path_str in paths:
        path = Path(path_str)
        
        # For write access with write_file restriction, delegate to validate_path
        # which enforces the single-file restriction
        if access_type == 'write' and filesystem_service.write_file is not None:
            resolved, error = filesystem_service.validate_path(path_str, access_type='write')
            if error:
                outside_paths.append(path_str)
        # If path is absolute, check if it's within the allowed directory
        elif path.is_absolute():
            resolved = path.resolve()
            try:
                resolved.relative_to(root_dir)
            except ValueError:
                outside_paths.append(path_str)
        else:
            # Use the service's validate_path method with the appropriate access type
            resolved, error = filesystem_service.validate_path(path_str, access_type=access_type)
            if error:
                outside_paths.append(path_str)

    if outside_paths:
        if len(outside_paths) == 1:
            return (False, f"Error: Access denied. Path '{outside_paths[0]}' is outside {dir_label}.")
        else:
            paths_str = "', '".join(outside_paths)
            return (False, f"Error: Access denied. Paths '{paths_str}' are outside {dir_label}.")

    return (True, None)


# ============================================================================
# File Operation Dataclass
# ============================================================================
@dataclass
class FileOperation:
    """Record of a file operation for undo/redo support."""
    operation_type: str
    timestamp: str
    file_path: str
    patch: Optional[str] = None
    compressed_before: Optional[bytes] = None
    compressed_after: Optional[bytes] = None
    operation_position: Optional[int] = None
    atomic_group_id: Optional[str] = None


# ============================================================================
# Diff utilities
# ============================================================================
def create_aggregate_diff(
    file_operations: list,
    operation_position: int,
    filesystem_service,
    context_lines: int = 3,
) -> str:
    """Create a unified diff aggregating all file changes up to current position.

    This diff can be passed to apply_patch after fully undoing all operations
    to reproduce the current state of files.

    Args:
        file_operations: Full list of FileOperation records
        operation_position: Current position in the operation history
        filesystem_service: For reading current file state if needed
        context_lines: Number of context lines to include around each change (default 3)

    Returns:
        Unified diff string showing net changes, or empty string if no changes
    """
    #print(f"DEBUG create_aggregate_diff {operation_position} {file_operations}")
    if not file_operations:
        return ""

    # Track file states: filepath -> (original_content, final_compressed_after, was_deleted)
    file_states: Dict[str, Tuple[Optional[str], Optional[bytes], bool]] = {}

    # Only process operations up to current position (undo/redo move position)
    for op in file_operations[:operation_position + 1] if operation_position >= 0 else []:
        assert op.operation_type not in ["undo", "redo"]
        filepath = op.file_path
        if filepath is None:
            assert op.operation_type == 'apply_patch'
            # TODO applied patches can affect multiple files
        else:
            is_deleted = op.operation_type == "delete_file"

            if filepath not in file_states:
                # First time seeing this file
                file_states[filepath] = (op.compressed_before, op.compressed_after, is_deleted)
            else:
                # Update existing file tracking
                original, _, _ = file_states[filepath]
                file_states[filepath] = (original, op.compressed_after, is_deleted)

    # Generate consolidated diff for each modified file
    diff_parts = []

    for filepath, (compressed_before, compressed_after, was_deleted) in file_states.items():
        # Determine "before" content and label
        if compressed_before is not None:
            before_content = decompress(compressed_before)
            before_label = f"a/{filepath}"
        else:
            # File was created during these operations
            before_content = ""
            before_label = "/dev/null"

        # Determine "after" content and label
        if was_deleted:
            after_content = ""
            after_label = "/dev/null"
        elif compressed_after is not None:
            after_content = decompress(compressed_after)
            after_label = f"b/{filepath}"
        else:
            assert False

        # Skip if no actual change
        if before_content == after_content:
            continue

        # Generate unified diff for this file
        before_lines = before_content.splitlines(keepends=True)
        after_lines = after_content.splitlines(keepends=True)

        # Ensure lines end with newlines for proper diff format
        if before_lines and not before_lines[-1].endswith('\n'):
            before_lines[-1] += '\n'
        if after_lines and not after_lines[-1].endswith('\n'):
            after_lines[-1] += '\n'

        file_diff = difflib.unified_diff(
            before_lines,
            after_lines,
            fromfile=before_label,
            tofile=after_label,
            n=context_lines,
            lineterm='\n'
        )

        diff_parts.extend(file_diff)

    return ''.join(diff_parts)


class ToolError(Exception):
    """Raised for tool validation/execution errors that should be returned to the model."""
    pass
