import re
import uuid
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timezone
from dataclasses import dataclass, field

from fluximp_agent.filesystem_service import check_line_mode_file_size
from fluximp_agent.tools.tool_definitions import ToolResult
from fluximp_agent.tools.common import FileOperation, compress


@dataclass
class PatchFileResult:
    """Per-file outcome of applying a unified diff.

    Captures before/after content so apply_patch can record a normal
    FileOperation (with compressed_before/compressed_after) per file, exactly
    like the other editing tools, and update the file view windows.
    """
    file_path: str
    old_content: str
    new_content: str
    existed_before: bool
    deleted: bool
    # (old_start_0idx, old_line_count, new_line_count) per hunk, in file order
    hunks: List[Tuple[int, int, int]] = field(default_factory=list)


def _apply_unified_diff(diff_text: str, filesystem_service) -> Tuple[bool, Optional[str], Optional[List[PatchFileResult]]]:
    """Apply a unified diff to files.

    Parses the unified diff and applies changes to files.
    Supports multiple files in a single patch.
    Validates context and remove lines before applying changes.
    Every file the patch cannot be read from is reported in one error message,
    before anything is written (so a failed patch changes nothing on disk).
    Returns (success, error_message, results) where results is a list of
    PatchFileResult (one per affected file), or None on failure.
    """
    try:
        # Normalize line endings to Unix style for consistent parsing
        normalized_diff = diff_text.replace('\r\n', '\n')
        lines = normalized_diff.splitlines(keepends=True)

        # Store parsed hunks for each file before validation
        # Structure: {filepath: [(hunk_old_start, hunk_old_count, hunk_lines), ...]}
        file_hunks: Dict[str, List[Tuple[int, int, List[Tuple[str, str]]]]] = {}
        # Store original file lines for each file
        file_original_lines: Dict[str, List[str]] = {}
        # Raw original content and existence, for compressed_before
        file_original_content: Dict[str, str] = {}
        file_existed: Dict[str, bool] = {}
        # Whether the patch deletes the file (+++ /dev/null)
        file_deleted: Dict[str, bool] = {}
        # Per-file problems (invalid path, unreadable, or oversized file)
        # collected during parsing so every affected file can be reported at
        # once. Each entry is a complete sentence naming the file.
        problems: Dict[str, str] = {}
        current_file = None

        i = 0
        while i < len(lines):
            line = lines[i]

            # Check for file header
            if line.startswith('--- '):
                # Extract old file path (strip trailing newline and whitespace)
                old_file = line[4:].split('\t')[0].strip()
                # Strip git-style a/ and b/ prefixes
                if old_file.startswith('a/'):
                    old_file = old_file[2:]
                if old_file == '/dev/null':
                    old_file = None
                i += 1
                if i < len(lines) and lines[i].startswith('+++ '):
                    new_file = lines[i][4:].split('\t')[0].strip()
                    # Strip git-style a/ and b/ prefixes
                    if new_file.startswith('b/'):
                        new_file = new_file[2:]
                    if new_file == '/dev/null':
                        new_file = None
                    current_file = new_file or old_file
                    # A file is deleted when the new side is /dev/null but the
                    # old side names a real file.
                    is_delete = (new_file is None and old_file is not None)

                    i += 1

                    # Read existing file content. Failures are collected rather
                    # than returned immediately so that every file the patch
                    # cannot read is reported to the model in one message.
                    if current_file:
                        path, error = filesystem_service.validate_path(current_file)
                        if error:
                            problems.setdefault(current_file, error)
                            path = None

                        existed = path is not None and path.exists()
                        raw_content = ''
                        old_lines = []
                        if existed:
                            size_error = check_line_mode_file_size(path, current_file)
                            if size_error:
                                problems.setdefault(current_file, size_error)
                            else:
                                try:
                                    # Raw text is kept for compressed_before. Note
                                    # that read_text translates CRLF to LF, so this
                                    # is not byte-exact for CRLF files.
                                    raw_content = path.read_text(encoding='utf-8')
                                except UnicodeDecodeError as e:
                                    problems.setdefault(
                                        current_file,
                                        f"Error reading file '{current_file}': {e} - apply_patch only handles UTF-8 text files"
                                    )
                                except Exception as e:
                                    problems.setdefault(current_file, f"Error reading file '{current_file}': {e}")
                                else:
                                    old_lines = raw_content.splitlines(keepends=False)
                        # Still register the file so later "@@" lines have a hunk
                        # bucket; the early return below keeps it out of validation.
                        file_original_lines[current_file] = old_lines
                        file_original_content[current_file] = raw_content
                        file_existed[current_file] = existed
                        file_deleted[current_file] = is_delete
                        file_hunks[current_file] = []
                continue

            # Check for hunk header
            if line.startswith('@@ '):
                # Parse hunk header: @@ -start,count +start,count @@
                match = re.match(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', line)
                if match and current_file:
                    hunk_old_start = int(match.group(1)) - 1  # 0-indexed
                    hunk_old_count = int(match.group(2)) if match.group(2) else 1
                    hunk_new_start = int(match.group(3)) - 1  # 0-indexed

                    i += 1

                    # Collect hunk operations
                    hunk_lines = []
                    while i < len(lines):
                        hunk_line = lines[i]
                        # Check for next file, hunk, or diff metadata
                        if hunk_line.startswith('@@ ') or hunk_line.startswith('--- ') or hunk_line.startswith('diff '):
                            break
                        # Handle removal lines (but not the file header '--- ')
                        if hunk_line.startswith('-') and not hunk_line.startswith('---'):
                            # Remove trailing newline from content
                            content = hunk_line[1:].rstrip('\n').rstrip('\r')
                            hunk_lines.append(('remove', content))
                        # Handle addition lines (but not the file header '+++ ')
                        elif hunk_line.startswith('+') and not hunk_line.startswith('+++'):
                            # Remove trailing newline from content
                            content = hunk_line[1:].rstrip('\n').rstrip('\r')
                            hunk_lines.append(('add', content))
                        # Handle context lines (lines starting with space)
                        elif hunk_line.startswith(' '):
                            # Remove trailing newline from content
                            content = hunk_line[1:].rstrip('\n').rstrip('\r')
                            hunk_lines.append(('context', content))
                        elif hunk_line == '\n' or hunk_line == '\r\n':
                            # Empty line in hunk (just newline chars) - treat as context
                            hunk_lines.append(('context', ''))
                        elif hunk_line == '':
                            # Truly empty line - treat as context
                            hunk_lines.append(('context', ''))
                        else:
                            # Unknown line type - treat as context for robustness
                            # This handles malformed diffs that don't have leading spaces
                            content = hunk_line.rstrip('\n').rstrip('\r')
                            hunk_lines.append(('context', content))
                        i += 1

                    # Store hunk for later validation and application
                    file_hunks[current_file].append((hunk_old_start, hunk_old_count, hunk_lines))
                else:
                    i += 1
                continue

            i += 1

        # Report every file that could not be read before validating anything.
        # Nothing has been written yet, so this preserves all-or-nothing.
        if problems:
            return (False, "\n".join(problems.values()), None)
        # Validate all hunks before applying any changes
        for filepath, hunks in file_hunks.items():
            old_lines = file_original_lines.get(filepath, [])

            for hunk_idx, (hunk_old_start, hunk_old_count, hunk_lines) in enumerate(hunks):
                # New-file hunks parse to old_start == -1; clamp to 0 (there are
                # only additions, so nothing is validated against the old file).
                old_idx = max(hunk_old_start, 0)
                removed_count = 0

                for op_type, content in hunk_lines:
                    if op_type == 'remove':
                        # Validate that the line to remove matches actual content
                        if old_idx >= len(old_lines):
                            return (False, f"Patch validation failed for '{filepath}': Cannot remove line at position {old_idx + 1} (file has only {len(old_lines)} lines)", None)

                        actual_content = old_lines[old_idx]
                        if actual_content != content:
                            return (False, f"Patch validation failed for '{filepath}': Context mismatch at line {old_idx + 1}. Expected '{content}' but found '{actual_content}'", None)

                        old_idx += 1
                        removed_count += 1
                    elif op_type == 'add':
                        # Additions don't need validation
                        pass
                    elif op_type == 'context':
                        # Validate that context line matches actual content
                        if old_idx >= len(old_lines):
                            return (False, f"Patch validation failed for '{filepath}': Context line at position {old_idx + 1} exceeds file length ({len(old_lines)} lines)", None)

                        actual_content = old_lines[old_idx]
                        if actual_content != content:
                            return (False, f"Patch validation failed for '{filepath}': Context mismatch at line {old_idx + 1}. Expected '{content}' but found '{actual_content}'", None)

                        old_idx += 1

        # All validations passed - now apply changes
        file_modifications: Dict[str, List[str]] = {}
        file_hunk_results: Dict[str, List[Tuple[int, int, int]]] = {}

        for filepath, hunks in file_hunks.items():
            old_lines = file_original_lines.get(filepath, [])
            new_lines = old_lines.copy()

            # Track how line numbers have shifted after each hunk is applied
            # (hunk_original_start, delta) - lines before original_start are unchanged
            applied_deltas: List[Tuple[int, int]] = []
            hunk_results: List[Tuple[int, int, int]] = []

            for hunk_old_start, hunk_old_count, hunk_lines in hunks:
                # Calculate the actual position in the current new_lines
                # by applying deltas from previously applied hunks
                actual_start = hunk_old_start
                for orig_start, delta in applied_deltas:
                    if hunk_old_start >= orig_start:
                        actual_start += delta
                # A new-file hunk ("@@ -0,0 +1,N @@") parses to old_start == -1;
                # clamp so the insertion lands at the start of the file instead
                # of indexing past the beginning.
                if actual_start < 0:
                    actual_start = 0

                # Apply hunk operations at the calculated position
                result_lines = new_lines[:actual_start]
                old_idx = actual_start
                removed_count = 0
                added_count = 0

                for op_type, content in hunk_lines:
                    if op_type == 'remove':
                        # Skip this line in the original (mark as removed)
                        if old_idx < len(new_lines) and removed_count < hunk_old_count:
                            old_idx += 1
                            removed_count += 1
                    elif op_type == 'add':
                        # Add this line to the result
                        result_lines.append(content)
                        added_count += 1
                    elif op_type == 'context':
                        # Keep context line from original
                        if old_idx < len(new_lines):
                            result_lines.append(new_lines[old_idx])
                            old_idx += 1

                # Add remaining lines after hunk
                while old_idx < len(new_lines):
                    result_lines.append(new_lines[old_idx])
                    old_idx += 1

                # Calculate delta for this hunk: (added - removed)
                delta = added_count - removed_count
                applied_deltas.append((hunk_old_start, delta))
                # Record the hunk span (old_start, old_line_count, new_line_count)
                # so view windows can be adjusted for each changed region.
                hunk_results.append((hunk_old_start, hunk_old_count, hunk_old_count + delta))

                new_lines = result_lines

            file_modifications[filepath] = new_lines
            file_hunk_results[filepath] = hunk_results

        # Write all modified files
        if not file_modifications:
            return (False, "No file found in patch", None)

        results: List[PatchFileResult] = []

        def _partial_note() -> str:
            """Note naming files already written when a write fails mid-loop.

            Write failures happen after earlier files in the same patch have
            already been written, so the model must be told that the workspace is
            now only partially patched.
            """
            if not results:
                return ""
            applied = ', '.join(r.file_path for r in results)
            return (f" ({len(results)} of {len(file_modifications)} files in this "
                    f"patch were already applied: {applied})")

        for filepath, file_lines in file_modifications.items():
            path, error = filesystem_service.validate_path(filepath)
            if error:
                return (False, error, None)

            deleted = file_deleted.get(filepath, False)
            if deleted:
                # A patch whose new side is /dev/null truly deletes the file.
                # create_file restores it if the model tries to delete-and-
                # recreate, so this stays safe.
                try:
                    if path.exists():
                        path.unlink()
                except Exception as e:
                    return (False, f"Error deleting file '{filepath}': {e}" + _partial_note(), None)
                new_content = ''
            else:
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    # Join lines with newlines - add trailing newline if file has content
                    if file_lines:
                        output_content = '\n'.join(file_lines)
                        # Ensure file ends with newline if original had content
                        output_content += '\n'
                    else:
                        output_content = ''
                    path.write_text(output_content, encoding='utf-8', newline='')
                except Exception as e:
                    return (False, f"Error writing file '{filepath}': {e}" + _partial_note(), None)
                new_content = output_content

            results.append(PatchFileResult(
                file_path=filepath,
                old_content=file_original_content.get(filepath, ''),
                new_content=new_content,
                existed_before=file_existed.get(filepath, False),
                deleted=deleted,
                hunks=file_hunk_results.get(filepath, []),
            ))

        return (True, None, results)

    except Exception as e:
        return (False, f"Error applying patch: {e}", None)


def _build_patch_view_updates(results: List[PatchFileResult], state_service) -> dict:
    """Add/adjust file view windows for the regions changed by a patch.

    Mirrors how the other editing tools keep views in sync: deleted files have
    their views removed, newly created files get a whole-file window, and edits
    adjust existing windows (adding a window when the change isn't visible).
    Hunks are processed in file order, threading the accumulated windows dict
    so later hunks account for the line shifts of earlier ones.
    """
    from fluximp_agent.state_service import FileViewRequest

    windows = state_service.file_view_windows
    for result in results:
        file_path = state_service.normalize_path(result.file_path)
        if result.deleted:
            update = state_service.build_file_view_windows_removal(
                [file_path], current_windows=windows)
            windows = update.get("file_view_windows", windows)
        elif not result.existed_before:
            # Newly created file: show the whole file, like create_file/write_file
            update = state_service.build_file_view_windows_update(
                [FileViewRequest(file_path=file_path, offset=1, limit=None)],
                source="edit",
                current_windows=windows,
            )
            windows = update.get("file_view_windows", windows)
        else:
            cum_delta = 0
            for (old_start, old_line_count, new_line_count) in result.hunks:
                start_line = old_start + cum_delta + 1
                update = state_service.build_file_view_windows_edit_adjustment(
                    file_path=file_path,
                    start_line=start_line,
                    old_line_count=old_line_count,
                    new_line_count=new_line_count,
                    source="edit",
                    current_windows=windows,
                )
                windows = update.get("file_view_windows", windows)
                cum_delta += new_line_count - old_line_count
    return {"file_view_windows": windows}


def apply_patch_tool(diff_text: str, **kwargs) -> ToolResult:
    """Apply a unified diff patch to files - STATE-MODIFYING tool.

    Applies a unified diff format patch to files. Supports multiple files in one patch.
    Records one FileOperation per affected file (with before/after content) so
    that undo/redo, the cumulative diff, scrutiny, and file view windows all
    work exactly as they do for the other editing tools. Multi-file patches are
    grouped with a shared atomic_group_id so they undo/redo as a unit.

    Args:
        diff_text: Unified diff format patch text

    Returns:
        ToolResult - with success status, content message, and state_updates dict
    """
    filesystem_service = kwargs.get('filesystem_service')
    state_service = kwargs.get('state_service')
    experiments_service = kwargs.get('experiments_service')

    if not diff_text:
        return ToolResult(success=False, content="Error: Empty patch")

    # Apply the patch
    success, error, results = _apply_unified_diff(diff_text, filesystem_service)
    if not success:
        print(f"DEBUG _apply_unified_diff failed: {error}")
        return ToolResult(success=False, content=f"Error applying patch: {error}")
    assert error is None
    assert results is not None

    # Record one operation per affected file, grouped atomically so that a
    # multi-file patch is undone/redone as a single unit.
    group_id = str(uuid.uuid4())
    ops: List[FileOperation] = []
    deleted_paths: List[str] = []
    for result in results:
        if result.deleted:
            ops.append(FileOperation(
                operation_type="delete_file",
                timestamp=datetime.now(timezone.utc).isoformat(),
                file_path=result.file_path,
                compressed_before=compress(result.old_content) if result.existed_before else None,
                compressed_after=None,
                atomic_group_id=group_id,
            ))
            deleted_paths.append(result.file_path)
        else:
            ops.append(FileOperation(
                operation_type="apply_patch",
                timestamp=datetime.now(timezone.utc).isoformat(),
                file_path=result.file_path,
                compressed_before=compress(result.old_content) if result.existed_before else None,
                compressed_after=compress(result.new_content),
                atomic_group_id=group_id,
            ))

    # Build state update
    state_update = state_service.build_new_file_operations(ops)

    # Track deleted files so create_file can restore them (create_file experiment)
    if deleted_paths and experiments_service and experiments_service.is_enabled("create_file"):
        state_update["deleted_file_paths"] = frozenset(deleted_paths)

    # Update file view windows if experiment is active
    file_view_update = {}
    if experiments_service and experiments_service.is_enabled("file_view_windows") and state_service:
        file_view_update = _build_patch_view_updates(results, state_service)

    return ToolResult(
        success=True,
        content="The patch has been successfully applied. The requested operation is complete.",
        state_updates={**state_update, **file_view_update}
    )
