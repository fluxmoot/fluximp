import re
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timezone

from fluximp_agent.tools.tool_definitions import ToolResult
from fluximp_agent.tools.common import FileOperation


def _apply_unified_diff(diff_text: str, filesystem_service) -> Tuple[bool, Optional[str], Optional[str]]:
    """Apply a unified diff to files.
    
    Parses the unified diff and applies changes to files.
    Supports multiple files in a single patch.
    Validates context and remove lines before applying changes.
    Returns (success, error_message, first_file_path).
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
        current_file = None
        first_file = None
        
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
                    
                    # Track the first file for return value
                    if first_file is None and current_file:
                        first_file = current_file
                    
                    i += 1
                    
                    # Read existing file content
                    if current_file:
                        path, error = filesystem_service.validate_path(current_file)
                        if error:
                            return (False, error, None)
                        
                        if path.exists():
                            try:
                                # Don't keep ends - we'll add newlines when writing
                                old_lines = path.read_text(encoding='utf-8').splitlines(keepends=False)
                            except:
                                old_lines = []
                        else:
                            old_lines = []
                        file_original_lines[current_file] = old_lines
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
        
        # Validate all hunks before applying any changes
        for filepath, hunks in file_hunks.items():
            old_lines = file_original_lines.get(filepath, [])
            
            for hunk_idx, (hunk_old_start, hunk_old_count, hunk_lines) in enumerate(hunks):
                old_idx = hunk_old_start
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
        
        for filepath, hunks in file_hunks.items():
            old_lines = file_original_lines.get(filepath, [])
            new_lines = old_lines.copy()
            
            # Track how line numbers have shifted after each hunk is applied
            # (hunk_original_start, delta) - lines before original_start are unchanged
            applied_deltas: List[Tuple[int, int]] = []
            
            for hunk_old_start, hunk_old_count, hunk_lines in hunks:
                # Calculate the actual position in the current new_lines
                # by applying deltas from previously applied hunks
                actual_start = hunk_old_start
                for orig_start, delta in applied_deltas:
                    if hunk_old_start >= orig_start:
                        actual_start += delta
                
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
                
                new_lines = result_lines
            
            file_modifications[filepath] = new_lines
        
        # Write all modified files
        if not file_modifications:
            return (False, "No file found in patch", None)
        
        for filepath, file_lines in file_modifications.items():
            path, error = filesystem_service.validate_path(filepath)
            if error:
                return (False, error, None)
            
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
                print(f"DEBUG after apply_patch for {filepath}:\n{output_content}")
            except Exception as e:
                return (False, f"Error writing file '{filepath}': {e}", None)
        
        return (True, None, first_file)
        
    except Exception as e:
        return (False, f"Error applying patch: {e}", None)


def apply_patch_tool(diff_text: str, **kwargs) -> ToolResult:
    """Apply a unified diff patch to files - STATE-MODIFYING tool.

    Applies a unified diff format patch to files. Supports multiple files in one patch.
    Stores the patch for reliable undo/redo functionality.

    Args:
        diff_text: Unified diff format patch text

    Returns:
        ToolResult - with success status, content message, and state_updates dict
    """
    filesystem_service = kwargs.get('filesystem_service')
    state_service = kwargs.get('state_service')

    if not diff_text:
        return ToolResult(success=False, content="Error: Empty patch")

    # Apply the patch
    success, error, first_file = _apply_unified_diff(diff_text, filesystem_service)
    if not success:
        print(f"DEBUG _apply_unified_diff failed: {error}")
        return ToolResult(success=False, content=f"Error applying patch: {error}")
    assert error is None

    # Create operation record with the patch
    op = FileOperation(
        operation_type="apply_patch",
        timestamp=datetime.now(timezone.utc).isoformat(),
        file_path=None,
        patch=diff_text  # Store patch for undo/redo
    )

    # Build state update
    state_update = state_service.build_new_file_operation(op)

    return ToolResult(
        success=True,
        content="The patch has been successfully applied. The requested operation is complete.",
        state_updates=state_update
    )
