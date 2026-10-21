#!/usr/bin/env python3
"""
Shared Tool Definitions

This module contains the authoritative definitions for all available tools.
Both main.py and test files import from here to ensure consistency.
"""

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Any, Dict, Union
from fluximp_agent.agent_config import ToolDef, ArgDef
from fluximp_agent.context_managment_constants import VIEW_BUDGET_DESC
from fluximp_agent.experiments import ExperimentsService


@dataclass
class ToolResult:
    """Return type for all tool functions.
    
    Attributes:
        success: Whether the tool execution succeeded
        content: Display content returned to the model (primarily for read-only tools)
        state_updates: State-modifying fields from this tool execution (file operations, 
                       file view windows, etc.)
    """
    success: bool
    content: str = ""
    state_updates: Dict[str, Any] = field(default_factory=dict)



# ============================================================================
# Example Values for Tool Arguments (used for error messages and documentation)
# ============================================================================
ARG_EXAMPLES: Dict[str, Union[str, int, bool, list]] = {
    # File paths
    'file_path': 'alice/main.py',
    
    # Content
    'content': 'Hello, world!',
    'new_content': 'Replacement text',
    'diff_text': '--- a/file.txt\n+++ b/file.txt\n@@ -1,2 +1,2 @@\n-old\n+new',
    
    # Patterns/lists
    'patterns': ['def \\w+', 'class \\w+'],
    
    # Optional list args
    'paths': ['alice', 'src'],
    'globs': ['*.py', '*.md'],
    
    # Numeric args
    'start_line': 1,
    'end_line': 10,
    'offset': 1,
    'limit': 50,
    'max_count': 100,
    'indent_delta': 1,
    'spaces_per_indent': 4,
    'count_leading_whitespace': 4,
    'whitespace_type': 'space',
    
    # Boolean args
    'line_numbers': True,
    
    # send_message args
    'tags': ['findings_report'],
    'format': 'markdown',
}

def get_tool_definitions(experiments_service: ExperimentsService = None) -> dict:
    """Create and return all tool definitions.
    
    Args:
        experiments_service: Optional ExperimentsService for checking experiment status.
                            Used to conditionally define tools with different signatures.
    
    Returns:
        Dict mapping tool_name to ToolDef instance
    """
    tools = {}
    
    # ripgrep_search
    tools["ripgrep_search"] = ToolDef(
        name="ripgrep_search",
        description="Search for patterns in the content in text files using ripgrep. Supports regex for content patterns and the glob format for path filtering.",
        category="read_only",
        args=[
            ArgDef(name="patterns", arg_type="array", description="List of regex patterns to search for", required=True, tags=frozenset({'query'})),
            ArgDef(name="globs", arg_type="array", description="List of glob patterns to include (e.g., ['*.py', '*.js']).  You must tailor the glob patterns to your current work and avoid excessively broad searches.  Never use ['*']", required=True, tags=frozenset({'query', 'scope'})),
            ArgDef(name="paths", arg_type="array", description="List of directories or files to search in", required=False, default=["."], tags=frozenset({'query', 'scope'})),
            ArgDef(name="max_count", arg_type="integer", description="Maximum number of matches to return", required=False, default=50, tags=frozenset({'config', 'limit'}))
        ],
        func=None,
        is_state_modifying=False
    )
    
    # read_file
    tools["read_file"] = ToolDef(
        name="read_file",
        description="Read file contents with optional line numbering and line range filtering.",
        category="read_only",
        args=[
            ArgDef(name="file_path", arg_type="string", description="Path to the file to read", required=True, tags=frozenset({'target'})),
            ArgDef(name="offset", arg_type="integer", description="Starting line number (1-indexed)", required=False, default=1, tags=frozenset({'position'})),
            ArgDef(name="limit", arg_type="integer", description="Maximum number of lines to read", required=False, default=2500, tags=frozenset({'config', 'limit'})),
            ArgDef(name="line_numbers", arg_type="boolean", description="Whether to include line numbers in output", required=False, default=True, tags=frozenset({'config'}))
        ],
        func=None,
        is_state_modifying=False
    )
    
    # find_files
    tools["find_files"] = ToolDef(
        name="find_files",
        description="Find files matching glob patterns in specified paths.",
        category="read_only",
        args=[
            ArgDef(name="globs", arg_type="array", description="List of glob patterns to match", required=True, tags=frozenset({'query'})),
            ArgDef(name="paths", arg_type="array", description="List of directories to search", required=False, default=["."], tags=frozenset({'query', 'scope'})),
            ArgDef(name="max_count", arg_type="integer", description="Maximum number of files to return", required=False, default=100, tags=frozenset({'config', 'limit'}))
        ],
        func=None,
        is_state_modifying=False
    )
    
    # send_message
    tools["send_message"] = ToolDef(
        name="send_message",
        description="Send a message to the user. This is the only way to communicate with the user.",
        category="communication",
        args=[
            ArgDef(name="tags", arg_type="array", description='''List of string ypted tags indicating what the message contains. Must have one or more of:
  "findings_report" - answers to the users questions, a report on the root cause of a defect, a report on an analaysis or research the user requested, etc.
  "user_feedback_requested" - questions for the user, risks raised or concerns raised that the user could address, etc.
  "complete_plan" - the message contains a fully complete plan that stands on its own
  "plan_update" - the message contains a partial update to the prior complete plan; not valid if there isn't a previous message with the complete_plan kind
  "work_completion_summary" - the agent has completed all planned work and the message includes a summary of the work completed since the last message tagged with work_completion_summary or since work began; the work_completion_summary should not be used in conjuction with the resolved_reported_issues tag as fixing the previously completed work is not a new instance of completing work so the resolved_reported_issues tag should just be used without the work_completion_summary tag
  "resolved_reported_issues" - the agents has fixed and/or resolved the issues with their work that were reported by the user and/or other agents after the work_completion_summary tag was used to signal all work has been completed; a message tagged with resolved_reported_issues should include a summary of each reported issue and what was done to address the issue
                   
  There is no tag that applies to reporting partial progress.  When none of the tags are applicable to the message content that is very stronger indicator that the message should not be sent.
''', required=True, tags=frozenset({'metadata'})),
            ArgDef(name="content", arg_type="string", description="The message content to send to the user", required=True, tags=frozenset({'message'})),
            ArgDef(name="format", arg_type="string", description='Format of the message content: "plain_text" or "markdown" (default: "markdown")', required=False, default='"markdown"', tags=frozenset({'metadata'}))
        ],
        func=None,
        is_state_modifying=False
    )
    
    # delete_file
    tools["delete_file"] = ToolDef(
        name="delete_file",
        description="Delete a file. The file content is preserved for undo operations.",
        category="filesystem",
        args=[
            ArgDef(name="file_path", arg_type="string", description="Path to the file to delete", required=True, tags=frozenset({'target'}))
        ],
        func=None,
        is_state_modifying=True
    )
    
    # write_file
    tools["write_file"] = ToolDef(
        name="write_file",
        description="Write content to a file, creating directories if needed. Overwrites existing files.",
        category="filesystem",
        args=[
            ArgDef(name="file_path", arg_type="string", description="Path to the file to write", required=True, tags=frozenset({'target'})),
            ArgDef(name="content", arg_type="string", description="Content to write to the file", required=True, tags=frozenset({'content'}))
        ],
        func=None,
        is_state_modifying=True
    )
    
    # create_file - Conditional definition based on experiment
    if experiments_service and experiments_service.is_enabled("create_file"):
        tools["create_file"] = ToolDef(
            name="create_file",
            description="Write content to a new file that does not already exist, creating directories if needed. This tool ONLY works for files that do not currently exist and were not deleted during this session. If a file was deleted during this session, it will be automatically restored and you must use edit_file_lines, insert_lines, or move_lines to modify it instead.",
            category="filesystem",
            args=[
                ArgDef(name="file_path", arg_type="string", description="Path to the file to create", required=True, tags=frozenset({'target'})),
                ArgDef(name="content", arg_type="string", description="Content to write to the file", required=True, tags=frozenset({'content'}))
            ],
            func=None,
            is_state_modifying=True
        )
    
    # edit_file_lines - Conditional definition based on experiment
    if experiments_service and experiments_service.is_enabled("edit_line_validation_first_line_whitespace"):
        # Validated version with whitespace control for first line
        tools["edit_file_lines"] = ToolDef(
            name="edit_file_lines",
            description="Edit specific lines in a file. Line numbers are 1-indexed and inclusive. This has the same effect as deleting the lines from start_line up to and including end_line and then inserting the content at start_line. Requires start_line_content and end_line_content to specify the expected content of the start and end lines for validation.",
            category="filesystem",
            args=[
                ArgDef(name="file_path", arg_type="string", description="Path to the file to edit", required=True, tags=frozenset({'target'})),
                ArgDef(name="start_line", arg_type="integer", description="Starting line number (1-indexed, inclusive)", required=True, tags=frozenset({'position'})),
                ArgDef(name="start_line_content", arg_type="string", description="Current content of the start line. Provide only one line not the entire current content for the block.", required=True, tags=frozenset({'validation'})),
                ArgDef(name="end_line", arg_type="integer", description="Ending line number (1-indexed, inclusive)", required=True, tags=frozenset({'position'})),
                ArgDef(name="end_line_content", arg_type="string", description="Expected content of the end line.  Provide only one line not the entire current content for the block.", required=True, tags=frozenset({'validation'})),
                ArgDef(name="new_content", arg_type="string", description="New content to replace the specified lines with any needed whitespace", required=True, tags=frozenset({'content'})),
                ArgDef(name="count_leading_whitespace_first_line", arg_type="integer", description="Exact number of whitespace characters to set at the start of the first line.  If the whitespace matches the leading whitespace of the first line of string provided for new_content then no changes will be made, otherwise the leading whitespace will be modified to match what is provided to count_leading_whitespace_first_line and whitespace_type_first_line.  This is workaround to the leading white space at the beginning of the string new_content string sometimes being stripped by the infrastructure between you and this harness.", required=True, tags=frozenset({'workaround'})),
                ArgDef(name="whitespace_type_first_line", arg_type="string", description="Type of whitespace to use - 'space' or 'tab' (default: 'space') for the first line", required=False, default="space", tags=frozenset({'workaround'}))
            ],
            func=None,
            is_state_modifying=True
        )
    elif experiments_service and experiments_service.is_enabled("edit_line_validation"):
        # Validated version with start_line_content and end_line_content required
        tools["edit_file_lines"] = ToolDef(
            name="edit_file_lines",
            description="Edit specific lines in a file. Line numbers are 1-indexed and inclusive. This has the same effect as deleting the lines from start_line up to and including end_line and then inserting the content at start_line. Requires start_line_content and end_line_content to specify the expected content of the start and end lines for validation.",
            category="filesystem",
            args=[
                ArgDef(name="file_path", arg_type="string", description="Path to the file to edit", required=True, tags=frozenset({'target'})),
                ArgDef(name="start_line", arg_type="integer", description="Starting line number (1-indexed, inclusive)", required=True, tags=frozenset({'position'})),
                ArgDef(name="start_line_content", arg_type="string", description="Current content of the start line. Provide only one line not the entire current content for the block.", required=True, tags=frozenset({'validation'})),
                ArgDef(name="end_line", arg_type="integer", description="Ending line number (1-indexed, inclusive)", required=True, tags=frozenset({'position'})),
                ArgDef(name="end_line_content", arg_type="string", description="Expected content of the end line.  Provide only one line not the entire current content for the block.", required=True, tags=frozenset({'validation'})),
                ArgDef(name="new_content", arg_type="string", description="New content to replace the specified lines with any needed whitespace", required=True, tags=frozenset({'content'}))
            ],
            func=None,
            is_state_modifying=True
        )
    elif experiments_service and experiments_service.is_enabled("edit_line_no_validation"):
        # Original version without content validation
        tools["edit_file_lines"] = ToolDef(
            name="edit_file_lines",
            description="Edit specific lines in a file. Line numbers are 1-indexed and inclusive. This has the same effect as deleting the lines from start_line up to and including end_line and then inserting the content at start_line.",
            category="filesystem",
            args=[
                ArgDef(name="file_path", arg_type="string", description="Path to the file to edit", required=True, tags=frozenset({'target'})),
                ArgDef(name="start_line", arg_type="integer", description="Starting line number (1-indexed, inclusive)", required=True, tags=frozenset({'position'})),
                ArgDef(name="end_line", arg_type="integer", description="Ending line number (1-indexed, inclusive)", required=True, tags=frozenset({'position'})),
                ArgDef(name="new_content", arg_type="string", description="New content to replace the specified lines with any needed whitespace", required=True, tags=frozenset({'content'}))
            ],
            func=None,
            is_state_modifying=True
        )
    
    # apply_patch
    tools["apply_patch"] = ToolDef(
        name="apply_patch",
        description="Apply a unified diff patch to files. Before/after content is stored per file for reliable undo/redo, and the changed regions are automatically added to the views and the cumulative diff.",
        category="filesystem",
        args=[
            ArgDef(name="diff_text", arg_type="string", description="Unified diff patch text to apply", required=True, tags=frozenset({'content'}))
        ],
        func=None,
        is_state_modifying=True
    )
    
    # undo - gated by "undo" experiment
    if experiments_service and experiments_service.is_enabled("undo"):
        tools["undo"] = ToolDef(
            name="undo",
            description="Undo the last file operation (delete_file, edit_file_lines, etc.).",
            category="filesystem",
            args=[],
            func=None,
            is_state_modifying=True
        )
    
    # redo - gated by "redo" experiment
    if experiments_service and experiments_service.is_enabled("redo"):
        tools["redo"] = ToolDef(
            name="redo",
            description="Redo the last undone file operation.",
            category="filesystem",
            args=[],
            func=None,
            is_state_modifying=True
        )
    
    # insert_lines - Conditional definition based on experiment
    if experiments_service and experiments_service.is_enabled("insert_line_validation_first_line_whitespace"):
        # Validated version with whitespace control for first line
        tools["insert_lines"] = ToolDef(
            name="insert_lines",
            description="Insert lines into a file BEFORE the specified line number (1-indexed). Use insert_line=1 to insert at the beginning, or insert_line=total_lines+1 to append at the end. Off-by-one values (total_lines+2) are also accepted and will be treated as appending at the end. Requires anchor_line_content to specify the expected content of the anchor line for validation.",
            category="filesystem",
            args=[
                ArgDef(name="file_path", arg_type="string", description="Path to the file to edit", required=True, tags=frozenset({'target'})),
                ArgDef(name="insert_line", arg_type="integer", description="Line number BEFORE which to insert (1-indexed)", required=True, tags=frozenset({'position'})),
                ArgDef(name="anchor_line_content", arg_type="string", description="Expected content of the anchor line", required=True, tags=frozenset({'validation'})),
                ArgDef(name="new_content", arg_type="string", description="New content to insert (will be split into lines)", required=True, tags=frozenset({'content'})),
                ArgDef(name="count_leading_whitespace_first_line", arg_type="integer", description="Exact number of whitespace characters to set at the start of the first line.  If the whitespace matches the leading whitespace of the first line of string provided for new_content then no changes will be made, otherwise the leading whitespace will be modified to match what is provided to count_leading_whitespace_first_line and whitespace_type_first_line.  This is workaround to the leading white space at the beginning of the string new_content string sometimes being stripped by the infrastructure between you and this harness.", required=True, tags=frozenset({'workaround'})),
                ArgDef(name="whitespace_type_first_line", arg_type="string", description="Type of whitespace to use - 'space' or 'tab' (default: 'space') for the first line", required=False, default="space", tags=frozenset({'workaround'}))
            ],
            func=None,
            is_state_modifying=True
        )
    elif experiments_service and experiments_service.is_enabled("insert_line_validation"):
        # Validated version with anchor_line_content required
        tools["insert_lines"] = ToolDef(
            name="insert_lines",
            description="Insert lines into a file BEFORE the specified line number (1-indexed). Use insert_line=1 to insert at the beginning, or insert_line=total_lines+1 to append at the end. Off-by-one values (total_lines+2) are also accepted and will be treated as appending at the end. Requires anchor_line_content to specify the expected content of the anchor line for validation.",
            category="filesystem",
            args=[
                ArgDef(name="file_path", arg_type="string", description="Path to the file to edit", required=True, tags=frozenset({'target'})),
                ArgDef(name="insert_line", arg_type="integer", description="Line number BEFORE which to insert (1-indexed)", required=True, tags=frozenset({'position'})),
                ArgDef(name="anchor_line_content", arg_type="string", description="Expected content of the anchor line", required=True, tags=frozenset({'validation'})),
                ArgDef(name="new_content", arg_type="string", description="New content to insert (will be split into lines)", required=True, tags=frozenset({'content'}))
            ],
            func=None,
            is_state_modifying=True
        )
    elif experiments_service and experiments_service.is_enabled("insert_lines"):
        # Non-validated version (experiment)
        tools["insert_lines"] = ToolDef(
            name="insert_lines",
            description="Insert lines into a file BEFORE the specified line number (1-indexed). Use insert_line=1 to insert at the beginning, or insert_line=total_lines+1 to append at the end. Off-by-one values (total_lines+2) are also accepted and will be treated as appending at the end.",
            category="filesystem",
            args=[
                ArgDef(name="file_path", arg_type="string", description="Path to the file to edit", required=True, tags=frozenset({'target'})),
                ArgDef(name="insert_line", arg_type="integer", description="Line number BEFORE which to insert (1-indexed)", required=True, tags=frozenset({'position'})),
                ArgDef(name="new_content", arg_type="string", description="New content to insert (will be split into lines)", required=True, tags=frozenset({'content'}))
            ],
            func=None,
            is_state_modifying=True
        )
    
    # move_lines - Conditional definition based on experiment
    if experiments_service and experiments_service.is_enabled("move_line_validation"):
        # Validated version with start_line_content, end_line_content, and anchor_line_content required
        tools["move_lines"] = ToolDef(
            name="move_lines",
            description="Move lines from one location to another within or between files. Removes lines from start_line to end_line in the source file and inserts them BEFORE insert_line in the destination file. If dest_file_path is not provided, lines are moved within the same file. If the destination file does not exist, it will be created with leading blank lines as needed to place the moved content at the requested line. For same-file moves, line numbers refer to the current state of the file before the move. Requires start_line_content and end_line_content to validate the source range, and anchor_line_content to validate the destination insert position.    You must use this tool to move large blocks of code (i.e. refactoring), such as whole functions, whole classes, and large container initialization. This function is purely mechanical so you can't make transcription errors with like you can when supplying the content for write_file and edit_file_lines.",
            category="filesystem",
            args=[
                ArgDef(name="file_path", arg_type="string", description="Source file path", required=True, tags=frozenset({'target'})),
                ArgDef(name="start_line", arg_type="integer", description="First line to move (1-indexed, inclusive)", required=True, tags=frozenset({'position'})),
                ArgDef(name="start_line_content", arg_type="string", description="Expected content of the start line in source.  Provide only one line not the entire current content for the block.", required=True, tags=frozenset({'validation'})),
                ArgDef(name="end_line", arg_type="integer", description="Last line to move (1-indexed, inclusive)", required=True, tags=frozenset({'position'})),
                ArgDef(name="end_line_content", arg_type="string", description="Expected content of the end line in source.  Provide only one line not the entire current content for the block.", required=True, tags=frozenset({'validation'})),
                ArgDef(name="insert_line", arg_type="integer", description="Line number BEFORE which to insert in destination (1-indexed)", required=True, tags=frozenset({'position'})),
                ArgDef(name="anchor_line_content", arg_type="string", description="Expected content of the anchor line in destination", required=True, tags=frozenset({'validation'})),
                ArgDef(name="dest_file_path", arg_type="string", description="Destination file path (defaults to same as source). If the file does not exist, it will be created.", required=False, default=None, tags=frozenset({'target'}))
            ],
            func=None,
            is_state_modifying=True
        )
    elif experiments_service and experiments_service.is_enabled("move_lines"):
        # Non-validated version (experiment)
        tools["move_lines"] = ToolDef(
            name="move_lines",
            description="Move lines from one location to another within or between files. Removes lines from start_line to end_line in the source file and inserts them BEFORE insert_line in the destination file. If dest_file_path is not provided, lines are moved within the same file. If the destination file does not exist, it will be created with leading blank lines as needed to place the moved content at the requested line. For same-file moves, line numbers refer to the current state of the file before the move.  You must use this tool to move large blocks of code (i.e. refactoring), such as whole functions, whole classes, and large container initialization. This function is purely mechanical so you can't make transcription errors with like you can when supplying the content for write_file and edit_file_lines.",
            category="filesystem",
            args=[
                ArgDef(name="file_path", arg_type="string", description="Source file path", required=True, tags=frozenset({'target'})),
                ArgDef(name="start_line", arg_type="integer", description="First line to move (1-indexed, inclusive)", required=True, tags=frozenset({'position'})),
                ArgDef(name="end_line", arg_type="integer", description="Last line to move (1-indexed, inclusive)", required=True, tags=frozenset({'position'})),
                ArgDef(name="insert_line", arg_type="integer", description="Line number BEFORE which to insert in destination (1-indexed)", required=True, tags=frozenset({'position'})),
                ArgDef(name="dest_file_path", arg_type="string", description="Destination file path (defaults to same as source). If the file does not exist, it will be created.", required=False, default=None, tags=frozenset({'target'}))
            ],
            func=None,
            is_state_modifying=True
        )
    
    # adjust_indent - Conditional definition based on experiment
    if experiments_service and experiments_service.is_enabled("adjust_indent_validation"):
        # Validated version with start_line_content and end_line_content required
        tools["adjust_indent"] = ToolDef(
            name="adjust_indent",
            description="Adjust the indent level of a block of code. Changes the indent level of lines from start_line to end_line (1-indexed, inclusive) by indent_delta levels. Negative values decrease indent, positive values increase it. Tab characters count as 1 indent level each. Spaces are grouped by spaces_per_indent to form indent levels, with any remainder rounded up. Mixed tabs and spaces are handled on a per-line basis. Requires start_line_content and end_line_content to validate the source range.",
            category="filesystem",
            args=[
                ArgDef(name="file_path", arg_type="string", description="Source file path", required=True, tags=frozenset({'target'})),
                ArgDef(name="start_line", arg_type="integer", description="First line to adjust (1-indexed, inclusive)", required=True, tags=frozenset({'position'})),
                ArgDef(name="start_line_content", arg_type="string", description="Expected content of the start line in source.  Provide only one line not the entire current content for the block.", required=True, tags=frozenset({'validation'})),
                ArgDef(name="end_line", arg_type="integer", description="Last line to adjust (1-indexed, inclusive)", required=True, tags=frozenset({'position'})),
                ArgDef(name="end_line_content", arg_type="string", description="Expected content of the end line in source.  Provide only one line not the entire current content for the block.", required=True, tags=frozenset({'validation'})),
                ArgDef(name="indent_delta", arg_type="integer", description="Number of indent levels to change (negative = decrease, positive = increase).  This is not the number of spaces, so if you specify indent_delta as 4 with the default of 4 spaces per 16 spaces will be prefixed to each line in the block.", required=True, tags=frozenset({'content'})),
                ArgDef(name="spaces_per_indent", arg_type="integer", description="Number of spaces per indent level (default: 4)", required=False, default=4, tags=frozenset({'content'}))
            ],
            func=None,
            is_state_modifying=True
        )
    elif experiments_service and experiments_service.is_enabled("adjust_indent"):
        # Non-validated version (experiment)
        tools["adjust_indent"] = ToolDef(
            name="adjust_indent",
            description="Adjust the indent level of a block of code. Changes the indent level of lines from start_line to end_line (1-indexed, inclusive) by indent_delta levels. Negative values decrease indent, positive values increase it. Tab characters count as 1 indent level each. Spaces are grouped by spaces_per_indent to form indent levels, with any remainder rounded up. Mixed tabs and spaces are handled on a per-line basis.",
            category="filesystem",
            args=[
                ArgDef(name="file_path", arg_type="string", description="Source file path", required=True, tags=frozenset({'target'})),
                ArgDef(name="start_line", arg_type="integer", description="First line to adjust (1-indexed, inclusive)", required=True, tags=frozenset({'position'})),
                ArgDef(name="end_line", arg_type="integer", description="Last line to adjust (1-indexed, inclusive)", required=True, tags=frozenset({'position'})),
                ArgDef(name="indent_delta", arg_type="integer", description="Number of indent levels to change (negative = decrease, positive = increase)", required=True, tags=frozenset({'content'})),
                ArgDef(name="spaces_per_indent", arg_type="integer", description="Number of spaces per indent level (default: 4)", required=False, default=4, tags=frozenset({'content'}))
            ],
            func=None,
            is_state_modifying=True
        )
    # adjust_indent_exact - Conditional definition based on experiment
    if experiments_service and experiments_service.is_enabled("adjust_indent_exact_validation"):
        # Validated version with start_line_content and end_line_content required
        tools["adjust_indent_exact"] = ToolDef(
            name="adjust_indent_exact",
            description="Set exact leading whitespace on a block of code. Changes the leading whitespace of lines from start_line to end_line (1-indexed, inclusive) to have exactly count_leading_whitespace characters of whitespace_type. Blank lines are left unchanged. Requires start_line_content and end_line_content to validate the source range.",
            category="filesystem",
            args=[
                ArgDef(name="file_path", arg_type="string", description="Source file path", required=True, tags=frozenset({'target'})),
                ArgDef(name="start_line", arg_type="integer", description="First line to adjust (1-indexed, inclusive)", required=True, tags=frozenset({'position'})),
                ArgDef(name="start_line_content", arg_type="string", description="Expected content of the start line in source.  Provide only one line not the entire current content for the block.", required=True, tags=frozenset({'validation'})),
                ArgDef(name="end_line", arg_type="integer", description="Last line to adjust (1-indexed, inclusive)", required=True, tags=frozenset({'position'})),
                ArgDef(name="end_line_content", arg_type="string", description="Expected content of the end line in source.  Provide only one line not the entire current content for the block.", required=True, tags=frozenset({'validation'})),
                ArgDef(name="count_leading_whitespace", arg_type="integer", description="Exact number of whitespace characters to set at the start of each line", required=True, tags=frozenset({'content'})),
                ArgDef(name="whitespace_type", arg_type="string", description="Type of whitespace to use - 'space' or 'tab' (default: 'space')", required=False, default="space", tags=frozenset({'content'}))
            ],
            func=None,
            is_state_modifying=True
        )
    elif experiments_service and experiments_service.is_enabled("adjust_indent_exact"):
        # Non-validated version (experiment)
        tools["adjust_indent_exact"] = ToolDef(
            name="adjust_indent_exact",
            description="Set exact leading whitespace on a block of code. Changes the leading whitespace of lines from start_line to end_line (1-indexed, inclusive) to have exactly count_leading_whitespace characters of whitespace_type. Blank lines are left unchanged.",
            category="filesystem",
            args=[
                ArgDef(name="file_path", arg_type="string", description="Source file path", required=True, tags=frozenset({'target'})),
                ArgDef(name="start_line", arg_type="integer", description="First line to adjust (1-indexed, inclusive)", required=True, tags=frozenset({'position'})),
                ArgDef(name="end_line", arg_type="integer", description="Last line to adjust (1-indexed, inclusive)", required=True, tags=frozenset({'position'})),
                ArgDef(name="count_leading_whitespace", arg_type="integer", description="Exact number of whitespace characters to set at the start of each line", required=True, tags=frozenset({'content'})),
                ArgDef(name="whitespace_type", arg_type="string", description="Type of whitespace to use - 'space' or 'tab' (default: 'space')", required=False, default="space", tags=frozenset({'content'}))
            ],
            func=None,
            is_state_modifying=True
        )
     
    # run_python_script
    tools["run_python_script"] = ToolDef(
        name="run_python_script",
        description="Execute a Python script and return its results. The script must be a .py file within the working directory. Output is redirected to a file in the .fluximp directory.",
        category="process",
        args=[
            ArgDef(name="script_path", arg_type="string", description="Path to the Python script (relative to working directory)", required=True, tags=frozenset({'target'})),
            ArgDef(name="args", arg_type="array", description="List of command-line arguments to pass to the script", required=False, default=[], tags=frozenset({'target'})),
            ArgDef(name="timeout", arg_type="integer", description="Timeout in seconds (default: 10)", required=False, default=10, tags=frozenset({'config'})),
            ArgDef(name="reason", arg_type="string", description="Required explanation for why the script needs to run", required=True, tags=frozenset({'intent'}))
        ],
        func=None,
        is_state_modifying=False
    )
    
    # run_tests
    tools["run_tests"] = ToolDef(
        name="run_tests",
        description="Execute test runners and return results. Tests are run in order of cost (cheapest first), stopping on first failure. Returns 'All tests passed' on success, or diagnostic information on failure.",
        category="process",
        args=[
            ArgDef(name="reason", arg_type="string", description="Required explanation for why tests need to run", required=True, tags=frozenset({'intent'})),
            ArgDef(name="test_identity", arg_type="string", description="All test are run when this is not specified.  Running all of the tests is the desired behavior unless there is very specific reason to run a specific test such as craeting new tests or investigating test failures.  When a test fails its will be provided as part of the diagnostic information.  Never guess at test indentity and instead rely on getting the test identifies from diagnostic information, explicitly from the user, from inspecting the tests source code and configs, or from your actual knowledge of new tests that you have created in this session.", required=False, default=None, tags=frozenset({'scope'}))
        ],
        func=None,
        is_state_modifying=False
    )
    
    # add_file_view
    tools["add_file_view"] = ToolDef(
        name="add_file_view",
        description="Add a file view window to track specific lines in a file. The file contents are read freshly before for every API request to you.  Calling this for lines that are already in view will not refresh them as what you see is already maximally fresh.  Avoid making multiple uses add_file_view for the same file in the same response except when you need to see non-continiguous sections of the file.  Its deterimental to use add_file_view multiple consecutive times with contiguous ranges, since each use generates a separate tool message in this conversation.  One of the most inefficent things you can do is use add_file_view on the same file on consecutive responses, so if your thinking of making using add_file_view again on the same file you just used on it make the value to limit larger so you can get what you need to see and move on.  This harness does not require that you use add_file_view on a block of lines before modifying that block, so if you can see the lines you want to modify now you have no reason to use add_file_view as your seeing those lines as they are in the filesystem now.",
        category="read_only",
        args=[
            ArgDef(name="file_path", arg_type="string", description="Path to the file to open a window to view", required=True, tags=frozenset({'target'})),
            ArgDef(name="offset", arg_type="integer", description="Starting line number (1-indexed) of the file view window", required=False, default=1, tags=frozenset({'position'})),
            ArgDef(name="limit", arg_type="integer", description="Maximum number of lines to include in the file view window", required=False, default=2500, tags=frozenset({'config', 'limit'}))
        ],
        func=None,
        is_state_modifying=False
    )
    
    # close_file_view and close_view - gated by "close_views" experiment
    if experiments_service and experiments_service.is_enabled("close_views"):
        tools["close_file_view"] = ToolDef(
            name="close_file_view",
            description="Close all file view windows for the specified files. Only use this if you're sure you have no need to view the file further such a transient output file.",
            category="read_only",
            args=[
                ArgDef(name="file_paths", arg_type="array", description="List of file paths to close view windows for", required=True, tags=frozenset({'target'}))
            ],
            func=None,
            is_state_modifying=False
        )
        
        tools["close_view"] = ToolDef(
            name="close_view",
            description=f"Close views by their IDs (shown in [id] in view headers). Works for both file views (all view windows for the file) and search/list query views. Its not necessary to use this.  When view capacity (a view content budget of {VIEW_BUDGET_DESC}) is exceeded the oldest views will be automatically closed.  That said if you really don't need to see what's in a view (i.e. a file view window on a transient output file) you should close it to keep the context tight to both save the user money on token costs and avoid context rot degrading your performance.  Using close_view on a view id for a file after used add_file_view on that file in the same request is a very braindead, so just don't do that.  It is definitely not productive to call close_view frequently such as on every other response.",
            category="read_only",
            args=[
                ArgDef(name="view_ids", arg_type="array", description="List of view IDs to close", required=True, tags=frozenset({'internal_id'}))
            ],
            func=None,
            is_state_modifying=False
        )
    
    # tavily_search tool - gated by "tavily_search" experiment
    if experiments_service and experiments_service.is_enabled("tavily_search"):
        tools["tavily_search"] = ToolDef(
            name="tavily_search",
            description=(
                "Search the web using Tavily's AI-optimized search API. "
                "Returns results with clean content snippets optimized for LLMs. "
                "Raw JSON response is saved to a file and added to your file view windows. "
                "Use include_answer=True to get an AI-generated summary of results. "
                "Use search_depth='advanced' for deeper, more thorough search. "
                "Use topic='news' for recent news results."
            ),
            category="filesystem_modify",
            args=[
                ArgDef(name="query", arg_type="string",
                       description="The search query string",
                       required=True),
                ArgDef(name="max_results", arg_type="integer",
                       description="Maximum number of results to return (default: 5)",
                       required=False),
                ArgDef(name="search_depth", arg_type="string",
                       description="Search depth: 'basic' (faster) or 'advanced' (thorough, default: 'basic')",
                       required=False),
                ArgDef(name="topic", arg_type="string",
                       description="Search topic: 'general' or 'news' (default: 'general')",
                       required=False),
                ArgDef(name="include_answer", arg_type="boolean",
                       description="Whether to include an AI-generated answer summarizing results (default: false)",
                       required=False),
                ArgDef(name="include_raw_content", arg_type="boolean",
                       description="Whether to include full raw page content for each result (default: false)",
                       required=False),
                ArgDef(name="include_domains", arg_type="array",
                       description='List of domains to include in search results (e.g., ["python.org", "github.com"])',
                       required=False),
                ArgDef(name="exclude_domains", arg_type="array",
                       description="List of domains to exclude from search results",
                       required=False),
            ],
            func=None,
            is_state_modifying=True
        )

    # tavily_extract tool - gated by "tavily_extract" experiment
    if experiments_service and experiments_service.is_enabled("tavily_extract"):
        tools["tavily_extract"] = ToolDef(
            name="tavily_extract",
            description=(
                "Extract clean content from one or more URLs using Tavily's extract API. "
                "Given URLs, retrieves and extracts the main content from those pages. "
                "Raw JSON response is saved to a file and added to your file view windows. "
                "Use extract_depth='advanced' for higher success rate and more data (tables, embedded content). "
                "Use format='text' for plain text instead of markdown (default)."
            ),
            category="filesystem_modify",
            args=[
                ArgDef(name="urls", arg_type="array",
                       description='List of URLs to extract content from (max 20), e.g., ["https://example.com/page1"]',
                       required=True),
                ArgDef(name="extract_depth", arg_type="string",
                       description="Extraction depth: 'basic' (faster) or 'advanced' (higher success rate, more data, default: 'basic')",
                       required=False),
                ArgDef(name="format", arg_type="string",
                       description="Output format: 'markdown' (default) or 'text'",
                       required=False),
                ArgDef(name="include_images", arg_type="boolean",
                       description="Whether to include images extracted from the URLs (default: false)",
                       required=False),
                ArgDef(name="include_favicon", arg_type="boolean",
                       description="Whether to include favicon URL for each result (default: false)",
                       required=False),
                ArgDef(name="include_usage", arg_type="boolean",
                       description="Whether to include credit usage info in response (default: false)",
                       required=False),
                ArgDef(name="query", arg_type="string",
                       description="User intent for reranking extracted content chunks. When provided, chunks_per_source can be used.",
                       required=False),
                ArgDef(name="chunks_per_source", arg_type="integer",
                       description="Number of content chunks per source (1-5). Only used when query is provided.",
                       required=False),
            ],
            func=None,
            is_state_modifying=True
        )

    # exa_search tool - gated by "exa_search" experiment
    if experiments_service and experiments_service.is_enabled("exa_search"):
        tools["exa_search"] = ToolDef(
            name="exa_search",
            description=(
                "Search the web using Exa's neural/semantic search engine. "
                "Exa uses neural embeddings to match semantic intent rather than keywords. "
                "Supports 6 search types from instant (250ms) to deep-reasoning (40s), "
                "and category-specific indexes for companies, research papers, people, news. "
                "Raw JSON response is saved to a file and added to your file view windows. "
                "Use type='instant' or type='fast' for real-time applications. "
                "Use type='deep' or type='deep-reasoning' for multi-step research. "
                "Use category='research paper' or category='company' for domain-specific search. "
                'Use contents={"highlights": True} for token-efficient relevant extracts.'
            ),
            category="filesystem_modify",
            args=[
                ArgDef(name="query", arg_type="string",
                       description="Natural-language search query string",
                       required=True),
                ArgDef(name="num_results", arg_type="integer",
                       description="Maximum number of results to return (default: 10)",
                       required=False),
                ArgDef(name="type", arg_type="string",
                       description="Search type: 'auto' (default, ~1s), 'instant' (~250ms), 'fast' (~450ms), 'deep-lite' (~4s), 'deep' (4-15s), 'deep-reasoning' (12-40s)",
                       required=False),
                ArgDef(name="category", arg_type="string",
                       description="Data category: 'company', 'research paper', 'news', 'personal site', 'financial report', 'people'",
                       required=False),
                ArgDef(name="include_domains", arg_type="array",
                       description='List of domains to include in search results (e.g., ["arxiv.org", "github.com"])',
                       required=False),
                ArgDef(name="exclude_domains", arg_type="array",
                       description="List of domains to exclude from search results",
                       required=False),
                ArgDef(name="start_published_date", arg_type="string",
                       description="ISO 8601 date string; only return results published after this date",
                       required=False),
                ArgDef(name="end_published_date", arg_type="string",
                       description="ISO 8601 date string; only return results published before this date",
                       required=False),
                ArgDef(name="contents", arg_type="object",
                       description='Content options dict, e.g. {"highlights": True} for token-efficient extracts, {"text": True} for full text, {"summary": True} for AI summaries, or False to disable',
                       required=False),
            ],
            func=None,
            is_state_modifying=True
        )

    # tinyfish_fetch tool - gated by "tinyfish_fetch" experiment
    if experiments_service and experiments_service.is_enabled("tinyfish_fetch"):
        tools["tinyfish_fetch"] = ToolDef(
            name="tinyfish_fetch",
            description=(
                "Fetch and extract clean content from URLs using TinyFish's browser-rendered fetch API. "
                "Uses real full-browser rendering (headless Chromium) to handle JavaScript-heavy pages, "
                "dynamic content, and SPAs. Returns clean markdown, HTML, or JSON. "
                "Raw JSON response is saved to a file and added to your file view windows. "
                "This tool is free (150 URLs/min) and uses browser rendering, making it a reliable "
                "fallback for pages where other extraction tools fail."
            ),
            category="filesystem_modify",
            args=[
                ArgDef(name="urls", arg_type="array",
                       description='List of URLs to fetch (max 10 per request), e.g., ["https://example.com/page1"]',
                       required=True),
                ArgDef(name="format", arg_type="string",
                       description="Output format: 'markdown' (default), 'html', or 'json'",
                       required=False),
                ArgDef(name="links", arg_type="boolean",
                       description="Whether to include extracted links in response (default: false)",
                       required=False),
                ArgDef(name="image_links", arg_type="boolean",
                       description="Whether to include image links in response (default: false)",
                       required=False),
                ArgDef(name="ttl", arg_type="integer",
                       description="Cache TTL in seconds (default: 3600)",
                       required=False),
                ArgDef(name="per_url_timeout_ms", arg_type="integer",
                       description="Per-URL fetch timeout in milliseconds (default: 45000)",
                       required=False),
            ],
            func=None,
            is_state_modifying=True
        )

    # exa_contents tool - gated by "exa_contents" experiment
    if experiments_service and experiments_service.is_enabled("exa_contents"):
        tools["exa_contents"] = ToolDef(
            name="exa_contents",
            description=(
                "Extract content from URLs using Exa's Contents API. "
                "Supports highlights (token-efficient relevant extracts, ~10x smaller than full text), "
                "full text, AI summaries, and subpage exploration (lightweight site crawling). "
                "Use highlights=true to scan many URLs efficiently before deciding which warrant full fetch. "
                "Use subpages=N to discover and fetch subpages of each URL. "
                "Raw JSON response is saved to a file and added to your file view windows."
            ),
            category="filesystem_modify",
            args=[
                ArgDef(name="urls", arg_type="array",
                       description='List of URLs to extract content from, e.g., ["https://example.com/page1"]',
                       required=True),
                ArgDef(name="text", arg_type="boolean",
                       description="Include full page text (default: true)",
                       required=False),
                ArgDef(name="highlights", arg_type="boolean",
                       description="Include token-efficient relevant extracts, ~10x smaller than full text (default: false)",
                       required=False),
                ArgDef(name="summary", arg_type="boolean",
                       description="Include AI-generated summary of page content (default: false)",
                       required=False),
                ArgDef(name="subpages", arg_type="integer",
                       description="Number of subpages to explore per URL (lightweight site crawling)",
                       required=False),
                ArgDef(name="subpage_target", arg_type="string",
                       description='Comma-separated subpage paths to target, e.g., "about,contact,blog"',
                       required=False),
            ],
            func=None,
            is_state_modifying=True
        )

    # tinyfish_search tool - gated by "tinyfish_search" experiment
    if experiments_service and experiments_service.is_enabled("tinyfish_search"):
        tools["tinyfish_search"] = ToolDef(
            name="tinyfish_search",
            description=(
                "Search the web using TinyFish's structured search API. "
                "Returns titles, snippets, and URLs designed for LLM consumption. "
                "Free (30 queries/min) with domain_type filters for web, news, and research papers. "
                "Supports search operators like site: and -site: in the query string. "
                "Raw JSON response is saved to a file and added to your file view windows."
            ),
            category="filesystem_modify",
            args=[
                ArgDef(name="query", arg_type="string",
                       description="Search query string. Supports operators: site:docs.python.org, -site:facebook.com",
                       required=True),
                ArgDef(name="domain_type", arg_type="string",
                       description="Filter: 'web' (default), 'news' (includes publisher+date), or 'research_paper' (includes authors+venue+citations)",
                       required=False),
                ArgDef(name="page", arg_type="integer",
                       description="Page number (0-10, default: 0)",
                       required=False),
                ArgDef(name="recency_minutes", arg_type="integer",
                       description="Freshness window in minutes (1-5256000). Cannot combine with date filters.",
                       required=False),
                ArgDef(name="after_date", arg_type="string",
                       description="YYYY-MM-DD; only results after this date. Must be <= before_date.",
                       required=False),
                ArgDef(name="before_date", arg_type="string",
                       description="YYYY-MM-DD; only results before this date.",
                       required=False),
                ArgDef(name="location", arg_type="string",
                       description="Geo-targeting code (e.g., 'US', 'GB')",
                       required=False),
                ArgDef(name="language", arg_type="string",
                       description="Language code (e.g., 'en', 'fr')",
                       required=False),
            ],
            func=None,
            is_state_modifying=True
        )

    # linkup_search tool - gated by "linkup_search" experiment
    if experiments_service and experiments_service.is_enabled("linkup_search"):
        tools["linkup_search"] = ToolDef(
            name="linkup_search",
            description=(
                "Search the web using Linkup agentic search API. "
                "Offers 3 depth modes: fast (<1s, keyword only), standard (1-3s, agentic), "
                "and deep (5-30s, multi-iteration agentic search-and-scrape). "
                "Offers 3 output types: searchResults (ranked URLs), "
                "sourcedAnswer (answer with citations), and structured (JSON matching schema). "
                "The deep mode performs multi-iteration search-and-scrape with evaluation, "
                "fundamentally different from keyword or semantic search. "
                "Note: deep mode costs 10x more than standard ($0.05 vs $0.005 per call). "
                "Raw JSON response is saved to a file and added to your file view windows."
            ),
            category="filesystem_modify",
            args=[
                ArgDef(name="query", arg_type="string",
                       description="Search query string",
                       required=True),
                ArgDef(name="depth", arg_type="string",
                       description="Search depth: fast (keyword only, <1s), standard (agentic, 1-3s), deep (multi-iteration agentic, 5-30s, costs 10x more)",
                       required=False),
                ArgDef(name="output_type", arg_type="string",
                       description="Output type: searchResults (ranked URLs+snippets), sourcedAnswer (answer+citations), structured (JSON matching schema)",
                       required=False),
                ArgDef(name="structured_output_schema", arg_type="object",
                       description="JSON schema dict for structured output (required when output_type=structured)",
                       required=False),
                ArgDef(name="include_images", arg_type="boolean",
                       description="Whether to include image results",
                       required=False),
                ArgDef(name="from_date", arg_type="string",
                       description="Start date filter (YYYY-MM-DD)",
                       required=False),
                ArgDef(name="to_date", arg_type="string",
                       description="End date filter (YYYY-MM-DD)",
                       required=False),
                ArgDef(name="include_domains", arg_type="array",
                       description="List of domains to restrict search to",
                       required=False),
                ArgDef(name="exclude_domains", arg_type="array",
                       description="List of domains to exclude from search",
                       required=False),
                ArgDef(name="max_results", arg_type="integer",
                       description="Maximum number of results to return",
                       required=False),
                ArgDef(name="include_inline_citations", arg_type="boolean",
                       description="For sourcedAnswer: include inline citations in the answer",
                       required=False),
                ArgDef(name="include_sources", arg_type="boolean",
                       description="For structured: include sources (changes response format)",
                       required=False),
            ],
            func=None,
            is_state_modifying=True
        )

    # linkup_fetch tool - gated by "linkup_fetch" experiment
    if experiments_service and experiments_service.is_enabled("linkup_fetch"):
        tools["linkup_fetch"] = ToolDef(
            name="linkup_fetch",
            description=(
                "Fetch and extract clean content from a URL using Linkup fetch API. "
                "Returns clean markdown with optional raw HTML and image extraction. "
                "Supports JavaScript rendering for dynamic pages (enabled by default). "
                "Accepts a single URL (not a list). "
                "Raw JSON response is saved to a file and added to your file view windows."
            ),
            category="filesystem_modify",
            args=[
                ArgDef(name="url", arg_type="string",
                       description="URL to fetch (single URL, not a list)",
                       required=True),
                ArgDef(name="render_js", arg_type="boolean",
                       description="Whether to render JavaScript before extraction (default: true, recommended for agentic pipelines)",
                       required=False),
                ArgDef(name="include_raw_html", arg_type="boolean",
                       description="Whether to include raw HTML in response (default: false)",
                       required=False),
                ArgDef(name="extract_images", arg_type="boolean",
                       description="Whether to extract images from the page (default: false)",
                       required=False),
            ],
            func=None,
            is_state_modifying=True
        )

    if False:
        # parallel_search tool - gated by "parallel_search" experiment
        if experiments_service and experiments_service.is_enabled("parallel_search"):
            tools["parallel_search"] = ToolDef(
                name="parallel_search",
                description=(
                    "Search the web using Parallel's declarative semantic search API. "
                    "Uses a proprietary web index of billions of pages. "
                    "Provide keyword search_queries (2-3 concise queries, 3-6 words each) "
                    "plus an optional natural-language objective that describes your goal. "
                    "Returns ranked URLs with token-dense compressed excerpts. "
                    "Use mode='turbo' for fastest, mode='basic' for low latency, "
                    "or mode='advanced' (default) for highest quality. "
                    "Use session_id to link search and extract calls. "
                    "Raw JSON response is saved to a file and added to your file view windows."
                ),
                category="filesystem_modify",
                args=[
                    ArgDef(name="search_queries", arg_type="array",
                        description='List of 2-3 concise keyword queries, 3-6 words each, e.g., ["python async guide", "asyncio tutorial"]',
                        required=True),
                    ArgDef(name="objective", arg_type="string",
                        description="Natural-language description of your underlying goal or question (max 5000 chars)",
                        required=False),
                    ArgDef(name="mode", arg_type="string",
                        description="Search mode: turbo (fastest), basic (low latency), advanced (highest quality, default)",
                        required=False),
                    ArgDef(name="max_chars_total", arg_type="integer",
                        description="Upper bound on total characters across all excerpts",
                        required=False),
                    ArgDef(name="session_id", arg_type="string",
                        description="Session ID to link search and extract calls for better contextual results",
                        required=False),
                ],
                func=None,
                is_state_modifying=True
            )

        # parallel_extract tool - gated by "parallel_extract" experiment
        if experiments_service and experiments_service.is_enabled("parallel_extract"):
            tools["parallel_extract"] = ToolDef(
                name="parallel_extract",
                description=(
                    "Extract content from URLs using Parallel's objective-driven extraction API. "
                    "Extracts content from specific URLs with optional objective-driven excerpts. "
                    "By default returns ranked excerpts; set full_content=true for full page markdown. "
                    "Handles JavaScript-rendered pages and PDFs. Up to 20 URLs per request. "
                    "Use objective to focus excerpts on relevant content. "
                    "Use session_id to link search and extract calls. "
                    "Raw JSON response is saved to a file and added to your file view windows."
                ),
                category="filesystem_modify",
                args=[
                    ArgDef(name="urls", arg_type="array",
                        description='List of URLs to extract content from (up to 20), e.g., ["https://example.com/page1"]',
                        required=True),
                    ArgDef(name="objective", arg_type="string",
                        description="Natural-language extraction goal. When provided, excerpts are focused on relevant content.",
                        required=False),
                    ArgDef(name="search_queries", arg_type="array",
                        description="Optional keyword queries to focus excerpts",
                        required=False),
                    ArgDef(name="full_content", arg_type="boolean",
                        description="If true, also return full page markdown content (default: false, excerpts only)",
                        required=False),
                    ArgDef(name="max_chars_total", arg_type="integer",
                        description="Upper bound on total excerpt characters",
                        required=False),
                    ArgDef(name="session_id", arg_type="string",
                        description="Session ID to link search and extract calls for better contextual results",
                        required=False),
                ],
                func=None,
                is_state_modifying=True
            )

    # perplexity_search tool - gated by "perplexity_search" experiment
    if experiments_service and experiments_service.is_enabled("perplexity_search"):
        tools["perplexity_search"] = ToolDef(
            name="perplexity_search",
            description=(
                "Search the web using Perplexity raw web search API. "
                "Returns ranked URLs with snippets. Supports multiple search modes: "
                "academic papers, SEC filings, and people search. "
                "Use search_mode='academic' for academic papers, "
                "search_mode='sec' for SEC filings, "
                "or search_type='people' for people search (supports >20 results). "
                "Use search_context_size to control content per result (low/medium/high). "
                "Supports multi-query search via query as a list of strings. "
                "Raw JSON response is saved to a file and added to your file view windows."
            ),
            category="filesystem_modify",
            args=[
                ArgDef(name="query", arg_type="string",
                       description="Search query string, or list of strings for multi-query search",
                       required=True),
                ArgDef(name="search_mode", arg_type="string",
                       description="Search mode: 'web' (general), 'academic' (academic papers), 'sec' (SEC filings)",
                       required=False),
                ArgDef(name="search_type", arg_type="string",
                       description="Search type: 'web' (general) or 'people' (people search, supports >20 results)",
                       required=False),
                ArgDef(name="search_context_size", arg_type="string",
                       description="Content per result: 'low', 'medium', or 'high'",
                       required=False),
                ArgDef(name="max_results", arg_type="integer",
                       description="Maximum number of results to return (default: 20 for web, higher for people)",
                       required=False),
                ArgDef(name="search_domain_filter", arg_type="array",
                       description="List of domains to include in search results",
                       required=False),
                ArgDef(name="search_recency_filter", arg_type="string",
                       description="Recency filter: 'hour', 'day', 'week', 'month', or 'year'",
                       required=False),
                ArgDef(name="search_language_filter", arg_type="array",
                       description="List of language filters",
                       required=False),
                ArgDef(name="country", arg_type="string",
                       description="Country code for geo-targeting (e.g., 'us', 'gb')",
                       required=False),
            ],
            func=None,
            is_state_modifying=True
        )

    # firecrawl_search tool - gated by "firecrawl_search" experiment
    if experiments_service and experiments_service.is_enabled("firecrawl_search"):
        tools["firecrawl_search"] = ToolDef(
            name="firecrawl_search",
            description=(
                "Search the web using Firecrawl's search API. "
                "Returns web, news, and image results from Firecrawl's index. "
                "Use 'site:example.com' in the query to limit results to a domain. "
                "Raw JSON response is saved to a file and added to your file view windows. "
                "Use sources=['web', 'news'] to search multiple sources in one call. "
                "Use categories=['research'] for academic results or ['pdf'] for PDF-focused results."
            ),
            category="filesystem_modify",
            args=[
                ArgDef(name="query", arg_type="string",
                       description="Search query string. Use 'site:example.com' to limit results to a domain.",
                       required=True),
                ArgDef(name="sources", arg_type="array",
                       description="List of sources to search: 'web', 'news', 'images' (default: web)",
                       required=False),
                ArgDef(name="categories", arg_type="array",
                       description="List of categories to filter results: 'developer', 'gov', 'research', 'pdf'",
                       required=False),
                ArgDef(name="limit", arg_type="integer",
                       description="Maximum number of results per source (default: 5)",
                       required=False),
                ArgDef(name="tbs", arg_type="string",
                       description="Time-based filter, e.g. 'qdr:d' (past day), 'qdr:w' (past week)",
                       required=False),
                ArgDef(name="location", arg_type="string",
                       description="Localized results, e.g. 'San Francisco,California,United States'",
                       required=False),
                ArgDef(name="ignore_invalid_urls", arg_type="boolean",
                       description="Drop URLs that cannot be scraped by other endpoints (default: false)",
                       required=False),
                ArgDef(name="timeout", arg_type="integer",
                       description="Request timeout in milliseconds (default: 300000)",
                       required=False),
            ],
            func=None,
            is_state_modifying=True
        )

    # firecrawl_scrape tool - gated by "firecrawl_scrape" experiment
    if experiments_service and experiments_service.is_enabled("firecrawl_scrape"):
        tools["firecrawl_scrape"] = ToolDef(
            name="firecrawl_scrape",
            description=(
                "Scrape a single URL using Firecrawl's scrape API. "
                "Returns clean markdown (default) plus any other requested formats. "
                "Handles JavaScript-rendered pages, PDFs, and anti-bot protection server-side. "
                "Accepts a single URL (not a list). "
                "Raw JSON response is saved to a file and added to your file view windows."
            ),
            category="filesystem_modify",
            args=[
                ArgDef(name="url", arg_type="string",
                       description="URL to scrape (single URL, not a list)",
                       required=True),
                ArgDef(name="formats", arg_type="array",
                       description="List of output formats: 'markdown' (default), 'html', 'rawHtml', 'links', 'images', 'screenshot', 'summary'",
                       required=False),
                ArgDef(name="only_main_content", arg_type="boolean",
                       description="Strip nav, footer, and other boilerplate (default: true)",
                       required=False),
                ArgDef(name="wait_for", arg_type="integer",
                       description="Milliseconds to wait for the page to render before scraping",
                       required=False),
                ArgDef(name="max_age", arg_type="integer",
                       description="Return cached content up to this age in milliseconds (avoids redundant scrapes)",
                       required=False),
                ArgDef(name="proxy", arg_type="string",
                       description="Proxy control: 'basic', 'stealth', 'enhanced', 'auto'",
                       required=False),
                ArgDef(name="timeout", arg_type="integer",
                       description="Request timeout in milliseconds",
                       required=False),
            ],
            func=None,
            is_state_modifying=True
        )
    return tools


# Backward compatibility alias
def create_tool_definitions() -> dict:
    """Create and return all tool definitions (backward compatibility)."""
    return get_tool_definitions(None)


def get_tool_definition(name: str, experiments_service: ExperimentsService = None) -> Optional[ToolDef]:
    """Get a specific tool definition by name.
    
    Args:
        name: Tool name to look up
        experiments_service: Optional ExperimentsService for checking experiment status.
                            Used to conditionally define tools with different signatures.
        
    Returns:
        ToolDef instance or None if not found
    """
    tools = get_tool_definitions(experiments_service)
    return tools.get(name)
