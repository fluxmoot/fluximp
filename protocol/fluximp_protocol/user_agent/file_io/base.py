#!/usr/bin/env python3
"""
Concrete shared base for the file I/O implementation of the user-agent protocol.

Both FileIOAgentProtocol and FileIOUserProtocol inherit from FileIOProtocolBase,
which provides common io_path, path resolution, and shared file-IO utilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from fluximp_protocol.user_agent.base import InputInfo
from fluximp_protocol.user_agent.file_io.defs import (
    get_input_filename,
    get_output_filename,
    get_end_filename,
)


# ============================================================================
# FileIOInputInfo - File-IO-specific input info with file paths
# ============================================================================

@dataclass
class FileIOInputInfo(InputInfo):
    """Result from checking for new input â file-IO-specific.
    
    Extends InputInfo with the concrete file paths that the file-IO
    implementation needs for handle management and output writing.
    
    Attributes:
        input_file: Path to the input file (None if end file found)
        output_file: Path to the output file (None if end file found)
        end_file: Path to the end file (None if input file found)
    """
    input_file: Optional[Path] = None
    output_file: Optional[Path] = None
    end_file: Optional[Path] = None


# ============================================================================
# FileIOProtocolBase - Concrete shared base for both sides
# ============================================================================

class FileIOProtocolBase:
    """Concrete base class with shared file-IO implementation.
    
    Both FileIOAgentProtocol and FileIOUserProtocol inherit from this.
    Provides common io_path and path resolution methods so both sides
    compute file paths consistently.
    """
    
    def __init__(self, io_path: Path):
        """
        Args:
            io_path: Directory containing the conversation files
        """
        self.io_path = io_path
    
    def input_path(self, fork: int, turn: int) -> Path:
        """Resolve the input file path for a given fork and turn."""
        return self.io_path / get_input_filename(fork, turn)
    
    def output_path(self, fork: int, turn: int) -> Path:
        """Resolve the output file path for a given fork and turn."""
        return self.io_path / get_output_filename(fork, turn)
    
    def end_path(self, fork: int, turn: int) -> Path:
        """Resolve the end signal file path for a given fork and turn."""
        return self.io_path / get_end_filename(fork, turn)
