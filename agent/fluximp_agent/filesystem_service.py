#!/usr/bin/env python3
"""
FileSystemService - Encapsulates filesystem operations and working directory state.

This module provides a service class that manages the working directory,
read root directory, write file restriction, path validation, and filesystem
state for tool functions. The working directory restricts write/modify
operations, while the read root directory (which must be an ancestor of or
equal to the working directory) allows read operations on a broader file tree.
When a write file is specified, write access is restricted to only that
single file (which must be within the working directory).
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List


class FileSystemService:
    """Service for managing filesystem operations and working directory state.
    
    This class encapsulates the working directory, read root directory, and
    optional write file restriction, providing path validation to ensure file
    operations stay within the appropriate allowed directories.
    
    - working_dir: restricts write/modify operations (to a subtree)
    - read_root_dir: restricts read/query operations (must be ancestor of or equal to working_dir)
    - write_file: when set, restricts write access to only this single file
                  (must be within working_dir)
    """
    
    def __init__(self, working_dir: str = ".", read_root_dir: Optional[str] = None, write_file: Optional[str] = None):
        """Initialize the filesystem service with working, read root, and write file directories.
        
        Args:
            working_dir: The base directory for all file operations.
                        All file paths will be validated to be within this directory
                        for write/modify operations (unless write_file is set).
            read_root_dir: The root directory for read/query operations. Must be an
                          ancestor of or equal to working_dir. If None, defaults to
                          the same as working_dir (no broader read access).
            write_file: When specified, restricts write access to only this single file.
                       The file path must be within working_dir. If None, write access
                       is allowed to any file within working_dir.
        """
        self.__working_dir = Path(working_dir).resolve()
        if read_root_dir is not None:
            self.__read_root_dir = Path(read_root_dir).resolve()
            # Validate that read_root_dir is an ancestor of or equal to working_dir
            try:
                self.__working_dir.relative_to(self.__read_root_dir)
            except ValueError:
                raise ValueError(
                    f"read_root_dir ({self.__read_root_dir}) must be an ancestor of "
                    f"or equal to working_dir ({self.__working_dir})"
                )
        else:
            self.__read_root_dir = self.__working_dir
        
        if write_file is not None:
            self.__write_file = Path(write_file).resolve()
            # Validate that write_file is within working_dir
            try:
                self.__write_file.relative_to(self.__working_dir)
            except ValueError:
                raise ValueError(
                    f"write_file ({self.__write_file}) must be within "
                    f"working_dir ({self.__working_dir})"
                )
        else:
            self.__write_file = None
    
    def ensure_state(self) -> None:
        """Ensure the OS current working directory is set to the working directory.
        
        This should be called before executing any tool that performs filesystem operations.
        """
        os.chdir(self.__working_dir)
    
    def validate_path(self, file_path: str, access_type: str = 'write') -> Tuple[Optional[Path], Optional[str]]:
        """Validate that a file path is within the allowed directory.
        
        Args:
            file_path: The path provided by the user/tool
            access_type: 'write' to validate against working_dir (default),
                        'read' to validate against read_root_dir
            
        Returns:
            Tuple of (resolved_path, error_message):
            - If valid: (Path object, None)
            - If invalid: (None, formatted error message suitable for LLM context)
        """
        try:
            # Resolve to absolute path
            resolved = (self.__working_dir / file_path).resolve()
            
            # For write access with write_file restriction, only allow that exact file
            if access_type == 'write' and self.__write_file is not None:
                if resolved != self.__write_file:
                    return (
                        None,
                        f"Error: Access denied. Path '{file_path}' is not the allowed write file. "
                        f"Write access is restricted to: {self.__write_file}"
                    )
                return (resolved, None)
            
            # Determine which root to validate against
            if access_type == 'read':
                root_dir = self.__read_root_dir
                dir_label = "the read root directory"
            else:
                root_dir = self.__working_dir
                dir_label = "the working directory"
            
            # Check if path is within the appropriate root directory
            try:
                resolved.relative_to(root_dir)
            except ValueError:
                return (
                    None,
                    f"Error: Access denied. Path '{file_path}' attempts to escape {dir_label}. "
                    f"All file operations must be within: {root_dir}"
                )
            
            return (resolved, None)
            
        except Exception as e:
            return (
                None,
                f"Error: Invalid path '{file_path}'. {str(e)}"
            )
    
    def get_default_paths(self) -> List[str]:
        """Get the default paths for tools that search the filesystem.
        
        Returns:
            List containing the working directory as a string.
        """
        return [str(self.__working_dir)]
    
    @property
    def working_dir(self) -> Path:
        """Get the working directory path.
        
        Returns:
            Path object for the working directory.
        """
        return self.__working_dir
    
    @property
    def read_root_dir(self) -> Path:
        """Get the read root directory path.
        
        The read root directory is the broadest directory that read/query
        operations can access. It is always an ancestor of or equal to
        the working directory.
        
        Returns:
            Path object for the read root directory.
        """
        return self.__read_root_dir
    
    @property
    def write_file(self) -> Optional[Path]:
        """Get the restricted write file path, if any.
        
        When set, write access is restricted to only this single file.
        When None, write access is allowed to any file within working_dir.
        
        Returns:
            Path object for the restricted write file, or None.
        """
        return self.__write_file
    
    def get_default_read_paths(self) -> List[str]:
        """Get the default paths for read-only tools that search the filesystem.
        
        Returns the read root directory, which may be broader than the
        working directory if a separate read_root_dir was configured.
        
        Returns:
            List containing the read root directory as a string.
        """
        return [str(self.__read_root_dir)]
    
    def get_fluximp_dir(self) -> Path:
        """Get (and create if needed) the .fluximp directory.
        
        The .fluximp directory is placed under the read root directory,
        which represents the workspace root. When only --working-dir is
        specified (no --read-root-dir), read_root_dir equals working_dir,
        so .fluximp is placed there as before. When --read-root-dir is
        specified, .fluximp is placed at the higher-level root.
        
        Returns:
            Path object for the .fluximp directory.
        """
        fluximp_dir = self.__read_root_dir / ".fluximp"
        fluximp_dir.mkdir(exist_ok=True)
        return fluximp_dir
    
    def get_fluximp_output_path(self, filename: str) -> Tuple[Optional[Path], Optional[str]]:
        """Get a validated path for an output file within .fluximp.
        
        Args:
            filename: The filename for the output file
            
        Returns:
            Tuple of (Path object, error_message):
            - If valid: (Path object, None)
            - If invalid: (None, formatted error message)
        """
        fluximp_dir = self.get_fluximp_dir()
        output_path = fluximp_dir / filename
        
        # Validate the path is within .fluximp
        try:
            output_path.relative_to(fluximp_dir)
        except ValueError:
            return (None, f"Error: Output path would escape .fluximp directory")
        
        return (output_path, None)
