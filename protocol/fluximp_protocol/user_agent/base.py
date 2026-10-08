#!/usr/bin/env python3
"""
Abstract base classes and shared types for the user-agent protocol.

This module defines the IO-mechanism-agnostic interfaces for both
sides of the user-agent protocol, as well as shared types used
across all implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Optional, Tuple, List, Any, Generator

if TYPE_CHECKING:
    from fluximp_protocol.user_agent.file_io.defs import ParsedHeader
# ============================================================================
# RequestLifecycleHooks
# ============================================================================

class RequestLifecycleHooks:
    """Hooks for request lifecycle events. Override methods as needed."""

    def on_request_start(self, start_time: datetime) -> None:
        """Called immediately before API request is made."""
        pass


class MaxRequestsExceeded(Exception):
    """Raised when the maximum number of API requests is reached."""
    pass


class TurnCancelled(Exception):
    """Raised when a cancel_turn is received while waiting for a permission response."""
    pass


# ============================================================================
# Permission types
# ============================================================================

class PermissionServiceProtocolError(Exception):
    """Fatal error in permission service protocol.
    
    This exception is raised when there is a protocol-level error
    in the permission service that indicates a bug or infrastructure
    problem. It should crash the agent rather than be sent to the model.
    """
    pass


class PermissionResult:
    """Result of a permission request.
    
    Attributes:
        approved: True if permission was granted
        user_explanation: Free-form text explanation from user
    """
    approved: bool
    user_explanation: str
    
    def __init__(self, approved: bool, user_explanation: str):
        self.approved = approved
        self.user_explanation = user_explanation


class ToolPermissionService(ABC):
    """Abstract base class for handling tool permission requests.
    
    Defines the interface for permission request/response flows.
    Concrete implementations handle the actual I/O mechanism
    (e.g., file-based streams, network, mock for testing).
    """
    
    @abstractmethod
    def has_pending_request(self, tool_name: str) -> bool:
        """Check if there is already a pending permission request for this tool.
        
        Args:
            tool_name: Name of the tool to check for pending request
            
        Returns:
            True if a pending request exists for this tool, False otherwise
        """
        pass
    
    @abstractmethod
    def request_permission(self, tool_name: str, request_data: dict) -> PermissionResult:
        """Request permission for a tool execution.

        Implementations write a tool_permission_request to the output stream,
        wait for a tool_permission_response from the input stream, and write the
        matching tool_permission_response_ack themselves before returning.
        During mid-turn recovery they may instead return a prior-approval result
        without writing anything, when the most recent acknowledged decision for
        ``tool_name`` was an approval.

        Args:
            tool_name: Name of the tool requesting permission
            request_data: Dictionary containing tool-specific request data

        Returns:
            PermissionResult with approval status and user explanation
        """
        pass


# ============================================================================
# InputInfo - IO-agnostic result from checking for new input
# ============================================================================

@dataclass
class InputInfo:
    """Result from checking for new input â IO-agnostic.
    
    Attributes:
        fork: Fork number from the input
        turn: Turn number from the input
        is_end_signal: True if this represents an end-of-conversation signal
    """
    fork: int
    turn: int
    is_end_signal: bool = False


# ============================================================================
# AgentProtocol - Abstract interface for the agent side
# ============================================================================

class AgentProtocol(ABC):
    """Abstract interface for the agent side of the user-agent protocol.
    
    Runs on the same machine as the agent process. Receives user input,
    sends agent output. Independent of transport mechanism (files, IPC, etc.).
    """
    
    # --- Lifecycle ---
    
    @abstractmethod
    def on_request_start(self, start_time: datetime) -> None:
        """Called immediately before API request is made."""
        pass
    
    # --- Input ---
    
    @abstractmethod
    def check_for_new_input(self, fork: int, turn: int) -> Optional[InputInfo]:
        """Non-blocking check for new input.
        
        Args:
            fork: Current fork number
            turn: Current turn number
            
        Returns:
            InputInfo if new input found, None if nothing new.
        """
        pass
    
    @abstractmethod
    def check_cancel_turn(self) -> bool:
        """Non-blocking check for cancel_turn.
        
        Returns True if cancel_turn found.
        """
        pass
    
    @abstractmethod
    def read_user_message(self) -> Tuple[str, str]:
        """Read user_message payload.
        
        Returns:
            Tuple of (msg_type, content) where msg_type should be 'user_message'
        """
        pass
    
    # --- Output ---
    
    @abstractmethod
    def write_response(self, content: str, reasoning_content: Optional[str],
                       metadata: dict) -> None:
        """Write response message with LLM content, reasoning, and metadata.
        
        Written immediately when the LLM API response arrives, before tool execution.
        """
        pass
    
    @abstractmethod
    def write_with_metadata(self, payload_type: str, content: str,
                            execution_successful: int = 0,
                            tool_calls: Optional[List[Tuple[str, dict]]] = None) -> None:
        """Write payload (tool_use, tool_use_result, etc.).

        Not a valid channel for ``model_message_to_user``: that message type
        requires ``tags`` and ``format`` header fields and has a dedicated
        5-field writer (:meth:`write_model_message_to_user`); routing it
        here would emit a 3-field header that every reader rejects as
        corrupt.
        """
        pass
    
    @abstractmethod
    def write_model_message_to_user(self, content: str, tags: list, format: str = "plain_text") -> None:
        """Write model_message_to_user with tags and format metadata.
        
        Args:
            content: The message content to send to the user
            tags: List of tags indicating what the message contains
            format: Format of the message content ('plain_text' or 'markdown')
        """
        pass
    
    
    @abstractmethod
    def write_cancel_ack(self) -> None:
        """Write cancel_turn_ack."""
        pass
    
    @abstractmethod
    def write_max_requests_reached(self, message: str) -> None:
        """Write max_requests_reached marker."""
        pass
    
    # --- Permission handling ---
    
    @abstractmethod
    def has_pending_request(self, tool_name: str) -> bool:
        """Check if there is already a pending permission request for this tool."""
        pass
    
    @abstractmethod
    def request_permission(self, tool_name: str, request_data: dict) -> PermissionResult:
        """Request permission for a tool execution.

        Implementations own the full request/response/ack exchange (see
        ToolPermissionService.request_permission).
        """
        pass

    
    def drain_stale_permission_responses(self):
        """Drain stale tool_permission_response messages from the input stream.

        Default no-op. File-based implementations override this to consume
        stale permission responses left over from mid-turn recovery.
        """
        pass

# ============================================================================
# UserProtocol - Abstract interface for the user side
# ============================================================================

class UserProtocol(ABC):
    """Abstract interface for the user side of the user-agent protocol.
    
    Runs on the user's machine. Sends user input, receives agent output.
    Independent of transport mechanism.
    """
    
    # --- Sending messages (user â agent) ---
    
    @abstractmethod
    def send_message(self, content: str, fork: int, turn: int) -> Tuple[int, int]:
        """Send a user message.
        
        Args:
            content: The message content
            fork: Fork number
            turn: Turn number
            
        Returns:
            Tuple of (fork, turn) where message was sent
        """
        pass
    
    @abstractmethod
    def cancel_current_turn(self, fork: int, turn: int) -> None:
        """Cancel the current turn."""
        pass
    
    @abstractmethod
    def send_permission_response(self, fork: int, turn: int,
                                 approved: bool, explanation: str = "") -> None:
        """Send a permission response."""
        pass
    
    # --- End signaling ---
    
    @abstractmethod
    def create_end_signal(self, fork: int, turn: int) -> None:
        """Create end signal to indicate conversation end."""
        pass
    
    @abstractmethod
    def remove_end_signal(self, fork: int, turn: int) -> None:
        """Remove end signal to allow conversation to continue."""
        pass
    
    # --- Output scanning/reading (agent â user) ---
    
    @abstractmethod
    def scan_output(self) -> List[Tuple[Any, int, int]]:
        """Scan for new output.
        
        Returns:
            List of (source, fork, turn) tuples for each output found
        """
        pass
    
    @abstractmethod
    def read_new_output(self, source: Any) -> Generator[Tuple['ParsedHeader', str, int], None, None]:
        """Incrementally read new output (LEGACY compatibility API).

        Backwards-compatible decode of unread output records from ``source``.
        Implementations based on the file I/O protocol (see
        :class:`fluximp_protocol.user_agent.file_io.user.FileIOUserProtocol`)
        back this by the positioned cursor and commit the read position
        *before* yielding each record (legacy semantics).

        New callers should prefer the positioned cursor API instead
        (``FileIOUserProtocol.new_output_records(source)`` together with
        ``commit_output_position``), which yields fully validated
        ``StreamedRecord`` objects without committing any position. The caller
        commits a record's ``next_pos`` only after it has fully processed that
        record, so a processing failure never silently skips a wire record and
        callers can stop at a message budget and resume exactly where they left
        off.

        Args:
            source: Implementation-specific output source

        Yields:
            Tuples of (header, payload, next_position)
            - header: ParsedHeader with structured access to header fields
            - payload: The decoded payload string
            - next_position: Byte position after this message (for incremental reading)
        """
        pass
