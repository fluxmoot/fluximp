#!/usr/bin/env python3
"""
File I/O implementation of the user side of the user-agent protocol.

FileIOUserProtocol writes input files and reads output files,
counterpart to FileIOAgentProtocol on the agent side.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Generator, List, Tuple, Any

from fluximp_protocol.user_agent.base import UserProtocol
from fluximp_protocol.user_agent.file_io.base import FileIOProtocolBase
from fluximp_protocol.user_agent.file_io.defs import (
    MessageType,
    ParsedHeader,
    get_input_filename,
    get_end_filename,
    extract_fork_turn_from_filename_strict,
    IOFilenameError,
    monotonic_stamp,
    parse_streamed_records,
)
from fluximp_protocol.user_agent.file_io.wire_format import (
    build_header,
    build_permission_response_header,
    build_cancel_turn_header,
)

# ============================================================================
# FileIOUserProtocol
# =============================================================================

class _PositionedRecordCursor:
    """Context-manager cursor over a positioned wire stream.

    Wraps the stat/seek/open lifecycle around :func:`parse_streamed_records`:

    - An unchanged ``st_size`` (equal to the recorded position) yields no
      records and performs no open/read.
    - A smaller size than the recorded position is corruption and fails fast.
    - A larger size opens the file and parses from the recorded position.

    The cursor yields :class:`StreamedRecord` objects; it does **not** commit
    any position itself. The caller commits a record's ``next_pos`` (via
    ``FileIOUserProtocol.commit_output_position`` / ``commit_input_position``)
    only after it has fully processed that record, so a processing failure
    never silently skips a wire record.
    """

    def __init__(self, path: Path, positions: Dict[Path, int]):
        self._path = path
        self._positions = positions
        self._file = None
        self._iter = None

    def open(self):
        # Fail fast on reuse: a cursor must be explicitly close()d before it is
        # re-opened. Reusing an open cursor would silently leak the prior file
        # handle (the early-return and reopen paths below otherwise orphan it),
        # so reject it rather than accumulate descriptors.
        if self._file is not None:
            raise ValueError(
                f"cursor for {self._path} is already open; close() it before "
                f"re-opening"
            )
        start = self._positions.get(self._path, 0)
        try:
            size = self._path.stat().st_size
        except FileNotFoundError:
            # A first-seen (untracked) missing file yields no records; there is
            # nothing to read yet. A file we had already committed a position
            # for that has since vanished is corruption -- fail fast rather
            # than silently skipping it (which could hide lost records).
            if start > 0:
                raise ValueError(
                    f"corrupt stream: {self._path} disappeared after being "
                    f"read to position {start}"
                )
            return self
        if size < start:
            raise ValueError(
                f"corrupt stream: {self._path} shrank from recorded position "
                f"{start} to size {size}"
            )
        if size == start:
            # No new data -> no open/read.
            return self
        try:
            self._file = open(self._path, 'rb')
            self._iter = parse_streamed_records(self._file, start_pos=start)
        except BaseException:
            if self._file is not None:
                self._file.close()
                self._file = None
            raise
        return self

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None
            self._iter = None

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __iter__(self):
        return self

    def __next__(self):
        if self._iter is None:
            raise StopIteration
        return next(self._iter)

class FileIOUserProtocol(UserProtocol, FileIOProtocolBase):
    """User-side file I/O protocol handler.
    
    Counterpart to FileIOAgentProtocol on the agent side. Manages writing
    input files and reading/parsing output files for the file-based
    communication protocol.
    
    While FileIOAgentProtocol reads .in.txt and writes .out.txt (agent side),
    FileIOUserProtocol writes .in.txt and reads .out.txt (user side).
    
    Both sides share wire format definitions and filename conventions
    from protocol.user_agent.file_io.defs.
    """
    
    def __init__(self, io_path: Path):
        """Initialize the FileIOUserProtocol.
        
        Args:
            io_path: Directory containing the conversation files
        """
        FileIOProtocolBase.__init__(self, io_path)
        # Separate read-position maps for output and input streams. Positions are
        # byte offsets immediately after the last fully *committed* record.
        self._output_file_positions: Dict[Path, int] = {}
        self._input_file_positions: Dict[Path, int] = {}
    
    # --- Input writing (client -> agent) ---
    
    def send_message(self, content: str, fork: int, turn: int) -> Tuple[int, int]:
        """Write user message to input file.
        
        Args:
            content: The message content
            fork: Fork number
            turn: Turn number
            
        Returns:
            Tuple of (fork, turn) where message was sent
            
        Raises:
            ValueError: If the turn already has input
        """
        input_file_path = self.io_path / get_input_filename(fork, turn)
        if input_file_path.exists():
            raise ValueError(f"Turn {fork}.{turn} already exists - cannot send another message")
        
        if not content.endswith('\n'):
            content += '\n'
        payload = content.encode('utf-8')
        header = build_header(payload, MessageType.USER_MESSAGE,
                              monotonic_stamp(str(input_file_path)))
        with open(input_file_path, 'wb') as f:
            f.write(header.encode('utf-8'))
            f.write(payload)
        
        return fork, turn
    
    def cancel_current_turn(self, fork: int, turn: int) -> None:
        """Append cancel_turn header to the turn's existing input file.

        Args:
            fork: Fork number
            turn: Turn number

        Raises:
            FileNotFoundError: If the turn's input file does not exist yet.
                Between turns there is no active turn to cancel; appending
                would silently create a coordinate-shaped input file whose
                only record is a cancel_turn, which the agent-side reader
                would then reject as an invalid input stream. Failing fast
                keeps an incorrect GUI/test call obvious instead.
        """
        # Fail fast when there is no active turn. The '.in.txt' input file
        # exists only from the moment send_message() writes the user_message
        # for this turn. Append mode ('ab') below would otherwise silently
        # create the file with only a cancel_turn header, and the agent-side
        # read_user_message() would then fail with the much less obvious
        # "user_message message expected" error.
        input_file = self.io_path / get_input_filename(fork, turn)
        if not input_file.is_file():
            raise FileNotFoundError(
                f"cannot cancel turn {fork}.{turn}: {input_file.name} does "
                f"not exist (no active turn to cancel)"
            )

        header = build_cancel_turn_header(monotonic_stamp(str(input_file)))

        # Append to existing input file (don't overwrite)
        with open(input_file, 'ab') as f:
            f.write(header.encode('utf-8'))
    
    def send_permission_response(self, fork: int, turn: int, approved: bool, explanation: str = "") -> None:
        """Append permission response to input file.
        
        Args:
            fork: Fork number
            turn: Turn number
            approved: Whether the permission is approved
            explanation: Optional explanation

        Raises:
            FileNotFoundError: If the turn's input file does not exist yet.
                A permission response can only be sent for an in-flight turn:
                the '.in.txt' input file exists only from the moment
                send_message() writes the user_message for this turn.
                Append mode ('ab') below would otherwise silently create a
                coordinate-shaped input file whose first record is a
                tool_permission_response, and the agent-side reader would
                then fail with the much less obvious "user_message message
                expected" error. Failing fast keeps an incorrect GUI/test
                call obvious instead (mirroring cancel_current_turn).
        """
        # Fail fast when there is no active turn, same as cancel_current_turn
        # above: the '.in.txt' input file exists only from the moment
        # send_message() writes the user_message for this turn. Append mode
        # ('ab') below would otherwise silently create the file with only a
        # tool_permission_response record.
        input_file = self.io_path / get_input_filename(fork, turn)
        if not input_file.is_file():
            raise FileNotFoundError(
                f"cannot send permission response for turn {fork}.{turn}: "
                f"{input_file.name} does not exist (no active turn to "
                f"respond to)"
            )
        if not explanation.endswith('\n'):
            explanation += '\n'
        payload = explanation.encode('utf-8')
        header = build_permission_response_header(
            payload, approved, monotonic_stamp(str(input_file)))
        
        # Append to existing input file
        with open(input_file, 'ab') as f:
            f.write(header.encode('utf-8'))
            f.write(payload)
    
    # --- End signal management ---
    
    def create_end_file(self, fork: int, turn: int) -> Path:
        """Create .end file to signal conversation end.
        
        Args:
            fork: Fork number
            turn: Turn number
            
        Returns:
            Path to the created end file
        """
        end_file = self.io_path / get_end_filename(fork, turn)
        end_file.touch()
        return end_file
    
    def remove_end_file(self, fork: int, turn: int) -> None:
        """Remove .end file to allow conversation to continue.
        
        Args:
            fork: Fork number
            turn: Turn number
        """
        end_file = self.io_path / get_end_filename(fork, turn)
        if end_file.exists():
            end_file.unlink()
    
    # --- UserProtocol ABC implementations ---
    
    def create_end_signal(self, fork: int, turn: int) -> None:
        """Create end signal to indicate conversation end.
        
        Implements UserProtocol.create_end_signal.
        Delegates to create_end_file.
        """
        self.create_end_file(fork, turn)
    
    def remove_end_signal(self, fork: int, turn: int) -> None:
        """Remove end signal to allow conversation to continue.
        
        Implements UserProtocol.remove_end_signal.
        Delegates to remove_end_file.
        """
        self.remove_end_file(fork, turn)
    
    # --- Output scanning/parsing (agent -> client) ---
    
    def scan_output_files(self) -> List[Tuple[Path, int, int]]:
        """Scan io_dir for output files.
        
        Returns:
            List of (path, fork, turn) tuples for each output file found
        """
        results = []
        if not self.io_path.exists():
            return results
        
        for out_file in self.io_path.glob("*.out.txt"):
            if not out_file.is_file():
                # A directory/special file shaped like a primary filename is not
                # an output stream; skip it exactly like the snapshot scanner.
                continue
            # Skip unrelated (non-coordinate-shaped) files, e.g. "notes.out.txt".
            # Coordinate-shaped but malformed names -- including extra
            # components ("0.1.2.out.txt") -- are kept so the strict parser
            # rejects them below instead of silently skipping them.
            parts = out_file.name.split('.')
            if not (len(parts) >= 4 and parts[-2] == 'out' and parts[-1] == 'txt'):
                continue
            # Coordinate-shaped but malformed names must fail fast, not be
            # silently skipped.
            try:
                fork, turn = extract_fork_turn_from_filename_strict(out_file.name)
            except IOFilenameError as e:
                raise Exception(
                    f"ERROR: malformed output filename: {out_file.name}"
                ) from e
            results.append((out_file, fork, turn))
        
        # Deterministic scan order so throttled/paced parsing and tests are
        # reproducible across ticks (raw glob order is unspecified).
        results.sort(key=lambda x: (x[1], x[2]))
        return results
    
    def scan_input_files(self) -> List[Tuple[Path, int, int]]:
        """Scan io_dir for coordinate-shaped ``*.in.txt`` input files.

        Uses the same ``_is_coordinate_shaped`` gate as output scanning:
        unrelated names (e.g. ``notes.in.txt``) are skipped, while
        coordinate-shaped but malformed names fail fast.

        Returns:
            List of (path, fork, turn) tuples, sorted by (fork, turn).
        """
        results = []
        if not self.io_path.exists():
            return results
        
        for in_file in self.io_path.glob("*.in.txt"):
            if not in_file.is_file():
                # A directory/special file shaped like a primary filename is not
                # an input stream; skip it exactly like the snapshot scanner.
                continue
            parts = in_file.name.split('.')
            if not (len(parts) >= 4 and parts[-2] == 'in' and parts[-1] == 'txt'):
                continue
            try:
                fork, turn = extract_fork_turn_from_filename_strict(in_file.name)
            except IOFilenameError as e:
                raise Exception(
                    f"ERROR: malformed input filename: {in_file.name}"
                ) from e
            results.append((in_file, fork, turn))
        
        results.sort(key=lambda x: (x[1], x[2]))
        return results
    
    def scan_output(self) -> List[Tuple[Any, int, int]]:
        """Scan for new output.
        
        Implements UserProtocol.scan_output.
        Delegates to scan_output_files.
        """
        return self.scan_output_files()
    
    def new_output_records(self, output_file: Path) -> _PositionedRecordCursor:
        """Open a positioned cursor over an output file's unread records.

        Use as a context manager; yields :class:`StreamedRecord` objects and
        commits nothing itself. After fully processing a record, call
        :meth:`commit_output_position` with the record's ``next_pos`` so a
        processing failure never silently skips a wire record.
        """
        return _PositionedRecordCursor(output_file, self._output_file_positions)

    def new_input_records(self, input_file: Path) -> _PositionedRecordCursor:
        """Open a positioned cursor over an input file's unread records.

        Mirrors :meth:`new_output_records` for ``*.in.txt`` streams (including
        conditional ``0.0.in.txt``), using the independent input-position map.
        """
        return _PositionedRecordCursor(input_file, self._input_file_positions)

    def commit_output_position(self, output_file: Path, next_pos: int) -> None:
        """Commit the read position after an output record has been consumed.

        Rejects regressions (a ``next_pos`` smaller than the stored position),
        which would otherwise silently re-consume committed records.
        """
        current = self._output_file_positions.get(output_file, 0)
        if next_pos < current:
            raise ValueError(
                f"commit_output_position regressed: {next_pos} < {current} "
                f"for {output_file}"
            )
        self._output_file_positions[output_file] = next_pos

    def commit_input_position(self, input_file: Path, next_pos: int) -> None:
        """Commit the read position after an input record has been consumed."""
        current = self._input_file_positions.get(input_file, 0)
        if next_pos < current:
            raise ValueError(
                f"commit_input_position regressed: {next_pos} < {current} "
                f"for {input_file}"
            )
        self._input_file_positions[input_file] = next_pos

    def read_new_output(self, output_file: Path) -> Generator[Tuple[ParsedHeader, str, int], None, None]:
        """Incrementally read new messages from an output file.

        Backwards-compatible wrapper over the positioned cursor. For each
        record it commits the position immediately before yielding (legacy
        semantics). New callers should use :meth:`new_output_records` plus
        :meth:`commit_output_position` for commit-after-consume semantics.

        Yields:
            Tuples of (header, payload, next_position).
        """
        with self.new_output_records(output_file) as records:
            for record in records:
                self.commit_output_position(output_file, record.next_pos)
                yield (record.header, record.payload, record.next_pos)

    def get_file_position(self, output_file: Path) -> int:
        """Get current read position for an output file."""
        return self._output_file_positions.get(output_file, 0)

    def set_file_position(self, output_file: Path, position: int) -> None:
        """Set read position for an output file (tests / manual adjustment)."""
        self._output_file_positions[output_file] = position

    def add_to_file_position(self, output_file: Path, delta: int) -> None:
        """Add delta to an output file's read position."""
        current_pos = self._output_file_positions.get(output_file, 0)
        self._output_file_positions[output_file] = current_pos + delta

    def get_input_file_position(self, input_file: Path) -> int:
        """Get current read position for an input file."""
        return self._input_file_positions.get(input_file, 0)

    def set_input_file_position(self, input_file: Path, position: int) -> None:
        """Set read position for an input file."""
        self._input_file_positions[input_file] = position

    def add_to_input_file_position(self, input_file: Path, delta: int) -> None:
        """Add delta to an input file's read position."""
        current_pos = self._input_file_positions.get(input_file, 0)
        self._input_file_positions[input_file] = current_pos + delta
