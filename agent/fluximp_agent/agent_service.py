#!/usr/bin/env python3
"""
AgentService - Service for running one-shot sub-agent analyses.

This service manages running one-shot sub-agents (scrutinator, summarizer)
via subprocess. It handles creating conversation directories, managing I/O,
and extracting results.
"""

import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

from fluximp_protocol.user_agent.file_io.user import FileIOUserProtocol
from fluximp_agent.process_utils import build_agent_command, terminate_process, build_launch_env
from fluximp_protocol.user_agent.file_io.defs import get_output_filename, parse_output_messages, MessageType
from fluximp_protocol.user_agent.file_io.wire_format import write_system_agent_started, write_system_agent_stopped


class AgentService:
    """Service for running one-shot sub-agent analyses (scrutinator, summarizer).
    
    This service creates unique conversation directories under the parent
    conversation directory and manages running sub-agents.
    """
    
    def __init__(self, io_dir: Path, primary_protocol, model_config=None, scrutinator_timeout: float = 660.0, summarizer_timeout: float = 900.0):
        """Initialize the agent service.
        
        Args:
            io_dir: The parent conversation directory
            primary_protocol: FileIOAgentProtocol of the primary agent.
                Used to write system_agent lifecycle messages to the
                primary agent's output stream.
            model_config: ModelConfig for the LLM (used by scrutinator)
            scrutinator_timeout: Timeout in seconds for scrutinator analysis
            summarizer_timeout: Timeout in seconds for summarizer analysis
        """
        self._io_dir = io_dir
        self._primary_protocol = primary_protocol
        self._scrutinator_timeout = scrutinator_timeout
        self._model_config = model_config
        self._summarizer_timeout = summarizer_timeout
    
    def run_scrutinator(self, diff_content: str) -> str:
        """Run the scrutinator agent on the given diff.
        
        Args:
            diff_content: The unified diff to analyze
            
        Returns:
            The scrutinator report as a string, or error string on failure
        """
        if self._scrutinator_timeout is None:
            print('DEBUG skipping scrutinator as self._scrutinator_timeout is None')
            return ''
        result = self._run_oneshot_agent(
            agent_type='scrutinator',
            subdir_prefix='scrutinator_',
            timeout=self._scrutinator_timeout,
            input_content=diff_content,
            message_type=MessageType.SCRUTINATOR_REPORT,
            max_requests=5,
            model_config=self._model_config,
        )
        if result is None:
            return 'Error: Scrutinator analysis failed'
        return result
    
    
    def run_summarizer(self, formatted_text: str, model_config) -> Optional[str]:
        """Run the summarizer agent on the given formatted text.
        
        Args:
            formatted_text: The formatted message text to summarize
            model_config: ModelConfig for the LLM
            
        Returns:
            The summary text string, or None if the summarizer failed
        """
        return self._run_oneshot_agent(
            agent_type='summarizer',
            subdir_prefix='summarizer_',
            timeout=self._summarizer_timeout,
            input_content=formatted_text,
            message_type=MessageType.SUMMARY,
            max_requests=3,
            model_config=model_config,
        )
    
    def _run_oneshot_agent(self, agent_type, subdir_prefix, timeout, input_content,
                           message_type, max_requests, model_config=None):
        """Run a one-shot agent in a subprocess and extract its output.
        
        Args:
            agent_type: Agent type string (e.g. 'summarizer', 'scrutinator')
            subdir_prefix: Prefix for the unique subdirectory
            timeout: Timeout in seconds (None to skip)
            input_content: The text to send as input
            message_type: MessageType enum value to extract from output
            max_requests: Maximum API requests allowed
            model_config: Optional ModelConfig (if needed for provider/model args)
            
        Returns:
            The output content as a string, or None if the agent failed
        """
        if timeout is None:
            print(f'DEBUG skipping {agent_type} as timeout is None')
            return None
        
        # Create unique subdirectory for this run
        run_id = str(uuid.uuid4())[:8]
        subdir = self._io_dir / f'{subdir_prefix}{run_id}'
        subdir.mkdir(parents=True, exist_ok=True)
        
        # Create client protocol and write input
        protocol = FileIOUserProtocol(subdir)
        protocol.send_message(input_content, 0, 0)
        
        # Build command
        cmd_kwargs = dict(
            agent_type=agent_type,
            io_dir=subdir,
            max_requests=max_requests,
            input_timeout=int(timeout),
        )
        if model_config is not None:
            cmd_kwargs['provider'] = model_config.provider.name
            cmd_kwargs['model'] = model_config.model
        cmd = build_agent_command(**cmd_kwargs)
        
        try:
            print(f'DEBUG AgentService: Starting {agent_type} in {subdir}')
            cmd_str = ' '.join(cmd)
            print(f'DEBUG AgentService: Command: {cmd_str}')
            print(f'DEBUG AgentService: Working dir: {self._io_dir.parent}')
            
            # Open stdout log file
            stdout_log_path = subdir / 'stdout.log'
            stdout_file = open(stdout_log_path, 'w', encoding='utf-8')
            
            # Start the process
            print(f'DEBUG AgentService: sys.executable={sys.executable}')
            process = subprocess.Popen(
                cmd,
                stdout=stdout_file,
                stderr=subprocess.PIPE,
                cwd=str(self._io_dir.parent),
                env=build_launch_env(),
            )
            print(f'DEBUG AgentService: {agent_type} process started with PID {process.pid}')
            
            # Write system_agent_started to primary output
            write_system_agent_started(
                self._primary_protocol._output_handle,
                agent_type,
                subdir.name,
            )
            
            # Poll for output file with timeout
            output_file = subdir / get_output_filename(0, 0)
            start_time = time.time()
            result = None
            
            while time.time() - start_time < timeout:
                if output_file.exists():
                    result = self._extract_output(output_file, message_type)
                    if result is not None:
                        print(f'DEBUG AgentService: received {agent_type} output')
                        protocol.create_end_file(0, 1)
                        break
                    if self._check_terminal_failure(output_file):
                        print(f'DEBUG AgentService: {agent_type} terminated with failure (max_requests_reached or infrastructure_error)')
                        break
                time.sleep(0.1)
            
            # Wait for process to exit
            terminate_process(process, timeout_graceful=10.0)
            
            # Determine success/failure for lifecycle message
            success = result is not None
            error = ""
            if not success:
                if time.time() - start_time >= timeout:
                    error = "timed out"
                else:
                    error = "did not produce output"
            
            # Write system_agent_stopped to primary output
            write_system_agent_stopped(
                self._primary_protocol._output_handle,
                agent_type,
                "true" if success else "false",
                error,
            )
            
            # Close stdout log file and capture stderr
            stdout_file.close()
            stderr_output = process.stderr.read().decode('utf-8') if process.stderr else ''
            if stderr_output:
                print(f'DEBUG AgentService: {agent_type} stderr: {stderr_output}')
            
            if result is None:
                if time.time() - start_time >= timeout:
                    print(f'DEBUG AgentService: {agent_type} timed out after {timeout}s')
                else:
                    print(f'DEBUG AgentService: {agent_type} did not produce output')
                return None
            
            return result
            
        except Exception as e:
            print(f'DEBUG AgentService: Error running {agent_type}: {str(e)}')
            # Write system_agent_stopped with error
            write_system_agent_stopped(
                self._primary_protocol._output_handle,
                agent_type,
                "false",
                str(e),
            )
            return None
    
    def _extract_output(self, output_file: Path, message_type: MessageType) -> Optional[str]:
        """Extract output of a specific message type from an output file.
        
        Args:
            output_file: Path to the output file
            message_type: MessageType to search for
            
        Returns:
            The content as a string, or None if not found
        """
        if not output_file.exists():
            return None

        content_bytes = output_file.read_bytes()

        for header, payload, next_pos in parse_output_messages(content_bytes):
            if header.is_type(message_type):
                return payload

        return None

    def _check_terminal_failure(self, output_file: Path) -> bool:
        """Check if the output file contains a terminal failure message.
        
        Terminal failure messages indicate the agent has stopped without
        producing the expected output type (e.g., MAX_REQUESTS_REACHED,
        INFRASTRUCTURE_ERROR). Detecting these allows the parent process
        to stop waiting immediately rather than polling until the timeout.
        
        Args:
            output_file: Path to the output file
            
        Returns:
            True if a terminal failure message was found, False otherwise
        """
        if not output_file.exists():
            return False

        content_bytes = output_file.read_bytes()

        for header, payload, next_pos in parse_output_messages(content_bytes):
            if header.is_type(MessageType.MAX_REQUESTS_REACHED) or header.is_type(MessageType.INFRASTRUCTURE_ERROR):
                return True

        return False
