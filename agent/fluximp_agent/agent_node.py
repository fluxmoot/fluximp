"""Base agent node for the LangGraph coding agent.

Contains the BaseAgentNode class which provides common LLM interaction logic
including retry with exponential backoff and response processing.
Also contains shared transformer types used by agent node subclasses.
"""

from __future__ import annotations

import json
import random
import time
from datetime import datetime, timezone
from typing import Optional, Callable, List, Dict, Any, Tuple
from copy import deepcopy

from openai import OpenAIError
import httpx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from fluximp_agent.agent_logger import AgentLogger
from fluximp_agent.openai_api_client import get_model_client, _convert_message_to_dict_with_reasoning
from fluximp_protocol.user_agent.base import MaxRequestsExceeded


# ============================================================================
# Transformer Types
# ============================================================================

class SystemPromptTransformer:
    """Transformer that always returns the system prompt, ignoring input."""
    
    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt
    
    def __call__(self, msg) -> SystemMessage:
        return SystemMessage(content=self.system_prompt)


class DropTransformer:
    """Transformer that always returns None to drop a message.

    `source` records which code path installed the drop (e.g. 'summarization',
    'failed_tool_prune') so _build_api_messages can log where each drop came from.
    """

    def __init__(self, source: str = "unknown"):
        self.source = source

    def __call__(self, msg):
        return None


class ContextInjectionTransformer:
    """Transformer that injects fixed context text into tool result messages.
    
    Unlike the previous implementation that called get_context() dynamically,
    this holds a fixed text string set at creation time. The ContextInjectionManager
    determines what text each instance should inject and where to place them.
    """
    
    def __init__(self, text_to_inject: str):
        self.text_to_inject = text_to_inject
    
    def __call__(self, msg):
        # Only inject context into tool result messages
        is_tool_result = isinstance(msg, ToolMessage)
        
        if not is_tool_result:
            return msg
        
        if self.text_to_inject:
            new_msg = msg.model_copy(deep=True)
            new_msg.content += self.text_to_inject
            return new_msg
        
        return msg


class ContextInjectionSystemTransformer:
    """Transformer that emits the original tool message followed by a system message.
    
    Used when the system_prompt_context_injection experiment is enabled.
    Returns a list [original_tool_message, SystemMessage(context_text)] when
    injecting, or the original message when not. This places context in system
    messages at the same positions where ContextInjectionTransformer would append
    to tool messages, preserving the ContextInjectionManager's prefix caching
    strategy.
    """
    
    def __init__(self, text_to_inject: str):
        self.text_to_inject = text_to_inject
    
    def __call__(self, msg):
        # Only inject context after tool result messages
        is_tool_result = isinstance(msg, ToolMessage)
        
        if not is_tool_result:
            return msg
        
        if self.text_to_inject:
            return [msg, SystemMessage(content=self.text_to_inject)]
        
        return msg



class AgentSpecificSystemTransformer:
    """Transformer that emits the original message followed by an agent-specific system message.
    
    Used for injecting the agent-specific preamble and tool instructions as a 
    follow-on system message when the two_part_system_prompt experiment is active.
    Placed at the first HumanMessage position when there are no tool messages yet.
    """
    
    def __init__(self, content: str):
        self.content = content
    
    def __call__(self, msg):
        if self.content:
            return [msg, SystemMessage(content=self.content)]
        return msg


class AgentSpecificToolTransformer:
    """Transformer that emits the original message followed by a tool-like message.
    
    Used for injecting the agent-specific preamble and tool instructions when the
    two_part_system_prompt experiment is active but system_prompt_context_injection
    is NOT enabled. Placed at the first HumanMessage position when there are no
    tool messages yet. Always emits a ToolMessage; the ToolMessageCompatibilityTransformer
    will convert to HumanMessage if the model doesn't support ToolMessage.
    """
    
    def __init__(self, content: str):
        self.content = content
    
    def __call__(self, msg):
        if self.content:
            tool_msg = ToolMessage(
                content=self.content,
                tool_call_id="agent_context",
                additional_kwargs={
                    'human_prefix': '',
                    'no_tool_use_error': True,
                },
            )
            return [msg, tool_msg]
        return msg



class ToolMessageCompatibilityTransformer:
    """Transformer that converts ToolMessage to HumanMessage for model compatibility.
    
    Some models cannot receive ToolMessage at all (e.g. Gemma-3, Mistral-Small-3.1),
    and some models can receive ToolMessage for normal results but not for 'no tool use'
    errors where there is no corresponding tool call in the previous AIMessage
    (e.g. MiniMax-M2.5, MiniMax-M2.7).
    
    This transformer runs as a second pass in _build_api_messages() after all
    position-specific transformers (context injection, summarization, drops) have
    been applied, so that the summarizer always sees ToolMessage for tool results.
    
    ToolMessages may carry markers in additional_kwargs:
    - 'human_prefix': str - prefix to add when converting to HumanMessage
    - 'no_tool_use_error': bool - True for 'no tool use' errors (no calls, duplicates, excess calls)
    
    Combinations:
    - (True, True): All ToolMessage pass through unchanged
    - (True, False): Only ToolMessages marked with no_tool_use_error=True
      are converted to HumanMessage
    - (False, *): All ToolMessage are converted to HumanMessage
    """
    
    def __init__(self, use_tool_messages: bool, use_tool_message_for_no_tool_error: bool):
        self.use_tool_messages = use_tool_messages
        self.use_tool_message_for_no_tool_error = use_tool_message_for_no_tool_error
    
    def transform_messages(self, messages: list) -> list:
        if self.use_tool_messages and self.use_tool_message_for_no_tool_error:
            # Combination (True, True): all ToolMessage pass through
            return messages
        
        result = []
        for msg in messages:
            if not isinstance(msg, ToolMessage):
                result.append(msg)
                continue
            
            should_convert = False
            if not self.use_tool_messages:
                # Combination (False, *): convert ALL ToolMessage
                should_convert = True
            elif msg.additional_kwargs.get('no_tool_use_error', False):
                # Combination (True, False): convert marked no-tool-use errors
                should_convert = True
            
            if should_convert:
                prefix = msg.additional_kwargs.get('human_prefix', '')
                human_msg = HumanMessage(
                    content=prefix + msg.content,
                    execution_successful=getattr(msg, 'execution_successful', True)
                )
                result.append(human_msg)
            else:
                result.append(msg)
        
        return result


def calculate_message_api_bytes(message):
    """Calculate the UTF-8 byte count of the LLM-visible content of a message.

    Serializes the message exactly as the LLM client will when building the
    OpenAI API request (via the monkey-patched
    _convert_message_to_dict_with_reasoning from openai_api_client) and counts
    the bytes of the string fields that carry the content the model actually
    sees:

    - content: a plain string, or the text of each text part when content is
      a list of content parts
    - tool_calls[].function.name
    - tool_calls[].function.arguments (a JSON string after serialization)
    - reasoning_content / reasoning (added at the message-dict level by the
      openai_api_client monkey-patch from additional_kwargs)

    Structural/metadata fields are skipped: role, tool_calls[].type (always
    'function'), tool_calls[].id and tool_call_id (routing identifiers), and
    the ToolMessage name field (stripped by _convert_message_to_dict for the
    chat/completions API).

    The calculated value is stashed in message.additional_kwargs under
    'api_content_bytes' so downstream consumers (e.g., the inline byte-count
    logging in AgentLogger.log_messages) can read it without re-serializing
    the message.

    Args:
        message: A LangChain BaseMessage. Should be a transformed API message
            (the exact object that will be sent to the LLM) so the count
            reflects drops, summaries, context injection, and compatibility
            transformations applied while building the request.

    Returns:
        Total byte count (int) of the LLM-visible content fields.
    """
    message_dict = _convert_message_to_dict_with_reasoning(message)

    total = 0

    # content: a string, a list of parts (plain strings or dicts), or None
    content = message_dict.get('content')
    if isinstance(content, str):
        total += len(content.encode('utf-8'))
    elif isinstance(content, (list, tuple)):
        for part in content:
            if isinstance(part, str):
                total += len(part.encode('utf-8'))
            elif isinstance(part, dict) and part.get('type') == 'text':
                text = part.get('text')
                if isinstance(text, str):
                    total += len(text.encode('utf-8'))
    # content is None or empty: contributes 0 bytes

    # tool_calls: count the function name and serialized arguments of each call
    for tool_call in message_dict.get('tool_calls') or []:
        func = tool_call.get('function') or {}
        name = func.get('name')
        if isinstance(name, str):
            total += len(name.encode('utf-8'))
        args = func.get('arguments')
        if isinstance(args, str):
            total += len(args.encode('utf-8'))
        elif args is not None:
            # arguments is a JSON string after serialization; defensively
            # serialize dicts/other values if that ever changes
            try:
                total += len(json.dumps(args, ensure_ascii=False).encode('utf-8'))
            except (TypeError, ValueError):
                total += len(str(args).encode('utf-8'))

    # Reasoning fields added at the message-dict level by the monkey-patch
    reasoning_content = message_dict.get('reasoning_content')
    if isinstance(reasoning_content, str):
        total += len(reasoning_content.encode('utf-8'))
    reasoning = message_dict.get('reasoning')
    if isinstance(reasoning, str):
        total += len(reasoning.encode('utf-8'))

    # Stash the computed value so later consumers (e.g., logging) can read it
    # without re-serializing the message
    additional_kwargs = getattr(message, 'additional_kwargs', None)
    if isinstance(additional_kwargs, dict):
        additional_kwargs['api_content_bytes'] = total

    return total

class BaseAgentNode:
    """Base class for agent nodes that call LLMs with retry logic.
    
    Provides common initialization, request counting, LLM invocation with
    transient error retry, and response processing. Subclasses customize
    behavior through hook methods.
    
    Hook methods (override in subclasses):
        _maybe_skip: Return a result dict to skip invocation, or None to proceed.
        _prepare_for_invoke: Called after request limit check, before building messages.
        _build_api_messages: Build the messages list sent to the LLM.
        _log_api_request: Log details about the API request.
        _on_request_start: Called at the start of each LLM request attempt.
        _invoke_client: Invoke the LLM client (allows custom invocation params).
        _process_response: Process a successful LLM response into a state update.
        _make_transient_error_result: Build state update when all retries exhausted.
    """
    
    def __init__(self, filesystem_service, agent_config, registry,
                 max_requests=5, max_transient_retries=8,
                 agent_name=None, model_client=None):
        self.filesystem_service = filesystem_service
        self.agent_config = agent_config
        self.registry = registry
        self.max_requests = max_requests
        self.max_transient_retries = max_transient_retries
        self.agent_name = agent_name or agent_config.agent_type.upper()
        
        # Use provided model client or create one
        if model_client is None:
            client = get_model_client(agent_config)
        else:
            client = model_client
        
        # Bind tools if using API protocol
        self.use_api_protocol = agent_config.use_api_protocol_for_tool_functions
        if self.use_api_protocol:
            client = client.bind_tools(registry.get_langchain_tools())
        
        self.client = client
        
        # Create logger
        self.logger = AgentLogger(agent_config, verbose=True)
        
        # Generate system prompt
        experiments_service = registry._experiments_service
        self.system_prompt = agent_config.generate_system_prompt(
            registry, experiments_service=experiments_service
        )
        
        # Log the system prompt
        self.logger.log_system_prompt(self.system_prompt)
        
        # ToolMessage compatibility transformer for models that don't support ToolMessage
        self._compatibility_transformer = ToolMessageCompatibilityTransformer(
            agent_config.use_tool_messages,
            agent_config.use_tool_message_for_no_tool_error
        )
    
    def __call__(self, state):
        """Process messages and generate a response."""
        # Allow subclass to skip invocation (e.g., mid-turn recovery)
        result = self._maybe_skip(state)
        if result is not None:
            return result
        
        # Check request limit
        request_count = self._check_request_limit(state)
        
        # Ensure filesystem state
        self.filesystem_service.ensure_state()
        
        # Prepare for invocation (subclass hook for pre-processing)
        self._prepare_for_invoke(state, request_count)
        
        # Build API messages
        messages, extra_log_context = self._build_api_messages(state)
        
        # Log API request (subclass hook)
        self._log_api_request(messages, request_count, extra_log_context)
        
        # Call the model with retry logic
        return self._invoke_with_retry(messages, request_count, state)
    
    # ââ Hook methods âââââââââââââââââââââââââââââââââââââââââââââ
    
    def _maybe_skip(self, state):
        """Hook for subclasses to skip invocation. Return None to proceed normally."""
        return None
    
    def _check_request_limit(self, state):
        """Check and return request count, raising MaxRequestsExceeded if exceeded."""
        request_count = state.get("request_count", 0) + 1
        if request_count > self.max_requests:
            raise MaxRequestsExceeded(
                f"{self.agent_name} maximum request limit reached: "
                f"{request_count}/{self.max_requests}"
            )
        print(f"\n[{self.agent_name} API REQUEST {request_count}/{self.max_requests}]")
        return request_count
    
    def _prepare_for_invoke(self, state, request_count):
        """Hook for subclasses to prepare before building API messages.
        
        Called after request limit check and filesystem state ensure.
        Default implementation logs the current state.
        """
        self.logger.log_state(state)
    
    def _build_api_messages(self, state):
        """Build the list of messages to send to the LLM.
        
        Default implementation prepends the system prompt, skips
        the placeholder SystemMessage at index 0, and applies the
        ToolMessage compatibility transformer.
        
        Returns:
            Tuple of (api_messages, extra_log_context) where extra_log_context
            is None by default or a dict of additional info for logging.
        """
        messages = state.get("messages", [])
        api_messages = [SystemMessage(content=self.system_prompt)]
        for msg in messages[1:]:  # Skip index 0 (placeholder SystemMessage)
            api_messages.append(msg)
        
        # Apply ToolMessage compatibility transformer for models that don't support ToolMessage
        api_messages = self._compatibility_transformer.transform_messages(api_messages)
        
        return api_messages, None
    
    def _log_api_request(self, messages, request_count, extra_log_context=None):
        """Log API request details.
        
        Logs the common fields shared by all agent types: agent type, provider,
        model, retry/timeout settings, temperature, streaming, extra_body,
        request count, and message summary.  Subclasses may override to add
        agent-specific logging (e.g. context injection details) and should
        call super()._log_api_request() first.
        """
        self.logger._log_header('API REQUEST')
        self.logger._log(f'agent: {self.agent_config.agent_type}')
        self.logger._log(f'provider: {self.agent_config.model_config.provider.name}')
        self.logger._log(f'model: {self.agent_config.model_config.model}')
        self.logger._log(f'max_retries: {self.agent_config.model_config.max_retries}')
        self.logger._log(f'timeout: {self.agent_config.model_config.timeout}s')
        self.logger._log(f'temperature: {self.agent_config.model_config.temperature}')
        self.logger._log(f'top_p: {self.agent_config.model_config.top_p}')
        self.logger._log(f'max_completion_tokens: {self.agent_config.model_config.max_completion_tokens}')
        self.logger._log(f'presence_penalty: {self.agent_config.model_config.presence_penalty}')
        self.logger._log(f'streaming: {self.agent_config.model_config.streaming}')
        self.logger._log(f'extra_body: {self.agent_config.model_config.extra_body}')
        self.logger._log(f'request count: {request_count}/{self.max_requests}')
        self.logger.log_messages(messages)
        self.logger.flush()
    
    def _on_request_start(self, start_dt):
        """Hook called at the start of each LLM request attempt."""
        pass
    
    def _invoke_client(self, messages):
        """Invoke the LLM client. Override for custom invocation logic."""
        return self.client.invoke(messages)

    def _invoke_client_streaming(self, messages):
        """Invoke the LLM client with streaming, accumulating chunks into a complete response."""
        accumulated = None
        reasoning_content_parts = []
        reasoning_parts = []
        last_token_usage = None
        last_usage_metadata = None

        for chunk in self.client.stream(messages):
            if accumulated is None:
                accumulated = chunk
            else:
                accumulated = accumulated + chunk

            # Accumulate reasoning content (strings that merge_dicts would overwrite)
            if hasattr(chunk, "additional_kwargs") and chunk.additional_kwargs:
                rc = chunk.additional_kwargs.get("reasoning_content")
                if rc:
                    reasoning_content_parts.append(rc)
                r = chunk.additional_kwargs.get("reasoning")
                if r:
                    reasoning_parts.append(r)

            # Capture token usage from response_metadata (typically only in last chunk)
            if hasattr(chunk, "response_metadata") and chunk.response_metadata:
                usage = chunk.response_metadata.get("token_usage")
                if usage:
                    last_token_usage = usage

            # Capture usage_metadata (LangChain structured token usage)
            if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                last_usage_metadata = chunk.usage_metadata

        if accumulated is not None:
            # Set accumulated reasoning content on the final chunk
            # to_message() preserves additional_kwargs, so set before conversion
            if reasoning_content_parts:
                accumulated.additional_kwargs["reasoning_content"] = "".join(reasoning_content_parts)
            if reasoning_parts:
                accumulated.additional_kwargs["reasoning"] = "".join(reasoning_parts)

            # Save response_metadata before to_message() which does NOT preserve it
            saved_response_metadata = accumulated.response_metadata or {}
            if last_token_usage is not None:
                saved_response_metadata["token_usage"] = last_token_usage

            # Ensure we return an AIMessage (not AIMessageChunk)
            if hasattr(accumulated, "to_message"):
                accumulated = accumulated.to_message()

            # Restore response_metadata after to_message() since it's not preserved
            accumulated.response_metadata = saved_response_metadata

            # Restore usage_metadata if present
            if last_usage_metadata is not None:
                accumulated.usage_metadata = last_usage_metadata

            # Normalize token usage from usage_metadata into response_metadata["token_usage"]
            # so _build_log_metadata always has a consistent data source, regardless of
            # whether the response came from streaming or non-streaming invocation.
            if "token_usage" not in accumulated.response_metadata:
                token_usage = self._extract_token_usage(accumulated)
                if token_usage and any(v is not None for v in token_usage.values()):
                    accumulated.response_metadata["token_usage"] = token_usage

            # Apply LenientChatOpenAI post-processing if needed
            if hasattr(self.client, "_postprocess_message"):
                accumulated = self.client._postprocess_message(accumulated)

        return accumulated
    
    def _is_transient_error(self, error, error_msg):
        """Check if an OpenAI error is transient (retryable)."""
        return (
            "429" in error_msg or
            "500" in error_msg or
            "503" in error_msg or
            "maximum capacity" in error_msg.lower() or
            "infrastructure" in error_msg.lower() or
            "temporarily unavailable" in error_msg.lower() or
            (hasattr(error, 'status_code') and error.status_code in (429, 500, 503))
        )
    
    @staticmethod
    def _extract_token_usage(response):
        """Extract and normalize token usage from an LLM response.
        
        Handles both streaming and non-streaming formats:
        - Non-streaming: token_usage in response_metadata (raw API format)
        - Streaming: usage_metadata (LangChain format) when token_usage is absent
        
        LangChain renames some fields in usage_metadata:
        - cached_tokens â cache_read (in input_token_details)
        - reasoning_tokens â reasoning (in output_token_details)
        
        This method maps them back to the standard API names so callers
        always get a consistent format regardless of the response source.
        
        Returns a dict with: prompt_tokens, completion_tokens, total_tokens,
        reasoning_tokens, cached_tokens
        """
        usage = {}
        if hasattr(response, 'response_metadata') and response.response_metadata:
            usage = response.response_metadata.get('token_usage') or {}
        
        # Fall back to usage_metadata when token_usage is not in response_metadata
        # (streaming responses put token counts in usage_metadata instead)
        usage_meta = getattr(response, 'usage_metadata', None)
        if not usage and usage_meta:
            usage = {
                'prompt_tokens': usage_meta.get('input_tokens'),
                'completion_tokens': usage_meta.get('output_tokens'),
                'total_tokens': usage_meta.get('total_tokens'),
            }
            # Map LangChain's renamed fields back to standard API names
            input_details = usage_meta.get('input_token_details')
            if input_details:
                prompt_tokens_details = {}
                if input_details.get('cache_read') is not None:
                    prompt_tokens_details['cached_tokens'] = input_details['cache_read']
                if input_details.get('audio') is not None:
                    prompt_tokens_details['audio_tokens'] = input_details['audio']
                usage['prompt_tokens_details'] = prompt_tokens_details
            output_details = usage_meta.get('output_token_details')
            if output_details:
                completion_tokens_details = {}
                if output_details.get('reasoning') is not None:
                    completion_tokens_details['reasoning_tokens'] = output_details['reasoning']
                if output_details.get('audio') is not None:
                    completion_tokens_details['audio_tokens'] = output_details['audio']
                usage['completion_tokens_details'] = completion_tokens_details
        
        # Extract reasoning_tokens from all possible locations:
        # 1. Top level of token_usage (some providers like GLM-5)
        # 2. Inside completion_tokens_details (standard OpenAI format)
        reasoning_tokens = usage.get('reasoning_tokens')
        if not reasoning_tokens:
            completion_tokens_details = usage.get('completion_tokens_details') or {}
            reasoning_tokens = completion_tokens_details.get('reasoning_tokens')
        
        # Extract cached_tokens from all possible locations:
        # 1. Top level of token_usage (unusual but possible)
        # 2. Inside prompt_tokens_details (standard OpenAI format)
        cached_tokens = usage.get('cached_tokens')
        if cached_tokens is None:
            prompt_tokens_details = usage.get('prompt_tokens_details') or {}
            cached_tokens = prompt_tokens_details.get('cached_tokens')
        
        return {
            'prompt_tokens': usage.get('prompt_tokens'),
            'completion_tokens': usage.get('completion_tokens'),
            'total_tokens': usage.get('total_tokens'),
            'reasoning_tokens': reasoning_tokens,
            'cached_tokens': cached_tokens,
        }
    
    def _build_log_metadata(self, response, start_dt, end_dt):
        """Build the metadata dict for logging an assistant response.
        
        Extracts timing, model, token usage, and other metadata from the
        LLM response. Uses _extract_token_usage for consistent token
        extraction across streaming and non-streaming paths.
        """
        token_usage = self._extract_token_usage(response)
        actual_model = response.response_metadata.get('model_name', 'unknown')
        
        response_metadata = {
            "start_time": start_dt.isoformat(),
            "end_time": end_dt.isoformat(),
            "model": actual_model,
            "finish_reason": response.response_metadata.get('finish_reason'),
            "prompt_tokens": token_usage.get('prompt_tokens'),
            "cached_tokens": token_usage.get('cached_tokens'),
            "completion_tokens": token_usage.get('completion_tokens'),
            "reasoning_tokens": token_usage.get('reasoning_tokens'),
            "total_tokens": token_usage.get('total_tokens'),
            "system_fingerprint": response.response_metadata.get('system_fingerprint'),
            "created": response.response_metadata.get('created'),
        }
        
        elapsed = (end_dt - start_dt).total_seconds()
        return {
            "response_time": f"{elapsed:.2f}s",
            **response_metadata
        }, response_metadata
    
    def _process_response(self, response, request_count, start_dt, end_dt, state):
        """Process a successful LLM response. Returns state update dict.
        
        Default implementation verifies the model name, logs the response,
        and returns an update with messages, request_count, and response_metadata.
        """
        # Verify model name matches what we requested
        requested_model = self.agent_config.model_config.model
        actual_model = response.response_metadata.get('model_name', 'unknown')
        self.agent_config.model_config.provider.verify_model_response(
            requested_model, actual_model
        )
        
        # Log the response with full metadata
        metadata, response_metadata = self._build_log_metadata(response, start_dt, end_dt)
        self.logger.log_assistant_message(response, metadata)
        
        return {
            "messages": [response],
            "request_count": request_count,
            "response_metadata": response_metadata,
        }
    
    def _make_transient_error_result(self, state, error_msg, request_count):
        """Build result dict when all transient error retries are exhausted."""
        error_content = (
            f"INFRASTRUCTURE_ERROR: API unavailable after "
            f"{self.max_transient_retries} retries. {error_msg}"
        )
        self.logger.log_assistant_message(
            error_content,
            {"error": "infrastructure", "retries": self.max_transient_retries}
        )
        return {
            "messages": [AIMessage(content=error_content)],
            "request_count": request_count,
        }
    
    # ââ Core retry logic âââââââââââââââââââââââââââââââââââââââââ
    
    def _invoke_with_retry(self, messages, request_count, state):
        """Call the LLM with transient-error retry. Returns state update dict."""
        base_delay = 5  # seconds
        last_error = None
        
        for attempt in range(self.max_transient_retries):
            try:
                start_dt = datetime.now(timezone.utc)
                self._on_request_start(start_dt)
                
                if getattr(self.agent_config.model_config, 'streaming', False):
                    response = self._invoke_client_streaming(messages)
                else:
                    response = self._invoke_client(messages)
                end_dt = datetime.now(timezone.utc)
                
                return self._process_response(
                    response, request_count, start_dt, end_dt, state
                )
                
            except (OpenAIError, httpx.RemoteProtocolError, AttributeError) as e:
                last_error = e
                error_msg = str(e)
                # Check if this is a transient error
                # Protocol errors (connection drops) are always transient

                if isinstance(last_error, AttributeError):
                    # Apparently this happens when the HTTP response has an empty body (i.e. the 'content-length': '0' header)
                    # DEBUG:openai._base_client:HTTP Response: POST https://api.inference.crusoecloud.com/v1/chat/completions "200 OK" Headers({'date': 'Wed, 22 Jul 2026 18:12:38 GMT', 'content-length': '0', 'connection': 'keep-alive', 'cf-cache-status': 'DYNAMIC', 'server': 'cloudflare', 'cf-ray': 'a1f46a118c6cc62f-SEA'})
                    is_transient = error_msg == "'str' object has no attribute 'model_dump'"
                else:
                    is_transient = isinstance(e, httpx.RemoteProtocolError) or self._is_transient_error(e, error_msg)

                
                if not is_transient:
                    raise  # Re-raise non-transient errors immediately
                
                if attempt == self.max_transient_retries - 1:
                    return self._make_transient_error_result(
                        state, error_msg, request_count
                    )
                
                # Exponential backoff with jitter
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                print(
                    f"[{self.agent_name}] Transient error "
                    f"(attempt {attempt + 1}/{self.max_transient_retries}): "
                    f"{error_msg[:100]}"
                )
                print(f"  Retrying in {delay:.1f}s...")
                time.sleep(delay)
                continue
        
        # Should not reach here, but handle it
        return self._make_transient_error_result(
            state, str(last_error), request_count
        )


