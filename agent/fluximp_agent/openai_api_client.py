import json
from typing import Optional, Any, Literal
from collections.abc import Mapping

import keyring
import langchain_openai.chat_models.base
import langchain_openai.chat_models._compat
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.messages.base import BaseMessage

from fluximp_agent.model_config import ModelProvider
from fluximp_agent.agent_config import AgentConfig

original_convert_dict_to_message = langchain_openai.chat_models.base._convert_dict_to_message

def _convert_dict_to_message_with_reasoning(_dict: Mapping[str, Any]) -> BaseMessage:
    message = original_convert_dict_to_message(_dict)
    if isinstance(message, AIMessage):
        if "reasoning_content" not in message.additional_kwargs:
            reasoning_content = _dict.get("reasoning_content")
            if reasoning_content:
                message.additional_kwargs["reasoning_content"] = reasoning_content
        if "reasoning" not in message.additional_kwargs:
            reasoning = _dict.get("reasoning")
            if reasoning:
                message.additional_kwargs["reasoning"] = reasoning
    return message

langchain_openai.chat_models.base._convert_dict_to_message = _convert_dict_to_message_with_reasoning

# Monkey-patch the streaming delta converter to extract reasoning_content.
# The default _convert_delta_to_message_chunk only handles content, function_call,
# and tool_calls â it drops reasoning_content and reasoning from the delta,
# so they never appear in AIMessageChunk.additional_kwargs during streaming.
original_convert_delta_to_message_chunk = langchain_openai.chat_models.base._convert_delta_to_message_chunk

def _convert_delta_to_message_chunk_with_reasoning(
    _dict: Mapping[str, Any], default_class: type
) -> BaseMessage:
    message_chunk = original_convert_delta_to_message_chunk(_dict, default_class)
    if isinstance(message_chunk, AIMessageChunk):
        if "reasoning_content" not in message_chunk.additional_kwargs:
            reasoning_content = _dict.get("reasoning_content")
            if reasoning_content:
                message_chunk.additional_kwargs["reasoning_content"] = reasoning_content
        if "reasoning" not in message_chunk.additional_kwargs:
            reasoning = _dict.get("reasoning")
            if reasoning:
                message_chunk.additional_kwargs["reasoning"] = reasoning
    return message_chunk

langchain_openai.chat_models.base._convert_delta_to_message_chunk = _convert_delta_to_message_chunk_with_reasoning

# Monkey-patch the usage metadata converter to handle top-level reasoning_tokens.
# Some providers (e.g., GLM-5) put reasoning_tokens at the top level of the usage
# dict instead of nesting it inside completion_tokens_details. The default
# _create_usage_metadata only looks in completion_tokens_details, so top-level
# reasoning_tokens is silently dropped and never reaches usage_metadata.
original_create_usage_metadata = langchain_openai.chat_models.base._create_usage_metadata

def _create_usage_metadata_with_top_level_reasoning(oai_token_usage, service_tier=None):
    if oai_token_usage and 'reasoning_tokens' in oai_token_usage:
        # If reasoning_tokens is at the top level but not in completion_tokens_details,
        # move it there so _create_usage_metadata can find it
        details = oai_token_usage.get('completion_tokens_details')
        if not details:
            oai_token_usage = dict(oai_token_usage)  # copy to avoid mutating original
            oai_token_usage['completion_tokens_details'] = {
                'reasoning_tokens': oai_token_usage['reasoning_tokens']
            }
        elif isinstance(details, dict) and 'reasoning_tokens' not in details:
            oai_token_usage = dict(oai_token_usage)  # copy to avoid mutating original
            oai_token_usage['completion_tokens_details'] = dict(details)
            oai_token_usage['completion_tokens_details']['reasoning_tokens'] = oai_token_usage['reasoning_tokens']
    return original_create_usage_metadata(oai_token_usage, service_tier)

langchain_openai.chat_models.base._create_usage_metadata = _create_usage_metadata_with_top_level_reasoning

_reasoning_return_fields = None
def set_curated_model(curated_model):
    global _reasoning_return_fields
    assert len(curated_model.reasoning_return_fields) > 0
    _reasoning_return_fields = list(curated_model.reasoning_return_fields)
    _reasoning_return_fields.sort()
    _reasoning_return_fields = tuple(_reasoning_return_fields)
    print("DEBUG _reasoning_return_fields:", _reasoning_return_fields)

original_convert_message_to_dict = langchain_openai.chat_models.base._convert_message_to_dict

def _convert_message_to_dict_with_reasoning(
    message: BaseMessage,
    api: Literal["chat/completions", "responses"] = "chat/completions",
) -> dict:
    message_dict = original_convert_message_to_dict(message, api)
    if isinstance(message, AIMessage):
        reasoning_received_field = None
        reasoning_content = message.additional_kwargs.get("reasoning_content")
        if reasoning_content is None:
            reasoning_content = message.additional_kwargs.get("reasoning")
            if reasoning_content is not None:
                reasoning_received_field = "reasoning"
        else:
            reasoning_received_field = "reasoning_content"

        if reasoning_received_field is not None:
            print("DEBUG _reasoning_return_fields 2:", _reasoning_return_fields)
            if _reasoning_return_fields:
                for f in _reasoning_return_fields:
                    print("DEBUG reasoning returned in field:", f)
                    message_dict[f] = reasoning_content
            else:
                print("DEBUG reasoning returned in field received in:", reasoning_received_field)
                message_dict[reasoning_received_field] = reasoning_content
        else:
            print("DEBUG reasoning_received_field is None")

        # Fix invalid tool_calls.function.arguments
        # LangChain _lc_invalid_tool_call_to_openai_tool_call passes args
        # directly as arguments without JSON serialization.  When the LLM
        # generates a tool call whose arguments are not valid JSON, the raw
        # string is sent to the API which rejects it with:
        #   400 - Invalid tool_calls.function.arguments value, expected JSON
        # We filter out tool calls whose arguments cannot be parsed as JSON
        # so the request is accepted.  Invalid tool calls are never executed
        # (parse_tool_calls only processes message.tool_calls, not
        # message.invalid_tool_calls), so no ToolMessage responses reference
        # their IDs and removing them from the serialized dict is safe.
        if "tool_calls" in message_dict:
            valid_tool_calls = []
            for tool_call in message_dict["tool_calls"]:
                func = tool_call.get("function")
                if func and "arguments" in func:
                    args_value = func["arguments"]
                    if isinstance(args_value, str):
                        try:
                            json.loads(args_value)
                            valid_tool_calls.append(tool_call)
                        except (json.JSONDecodeError, ValueError):
                            # Skip tool calls with invalid JSON arguments
                            pass
                    elif args_value is None:
                        func["arguments"] = "{}"
                        valid_tool_calls.append(tool_call)
                    else:
                        # arguments should be a JSON string; serialize it
                        func["arguments"] = json.dumps(args_value)
                        valid_tool_calls.append(tool_call)
                else:
                    valid_tool_calls.append(tool_call)

            if valid_tool_calls:
                message_dict["tool_calls"] = valid_tool_calls
            else:
                del message_dict["tool_calls"]
                # When tool_calls is removed, content should not be None
                if message_dict.get("content") is None:
                    message_dict["content"] = ""

    return message_dict

langchain_openai.chat_models.base._convert_message_to_dict = _convert_message_to_dict_with_reasoning

original_convert_from_v1_to_chat_completions = langchain_openai.chat_models._compat._convert_from_v1_to_chat_completions

def _convert_from_v1_to_chat_completions_noop(message: AIMessage) -> AIMessage:
    assert not isinstance(message.content, list)
    return message

langchain_openai.chat_models._compat._convert_from_v1_to_chat_completions = _convert_from_v1_to_chat_completions_noop
langchain_openai.chat_models.base._convert_from_v1_to_chat_completions = _convert_from_v1_to_chat_completions_noop

def get_api_key(provider: ModelProvider):
    """Get the API key from environment or secure storage."""
    # Add by running: python -m keyring set <value of provider.name> api_key
    api_key = keyring.get_password(provider.name, "api_key")
    if not api_key:
        raise Exception(f"No api_key for {provider.name}")
    return api_key


class LenientChatOpenAI(ChatOpenAI):
    """ChatOpenAI that handles JSON tool calls in content field.

    Some models (e.g., Mistral-Small-3.1) output tool calls as JSON in the content
    field instead of using the native tool_calls API field. This class extracts
    those JSON tool calls and converts them to the proper format.
    """

    def _create_chat_result(
        self,
        response,
        generation_info: Optional[dict] = None,
    ):
        # Call parent's method first
        result = super()._create_chat_result(response, generation_info)

        # Post-process to extract tool calls from content if needed
        return self._postprocess_chat_result(result)

    def _extract_json_tool_call(self, content: str) -> Optional[dict]:
        """Extract tool call from JSON content, handling markdown code blocks."""
        if not content:
            return None

        content = content.strip()

        # Strip markdown code blocks (handles ```json, ```JSON, or just ```)
        if content.startswith("```") and content.endswith('```'):
            content = content[:-3]
            if content[:7].lower() == "```json":
                content = content[7:]
            else:
                content = content[3:]
            print(f"DEBUG: updated content after removiong ``` block markers:\n{content}")

        try:
            data = json.loads(content)
        except (json.JSONDecodeError, KeyError, TypeError):
            print(f"DEBUG: failed to decode JSON")
            return None

        if isinstance(data, dict):
            print(f"DEBUG: JSON was parsed to dict: {data}")

            if 'arguments' in data:
                args = data['arguments']
            elif 'args' in data:
                args = data['args']
            else:
                args = None

            if 'name' in data:
                tool_name = data['name']
            elif 'tool' in data:
                tool_name = data['tool']
            elif 'tool' in data:
                tool_name = data['tool_usage']
            else:
                tool_name = None

            print(f"DEBUG: {tool_name} {args}")

            if tool_name is not None:
                extracted = {
                    "id": f"call_{hash(content) & 0xFFFFFFFF:08x}",
                    "type": "function",
                    "function": {
                        "name": tool_name,
                    }
                }
                if args is not None:
                    extracted["function"]["arguments"] = json.dumps(args)
                return extracted

        return None

    def _postprocess_message(self, msg):
        """Post-process a single AIMessage to extract tool calls from content if needed."""
        from langchain_core.output_parsers.openai_tools import parse_tool_call

        if isinstance(msg, AIMessage) and not msg.tool_calls and msg.content:
            # Try to extract from content
            tool_call = self._extract_json_tool_call(msg.content)
            print(f"DEBUG extracted tool call: {tool_call}")
            if tool_call:
                # Parse the tool call to match expected format
                try:
                    parsed = parse_tool_call(tool_call, return_id=True)
                    print(f"DEBUG extracted tool call: {parsed}")
                    msg.tool_calls = [parsed]
                    msg.content = ""
                except Exception:
                    print(f"DEBUG parse_tool_call failed")
                    pass
        return msg

    def _postprocess_chat_result(self, result):
        """Post-process to extract tool calls from content if needed."""
        for gen in result.generations:
            msg = gen.message
            print("DEBUG gen.message", msg)
            self._postprocess_message(msg)
        return result


def get_model_client(agent_config: AgentConfig):
    """Get the LangChain ChatOpenAI client configured for synthetic.new API with retry support."""
    # Get API key from secure storage
    api_key = get_api_key(agent_config.model_config.provider)

    client_type = LenientChatOpenAI if agent_config.use_api_protocol_for_tool_functions else ChatOpenAI

    #client_type()

    client = client_type(
        model=agent_config.model_config.model,
        base_url=agent_config.model_config.base_url,
        api_key=api_key,
        max_retries=agent_config.model_config.max_retries,
        timeout=agent_config.model_config.timeout,  # timeout in seconds per request
        temperature=agent_config.model_config.temperature,
        presence_penalty=agent_config.model_config.presence_penalty,
        top_p=agent_config.model_config.top_p,
        max_completion_tokens=agent_config.model_config.max_completion_tokens,
        #reasoning
        reasoning_effort=agent_config.model_config.reasoning_effort,
        extra_body=agent_config.model_config.extra_body,
        streaming=agent_config.model_config.streaming,
        # Include token usage in streaming responses so _invoke_client_streaming
        # can capture prompt_tokens, completion_tokens, reasoning_tokens etc.
        stream_usage=agent_config.model_config.streaming,
        # Might also need
        #model_kwargs
        #frequency_penalty
    )
    return client
