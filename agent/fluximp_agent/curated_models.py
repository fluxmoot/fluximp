from dataclasses import dataclass
from typing import FrozenSet

from fluximp_agent import model_config

@dataclass(frozen=True)
class CuratedModel:
    reasoning_return_fields: FrozenSet[str]
    # If True then reasoning must be returned on all fields
    # otherwise any field is fine
    all_fields: bool = False
    def __post_init__(self):
        unexpected_fields = self.reasoning_return_fields - frozenset(("reasoning_content", "reasoning"))
        if unexpected_fields:
            raise ValueError(f"Unexpected fields in reasoning_return_fields: {unexpected_fields}")
        if len(self.reasoning_return_fields) == 0:
            raise ValueError("reasoning_return_fields must have at least one field")
        if self.all_fields:
            raise ValueError("all_fields currently must be False")

reasoning_content_only = CuratedModel(frozenset(("reasoning_content",)))
reasoning_only = CuratedModel(frozenset(("reasoning",)))
both_fields = CuratedModel(frozenset(("reasoning", "reasoning_content")))

curated_model_by_model_name_by_provider_name = {
    model_config.SiliconflowComProvider.name: {
        "deepseek-ai/DeepSeek-V4-Flash-0731": reasoning_content_only,
    },
    model_config.FireworksAIProvider.name: {
        "accounts/fireworks/models/glm-5p3": reasoning_content_only,
        "accounts/fireworks/models/glm-5p3-flash": reasoning_content_only,
    },
    model_config.WandBAIProvider.name: {
        "deepseek-ai/DeepSeek-V4.1-Flash": reasoning_only,
    },
    model_config.DeepinfraComProvider.name: {
        "deepseek-ai/DeepSeek-V4.1-Flash": reasoning_content_only,
        "deepseek-ai/DeepSeek-V4-Pro-0813": reasoning_content_only,
        "deepseek-ai/DeepSeek-V4-Flash-0731": reasoning_content_only,
        "Qwen/Qwen3.8-27B": reasoning_only,
    },
    model_config.TogetherAIProvider.name: {
        "zai-org/GLM-5.3": reasoning_content_only,
        "zai-org/GLM-5.3-Flash": reasoning_content_only,
        "deepseek-ai/DeepSeek-V4.1-Flash": reasoning_content_only,
        "MiniMaxAI/MiniMax-M3": reasoning_content_only,
        "moonshotai/Kimi-K3": reasoning_content_only,
        "Qwen/Qwen3.8-2.4T-A95B": reasoning_content_only,
    },
    model_config.FriendliAIProvider.name: {
        "zai-org/GLM-5.3": reasoning_content_only,
        "zai-org/GLM-5.3-Flash": reasoning_content_only,
    },
    model_config.MakoraComProvider.name: {
        "deepseek-ai/DeepSeek-V4.1-Flash": reasoning_content_only,
        "moonshotai/Kimi-K3": both_fields,
    }
}


def get_curated_provider_names() -> list[str]:
    """Get a sorted list of provider names that have curated models.

    Returns:
        A sorted list of provider name strings (the keys of the curated map).
    """
    return sorted(curated_model_by_model_name_by_provider_name.keys())


def get_curated_model_names(provider_name: str) -> list[str]:
    """Get a sorted list of curated model names for the given provider.

    Returns:
        A sorted list of model identifier strings; empty when the provider
        has no curated models.
    """
    curated_model_by_model_name = curated_model_by_model_name_by_provider_name.get(provider_name)
    if curated_model_by_model_name is None:
        return []
    return sorted(curated_model_by_model_name.keys())


def filter_provider_names(provider_names: list[str], limit_to_curated: bool) -> list[str]:
    """Restrict provider_names to the curated set when limit_to_curated.

    The filter is a pure narrowing (no fallback): an empty result means the
    curated filter removed every provider, and the caller decides how to
    recover (the GUI offers to turn the filter off or exit).
    """
    if not limit_to_curated:
        return list(provider_names)
    curated = set(get_curated_provider_names())
    return [name for name in provider_names if name in curated]


def filter_model_names(provider_name: str, model_names: list[str], limit_to_curated: bool) -> list[str]:
    """Restrict model_names to the provider's curated set when limit_to_curated.

    Intersects with the provider's actual model list so stale curated
    entries (a curated model the provider no longer serves) can't leak in.
    Falls back to the full list when the intersection is empty: a provider
    is only offered because it is curated, so an empty model list would be
    a dead end.
    """
    if not limit_to_curated:
        return list(model_names)
    curated = set(get_curated_model_names(provider_name))
    filtered = [name for name in model_names if name in curated]
    return filtered if filtered else list(model_names)
