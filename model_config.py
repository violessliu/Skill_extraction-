#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model presets for the Lightcast extraction pipeline on Together AI.

改模型只需要动这个文件：换 DEFAULT_MODEL_PRESET，或往 MODEL_PRESETS 里加一条。
To change models, edit this file only: switch DEFAULT_MODEL_PRESET, or add a
new entry to MODEL_PRESETS.

Together's catalogue changes over time. Run the pipeline with --list_models to
print the chat models your account can currently serve, then copy the exact
model string into a preset below.
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass(frozen=True)
class ModelSpec:
    """One LLM configuration.

    model_id:       exact Together model string, e.g. "Qwen/Qwen2.5-72B-Instruct-Turbo"
    json_mode:      send response_format={"type": "json_object"}. Automatically
                    disabled at runtime if the model rejects it.
    chat_template_kwargs:
                    provider-specific chat-template options, such as disabling
                    reasoning for models that otherwise emit thought text.
    strip_thinking: the model emits <think>...</think> reasoning before its
                    answer, which must be removed before JSON parsing.
    """

    model_id: str
    json_mode: bool = True
    stream: bool = False
    chat_template_kwargs: Dict[str, object] = field(
        default_factory=dict
    )
    strip_thinking: bool = False
    temperature: float = 0.0
    max_infer_tokens: int = 3000
    max_match_tokens: int = 2500
    notes: str = ""


MODEL_PRESETS: Dict[str, ModelSpec] = {
    "llama3.3-70b": ModelSpec(
        model_id="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        notes="Balanced default: reliable JSON mode, broad availability.",
    ),
    "llama3.1-8b": ModelSpec(
        model_id="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        notes="Cheapest option. Expect more unaligned evidence spans.",
    ),
    "llama4-maverick": ModelSpec(
        model_id="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        notes="Long context, strong instruction following.",
    ),
    "qwen2.5-72b": ModelSpec(
        model_id="Qwen/Qwen2.5-72B-Instruct-Turbo",
        notes="Closest hosted successor to the original local Qwen3-14B setup.",
    ),
    "qwen2.5-7b": ModelSpec(
        model_id="Qwen/Qwen2.5-7B-Instruct-Turbo",
        notes="Small Qwen for smoke tests.",
    ),
    "qwen3-235b": ModelSpec(
        model_id="Qwen/Qwen3-235B-A22B-fp8-tput",
        strip_thinking=True,
        max_infer_tokens=6000,
        max_match_tokens=4000,
        notes="Reasoning model. Emits <think> blocks; needs a larger token budget.",
    ),
    "qwen3.8-flash": ModelSpec(
        model_id="Qwen/Qwen3.8-Flash",
        stream=True,
        notes="Together requires streaming responses for this model.",
    ),
    "qwen3.5-9b": ModelSpec(
        model_id="Qwen/Qwen3.5-9B",
        chat_template_kwargs={"enable_thinking": False},
        notes="Serverless Together model; disables thinking for reliable JSON output.",
    ),
    "deepseek-v3": ModelSpec(
        model_id="deepseek-ai/DeepSeek-V3",
        notes="Strong extraction quality, non-reasoning.",
    ),
    "deepseek-v4-flash": ModelSpec(
        model_id="deepseek-ai/DeepSeek-V4-Flash-0731",
        notes="Together model also used by the job_skill_rag pipeline.",
    ),
    "deepseek-v4.1-flash": ModelSpec(
        model_id="deepseek-ai/DeepSeek-V4.1-Flash",
        notes="Together model also used by the job_skill_rag pipeline.",
    ),
    "mistral-small-24b": ModelSpec(
        model_id="mistralai/Mistral-Small-24B-Instruct-2501",
        notes="Replaces the local Ministral preset.",
    ),
    "gemma2-27b": ModelSpec(
        model_id="google/gemma-2-27b-it",
        json_mode=False,
        notes="Gemma has no JSON mode on Together; relies on prompt-only JSON.",
    ),
    "gpt-oss-120b": ModelSpec(
        model_id="openai/gpt-oss-120b",
        strip_thinking=True,
        max_infer_tokens=6000,
        max_match_tokens=4000,
        notes="Open-weight reasoning model.",
    ),
}

DEFAULT_MODEL_PRESET = "deepseek-v4-flash"


# ============================================================
# Embeddings
# ============================================================

# Runs locally via sentence-transformers. Taxonomy vectors are encoded once and
# cached to <output_dir>/embedding_cache; only the per-run query concepts are
# re-encoded. Changing this value invalidates the cache automatically.
DEFAULT_EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"


# ============================================================
# Retrieval defaults
# ============================================================

# None means no concept-count cap: extract every supported skill concept.
# Individual models still enforce max_infer_tokens as a transport safeguard.
DEFAULT_MAX_CONCEPTS = None
DEFAULT_MATCH_BATCH_SIZE = 4
DEFAULT_RETRIEVE_TOP_K = 5
DEFAULT_LEXICAL_TOP_K = 3
DEFAULT_LEXICAL_MIN_SCORE = 0.65


def get_model_spec(preset: str, model_id_override: str = None) -> ModelSpec:
    """Resolve a preset, optionally swapping in an exact model string."""
    if preset not in MODEL_PRESETS:
        raise ValueError(
            f"Unknown model preset: {preset}\n"
            f"Available presets: {', '.join(sorted(MODEL_PRESETS))}"
        )

    spec = MODEL_PRESETS[preset]

    if model_id_override:
        return ModelSpec(
            model_id=model_id_override.strip(),
            json_mode=spec.json_mode,
            stream=spec.stream,
            chat_template_kwargs=dict(spec.chat_template_kwargs),
            strip_thinking=spec.strip_thinking,
            temperature=spec.temperature,
            max_infer_tokens=spec.max_infer_tokens,
            max_match_tokens=spec.max_match_tokens,
            notes=f"Override of preset '{preset}'.",
        )

    return spec


def describe_presets() -> str:
    lines = []
    width = max(len(name) for name in MODEL_PRESETS)

    for name in sorted(MODEL_PRESETS):
        spec = MODEL_PRESETS[name]
        default_tag = " (default)" if name == DEFAULT_MODEL_PRESET else ""
        lines.append(f"  {name:<{width}}  {spec.model_id}{default_tag}")
        if spec.notes:
            lines.append(f"  {'':<{width}}  {spec.notes}")

    return "\n".join(lines)
