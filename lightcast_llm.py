#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Together AI calls plus Infer and Context Match stages."""

import json
import random
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)

import api_config
from model_config import ModelSpec
from lightcast_data import (
    extract_json_object,
    normalize_concepts,
)

JSON_MODE_DISABLED: set[str] = set()
JSON_MODE_LOCK = threading.Lock()

def strip_thinking_blocks(text: str) -> str:
    """Remove <think> reasoning so it cannot confuse JSON extraction."""
    text = re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # A block opened before the response started leaves a stray closing tag.
    text = re.sub(
        r"^.*?</think>",
        "",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    return text.strip()


def retry_delay(
    attempt: int,
    error: Exception,
) -> float:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)

    if headers:
        raw = headers.get("retry-after")
        if raw:
            try:
                return min(
                    float(raw),
                    api_config.RETRY_MAX_DELAY,
                )
            except (TypeError, ValueError):
                pass

    delay = min(
        api_config.RETRY_BASE_DELAY * (2 ** (attempt - 1)),
        api_config.RETRY_MAX_DELAY,
    )

    return random.uniform(delay * 0.5, delay)


def call_llm(
    client: OpenAI,
    spec: ModelSpec,
    messages: List[Dict[str, str]],
    max_tokens: int,
) -> str:
    """
    One Together AI chat completion.

    Retries rate limits and transient transport failures with exponential
    backoff, and permanently drops JSON mode for a model that rejects it.
    """
    last_error: Optional[Exception] = None

    for attempt in range(
        1,
        api_config.MAX_TRANSPORT_RETRIES + 1,
    ):
        use_json_mode = (
            spec.json_mode
            and spec.model_id not in JSON_MODE_DISABLED
        )

        kwargs: Dict[str, Any] = dict(
            model=spec.model_id,
            messages=messages,
            temperature=spec.temperature,
            max_tokens=max_tokens,
        )

        if spec.stream:
            kwargs["stream"] = True

        if spec.chat_template_kwargs:
            kwargs["extra_body"] = {
                "chat_template_kwargs": dict(
                    spec.chat_template_kwargs
                )
            }

        if use_json_mode:
            kwargs["response_format"] = {
                "type": "json_object"
            }

        try:
            response = client.chat.completions.create(
                **kwargs
            )
            if spec.stream:
                content = "".join(
                    chunk.choices[0].delta.content or ""
                    for chunk in response
                    if chunk.choices
                )
            else:
                content = response.choices[0].message.content or ""
        except BadRequestError:
            if not use_json_mode:
                raise

            with JSON_MODE_LOCK:
                JSON_MODE_DISABLED.add(
                    spec.model_id
                )

            print(
                f"[LLM] {spec.model_id} rejected JSON mode; "
                f"falling back to prompt-only JSON."
            )
            continue

        except (
            RateLimitError,
            APIConnectionError,
            APITimeoutError,
            APIStatusError,
        ) as e:
            status = getattr(
                e,
                "status_code",
                None,
            )

            # Auth, quota and payload errors will not fix themselves.
            if (
                isinstance(e, APIStatusError)
                and status is not None
                and status < 500
                and status != 429
            ):
                raise

            last_error = e

            if attempt >= api_config.MAX_TRANSPORT_RETRIES:
                break

            time.sleep(
                retry_delay(attempt, e)
            )
            continue

        return strip_thinking_blocks(content)

    raise RuntimeError(
        "Together AI call failed after "
        f"{api_config.MAX_TRANSPORT_RETRIES} attempts: "
        f"{last_error!r}"
    )


def infer_one(
    client: OpenAI,
    spec: ModelSpec,
    system_prompt: str,
    jd_text: str,
    jd_units: List[str],
    max_concepts: Optional[int],
    max_retries: int = api_config.MAX_CONTENT_RETRIES,
) -> List[Dict[str, Any]]:
    last_error = None

    for attempt in range(
        1,
        max_retries + 1,
    ):
        try:
            content = call_llm(
                client=client,
                spec=spec,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": f"JOB DESCRIPTION:\n{jd_text}",
                    },
                ],
                max_tokens=spec.max_infer_tokens,
            )

            obj = extract_json_object(
                content
            )

            concepts = normalize_concepts(
                obj.get(
                    "skills",
                    [],
                ),
                jd_units,
                max_concepts,
            )

            if concepts:
                return concepts

            raise ValueError(
                "LLM returned no aligned skills."
            )

        except Exception as e:
            last_error = e

            if attempt < max_retries:
                time.sleep(
                    2 * attempt
                )

    raise RuntimeError(
        "Infer failed after "
        f"{max_retries} attempts: "
        f"{last_error}"
    )


# ============================================================
# Context Match
# ============================================================

def build_match_user_prompt(
    concept_groups: List[Dict[str, Any]],
) -> str:
    payload = []

    for (
        concept_no,
        group,
    ) in enumerate(
        concept_groups
    ):
        candidate_payload = []

        for (
            candidate_no,
            m,
        ) in enumerate(
            group["candidates"]
        ):
            candidate_payload.append(
                {
                    "candidate_no":
                        candidate_no,
                    "skill":
                        str(
                            m["skill"]
                        ),
                }
            )

        payload.append(
            {
                "concept_no":
                    concept_no,
                "source_concept":
                    str(
                        group["concept"]
                    ),
                "candidates":
                    candidate_payload,
            }
        )

    return (
        "SOURCE CONCEPTS AND CANDIDATES:\n"
        + json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_match_selections(
    obj: Dict[str, Any],
    concept_groups: List[Dict[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    items = obj.get(
        "selections",
        [],
    )

    if not isinstance(
        items,
        list,
    ):
        raise ValueError(
            "Matcher JSON field "
            "'selections' is not a list."
        )

    parsed: Dict[
        int,
        Dict[str, Any],
    ] = {}

    for item in items:
        if not isinstance(
            item,
            dict,
        ):
            continue

        try:
            concept_no = int(
                item.get(
                    "concept_no"
                )
            )
        except Exception:
            continue

        if not (
            0
            <= concept_no
            < len(concept_groups)
        ):
            continue

        if concept_no in parsed:
            continue

        raw_candidate = item.get(
            "candidate_no"
        )

        if raw_candidate is None:
            candidate_no = None
        elif (
            isinstance(
                raw_candidate,
                str,
            )
            and raw_candidate
            .strip()
            .casefold()
            in {
                "",
                "none",
                "null",
                "reject",
                "no_match",
                "no match",
            }
        ):
            candidate_no = None
        else:
            try:
                candidate_no = int(
                    raw_candidate
                )
            except Exception:
                continue

            if not (
                0
                <= candidate_no
                < len(
                    concept_groups[
                        concept_no
                    ]["candidates"]
                )
            ):
                continue

        reason = re.sub(
            r"\s+",
            " ",
            str(
                item.get(
                    "reason",
                    "",
                )
            ).strip(),
        )

        parsed[concept_no] = {
            "candidate_no":
                candidate_no,
            "reason":
                reason,
        }

    missing = [
        i
        for i in range(
            len(concept_groups)
        )
        if i not in parsed
    ]

    if missing:
        raise ValueError(
            "Matcher omitted/invalid "
            f"selections for concept_no: "
            f"{missing}"
        )

    return parsed


def match_one_batch(
    client: OpenAI,
    spec: ModelSpec,
    match_system_prompt: str,
    concept_groups: List[Dict[str, Any]],
    max_retries: int = api_config.MAX_CONTENT_RETRIES,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    if not concept_groups:
        return [], []

    last_error = None

    for attempt in range(
        1,
        max_retries + 1,
    ):
        try:
            max_tokens = min(
                spec.max_match_tokens,
                max(
                    700,
                    120
                    * len(
                        concept_groups
                    ),
                ),
            )

            content = call_llm(
                client=client,
                spec=spec,
                messages=[
                    {
                        "role":
                            "system",
                        "content":
                            match_system_prompt,
                    },
                    {
                        "role":
                            "user",
                        "content":
                            build_match_user_prompt(
                                concept_groups
                            ),
                    },
                ],
                max_tokens=max_tokens,
            )

            obj = extract_json_object(
                content
            )

            selections = (
                parse_match_selections(
                    obj,
                    concept_groups,
                )
            )

            audit: List[
                Dict[str, Any]
            ] = []

            selected: List[
                Dict[str, Any]
            ] = []

            for (
                concept_no,
                group,
            ) in enumerate(
                concept_groups
            ):
                decision = selections[
                    concept_no
                ]

                chosen_no = decision[
                    "candidate_no"
                ]
                reason = decision[
                    "reason"
                ]

                for (
                    candidate_no,
                    m,
                ) in enumerate(
                    group["candidates"]
                ):
                    item = dict(m)

                    item[
                        "source_concepts"
                    ] = [
                        str(
                            group[
                                "concept"
                            ]
                        )
                    ]

                    item[
                        "source_sentences"
                    ] = [
                        str(
                            group.get(
                                "source_sentence",
                                "",
                            )
                        )
                    ]

                    is_chosen = (
                        chosen_no
                        is not None
                        and candidate_no
                        == chosen_no
                    )

                    item[
                        "match_selected"
                    ] = bool(
                        is_chosen
                    )

                    if is_chosen:
                        item[
                            "match_reason"
                        ] = reason
                        selected.append(
                            item
                        )
                    elif (
                        chosen_no
                        is None
                    ):
                        item[
                            "match_reason"
                        ] = (
                            reason
                            or "No equivalent "
                            "supported candidate."
                        )
                    else:
                        item[
                            "match_reason"
                        ] = (
                            f"Not selected; "
                            f"candidate "
                            f"{chosen_no} "
                            f"was chosen. "
                            f"{reason}"
                        ).strip()

                    audit.append(
                        item
                    )

            return (
                audit,
                selected,
            )

        except Exception as e:
            last_error = e

            if attempt < max_retries:
                time.sleep(
                    2 * attempt
                )

    raise RuntimeError(
        "Match failed after "
        f"{max_retries} attempts: "
        f"{last_error}"
    )


def match_candidates(
    client: OpenAI,
    spec: ModelSpec,
    match_system_prompt: str,
    candidates: List[Dict[str, Any]],
    batch_size: int,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    if not candidates:
        return [], []

    grouped: Dict[
        str,
        Dict[str, Any],
    ] = {}

    concept_order: List[str] = []

    for m in candidates:
        concept = str(
            m["concept"]
        )

        if concept not in grouped:
            grouped[concept] = {
                "concept":
                    concept,
                "source_sentence":
                    str(
                        m.get(
                            "source_sentence",
                            "",
                        )
                    ),
                "candidates":
                    [],
            }
            concept_order.append(
                concept
            )

        grouped[
            concept
        ]["candidates"].append(
            m
        )

    batch_size = max(
        1,
        int(batch_size),
    )

    all_audit: List[
        Dict[str, Any]
    ] = []

    all_selected: List[
        Dict[str, Any]
    ] = []

    for start in range(
        0,
        len(concept_order),
        batch_size,
    ):
        batch_concepts = (
            concept_order[
                start:
                start + batch_size
            ]
        )

        concept_groups = [
            grouped[concept]
            for concept
            in batch_concepts
        ]

        (
            audit,
            selected,
        ) = match_one_batch(
            client=client,
            spec=spec,
            match_system_prompt=match_system_prompt,
            concept_groups=concept_groups,
        )

        all_audit.extend(
            audit
        )
        all_selected.extend(
            selected
        )

    return (
        all_audit,
        all_selected,
    )


def deduplicate_matches(
    matches: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    grouped: Dict[
        str,
        Dict[str, Any],
    ] = {}

    for m in matches:
        sid = str(
            m["skill_id"]
        )

        concept = str(
            m.get(
                "concept",
                "",
            )
        ).strip()

        source_sentence = str(
            m.get(
                "source_sentence",
                "",
            )
        ).strip()

        evidence_id = str(
            m.get(
                "evidence_id",
                "",
            )
        ).strip()

        if sid not in grouped:
            grouped[sid] = dict(m)

            grouped[sid][
                "source_concepts"
            ] = (
                [concept]
                if concept
                else []
            )

            grouped[sid][
                "source_sentences"
            ] = (
                [source_sentence]
                if source_sentence
                else []
            )

            grouped[sid][
                "evidence_ids"
            ] = (
                [evidence_id]
                if evidence_id
                else []
            )
        else:
            if (
                concept
                and concept
                not in grouped[
                    sid
                ][
                    "source_concepts"
                ]
            ):
                grouped[
                    sid
                ][
                    "source_concepts"
                ].append(
                    concept
                )

            if (
                source_sentence
                and source_sentence
                not in grouped[
                    sid
                ][
                    "source_sentences"
                ]
            ):
                grouped[
                    sid
                ][
                    "source_sentences"
                ].append(
                    source_sentence
                )

            if (
                evidence_id
                and evidence_id
                not in grouped[
                    sid
                ][
                    "evidence_ids"
                ]
            ):
                grouped[
                    sid
                ][
                    "evidence_ids"
                ].append(
                    evidence_id
                )

            if float(
                m["similarity"]
            ) > float(
                grouped[
                    sid
                ][
                    "similarity"
                ]
            ):
                concepts = grouped[
                    sid
                ][
                    "source_concepts"
                ]
                source_sentences = grouped[
                    sid
                ][
                    "source_sentences"
                ]
                evidence_ids = grouped[
                    sid
                ][
                    "evidence_ids"
                ]

                grouped[sid] = dict(m)

                grouped[sid][
                    "source_concepts"
                ] = concepts
                grouped[sid][
                    "source_sentences"
                ] = source_sentences
                grouped[sid][
                    "evidence_ids"
                ] = evidence_ids

    return list(
        grouped.values()
    )


