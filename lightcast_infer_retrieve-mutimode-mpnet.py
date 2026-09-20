#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Command-line runner for the Lightcast skill extraction pipeline."""

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import httpx
import pandas as pd
import torch
from openai import OpenAI
from sentence_transformers import SentenceTransformer

import api_config
from model_config import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_LEXICAL_MIN_SCORE,
    DEFAULT_LEXICAL_TOP_K,
    DEFAULT_MATCH_BATCH_SIZE,
    DEFAULT_MAX_CONCEPTS,
    DEFAULT_MODEL_PRESET,
    DEFAULT_RETRIEVE_TOP_K,
    MODEL_PRESETS,
    describe_presets,
    get_model_spec,
)
from lightcast_data import (
    DEFAULT_TEXT_COL,
    build_exact_skill_lookup,
    build_faiss_index,
    build_lexical_skill_index,
    build_or_load_taxonomy_embeddings,
    json_default,
    load_jd_data,
    load_prompt,
    load_taxonomy,
    retrieve_concepts,
    split_jd_units,
)
from lightcast_llm import deduplicate_matches, infer_one, match_candidates

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROMPT_FILE = SCRIPT_DIR / "infer_prompt.txt"
DEFAULT_MATCH_PROMPT_FILE = SCRIPT_DIR / "match_prompt.txt"
DEFAULT_JD_CSV = Path(
    r"D:\extraction_skill\job_join_rag_level3_parallel\job_join_rag_level3\data\sample_2021_2025_400.csv"
)
DEFAULT_TAXONOMY_CSV = SCRIPT_DIR / "final_lightcast_taxonomy.csv"

# ============================================================
# Progress / resume
# ============================================================

def load_completed_rows(
    progress_jsonl: Path,
) -> set:
    latest_success = {}

    if not progress_jsonl.exists():
        return set()

    with progress_jsonl.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(
                    line
                )
                row_index = int(obj["row_index"])
                latest_success[row_index] = not bool(obj.get("error"))
            except Exception:
                continue

    return {
        row_index
        for row_index, succeeded in latest_success.items()
        if succeeded
    }


def read_progress(
    progress_jsonl: Path,
) -> List[Dict[str, Any]]:
    records = []

    if not progress_jsonl.exists():
        return records

    with progress_jsonl.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                records.append(
                    json.loads(
                        line
                    )
                )
            except Exception:
                continue

    return records


def print_available_models(base_url: str, api_key: str) -> None:
    """Print Together models, accepting both list and data-wrapped responses."""
    response = httpx.get(
        f"{base_url}/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=api_config.REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    models = payload if isinstance(payload, list) else payload.get("data", [])
    ids = [str(model["id"]) for model in models
           if isinstance(model, dict) and model.get("id")]

    print("\nTogether models available to this key:")
    for model_id in sorted(set(ids), key=str.casefold):
        print(f"  {model_id}")


def validate_run_arguments(args: argparse.Namespace) -> None:
    """Fail before loading models or making billable API requests."""
    positive_names = (
        "infer_workers",
        "match_batch_size",
        "retrieve_top_k",
        "lexical_top_k",
    )
    for name in positive_names:
        if getattr(args, name) < 1:
            raise ValueError(f"--{name} must be at least 1.")

    if args.max_concepts is not None and args.max_concepts < 1:
        raise ValueError("--max_concepts must be at least 1 when supplied.")

    if not 0.0 <= args.lexical_min_score <= 1.15:
        raise ValueError("--lexical_min_score must be between 0 and 1.15.")

    if args.start_row < 0:
        raise ValueError("--start_row cannot be negative.")

    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1 when supplied.")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Lightcast Infer -> Evidence Align -> "
            "Hybrid Retrieve -> Context Match"
        )
    )

    parser.add_argument(
        "--jd_csv",
        default=str(DEFAULT_JD_CSV),
        help="Job-description CSV (defaults to the CSV beside this script).",
    )
    parser.add_argument(
        "--taxonomy_csv",
        default=str(DEFAULT_TAXONOMY_CSV),
        help="Lightcast taxonomy CSV (defaults to the CSV beside this script).",
    )
    parser.add_argument(
        "--output_dir",
        default=str(SCRIPT_DIR / "lightcast_sample_2021_2025_400"),
    )
    parser.add_argument(
        "--text_col",
        default=DEFAULT_TEXT_COL,
    )
    parser.add_argument(
        "--prompt_file",
        default=str(
            DEFAULT_PROMPT_FILE
        ),
    )
    parser.add_argument(
        "--match_prompt_file",
        default=str(
            DEFAULT_MATCH_PROMPT_FILE
        ),
        help="Static Context Match prompt, read once at startup.",
    )
    parser.add_argument(
        "--model_preset",
        choices=list(MODEL_PRESETS.keys()),
        default=DEFAULT_MODEL_PRESET,
        help=(
            "Preset from model_config.py. Use --llm_model to override "
            "with any exact Together model ID."
        ),
    )
    parser.add_argument(
        "--llm_model",
        default=None,
        help=(
            "Exact Together AI model string. Overrides the preset's "
            "model_id while keeping its other settings."
        ),
    )
    parser.add_argument(
        "--llm_base_url",
        default=None,
        help="Defaults to the Together endpoint in api_config.py.",
    )
    parser.add_argument(
        "--llm_api_key",
        default=None,
        help=(
            "Overrides the key from the environment and api_config.py."
        ),
    )
    parser.add_argument(
        "--list_models",
        action="store_true",
        help=(
            "Print the local presets and the chat models your Together "
            "account can currently serve, then exit."
        ),
    )
    parser.add_argument(
        "--embed_model",
        default=DEFAULT_EMBED_MODEL,
    )
    parser.add_argument(
        "--device",
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )
    parser.add_argument(
        "--max_concepts",
        type=int,
        default=DEFAULT_MAX_CONCEPTS,
    )
    parser.add_argument(
        "--infer_workers",
        type=int,
        default=api_config.INFER_WORKERS,
    )
    parser.add_argument(
        "--match_batch_size",
        type=int,
        default=DEFAULT_MATCH_BATCH_SIZE,
    )
    parser.add_argument(
        "--retrieve_top_k",
        type=int,
        default=DEFAULT_RETRIEVE_TOP_K,
    )
    parser.add_argument(
        "--lexical_top_k",
        type=int,
        default=DEFAULT_LEXICAL_TOP_K,
    )
    parser.add_argument(
        "--lexical_min_score",
        type=float,
        default=DEFAULT_LEXICAL_MIN_SCORE,
        help=(
            "Minimum score for candidates contributed only by lexical "
            "retrieval. Semantic candidates are never removed by this value."
        ),
    )
    parser.add_argument(
        "--start_row",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
    )

    args = parser.parse_args()
    validate_run_arguments(args)

    spec = get_model_spec(
        args.model_preset,
        args.llm_model,
    )
    resolved_base_url = api_config.resolve_base_url(
        args.llm_base_url
    )

    print(
        f"[LLM] preset={args.model_preset} "
        f"model={spec.model_id}"
    )
    if args.list_models:
        print("\nConfigured local presets:")
        print(describe_presets())
        try:
            list_key = api_config.resolve_api_key(
                args.llm_api_key
            )
        except RuntimeError:
            print(
                "\nNo Together API key is configured, so account "
                "models were not requested. Set TOGETHER_API_KEY (or "
                "pass --llm_api_key) to list them."
            )
            return

        print_available_models(resolved_base_url, list_key)
        return

    resolved_api_key = api_config.resolve_api_key(
        args.llm_api_key
    )
    print(
        f"[LLM] endpoint={resolved_base_url} "
        f"key={api_config.mask_api_key(resolved_api_key)}"
    )

    client = OpenAI(
        base_url=resolved_base_url,
        api_key=resolved_api_key,
        timeout=api_config.REQUEST_TIMEOUT,
    )

    output_dir = Path(
        args.output_dir
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_dir = (
        output_dir
        / "embedding_cache"
    )
    progress_jsonl = (
        output_dir
        / "progress.jsonl"
    )
    wide_csv = (
        output_dir
        / "jd_with_lightcast_skills.csv"
    )
    long_csv = (
        output_dir
        / "jd_skill_matches_long.csv"
    )
    match_audit_csv = (
        output_dir
        / "jd_skill_match_audit_long.csv"
    )

    jd_df, text_col = load_jd_data(
        args.jd_csv,
        args.text_col,
    )
    taxonomy = load_taxonomy(
        args.taxonomy_csv
    )
    system_prompt = load_prompt(
        args.prompt_file
    )
    match_system_prompt = load_prompt(
        args.match_prompt_file
    )

    print(
        f"[JD] Rows: "
        f"{len(jd_df):,}"
    )
    print(
        f"[Taxonomy] Rows: "
        f"{len(taxonomy):,}"
    )
    print(
        f"[Infer Prompt] "
        f"{args.prompt_file}"
    )
    print(
        f"[Match Prompt] "
        f"{args.match_prompt_file}"
    )
    print(
        f"[Match] Concept batch size: "
        f"{args.match_batch_size}"
    )
    print(
        f"[Retrieve] Semantic Top-K: "
        f"{args.retrieve_top_k}"
    )
    print(
        f"[Retrieve] Lexical Top-K: "
        f"{args.lexical_top_k}"
    )
    print(
        f"[Retrieve] Pure lexical minimum score: "
        f"{args.lexical_min_score}"
    )

    embedder = SentenceTransformer(
        args.embed_model,
        device=args.device,
    )

    taxonomy_embeddings = (
        build_or_load_taxonomy_embeddings(
            taxonomy=taxonomy,
            embedder=embedder,
            cache_dir=cache_dir,
            embed_model_name=
                args.embed_model,
        )
    )
    faiss_index = build_faiss_index(
        taxonomy_embeddings
    )
    print(
        "[Retrieve] Built FAISS IndexFlatIP "
        f"with {faiss_index.ntotal:,} taxonomy vectors."
    )

    exact_lookup = (
        build_exact_skill_lookup(
            taxonomy
        )
    )
    lexical_index = (
        build_lexical_skill_index(
            taxonomy
        )
    )

    start = max(
        0,
        args.start_row,
    )
    stop = len(jd_df)

    if args.limit is not None:
        stop = min(
            stop,
            start + args.limit,
        )

    selected = list(
        range(
            start,
            stop,
        )
    )

    completed = set()

    if not args.no_resume:
        completed = (
            load_completed_rows(
                progress_jsonl
            )
        )

    todo = [
        i
        for i in selected
        if i not in completed
    ]

    print(
        f"[Run] Selected rows: "
        f"{len(selected):,}"
    )
    print(
        f"[Run] Already completed: "
        f"{len(selected)-len(todo):,}"
    )
    print(
        f"[Run] To process: "
        f"{len(todo):,}"
    )
    print(
        "[Run] Pipeline: Infer + exact evidence "
        "-> Evidence Align -> Hybrid Retrieve "
        "-> Context Match -> Deduplicate"
    )

    if todo:
        progress_f = (
            progress_jsonl.open(
                "a",
                encoding="utf-8",
                buffering=1,
            )
        )

        embed_lock = (
            threading.Lock()
        )

        def process_job(
            row_idx: int,
        ):
            stage_times = {
                "infer": None,
                "retrieve": None,
                "match": None,
            }

            row = jd_df.iloc[
                row_idx
            ]
            jd_text = row.get(
                text_col
            )

            if (
                pd.isna(jd_text)
                or not str(
                    jd_text
                ).strip()
            ):
                return (
                    row_idx,
                    [],
                    [],
                    [],
                    "empty_text",
                    stage_times,
                )

            jd_text = str(
                jd_text
            )
            jd_units = split_jd_units(
                jd_text
            )

            stage_started = time.perf_counter()

            try:
                concepts = infer_one(
                    client=client,
                    spec=spec,
                    system_prompt=
                        system_prompt,
                    jd_text=jd_text,
                    jd_units=jd_units,
                    max_concepts=
                        args.max_concepts,
                )
            except Exception as e:
                stage_times["infer"] = (
                    time.perf_counter()
                    - stage_started
                )
                return (
                    row_idx,
                    [],
                    [],
                    [],
                    "infer_error: "
                    + repr(e),
                    stage_times,
                )

            stage_times["infer"] = (
                time.perf_counter()
                - stage_started
            )

            stage_started = time.perf_counter()

            try:
                raw_candidates = (
                    retrieve_concepts(
                        concepts=
                            concepts,
                        taxonomy=
                            taxonomy,
                        faiss_index=
                            faiss_index,
                        embedder=
                            embedder,
                        exact_lookup=
                            exact_lookup,
                        lexical_index=
                            lexical_index,
                        top_k=
                            args.retrieve_top_k,
                        lexical_top_k=
                            args.lexical_top_k,
                        lexical_min_score=
                            args.lexical_min_score,
                        embed_lock=
                            embed_lock,
                    )
                )
            except Exception as e:
                stage_times["retrieve"] = (
                    time.perf_counter()
                    - stage_started
                )
                return (
                    row_idx,
                    concepts,
                    [],
                    [],
                    "retrieve_error: "
                    + repr(e),
                    stage_times,
                )

            stage_times["retrieve"] = (
                time.perf_counter()
                - stage_started
            )

            stage_started = time.perf_counter()

            try:
                (
                    match_audit,
                    selected_matches,
                ) = match_candidates(
                    client=client,
                    spec=spec,
                    match_system_prompt=
                        match_system_prompt,
                    candidates=
                        raw_candidates,
                    batch_size=
                        args.match_batch_size,
                )

                final_matches = (
                    deduplicate_matches(
                        selected_matches
                    )
                )
            except Exception as e:
                stage_times["match"] = (
                    time.perf_counter()
                    - stage_started
                )
                return (
                    row_idx,
                    concepts,
                    [],
                    [],
                    "match_error: "
                    + repr(e),
                    stage_times,
                )

            stage_times["match"] = (
                time.perf_counter()
                - stage_started
            )

            return (
                row_idx,
                concepts,
                match_audit,
                final_matches,
                "",
                stage_times,
            )

        def compact_match(
            m: Dict[str, Any],
        ) -> Dict[str, Any]:
            evidence_ids = (
                m.get(
                    "evidence_ids"
                )
                or (
                    [
                        m.get(
                            "evidence_id"
                        )
                    ]
                    if m.get(
                        "evidence_id"
                    )
                    else []
                )
            )

            return {
                "skill_id":
                    m["skill_id"],
                "skill":
                    m["skill"],
                "category":
                    m["category"],
                "subcategory":
                    m["subcategory"],
                "evidence_ids":
                    evidence_ids,
                "source_concepts":
                    m.get(
                        "source_concepts",
                        (
                            [
                                m.get(
                                    "concept"
                                )
                            ]
                            if m.get(
                                "concept"
                            )
                            else []
                        ),
                    ),
                "source_sentences":
                    m.get(
                        "source_sentences",
                        (
                            [
                                m.get(
                                    "source_sentence"
                                )
                            ]
                            if m.get(
                                "source_sentence"
                            )
                            else []
                        ),
                    ),
                "retrieval_method":
                    m[
                        "retrieval_method"
                    ],
                "cosine_similarity_audit":
                    round(
                        float(
                            m.get(
                                "similarity",
                                0.0,
                            )
                        ),
                        6,
                    ),
                "match_selected":
                    bool(
                        m.get(
                            "match_selected",
                            False,
                        )
                    ),
                "match_reason":
                    str(
                        m.get(
                            "match_reason",
                            "",
                        )
                    ),
            }

        try:
            workers = max(
                1,
                args.infer_workers,
            )

            with ThreadPoolExecutor(
                max_workers=workers
            ) as executor:

                for (
                    batch_no,
                    batch_start,
                ) in enumerate(
                    range(
                        0,
                        len(todo),
                        workers,
                    ),
                    start=1,
                ):
                    batch_indices = (
                        todo[
                            batch_start:
                            batch_start
                            + workers
                        ]
                    )

                    t0 = (
                        time.perf_counter()
                    )

                    batch_stage_times = {
                        "infer": [],
                        "retrieve": [],
                        "match": [],
                    }

                    futures = {
                        executor.submit(
                            process_job,
                            idx,
                        ): idx
                        for idx
                        in batch_indices
                    }

                    for future in (
                        as_completed(
                            futures
                        )
                    ):
                        (
                            row_idx,
                            concepts,
                            match_audit,
                            final_matches,
                            error,
                            stage_times,
                        ) = future.result()

                        for (
                            stage_name,
                            stage_seconds,
                        ) in stage_times.items():
                            if stage_seconds is not None:
                                batch_stage_times[
                                    stage_name
                                ].append(
                                    float(stage_seconds)
                                )

                        row = jd_df.iloc[
                            row_idx
                        ]

                        compact_audit = [
                            compact_match(m)
                            for m
                            in match_audit
                        ]
                        compact_final = [
                            compact_match(m)
                            for m
                            in final_matches
                        ]

                        obj = {
                            "row_index":
                                row_idx,
                            "sample_year":
                                row.get(
                                    "sample_year",
                                    row.get(
                                        "created_year",
                                        "",
                                    ),
                                ),
                            "JOB_HASH":
                                row.get(
                                    "JOB_HASH",
                                    "",
                                ),
                            "COMPANY_NAME":
                                row.get(
                                    "COMPANY_NAME",
                                    "",
                                ),
                            "TITLE":
                                row.get(
                                    "TITLE",
                                    "",
                                ),
                            "infer_concepts":
                                [
                                    c.get(
                                        "skill",
                                        "",
                                    )
                                    for c
                                    in concepts
                                ],
                            "match_audit":
                                compact_audit,
                            "matched_skills":
                                compact_final,
                            "error":
                                error,
                        }

                        progress_f.write(
                            json.dumps(
                                obj,
                                ensure_ascii=False,
                                default=json_default,
                            )
                            + "\n"
                        )

                    dt = (
                        time.perf_counter()
                        - t0
                    )

                    stage_totals = {
                        name: sum(values)
                        for name, values
                        in batch_stage_times.items()
                    }

                    total_stage_work = sum(
                        stage_totals.values()
                    )

                    wall_shares = {
                        name: (
                            dt * stage_total
                            / total_stage_work
                            if total_stage_work > 0.0
                            else 0.0
                        )
                        for name, stage_total
                        in stage_totals.items()
                    }

                    displayed_dt_tenths = round(
                        dt * 10
                    )
                    displayed_infer_tenths = round(
                        wall_shares["infer"] * 10
                    )
                    displayed_retrieve_tenths = round(
                        wall_shares["retrieve"] * 10
                    )
                    displayed_match_tenths = (
                        displayed_dt_tenths
                        - displayed_infer_tenths
                        - displayed_retrieve_tenths
                    )

                    print(
                        f"[Batch {batch_no} | "
                        f"{len(batch_indices)} descriptions | "
                        f"dt={displayed_dt_tenths / 10:.1f}s | "
                        f"estimated wall share: "
                        f"infer={displayed_infer_tenths / 10:.1f}s | "
                        f"retrieve={displayed_retrieve_tenths / 10:.1f}s | "
                        f"match={displayed_match_tenths / 10:.1f}s]"
                    )
        finally:
            progress_f.close()

    progress_records = read_progress(
        progress_jsonl
    )

    selected_set = set(
        selected
    )

    progress_records = [
        r
        for r
        in progress_records
        if int(
            r.get(
                "row_index",
                -1,
            )
        )
        in selected_set
    ]

    latest_by_row = {}

    for r in progress_records:
        latest_by_row[
            int(
                r["row_index"]
            )
        ] = r

    progress_records = [
        latest_by_row[i]
        for i
        in sorted(
            latest_by_row
        )
    ]

    wide_records = []
    long_records = []
    audit_records = []

    taxonomy_by_id = (
        taxonomy
        .set_index(
            "skill_id",
            drop=False,
        )
    )

    for p in progress_records:
        row_idx = int(
            p["row_index"]
        )
        row = jd_df.iloc[
            row_idx
        ]

        wide = row.to_dict()

        wide[
            "INFERRED_CONCEPTS"
        ] = json.dumps(
            p.get(
                "infer_concepts",
                [],
            ),
            ensure_ascii=False,
        )

        wide[
            "LIGHTCAST_RETRIEVED_CANDIDATE_COUNT"
        ] = len(
            p.get(
                "match_audit",
                [],
            )
        )

        wide[
            "LIGHTCAST_MATCH_AUDIT"
        ] = json.dumps(
            p.get(
                "match_audit",
                [],
            ),
            ensure_ascii=False,
        )

        wide[
            "LIGHTCAST_SKILL_COUNT"
        ] = len(
            p.get(
                "matched_skills",
                [],
            )
        )

        wide[
            "LIGHTCAST_SKILLS"
        ] = json.dumps(
            p.get(
                "matched_skills",
                [],
            ),
            ensure_ascii=False,
        )

        wide[
            "EXTRACTION_ERROR"
        ] = p.get(
            "error",
            "",
        )

        wide_records.append(
            wide
        )

        for m in p.get(
            "matched_skills",
            [],
        ):
            skill_id = str(
                m["skill_id"]
            )

            definition = ""

            if skill_id in (
                taxonomy_by_id.index
            ):
                definition = (
                    taxonomy_by_id.loc[
                        skill_id,
                        "definition",
                    ]
                )
                if isinstance(
                    definition,
                    pd.Series,
                ):
                    definition = (
                        definition.iloc[0]
                    )

            long_records.append(
                {
                    "row_index":
                        row_idx,
                    "sample_year":
                        p.get(
                            "sample_year",
                            "",
                        ),
                    "JOB_HASH":
                        p.get(
                            "JOB_HASH",
                            "",
                        ),
                    "COMPANY_NAME":
                        p.get(
                            "COMPANY_NAME",
                            "",
                        ),
                    "TITLE":
                        p.get(
                            "TITLE",
                            "",
                        ),
                    "evidence_ids":
                        json.dumps(
                            m.get(
                                "evidence_ids",
                                [],
                            ),
                            ensure_ascii=False,
                        ),
                    "infer_concepts":
                        json.dumps(
                            m.get(
                                "source_concepts",
                                [],
                            ),
                            ensure_ascii=False,
                        ),
                    "source_sentences":
                        json.dumps(
                            m.get(
                                "source_sentences",
                                [],
                            ),
                            ensure_ascii=False,
                        ),
                    "skill_id":
                        m["skill_id"],
                    "skill":
                        m["skill"],
                    "category":
                        m["category"],
                    "subcategory":
                        m["subcategory"],
                    "definition":
                        definition,
                    "retrieval_method":
                        m[
                            "retrieval_method"
                        ],
                    "cosine_similarity_audit":
                        m[
                            "cosine_similarity_audit"
                        ],
                    "match_selected":
                        m.get(
                            "match_selected",
                            True,
                        ),
                    "match_reason":
                        m.get(
                            "match_reason",
                            "",
                        ),
                }
            )

        for m in p.get(
            "match_audit",
            [],
        ):
            skill_id = str(
                m["skill_id"]
            )

            definition = ""

            if skill_id in (
                taxonomy_by_id.index
            ):
                definition = (
                    taxonomy_by_id.loc[
                        skill_id,
                        "definition",
                    ]
                )
                if isinstance(
                    definition,
                    pd.Series,
                ):
                    definition = (
                        definition.iloc[0]
                    )

            audit_records.append(
                {
                    "row_index":
                        row_idx,
                    "sample_year":
                        p.get(
                            "sample_year",
                            "",
                        ),
                    "JOB_HASH":
                        p.get(
                            "JOB_HASH",
                            "",
                        ),
                    "COMPANY_NAME":
                        p.get(
                            "COMPANY_NAME",
                            "",
                        ),
                    "TITLE":
                        p.get(
                            "TITLE",
                            "",
                        ),
                    "evidence_ids":
                        json.dumps(
                            m.get(
                                "evidence_ids",
                                [],
                            ),
                            ensure_ascii=False,
                        ),
                    "infer_concepts":
                        json.dumps(
                            m.get(
                                "source_concepts",
                                [],
                            ),
                            ensure_ascii=False,
                        ),
                    "source_sentences":
                        json.dumps(
                            m.get(
                                "source_sentences",
                                [],
                            ),
                            ensure_ascii=False,
                        ),
                    "skill_id":
                        m["skill_id"],
                    "skill":
                        m["skill"],
                    "category":
                        m["category"],
                    "subcategory":
                        m["subcategory"],
                    "definition":
                        definition,
                    "retrieval_method":
                        m[
                            "retrieval_method"
                        ],
                    "cosine_similarity_audit":
                        m[
                            "cosine_similarity_audit"
                        ],
                    "match_selected":
                        m.get(
                            "match_selected",
                            False,
                        ),
                    "match_reason":
                        m.get(
                            "match_reason",
                            "",
                        ),
                }
            )

    pd.DataFrame(
        wide_records
    ).to_csv(
        wide_csv,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        long_records
    ).to_csv(
        long_csv,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        audit_records
    ).to_csv(
        match_audit_csv,
        index=False,
        encoding="utf-8-sig",
    )

    print("\nFinished.")
    print(
        f"Wide output: "
        f"{wide_csv}"
    )
    print(
        f"Skill long output: "
        f"{long_csv}"
    )
    print(
        f"Match audit output: "
        f"{match_audit_csv}"
    )
    print(
        f"Checkpoint: "
        f"{progress_jsonl}"
    )


if __name__ == "__main__":
    main()
