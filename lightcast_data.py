#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Data loading, evidence alignment, and Lightcast candidate retrieval."""

import ast
import hashlib
import html
import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
try:
    import faiss
except ImportError as exc:
    raise RuntimeError(
        "FAISS is required for candidate retrieval. "
        "Install it with: pip install faiss-cpu"
    ) from exc
from sentence_transformers import SentenceTransformer

from model_config import (
    DEFAULT_LEXICAL_MIN_SCORE,
    DEFAULT_LEXICAL_TOP_K,
    DEFAULT_RETRIEVE_TOP_K,
)

DEFAULT_TEXT_COL = "DESCRIPTION_CLEAN"
FALLBACK_TEXT_COL = "DESCRIPTION"
REQUIRED_TAXONOMY_COLUMNS = [
    "category", "subcategory", "skill", "skill_id", "definition",
]

# ============================================================
# JSON helper
# ============================================================

def json_default(obj: Any):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if obj is pd.NA:
        return None
    raise TypeError(
        f"Object of type {type(obj).__name__} is not JSON serializable"
    )


# ============================================================
# Conservative JD cleaning
# ============================================================

def decode_literal_unicode(text: str) -> str:
    text = re.sub(
        r"\\u([0-9a-fA-F]{4})",
        lambda m: chr(int(m.group(1), 16)),
        text,
    )
    text = re.sub(
        r"\\U([0-9a-fA-F]{8})",
        lambda m: chr(int(m.group(1), 16)),
        text,
    )
    return text


def clean_description(value: Any) -> Any:
    if pd.isna(value):
        return value

    text = html.unescape(str(value))

    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(
        r"(?i)</(?:p|div|section|article|h[1-6])\s*>",
        "\n\n",
        text,
    )
    text = re.sub(r"(?i)<li[^>]*>", "\n- ", text)
    text = re.sub(r"(?i)</li\s*>", "", text)
    text = re.sub(r"<[^>]+>", "", text)

    text = decode_literal_unicode(text)

    unicode_spaces = [
        "\u00a0", "\u1680", "\u180e",
        "\u2000", "\u2001", "\u2002", "\u2003", "\u2004",
        "\u2005", "\u2006", "\u2007", "\u2008", "\u2009",
        "\u200a", "\u202f", "\u205f", "\u3000", "\ufeff",
    ]
    for ch in unicode_spaces:
        text = text.replace(ch, " ")

    text = re.sub(
        r'https?://[^\s<>"\']+|www\.[^\s<>"\']+',
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    chars = []
    for ch in text:
        if ch == "\n":
            chars.append(ch)
        elif ch == "\t":
            chars.append(" ")
        elif unicodedata.category(ch) == "Cc":
            continue
        else:
            chars.append(ch)
    text = "".join(chars)

    text = re.sub(r"(?m)^[ \t]*[•●▪◦‣⁃]\s*", "- ", text)
    text = re.sub(r"(?m)^[ \t]*\*+\s+", "- ", text)
    text = re.sub(r"(?m)^[ \t]*-\s+", "- ", text)

    text = re.sub(r"[ \t]+", " ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ============================================================
# Input loading
# ============================================================

def load_jd_data(
    path: str,
    preferred_text_col: str,
) -> Tuple[pd.DataFrame, str]:
    df = pd.read_csv(path)

    if preferred_text_col in df.columns:
        print(f"[JD] Using text column: {preferred_text_col}")
        return df, preferred_text_col

    if FALLBACK_TEXT_COL not in df.columns:
        raise ValueError(
            f"Neither '{preferred_text_col}' nor '{FALLBACK_TEXT_COL}' exists.\n"
            f"Available columns: {df.columns.tolist()}"
        )

    print(
        f"[JD] '{preferred_text_col}' not found. "
        f"Creating it from '{FALLBACK_TEXT_COL}'."
    )
    df[preferred_text_col] = df[FALLBACK_TEXT_COL].apply(clean_description)
    return df, preferred_text_col


def load_taxonomy(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing = [
        c for c in REQUIRED_TAXONOMY_COLUMNS
        if c not in df.columns
    ]
    if missing:
        raise ValueError(
            f"Taxonomy missing columns: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    df = df.copy()

    for col in [
        "skill",
        "skill_id",
        "definition",
        "category",
        "subcategory",
    ]:
        df[col] = df[col].fillna("").astype(str).str.strip()

    if (df["skill"] == "").any():
        raise ValueError("Taxonomy contains empty skill names.")

    if (df["skill_id"] == "").any():
        raise ValueError("Taxonomy contains empty skill_id values.")

    df["retrieval_text"] = (
        "Skill name: "
        + df["skill"]
        + "\nSkill definition: "
        + df["definition"]
    )

    return df.reset_index(drop=True)


# ============================================================
# Taxonomy embeddings
# ============================================================

def taxonomy_fingerprint(
    df: pd.DataFrame,
    embed_model_name: str,
) -> str:
    h = hashlib.sha256()
    h.update(embed_model_name.encode("utf-8"))

    for row in df[
        ["skill_id", "skill", "definition"]
    ].itertuples(index=False):
        h.update(str(row.skill_id).encode("utf-8"))
        h.update(b"\0")
        h.update(str(row.skill).encode("utf-8"))
        h.update(b"\0")
        h.update(str(row.definition).encode("utf-8"))
        h.update(b"\n")

    return h.hexdigest()[:16]


def build_or_load_taxonomy_embeddings(
    taxonomy: pd.DataFrame,
    embedder: SentenceTransformer,
    cache_dir: Path,
    embed_model_name: str,
) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)

    fingerprint = taxonomy_fingerprint(
        taxonomy,
        embed_model_name,
    )
    safe_model = re.sub(
        r"[^A-Za-z0-9_.-]+",
        "_",
        embed_model_name,
    )
    cache_file = (
        cache_dir
        / f"taxonomy_{safe_model}_{fingerprint}.npy"
    )

    if cache_file.exists():
        print(
            f"[Retrieve] Loading cached taxonomy embeddings: "
            f"{cache_file}"
        )
        arr = np.load(cache_file)
    else:
        print(
            "[Retrieve] Encoding taxonomy "
            "(skill + definition)..."
        )
        arr = embedder.encode(
            taxonomy["retrieval_text"].tolist(),
            batch_size=256,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        np.save(cache_file, arr)
        print(
            f"[Retrieve] Saved taxonomy embedding cache: "
            f"{cache_file}"
        )

    return np.ascontiguousarray(
        arr,
        dtype=np.float32,
    )


def build_faiss_index(
    taxonomy_embeddings: np.ndarray,
) -> Any:
    """Build an exact cosine-similarity FAISS index for taxonomy vectors.

    Embeddings are normalized at encode time, so inner product is equivalent
    to cosine similarity.  This index deliberately stays exact (IndexFlatIP)
    to preserve the retrieval behavior of the previous torch implementation.
    """
    vectors = np.ascontiguousarray(
        taxonomy_embeddings,
        dtype=np.float32,
    )

    if vectors.ndim != 2 or len(vectors) == 0:
        raise ValueError(
            "Taxonomy embeddings must be a non-empty 2D array."
        )

    index = faiss.IndexFlatIP(int(vectors.shape[1]))
    index.add(vectors)
    return index


# ============================================================
# Infer prompt + evidence alignment
# ============================================================

def load_prompt(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {p}"
        )

    text = p.read_text(
        encoding="utf-8"
    ).strip()

    if not text:
        raise ValueError(
            f"Prompt file is empty: {p}"
        )

    return text


def split_jd_units(jd_text: str) -> List[str]:
    """
    Split a cleaned JD into sentence/requirement units.
    This is used only after Infer to recover the full source unit.
    The LLM does not see or predict these unit IDs.
    """
    units: List[str] = []

    for raw_line in str(jd_text).splitlines():
        line = raw_line.strip()
        if not line:
            continue

        line = re.sub(
            r"^\s*[-•*]\s*",
            "",
            line,
        ).strip()

        if not line:
            continue

        parts = re.split(
            r"(?<=[.!?])\s+(?=[A-Z0-9])",
            line,
        )

        for part in parts:
            part = part.strip()
            if part:
                units.append(part)

    if not units and str(jd_text).strip():
        units = [str(jd_text).strip()]

    return units


def extract_json_object(text: str) -> Dict[str, Any]:
    text = (text or "").strip()

    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    match = re.search(
        r"\{.*\}",
        text,
        flags=re.DOTALL,
    )
    if match:
        try:
            obj = json.loads(
                match.group(0)
            )
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    try:
        obj = ast.literal_eval(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    raise ValueError(
        "Could not parse LLM JSON output: "
        + text[:1500]
    )


def normalize_alignment_text(
    text: str,
) -> str:
    """
    Only whitespace and case are normalized.
    No fuzzy semantic matching is used.
    """
    return re.sub(
        r"\s+",
        " ",
        str(text),
    ).strip().casefold()


def align_evidence_to_unit(
    evidence: str,
    jd_units: List[str],
) -> Optional[int]:
    """
    Evidence must align to exactly one JD unit.
    Ambiguous or non-verbatim evidence is rejected.
    """
    evidence_key = normalize_alignment_text(
        evidence
    )

    if not evidence_key:
        return None

    exact_matches = [
        i
        for i, unit in enumerate(jd_units)
        if normalize_alignment_text(unit)
        == evidence_key
    ]

    if len(exact_matches) == 1:
        return exact_matches[0]

    containing_matches = [
        i
        for i, unit in enumerate(jd_units)
        if evidence_key
        in normalize_alignment_text(unit)
    ]

    if len(containing_matches) == 1:
        return containing_matches[0]

    return None


def normalize_concepts(
    items: Any,
    jd_units: List[str],
    max_concepts: Optional[int],
) -> List[Dict[str, Any]]:
    if not isinstance(items, list):
        return []

    out: List[Dict[str, Any]] = []
    seen = set()

    for item in items:
        if not isinstance(item, dict):
            continue

        skill = re.sub(
            r"\s+",
            " ",
            str(
                item.get(
                    "skill",
                    "",
                )
            ).strip(),
        )

        evidence = re.sub(
            r"\s+",
            " ",
            str(
                item.get(
                    "evidence",
                    "",
                )
            ).strip(),
        )

        if not skill or not evidence:
            continue

        key = skill.casefold()
        if key in seen:
            continue

        unit_index = align_evidence_to_unit(
            evidence,
            jd_units,
        )

        # Unaligned evidence never enters Retrieve.
        if unit_index is None:
            continue

        seen.add(key)

        out.append(
            {
                "skill": skill,
                "evidence_id": f"E{unit_index}",
                "source_sentence": jd_units[
                    unit_index
                ],
            }
        )

        if (
            max_concepts is not None
            and len(out) >= max_concepts
        ):
            break

    return out


# ============================================================
# Retrieve
# ============================================================

def normalize_skill_key(
    s: str,
) -> str:
    s = str(s).casefold().strip()
    return re.sub(
        r"\s+",
        " ",
        s,
    )


def build_exact_skill_lookup(
    taxonomy: pd.DataFrame,
) -> Dict[str, int]:
    lookup: Dict[str, int] = {}

    for idx, skill in taxonomy[
        "skill"
    ].items():
        lookup.setdefault(
            normalize_skill_key(skill),
            int(idx),
        )

    return lookup


def lexical_tokens(
    s: str,
) -> set:
    return set(
        re.findall(
            r"[a-z0-9+#.]+",
            str(s).casefold(),
        )
    )


def build_lexical_skill_index(
    taxonomy: pd.DataFrame,
) -> Dict[str, Any]:
    names: List[str] = []
    token_sets: List[set] = []
    inverted: Dict[
        str,
        List[int],
    ] = {}

    for idx, skill in taxonomy[
        "skill"
    ].items():
        name = normalize_skill_key(
            skill
        )
        tokens = lexical_tokens(
            name
        )

        names.append(name)
        token_sets.append(tokens)

        for token in tokens:
            inverted.setdefault(
                token,
                [],
            ).append(
                int(idx)
            )

    return {
        "names": names,
        "token_sets": token_sets,
        "inverted": inverted,
    }


def lexical_top_candidates(
    query: str,
    lexical_index: Dict[str, Any],
    top_k: int,
) -> List[Tuple[int, float]]:
    q_norm = normalize_skill_key(
        query
    )
    q_tokens = lexical_tokens(
        q_norm
    )

    pool = set()

    for token in q_tokens:
        pool.update(
            lexical_index[
                "inverted"
            ].get(
                token,
                [],
            )
        )

    if not pool:
        return []

    scored: List[
        Tuple[int, float]
    ] = []

    for idx in pool:
        name = lexical_index[
            "names"
        ][idx]
        name_tokens = lexical_index[
            "token_sets"
        ][idx]

        seq = SequenceMatcher(
            None,
            q_norm,
            name,
        ).ratio()

        union = (
            q_tokens
            | name_tokens
        )

        jaccard = (
            len(
                q_tokens
                & name_tokens
            )
            / len(union)
            if union
            else 0.0
        )

        score = (
            0.65 * seq
            + 0.35 * jaccard
        )

        if q_norm and (
            q_norm in name
            or name in q_norm
        ):
            score += 0.15

        scored.append(
            (
                int(idx),
                float(score),
            )
        )

    scored.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    return scored[
        :max(
            1,
            int(top_k),
        )
    ]


def retrieve_concepts(
    concepts: List[Dict[str, Any]],
    taxonomy: pd.DataFrame,
    faiss_index: Any,
    embedder: SentenceTransformer,
    exact_lookup: Dict[str, int],
    lexical_index: Dict[str, Any],
    top_k: int = DEFAULT_RETRIEVE_TOP_K,
    lexical_top_k: int = DEFAULT_LEXICAL_TOP_K,
    lexical_min_score: float = DEFAULT_LEXICAL_MIN_SCORE,
    embed_lock: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    if not concepts:
        return []

    exact_results: Dict[
        int,
        Dict[str, Any],
    ] = {}

    semantic_positions: List[int] = []
    semantic_queries: List[str] = []

    for pos, concept_item in enumerate(
        concepts
    ):
        concept = str(
            concept_item["skill"]
        )
        key = normalize_skill_key(
            concept
        )

        if key in exact_lookup:
            idx = exact_lookup[key]
            row = taxonomy.iloc[idx]

            exact_results[pos] = {
                "concept": concept,
                "evidence_id":
                    concept_item.get(
                        "evidence_id"
                    ),
                "source_sentence":
                    concept_item.get(
                        "source_sentence",
                        "",
                    ),
                "taxonomy_index": idx,
                "skill_id":
                    row["skill_id"],
                "skill":
                    row["skill"],
                "category":
                    row["category"],
                "subcategory":
                    row["subcategory"],
                "definition":
                    row["definition"],
                "retrieval_method":
                    "exact_name",
                "semantic_rank": None,
                "lexical_rank": None,
                "lexical_similarity_audit":
                    None,
                "similarity": 1.0,
            }
        else:
            semantic_positions.append(
                pos
            )

            source_sentence = str(
                concept_item.get(
                    "source_sentence",
                    "",
                )
            ).strip()

            semantic_queries.append(
                "Skill concept: "
                + concept
                + "\nSource sentence: "
                + source_sentence
            )

    candidates_by_pos: Dict[
        int,
        List[Dict[str, Any]],
    ] = {}

    if semantic_queries:
        def encode_queries() -> np.ndarray:
            return embedder.encode(
                semantic_queries,
                batch_size=128,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype("float32")

        if embed_lock is None:
            q_np = encode_queries()
        else:
            with embed_lock:
                q_np = encode_queries()

        semantic_k = max(
            1,
            min(
                int(top_k),
                int(faiss_index.ntotal),
            ),
        )

        (
            top_scores,
            top_indices,
        ) = faiss_index.search(
            np.ascontiguousarray(q_np),
            semantic_k,
        )

        for (
            local_i,
            original_pos,
        ) in enumerate(
            semantic_positions
        ):
            concept_item = concepts[
                original_pos
            ]
            concept = str(
                concept_item["skill"]
            )

            semantic_meta: Dict[
                int,
                Tuple[int, float],
            ] = {}

            for rank in range(
                semantic_k
            ):
                idx = int(
                    top_indices[
                        local_i,
                        rank,
                    ]
                )
                score = float(
                    top_scores[
                        local_i,
                        rank,
                    ]
                )
                semantic_meta[idx] = (
                    rank + 1,
                    score,
                )

            lexical_pairs = (
                lexical_top_candidates(
                    concept,
                    lexical_index,
                    lexical_top_k,
                )
            )

            lexical_meta = {
                idx: (
                    rank + 1,
                    score,
                )
                for (
                    rank,
                    (
                        idx,
                        score,
                    ),
                ) in enumerate(
                    lexical_pairs
                )
            }

            ordered_indices: List[
                int
            ] = []
            seen_indices = set()

            for rank in range(
                semantic_k
            ):
                idx = int(
                    top_indices[
                        local_i,
                        rank,
                    ]
                )

                if idx not in seen_indices:
                    ordered_indices.append(
                        idx
                    )
                    seen_indices.add(
                        idx
                    )

            for idx, lexical_score in lexical_pairs:
                if (
                    idx not in seen_indices
                    and lexical_score >= lexical_min_score
                ):
                    ordered_indices.append(
                        idx
                    )
                    seen_indices.add(
                        idx
                    )

            concept_candidates: List[
                Dict[str, Any]
            ] = []

            for idx in ordered_indices:
                row = taxonomy.iloc[idx]

                semantic_info = (
                    semantic_meta.get(
                        idx
                    )
                )
                lexical_info = (
                    lexical_meta.get(
                        idx
                    )
                )

                if (
                    semantic_info
                    and lexical_info
                ):
                    method = (
                        "semantic+lexical"
                    )
                elif semantic_info:
                    method = "semantic"
                else:
                    method = "lexical"

                concept_candidates.append(
                    {
                        "concept": concept,
                        "evidence_id":
                            concept_item.get(
                                "evidence_id"
                            ),
                        "source_sentence":
                            concept_item.get(
                                "source_sentence",
                                "",
                            ),
                        "taxonomy_index":
                            idx,
                        "skill_id":
                            row["skill_id"],
                        "skill":
                            row["skill"],
                        "category":
                            row["category"],
                        "subcategory":
                            row["subcategory"],
                        "definition":
                            row["definition"],
                        "retrieval_method":
                            method,
                        "semantic_rank":
                            (
                                semantic_info[0]
                                if semantic_info
                                else None
                            ),
                        "lexical_rank":
                            (
                                lexical_info[0]
                                if lexical_info
                                else None
                            ),
                        "lexical_similarity_audit":
                            (
                                lexical_info[1]
                                if lexical_info
                                else None
                            ),
                        "similarity":
                            float(
                                semantic_info[1]
                                if semantic_info
                                else np.dot(
                                    q_np[local_i],
                                    faiss_index.reconstruct(idx),
                                )
                            ),
                    }
                )

            candidates_by_pos[
                original_pos
            ] = concept_candidates

    results: List[
        Dict[str, Any]
    ] = []

    for pos in range(
        len(concepts)
    ):
        if pos in exact_results:
            results.append(
                exact_results[pos]
            )
        else:
            results.extend(
                candidates_by_pos.get(
                    pos,
                    [],
                )
            )

    return results


