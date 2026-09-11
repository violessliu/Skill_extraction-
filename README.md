# Lightcast Skill Extraction with RAG

A job-description skill extraction pipeline grounded in the **Lightcast skill taxonomy**.

The pipeline uses **Qwen3-Embedding** for candidate retrieval, **Qwen3-Reranker** for candidate reranking, and **Qwen3.5** for final sentence-level skill selection.

## Pipeline

```text
Job Description
      ↓
Python sentence splitting
      ↓
Qwen3-Embedding-4B
      ↓
Retrieve Top-K Lightcast skills
      ↓
Qwen3-Reranker-4B
      ↓
Keep best candidates per sentence
      ↓
Qwen3.5-9B
      ↓
Sentence-level Lightcast skills
```

The Lightcast taxonomy is flattened to use only the original **level-3 skill names**. Category and subcategory information are not provided to the final LLM.

## Models

- **Embedding:** `Qwen/Qwen3-Embedding-4B`
- **Reranker:** `Qwen/Qwen3-Reranker-4B`
- **Final LLM:** `Qwen/Qwen3.5-9B`
- **Vector search:** FAISS

## Retrieval Design

For each sentence in a job description:

1. Qwen3-Embedding retrieves a broad candidate set from the Lightcast taxonomy.
2. Exact skill-name matching is added as a lexical safety net.
3. Qwen3-Reranker reranks the candidate skills.
4. The highest-ranked candidates are passed to the final LLM.
5. Qwen3.5 decides which candidates are actually supported by the sentence.

Default settings:

```text
retrieve_k = 32
top_k      = 8
```

Retrieval and reranking only generate candidates. A high similarity or reranker score does **not** automatically mean that a skill is extracted.

## JD-Level Batch Processing

The pipeline supports batching multiple job descriptions for better GPU utilization.

For example:

```text
8 JDs
  ↓
Batch sentence embedding
  ↓
Batch candidate reranking
  ↓
8 independent JD prompts
  ↓
vLLM batched generation
```

Each job description remains an independent prompt. Different JDs are **not concatenated into one context**.

The main batch controls are:

```text
--batch-size
--embedding-batch-size
--reranker-batch-size
```

## Installation

```bash
pip install -r requirements.txt
```

Main dependencies include:

- PyTorch
- vLLM
- Transformers
- Sentence Transformers
- FAISS
- pandas
- NumPy

## Build the Taxonomy Index

Before running extraction, build the FAISS index:

```bash
python build_taxonomy_index.py \
  --embedding-model Qwen/Qwen3-Embedding-4B \
  --device cuda \
  --batch-size 32
```

By default, only skill names are embedded.

To include Lightcast definitions:

```bash
python build_taxonomy_index.py \
  --embedding-model Qwen/Qwen3-Embedding-4B \
  --device cuda \
  --batch-size 32 \
  --use-definition
```

The same `--use-definition` setting must be used when running extraction.

## Run Extraction

Example:

```bash
python -u run_extraction.py \
  --model qwen3.5-9b \
  --embedding-device cuda:1 \
  --reranker-device cuda:1 \
  --tensor-parallel-size 1 \
  --retrieve-k 32 \
  --top-k 8 \
  --batch-size 8 \
  --embedding-batch-size 32 \
  --reranker-batch-size 8 \
  --checkpoint-every 10
```

For datasets with a row count different from the default pilot dataset, use:

```bash
--expected-rows 0
```

## Output

Sentence-level predictions are written to:

```text
predictions/<model>/sentence_skills.csv
```

Main columns:

```text
source_index
JOB_HASH
model_alias
sentence_id
sentence
skill
```

Example:

```text
Experience with Python and SQL is required.
→ Python
→ SQL
```

## Project Structure

```text
.
├── run_extraction.py
├── build_taxonomy_index.py
├── clean_job_descriptions.py
├── requirements.txt
├── prompts/
│   ├── system.txt
│   └── extract_sentences.txt
├── src/job_skill_rag/
├── data/
├── taxonomy/
├── index/
└── predictions/
```

## Notes

- A sentence may contain zero, one, or multiple skills.
- The final LLM can only select from the candidates retrieved for that sentence.
- Skill names are preserved exactly as defined in the Lightcast taxonomy.
- If GPU memory is limited, reduce `--batch-size` first.

## License

This repository is intended for research and academic use. Please ensure that any use of the Lightcast taxonomy complies with the applicable Lightcast data license and terms.
