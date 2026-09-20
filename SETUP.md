# Source-only setup

This repository intentionally excludes job-description data, the Lightcast
taxonomy CSV, run outputs, embedding caches, checkpoints, and credentials.

1. Create and activate a Python 3.10+ virtual environment.
2. Install dependencies with `python -m pip install -r requirements.txt`.
3. Copy `api_config.example.py` to `api_config.py`.
4. Set `TOGETHER_API_KEY` as an environment variable (recommended), or add it
   only to your untracked local `api_config.py`.
5. Supply your own licensed taxonomy CSV and job-description CSV when running
   `lightcast_infer_retrieve-mutimode-mpnet.py`.

The first run downloads `sentence-transformers/all-mpnet-base-v2` and creates
an `embedding_cache` inside the selected output directory.
