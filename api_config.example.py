#!/usr/bin/env python3
"""Together AI credentials and transport settings.

Copy this file to ``api_config.py`` and keep the real file out of version
control. Prefer setting TOGETHER_API_KEY in the environment.
"""

import os
from typing import Optional


TOGETHER_API_KEY = ""

API_KEY_ENV_VARS = (
    "TOGETHER_API_KEY",
    "TOGETHERAI_API_KEY",
)

TOGETHER_BASE_URL = "https://api.together.xyz/v1"
BASE_URL_ENV_VAR = "TOGETHER_BASE_URL"

REQUEST_TIMEOUT = 180.0
MAX_TRANSPORT_RETRIES = 6
RETRY_BASE_DELAY = 2.0
RETRY_MAX_DELAY = 60.0
MAX_CONTENT_RETRIES = 3
INFER_WORKERS = 8


def resolve_api_key(cli_key: Optional[str] = None) -> str:
    if cli_key:
        return cli_key.strip()

    for env_var in API_KEY_ENV_VARS:
        value = os.environ.get(env_var, "").strip()
        if value:
            return value

    if TOGETHER_API_KEY.strip():
        return TOGETHER_API_KEY.strip()

    raise RuntimeError(
        "No Together AI API key found. "
        "Set TOGETHER_API_KEY or pass --llm_api_key."
    )


def resolve_base_url(cli_base_url: Optional[str] = None) -> str:
    if cli_base_url:
        return cli_base_url.strip().rstrip("/")

    env_value = os.environ.get(BASE_URL_ENV_VAR, "").strip()
    if env_value:
        return env_value.rstrip("/")

    return TOGETHER_BASE_URL.rstrip("/")


def mask_api_key(api_key: str) -> str:
    if not api_key:
        return "<empty>"
    if len(api_key) <= 8:
        return "*" * len(api_key)
    return f"{api_key[:4]}...{api_key[-4:]}"
