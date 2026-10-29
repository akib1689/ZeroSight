"""ChatLiteLLM factory — builds langchain-compatible models from configs/models.yaml.

Direct-mode only: calls providers (NVIDIA NIM, Z.AI, Groq) via litellm prefixes.
No proxy server required.
"""

import logging
import os
from pathlib import Path

import litellm
import yaml
from langchain_litellm import ChatLiteLLM

litellm.suppress_debug_info = True
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

_CONFIGS_DIR = Path(__file__).resolve().parent / "configs"


def _load_models_yaml() -> dict:
    with open(_CONFIGS_DIR / "models.yaml") as f:
        return yaml.safe_load(f)


def list_models() -> list[str]:
    """Return available model config names."""
    raw = _load_models_yaml()
    return sorted(k for k in raw if not k.endswith("-verdict"))


def build_chat_model(config_name: str) -> ChatLiteLLM:
    """Build a ChatLiteLLM from configs/models.yaml.

    Args:
        config_name: Key in models.yaml (e.g. "nvidia-gpt-oss-120b").

    Returns:
        A ChatLiteLLM instance configured for direct provider calls.
    """
    raw = _load_models_yaml()

    if config_name not in raw:
        available = sorted(raw.keys())
        raise ValueError(
            f"Model config '{config_name}' not found in models.yaml. "
            f"Available: {available}"
        )

    cfg = raw[config_name]

    api_key_env = cfg.get("api_key_env", "LITELLM_API_KEY")
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise ValueError(
            f"{api_key_env} environment variable is not set "
            f"(required by model config '{config_name}')"
        )

    kwargs = dict(
        model=cfg["model"],
        api_key=api_key,
        temperature=cfg.get("temperature", 0.0),
        max_tokens=cfg.get("max_completion_tokens", 1024),
        timeout=cfg.get("timeout", 120),
    )

    base_url = cfg.get("api_base")
    if not base_url and cfg.get("api_base_env"):
        api_base_env = cfg["api_base_env"]
        base_url = os.getenv(api_base_env)
        if not base_url:
            raise ValueError(
                f"{api_base_env} environment variable is not set "
                f"(required by model config '{config_name}')"
            )
    if base_url:
        kwargs["api_base"] = base_url

    num_retries = cfg.get("num_retries")
    if num_retries is not None:
        kwargs["num_retries"] = num_retries

    top_p = cfg.get("top_p")
    if top_p is not None:
        kwargs["top_p"] = top_p

    extra_body = cfg.get("extra_body")
    if extra_body:
        kwargs["model_kwargs"] = {"extra_body": extra_body}

    return ChatLiteLLM(**kwargs)
