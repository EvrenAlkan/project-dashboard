"""
ai_client.py
────────────
A thin, provider-agnostic wrapper around the OpenAI-compatible
chat-completions endpoint.

Supports:
  • Local runners  – LM Studio  (http://localhost:1234/v1)
                   – Ollama     (http://localhost:11434/v1)
  • Online APIs    – OpenAI     (https://api.openai.com/v1)
                   – Gemini     (https://generativelanguage.googleapis.com/v1beta/openai)
                   – Any other OpenAI-compatible proxy

All settings are read from environment variables (with sensible defaults)
so nothing needs to be hard-coded.  Override them in a .env file or in
your shell before starting the Flask server:

    AI_BASE_URL   – Base URL up to /v1  (default: http://localhost:1234/v1)
    AI_MODEL      – Model identifier    (default: "auto" → query /v1/models)
    AI_API_KEY    – API key             (default: "lm-studio" – ignored by local runners)
    AI_TIMEOUT    – Request timeout s   (default: 300)
    AI_MAX_TOKENS – Max tokens per call (default: 2048, 0 = provider default)
    AI_TEMPERATURE– Sampling temp       (default: 0.2)
    CONTEXT_TURNS – How many past turns to keep in history (default: 4)
"""

from __future__ import annotations

import json
import logging
import os
import time
import requests

logger = logging.getLogger(__name__)

# ── Default configuration (overridden by env vars) ─────────────────────────

DEFAULT_CONFIG: dict = {
    "base_url":      "http://localhost:1234/v1",
    "model":         "google/gemma-3-12b",  # override with AI_MODEL in .env
    "api_key":       "lm-studio",   # local runners ignore this value
    "timeout":       300,
    "max_tokens":    2048,          # set to 0 to omit (use provider default)
    "temperature":   0.2,
    "context_turns": 1,             # past turn pairs to keep
}


def _load_config() -> dict:
    """Read config from env vars, falling back to DEFAULT_CONFIG."""
    cfg = dict(DEFAULT_CONFIG)
    if val := os.getenv("AI_BASE_URL"):
        cfg["base_url"] = val.rstrip("/")
    if val := os.getenv("AI_MODEL"):
        cfg["model"] = val
    if val := os.getenv("AI_API_KEY"):
        cfg["api_key"] = val
    if val := os.getenv("AI_TIMEOUT"):
        cfg["timeout"] = int(val)
    if val := os.getenv("AI_MAX_TOKENS"):
        cfg["max_tokens"] = int(val)
    if val := os.getenv("AI_TEMPERATURE"):
        cfg["temperature"] = float(val)
    if val := os.getenv("CONTEXT_TURNS"):
        cfg["context_turns"] = int(val)
    return cfg


# Loaded once at import time; call reload_config() to refresh.
_config: dict = _load_config()


def reload_config() -> dict:
    """Re-read environment variables and return the updated config."""
    global _config
    _config = _load_config()
    logger.info("AI client config reloaded: base_url=%s model=%s",
                _config["base_url"], _config["model"])
    return _config


def get_config() -> dict:
    """Return a copy of the current configuration."""
    return dict(_config)



# ── Core call ───────────────────────────────────────────────────────────────

def chat(
    system_prompt: str,
    conversation: list[dict],
    user_message: str,
    *,
    config: dict | None = None,
) -> str:
    """
    Send a chat-completions request and return the assistant's reply text.

    Parameters
    ----------
    system_prompt : str
        The system role message prepended to every request.
    conversation : list[dict]
        Prior turns as ``[{"role": "user"|"assistant", "content": str}, …]``.
        Only the last ``config["context_turns"]`` entries are forwarded.
    user_message : str
        The latest user turn to add.
    config : dict | None
        Optional per-call config overrides (same keys as DEFAULT_CONFIG).
        Falls back to the module-level config if None.

    Returns
    -------
    str
        The assistant reply, or an empty string on parse failure.
    """
    cfg = dict(_config)
    if config:
        cfg.update(config)

    model = cfg["model"]
    if not model:
        raise RuntimeError(
            "AI_MODEL is not set. Add it to your .env file, e.g.:\n"
            "  AI_MODEL=google/gemma-3-12b"
        )

    # Trim history to stay within context limits
    max_turns = cfg["context_turns"]
    trimmed = conversation[-(max_turns * 2):] if conversation else []

    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    messages.extend(trimmed)
    messages.append({"role": "user", "content": user_message})

    payload: dict = {
        "model":       model,
        "messages":    messages,
        "temperature": cfg["temperature"],
    }
    if cfg["max_tokens"]:
        payload["max_tokens"] = cfg["max_tokens"]

    headers = {
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {cfg['api_key']}",
    }

    url = f"{cfg['base_url']}/chat/completions"
    payload_chars = sum(len(m["content"]) for m in messages)
    logger.info("⇒ POST %s  model=%s  messages=%d  ~%d chars",
                url, model, len(messages), payload_chars)

    t0 = time.time()
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=cfg["timeout"], verify=False)
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            f"\n❌  Cannot connect to AI server at: {cfg['base_url']}\n"
            "   Check that LM Studio / Ollama / your API provider is running\n"
            "   and that AI_BASE_URL in .env points to the correct address.\n"
            f"   Detail: {exc}"
        ) from exc
    except requests.exceptions.Timeout:
        elapsed = time.time() - t0
        raise RuntimeError(
            f"\n⏱  AI request timed out after {elapsed:.0f}s "
            f"(limit = {cfg['timeout']}s).\n"
            "   The model may still be loading — try again in a moment,\n"
            "   or increase AI_TIMEOUT in your .env file."
        ) from None
    elapsed = time.time() - t0

    if not resp.ok:
        try:
            err_body = resp.json()
        except Exception:
            err_body = resp.text
        raise RuntimeError(
            f"\n❌  AI server returned HTTP {resp.status_code} {resp.reason}\n"
            f"   Provider message: {err_body}"
        )

    try:
        data = resp.json()
    except Exception:
        data = resp.text
        
    logger.info("← %s  %.1fs", model, elapsed)

    # Standard OpenAI shape
    if isinstance(data, dict):
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            text = (
                data.get("content")
                or data.get("response")
                or data.get("text")
                or ""
            )
    elif isinstance(data, str):
        text = data
    else:
        text = str(data)
        
    if not text:
        logger.warning("Unexpected response shape from %s: %s", model, data)

    # Always return a string
    if not isinstance(text, str):
        text = str(text)

    # Print a preview of the response for the user to debug
    preview = text if len(text) < 300 else text[:300] + " ... [truncated]"
    logger.info("Raw AI Response:\n%s\n" % preview)

    return text
