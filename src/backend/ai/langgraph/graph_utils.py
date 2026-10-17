import os
import dotenv
import hashlib
from typing import Optional
from functools import lru_cache
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig

from src.backend.ai.langgraph.graph_states import AgentState

dotenv.load_dotenv()


@lru_cache(maxsize=8)
def _build_llm(model_name: str, base_url: str, temperature: float) -> ChatOpenAI:
    """Cached so repeated nodes reuse one client. Args are hashable on purpose.

    NOTE: no fixed seed here. A fixed seed (the old seed=42) makes identical prompts return
    identical text, which collapses pass@N -> pass@1. Sampling pins a FRESH random seed per call
    via `_seeded` — only meaningful at temperature > 0 (at temp 0 decoding is greedy/deterministic
    and the seed has no effect).
    """
    return ChatOpenAI(
        base_url=base_url,
        api_key="not-needed",
        model=model_name,
        temperature=temperature,
        max_tokens=2048,
        max_retries=0,
        timeout=120,
    )


def get_llm(
    config: RunnableConfig,
) -> ChatOpenAI:
    """Build (or reuse) the local llama.cpp-backed client.
    - model_name comes from config; base_url from env (defaults to the local llama-server).
    """
    cfg = config["configurable"]
    model_name = cfg.get("model_name", "qwen")
    base_url = os.getenv("LOCAL_LLM_URL", "http://localhost:18000/v1")
    temp = cfg.get("temperature", 0.0)

    llm = _build_llm(model_name, base_url, temp)

    return llm


def _seeded(llm: ChatOpenAI, seed: int) -> ChatOpenAI:
    """Return a copy of the cached client pinned to a fresh `seed`, sharing the same underlying
    HTTP client (verified: model_copy does not rebuild the connection pool). Distinct seeds make
    the N samples genuinely diverse instead of duplicates (only at temperature > 0)."""
    return llm.model_copy(update={"seed": seed})


# ───────────────────────── Answer extraction / comparison ─────────────────────────


def _sample_idx(state: AgentState, config: RunnableConfig) -> str:
    """Per-sample tag for log lines, so logs from concurrently-running GSM8K questions don't
    interleave into something that looks impossible. Prefers config['configurable']['sample_idx']
    (set by the eval harness); otherwise falls back to a short stable hash of the question so every
    log line within one invocation shares a tag even before the eval wires sample_idx. Logging only:
    not stored in state, not a metric."""
    cfg = config.get("configurable", {}) if isinstance(config, dict) else {}
    idx = cfg.get("sample_idx")
    if idx is not None:
        return str(idx)
    q = state.get("user_query") or ""
    return hashlib.sha1(q.encode("utf-8")).hexdigest()[:6] if q else "?"


# ── TEMPORARY token-usage debug logging (console only; nothing stored in state/CSV/metrics) ──
# Remove this block when context-usage inspection is no longer needed.


def _estimate_tokens(text: str) -> int:
    """Approximate token count when the API doesn't report usage. Prefers tiktoken; falls back to a
    ~4-chars/token heuristic. Estimate only — used purely for the temporary debug print."""
    if not text:
        return 0
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return max(1, (len(text) + 3) // 4)


def _token_usage_from_message(msg) -> Optional[tuple]:
    """(prompt, completion, total) from an LLM response message if the API exposed usage, else None.
    Checks usage_metadata first, then response_metadata['token_usage'] / ['usage']."""
    if msg is None:
        return None
    um = getattr(msg, "usage_metadata", None)
    if um:
        p, c, t = um.get("input_tokens"), um.get("output_tokens"), um.get("total_tokens")
        if p is not None or c is not None or t is not None:
            p, c = p or 0, c or 0
            return p, c, (t if t is not None else p + c)
    rm = getattr(msg, "response_metadata", None) or {}
    tu = rm.get("token_usage") or rm.get("usage")
    if tu:
        p = tu.get("prompt_tokens", tu.get("input_tokens"))
        c = tu.get("completion_tokens", tu.get("output_tokens"))
        t = tu.get("total_tokens")
        if p is not None or c is not None or t is not None:
            p, c = p or 0, c or 0
            return p, c, (t if t is not None else p + c)
    return None


def _log_token_usage(
    idx: str, label: str, msg=None, prompt_text: str = "", completion_text: str = ""
) -> None:
    """TEMPORARY: print prompt/completion/total tokens for one LLM call. Uses real usage metadata
    when available, otherwise an approximate tokenizer estimate. Console only — stores nothing."""
    usage = _token_usage_from_message(msg)
    if usage is not None:
        p, c, t = usage
    else:
        p = _estimate_tokens(prompt_text)
        c = _estimate_tokens(completion_text)
        t = p + c
    print(f"[IDX {idx}] {label}")
    print(f"Prompt Tokens: {p}")
    print(f"Completion Tokens: {c}")
    print(f"Total Tokens: {t}\n")


async def invoke_model(model, messages):
    """Invoke one model request and distinguish uncertain remote compute failures.

    Args:
        model: Chat client or bound runnable exposing async ainvoke.
        messages: Role/content messages passed to this single inference request.

    Returns:
        Model response after the request completes normally.

    Raises:
        InferenceStateUnknown: Connection loss, timeout or server failure leaves
            remote computation unconfirmed. No automatic model retry is attempted.
        Exception: Other model errors propagate; no error text becomes an answer.
    """
    import httpx
    from openai import APIConnectionError, APIStatusError
    from src.backend.ai.errors import InferenceStateUnknown

    try:
        return await model.ainvoke(messages)
    except (APIConnectionError, httpx.TransportError, TimeoutError) as error:
        raise InferenceStateUnknown("Remote inference state is unknown") from error
    except APIStatusError as error:
        if error.status_code >= 500:
            raise InferenceStateUnknown("Remote inference state is unknown") from error
        raise
