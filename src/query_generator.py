import json
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Provider switch ─────────────────────────────────────────────────────────────
# The CLI tool's query generator can run on a cloud model (default) or a local
# Ollama model, so the whole tool can operate fully offline / on-prem.
#   QUERYFORGE_PROVIDER = groq | openai | ollama   (default: groq)
#   QUERYFORGE_MODEL    = model id                 (default: llama-3.3-70b-versatile)
# Examples:
#   QUERYFORGE_PROVIDER=ollama QUERYFORGE_MODEL=qwen2.5:7b-instruct python cli.py "..."
QUERYFORGE_PROVIDER = os.getenv("QUERYFORGE_PROVIDER", "groq").lower()
QUERYFORGE_MODEL = os.getenv("QUERYFORGE_MODEL", "llama-3.3-70b-versatile")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")

_client = None


def _get_client():
    """Lazily build the chat client for the configured provider.

    All three providers expose the OpenAI chat-completions API, so the rest of
    this module is provider-agnostic.
    """
    global _client
    if _client is None:
        if QUERYFORGE_PROVIDER == "groq":
            from groq import Groq
            _client = Groq()
        elif QUERYFORGE_PROVIDER == "openai":
            import openai
            _client = openai.OpenAI()
        elif QUERYFORGE_PROVIDER == "ollama":
            import openai
            # Ollama's OpenAI-compatible endpoint; no real API key required.
            _client = openai.OpenAI(base_url=f"{OLLAMA_BASE_URL}/v1", api_key="ollama")
        else:
            raise ValueError(
                f"Unsupported QUERYFORGE_PROVIDER={QUERYFORGE_PROVIDER!r}. "
                "Use groq, openai, or ollama."
            )
    return _client


# ── Prompts ─────────────────────────────────────────────────────────────────────
# Each prompt is a standalone file under prompts/ (see prompts/CHANGELOG.md).

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def _load_prompt(name: str) -> str:
    return (_PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8")


# The active, engineered "with-harness" prompt.
PROMPT_VERSION = "v2_checklist_enforcement"
SYSTEM_PROMPT = _load_prompt(PROMPT_VERSION)

# Bare "no-harness" baseline: task + schema only, no hardening rules. Used to
# measure a model's raw structured-output ability before any prompt engineering.
MINIMAL_PROMPT_VERSION = "minimal_noharness"
MINIMAL_PROMPT = _load_prompt(MINIMAL_PROMPT_VERSION)

SYNTHESIS_PROMPT = """You are a vehicle safety expert. The user asked a question and we fetched data from multiple NHTSA API queries to answer it completely.

Summarize the findings clearly and concisely for the user. Group results by query type if there are multiple. Respond in the same language as the user's original query."""


def generate_query(nl_input: str) -> dict:
    response = _get_client().chat.completions.create(
        model=QUERYFORGE_MODEL,
        max_tokens=512,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": nl_input},
        ],
    )
    return json.loads(response.choices[0].message.content)


def synthesize_results(original_query: str, all_results: list) -> str:
    results_text = json.dumps(all_results, ensure_ascii=False, indent=2)
    response = _get_client().chat.completions.create(
        model=QUERYFORGE_MODEL,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": SYNTHESIS_PROMPT},
            {"role": "user", "content": f"User's original query: {original_query}\n\nAPI results:\n{results_text}"},
        ],
    )
    return response.choices[0].message.content
