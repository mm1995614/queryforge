import json
from pathlib import Path
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

_client = Groq()

PROMPT_VERSION = "v2_checklist_enforcement"
SYSTEM_PROMPT = (Path(__file__).parent.parent / "prompts" / f"{PROMPT_VERSION}.txt").read_text(encoding="utf-8")

SYNTHESIS_PROMPT = """You are a vehicle safety expert. The user asked a question and we fetched data from multiple NHTSA API queries to answer it completely.

Summarize the findings clearly and concisely for the user. Group results by query type if there are multiple. Respond in the same language as the user's original query."""


def generate_query(nl_input: str) -> dict:
    response = _client.chat.completions.create(
        model="llama-3.3-70b-versatile",
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
    response = _client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=1024,
        messages=[
            {"role": "system", "content": SYNTHESIS_PROMPT},
            {"role": "user", "content": f"User's original query: {original_query}\n\nAPI results:\n{results_text}"},
        ],
    )
    return response.choices[0].message.content
