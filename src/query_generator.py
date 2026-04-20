import json
import anthropic
from dotenv import load_dotenv

load_dotenv()

_client = anthropic.Anthropic()

SYSTEM_PROMPT = """You are a query parser for the NHTSA (National Highway Traffic Safety Administration) vehicle safety database.

Convert natural language questions about vehicle safety into structured JSON queries.

Available endpoints:
- recalls: Vehicle recall campaigns issued by manufacturers
- complaints: Consumer safety complaints filed with NHTSA
- safetyRatings: NHTSA crash test safety ratings

Output format for a valid query:
{"endpoint": "recalls|complaints|safetyRatings", "make": "UPPERCASE", "model": "UPPERCASE", "year": "YYYY"}

Output format for an error:
{"error": "missing_year|missing_make|out_of_scope|ambiguous_input", "message": "brief explanation"}

Rules:
1. Output ONLY a valid JSON object. No markdown, no explanation, nothing else.
2. Normalize make and model to UPPERCASE.
3. Correct typos: Toyata→TOYOTA, Hond→HONDA, Chevvy→CHEVROLET, kia sorrento→KIA SORENTO.
4. Infer make when unambiguous: Camry→TOYOTA, Civic/Accord→HONDA, Mustang/F-150/Explorer→FORD, Silverado→CHEVROLET.
5. Infer endpoint from keywords:
   - "recall", "召回", "campaign" → recalls
   - "complaint", "problem", "issue", "投訴", "問題" → complaints
   - "safety rating", "crash test", "安全評等", "安全評分" → safetyRatings
   - Default to recalls if endpoint is ambiguous.
6. If year is missing, return: {"error": "missing_year", "message": "Please specify the model year."}
7. If make cannot be determined, return: {"error": "missing_make", "message": "Please specify the vehicle make and model."}
8. If unrelated to vehicle safety, return: {"error": "out_of_scope", "message": "This tool only handles NHTSA vehicle safety queries."}
9. Handle English, Traditional Chinese, and Simplified Chinese input.
10. Write the message in the same language as the input query."""

_SCHEMA = {
    "type": "object",
    "properties": {
        "endpoint": {"type": "string"},
        "make": {"type": "string"},
        "model": {"type": "string"},
        "year": {"type": "string"},
        "error": {"type": "string"},
        "message": {"type": "string"},
    },
    "additionalProperties": False,
}


def generate_query(nl_input: str) -> dict:
    response = _client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=256,
        system=[{
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }],
        messages=[{"role": "user", "content": nl_input}],
        output_config={"format": {"type": "json_schema", "schema": _SCHEMA}},
    )
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)
