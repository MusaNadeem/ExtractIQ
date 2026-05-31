import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from app.schemas.invoice import InvoiceData

load_dotenv()

_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

_SYSTEM_PROMPT = """\
You are a document data extraction assistant. Extract invoice data from the provided text and return ONLY valid JSON — no markdown, no explanation, no code fences.

The JSON must match this schema exactly:
{
  "vendor_name": string | null,
  "vendor_address": string | null,
  "invoice_number": string | null,
  "invoice_date": "YYYY-MM-DD" | null,
  "due_date": "YYYY-MM-DD" | null,
  "line_items": [
    {
      "description": string | null,
      "quantity": number | null,
      "unit_price": number | null,
      "total": number | null
    }
  ],
  "subtotal": number | null,
  "tax": number | null,
  "total": number | null,
  "currency": string | null,
  "uncertain_fields": [list of field names you are not confident about]
}

Rules:
- Use null for any field not found or not determinable from the text.
- Dates must be in YYYY-MM-DD format.
- Numbers must be plain numeric values (no currency symbols).
- List field names in uncertain_fields whenever you are guessing or the source text is ambiguous.
- Return ONLY the JSON object.\
"""


def _call_openai(text: str) -> str:
    response = _client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
    )
    return response.choices[0].message.content.strip()


def _parse_response(raw: str) -> InvoiceData:
    # Strip accidental code fences the model may still emit
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return InvoiceData.model_validate(json.loads(raw))


def extract_invoice(text: str) -> InvoiceData:
    raw = _call_openai(text)
    try:
        return _parse_response(raw)
    except (json.JSONDecodeError, ValueError):
        # Retry once before giving up
        raw = _call_openai(text)
        try:
            return _parse_response(raw)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(
                f"OpenAI returned malformed JSON after two attempts.\n"
                f"Last response:\n{raw}"
            ) from exc
