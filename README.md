# ExtractIQ

AI-powered invoice data extraction. Upload a PDF or image and get back structured JSON.

## How it works

1. **Text extraction** — pdfplumber pulls text from digital PDFs directly.
2. **Routing** — if the text is long and clean enough, it goes straight to Gemini. If not (scanned/image-based), pages are converted to images and sent via the Gemini vision API. Tesseract OCR is a last-resort fallback.
3. **Validation** — extracted fields are checked for date logic, arithmetic consistency across line items, and numeric sanity. Issues are flagged in `uncertain_fields` rather than hard-failing.
4. **Review** — a Streamlit UI shows the original document alongside an editable form. Uncertain fields are highlighted. The result can be downloaded as JSON.

## Stack

| Layer | Library |
|---|---|
| LLM / Vision | Google Gemini (via `google-genai`) |
| PDF text | pdfplumber |
| PDF → images | pdf2image |
| OCR fallback | pytesseract |
| Data models | Pydantic v2 |
| API server | FastAPI + uvicorn |
| Frontend | Streamlit |

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# add your GEMINI_API_KEY to .env
```

## Run

```bash
# Streamlit UI
streamlit run frontend/streamlit_app.py

# FastAPI server
uvicorn app.main:app --reload
```
