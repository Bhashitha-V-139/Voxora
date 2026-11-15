# 🧠 Voxora – AI-Powered Meeting Intelligence Platform

Upload a meeting transcript and get a summary, key discussion points, decisions,
action items (with owners and deadlines **only when explicitly stated**), structured
JSON, and a Q&A chat grounded in the transcript.

**Stack:** Python · Streamlit · Gemini API (`google-genai`) · python-dotenv

## Features

- Upload `.txt` transcripts and preview them
- Meeting summary and key discussion points
- Decisions extracted only when clearly agreed on
- Action items table (owner / deadline show "Not specified" if absent) with CSV export
- Structured JSON output with download
- Meeting Q&A chat (answers only from the transcript)
- Loading spinners, input validation, and friendly API error messages

## Setup

Requires Python 3.9+.

```bash
# 1. Enter the project folder
cd voxora

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows (PowerShell / CMD)

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure your API key
cp .env.example .env             # Windows: copy .env.example .env
# Open .env and set GEMINI_API_KEY=<your key>
```

Get a free API key at <https://aistudio.google.com/apikey>.

## Run

```bash
streamlit run app.py
```

Open <http://localhost:8501>, then either upload a `.txt` transcript or click
**Load sample transcript**, and press **✨ Analyze Meeting**.

## How it works

| Step               | Detail                                                                                                                            |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------- |
| Analysis           | One Gemini call with `response_mime_type="application/json"` and a Pydantic `response_schema`, so the output is always structured |
| Anti-hallucination | Low temperature, strict system prompt, nullable `owner`/`deadline`, and an `evidence` quote per action item                       |
| Q&A                | Transcript goes in the system instruction; recent chat turns are sent as conversation history                                     |

## Configuration

| Variable         | Required | Default            |
| ---------------- | -------- | ------------------ |
| `GEMINI_API_KEY` | Yes      | –                  |
| `GEMINI_MODEL`   | No       | `gemini-2.5-flash` |

## Troubleshooting

- **"GEMINI_API_KEY not found"** – make sure `.env` exists in the same folder as `app.py`, then restart Streamlit.
- **"Model not found"** – set `GEMINI_MODEL` in `.env` to a model available to your key.
- **Rate limit / quota** – wait a minute and retry (free tier has per-minute limits).
- **Garbled text** – save your transcript as UTF-8.

## Project structure

```
voxora/
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Ideas for next steps

- Support PDF/DOCX/VTT transcripts
- Persist past meetings in SQLite or MongoDB
- Add a Flask/FastAPI backend with a React or Vue frontend
