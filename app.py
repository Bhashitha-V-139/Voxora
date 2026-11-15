"""
Voxora - AI-Powered Meeting Intelligence Platform
Streamlit + Gemini (google-genai) MVP
"""

import hashlib
import json
import os
import time
import re
from datetime import datetime
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
load_dotenv()

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
MIN_CHARS = 100          # reject transcripts shorter than this
MAX_CHARS = 150_000      # keep requests within a safe size
CHAT_HISTORY_LIMIT = 10  # last N chat messages sent to the model

st.set_page_config(page_title="Voxora", page_icon="🧠", layout="wide")


# --------------------------------------------------------------------------
# Output schema (Gemini is forced to return JSON matching this shape)
# --------------------------------------------------------------------------
class Decision(BaseModel):
    decision: str
    context: Optional[str] = None


class ActionItem(BaseModel):
    task: str
    owner: Optional[str] = None
    deadline: Optional[str] = None
    evidence: Optional[str] = None


class MeetingAnalysis(BaseModel):
    meeting_title: Optional[str] = None
    participants: list[str] = []
    summary: str
    key_discussion_points: list[str] = []
    decisions: list[Decision] = []
    action_items: list[ActionItem] = []


ANALYSIS_SYSTEM_PROMPT = """You are Voxora, a meticulous meeting analyst.
Analyze the meeting transcript and return structured data.

STRICT RULES:
1. Use ONLY information found in the transcript. Never invent facts.
2. "decisions": include only things the group clearly agreed on or decided.
   Do not turn suggestions, open questions or ideas into decisions.
   If none were made, return an empty list.
3. "action_items": include only tasks someone clearly committed to or was
   assigned.
   - "owner": a person's name ONLY if explicitly stated; otherwise null.
   - "deadline": ONLY if a date/time frame is explicitly stated
     (e.g. "by Friday", "next sprint"); otherwise null. Never guess.
   - "evidence": a short quote (max 25 words) from the transcript.
4. "summary": 3-5 sentences, neutral and factual.
5. "key_discussion_points": 3-8 concise bullet-style strings.
6. "participants": speaker names that appear in the transcript (empty list
   if none can be identified).
7. "meeting_title": a short descriptive title, or null if unclear.
"""

QA_SYSTEM_PROMPT = """You are Voxora, an assistant that answers questions
about ONE meeting transcript.

RULES:
- Answer ONLY from the transcript below. Do not use outside knowledge.
- If the answer is not in the transcript, reply exactly:
  "That wasn't discussed in the transcript."
- Be concise. Mention who said something when it is relevant.
- Never invent decisions, owners or deadlines.

TRANSCRIPT:
"""

SAMPLE_TRANSCRIPT = """Product Sync - Q3 Launch Planning

Priya: Thanks everyone for joining. Let's talk about the dashboard launch. Where are we on the beta feedback?
Marcus: We collected 42 responses. Most people like the new layout, but the export feature is confusing.
Priya: Okay. Should we delay launch to fix export?
Sofia: I don't think we need to delay. We can ship export as-is and improve it in the next release.
Marcus: Agreed. The core dashboard is solid.
Priya: Alright, decision made: we launch on the planned date and treat export improvements as a follow-up.
Sofia: I'll write the release notes and send them to the team by Thursday.
Priya: Great. Marcus, can you look into the export confusion and propose a redesign?
Marcus: Sure, I'll start on that.
Priya: We also need someone to update the pricing page. Any volunteers?
Sofia: Let me think about it, I'm not sure about my bandwidth.
Priya: No problem, we'll come back to that. Let's also keep an eye on the support ticket volume after launch.
Marcus: One concern is the mobile layout, it still breaks on small screens.
Priya: Noted. We'll discuss mobile in the next meeting. Thanks, everyone.
"""



# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --bg: #080b16;
        --panel: rgba(19, 25, 45, .78);
        --panel-strong: #121a30;
        --line: rgba(148, 163, 184, .15);
        --muted: #98a5c2;
        --text: #edf2ff;
        --accent: #8b7cff;
        --accent-2: #36c8d7;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background:
          radial-gradient(ellipse at 12% 0%, rgba(83, 67, 190, .22), transparent 34%),
          radial-gradient(ellipse at 88% 8%, rgba(20, 169, 190, .13), transparent 30%),
          linear-gradient(145deg, #080b16 0%, #0b1020 52%, #080d19 100%);
        color: var(--text);
    }

    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"] { right: 1rem; }

    .block-container {
        padding-top: 2.2rem;
        padding-bottom: 4rem;
        max-width: 1440px;
    }

    h1, h2, h3, h4 {
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: -.035em;
        color: #f5f7ff;
    }

    p, label, .stCaption, [data-testid="stMarkdownContainer"] {
        color: #d5def2;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #10172a 0%, #0b1120 100%);
        border-right: 1px solid var(--line);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.5rem;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #f4f6ff;
    }

    [data-testid="stFileUploader"] {
        background: rgba(139, 124, 255, .07);
        border: 1px dashed rgba(139, 124, 255, .48);
        border-radius: 16px;
        padding: .55rem;
        transition: all .3s ease;
    }

    [data-testid="stFileUploader"]:hover {
        background: rgba(139, 124, 255, .13);
        border-color: #8b7cff;
        box-shadow: 0 8px 25px rgba(105, 87, 220, .16);
    }

    [data-testid="stFileUploader"] section {
        border: 0;
    }

    .hero {
        position: relative;
        overflow: hidden;
        background:
          radial-gradient(circle at 88% 12%, rgba(54, 200, 215, .24), transparent 27%),
          linear-gradient(115deg, rgba(80, 65, 190, .96), rgba(85, 68, 190, .88) 48%, rgba(24, 119, 151, .9));
        padding: 2.15rem 2.3rem;
        border: 1px solid rgba(255,255,255,.17);
        border-radius: 25px;
        color: white;
        margin-bottom: 1.6rem;
        box-shadow: 0 22px 70px rgba(0,0,0,.22);
    }

    .hero:after {
        content: "✦";
        position: absolute;
        right: 8%;
        top: 4%;
        font-size: 8rem;
        color: rgba(255,255,255,.09);
        transform: rotate(12deg);
        pointer-events: none;
    }

    .hero h1 {
        margin: 0;
        font-size: clamp(2.1rem, 4vw, 3rem);
        font-weight: 700;
        color: white;
    }

    .hero p {
        margin: .55rem 0 0;
        max-width: 680px;
        color: rgba(255,255,255,.82);
        font-size: 1.02rem;
    }

    .card {
        background: linear-gradient(145deg, rgba(23, 32, 56, .88), rgba(15, 22, 40, .82));
        border: 1px solid var(--line);
        border-radius: 19px;
        padding: 1.25rem 1.4rem;
        margin-bottom: 1rem;
        box-shadow: 0 12px 35px rgba(0,0,0,.13);
        backdrop-filter: blur(12px);
    }

    .card h4 {
        margin-top: 0;
        color: #f5f7ff;
    }

    .pill {
        display: inline-block;
        padding: 5px 12px;
        margin: 3px 5px 3px 0;
        border-radius: 999px;
        background: rgba(139,124,255,.13);
        border: 1px solid rgba(139,124,255,.32);
        color: #dcd7ff;
        font-size: .82rem;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(25, 34, 61, .94), rgba(16, 24, 44, .9));
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        box-shadow: 0 10px 30px rgba(0,0,0,.12);
        transition: transform .25s ease, border-color .25s ease, box-shadow .25s ease;
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-4px);
        border-color: rgba(139,124,255,.65);
        box-shadow: 0 14px 35px rgba(105,87,220,.17);
    }

    [data-testid="stMetricLabel"] {
        color: #9eaccb !important;
    }

    [data-testid="stMetricValue"] {
        color: #f5f7ff !important;
        font-family: 'Space Grotesk', sans-serif;
    }

    /* ==================================================
       VOXORA — MODERN TABS
       ================================================== */

    [data-testid="stTabs"] {
        margin-top: 1.8rem;
    }

    [data-testid="stTabs"] [role="tablist"] {
        display: flex !important;
        width: 100%;
        justify-content: space-between;
        align-items: stretch;
        gap: 12px !important;
        padding: 12px !important;
        background: rgba(15, 22, 40, .88) !important;
        border: 1px solid rgba(139,124,255,.22) !important;
        border-radius: 19px !important;
        overflow: visible !important;
    }

    [data-testid="stTabs"] [role="tab"] {
        flex: 1 1 0 !important;
        min-width: 0 !important;
        min-height: 60px !important;
        padding: 15px 10px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        border-radius: 13px !important;
        background: rgba(139,124,255,.07) !important;
        color: #b9c5df !important;
        border: 1px solid rgba(139,124,255,.15) !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 14px !important;
        font-weight: 700 !important;
        transition: all .28s cubic-bezier(.2,.8,.2,1) !important;
        cursor: pointer !important;
        transform: translateY(0) scale(1) !important;
        box-shadow: none !important;
    }

    [data-testid="stTabs"] [role="tab"] p,
    [data-testid="stTabs"] [role="tab"] div,
    [data-testid="stTabs"] [role="tab"] span {
        color: inherit !important;
        font-weight: 700 !important;
        transition: color .25s ease !important;
    }

    /* Hover */
    [data-testid="stTabs"] [role="tab"]:hover {
        background: linear-gradient(115deg, #6554d9, #477fbd) !important;
        color: #ffffff !important;
        border-color: rgba(190,180,255,.8) !important;
        transform: translateY(-5px) scale(1.025) !important;
        box-shadow:
            0 10px 25px rgba(105,87,220,.38),
            0 0 18px rgba(139,124,255,.22) !important;
        z-index: 2;
    }

    /* Selected tab */
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        background: linear-gradient(115deg, #7865ef, #398eae) !important;
        color: #ffffff !important;
        border-color: rgba(190,180,255,.7) !important;
        box-shadow: 0 6px 18px rgba(105,87,220,.30) !important;
    }

    [data-testid="stTabs"] [role="tab"][aria-selected="true"]:hover {
        background: linear-gradient(115deg, #927fff, #42b4c7) !important;
        color: #ffffff !important;
        transform: translateY(-5px) scale(1.025) !important;
        box-shadow:
            0 12px 28px rgba(105,87,220,.42),
            0 0 22px rgba(54,200,215,.22) !important;
    }

    /* Remove default tab indicators */
    [data-testid="stTabs"] [data-baseweb="tab-highlight"],
    [data-testid="stTabs"] [data-testid="stTabBar"] > div > div:last-child {
        display: none !important;
    }

    [data-testid="stTabs"] [role="tab"]:focus-visible {
        outline: 2px solid #8b7cff !important;
        outline-offset: 3px;
    }

    [data-testid="stTabs"] [role="tabpanel"] {
        padding-top: 1.3rem;
    }

    /* ==================================================
       VOXORA — BUTTONS
       ================================================== */

    .stButton > button,
    .stDownloadButton > button,
    [data-testid="stFileUploader"] button {
        position: relative;
        overflow: hidden;
        border-radius: 12px !important;
        border: 1px solid rgba(139,124,255,.45) !important;
        background: linear-gradient(110deg, #6e5be5, #4e78c9) !important;
        color: #ffffff !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 700 !important;
        min-height: 2.8rem;
        transition: all .25s cubic-bezier(.2,.8,.2,1) !important;
        box-shadow: 0 5px 15px rgba(93,78,206,.12);
    }

    .stButton > button p,
    .stDownloadButton > button p,
    [data-testid="stFileUploader"] button p {
        color: #ffffff !important;
        font-weight: 700 !important;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover,
    [data-testid="stFileUploader"] button:hover {
        color: #ffffff !important;
        border-color: rgba(190,180,255,.9) !important;
        background: linear-gradient(110deg, #8875ff, #35a9c5) !important;
        transform: translateY(-3px) scale(1.015) !important;
        box-shadow:
            0 12px 28px rgba(93,78,206,.30),
            0 0 18px rgba(54,200,215,.14) !important;
        filter: brightness(1.08);
    }

    .stButton > button:active,
    .stDownloadButton > button:active {
        transform: translateY(-1px) scale(.99) !important;
    }

    .stButton > button:focus-visible,
    .stDownloadButton > button:focus-visible {
        outline: 2px solid #8b7cff !important;
        outline-offset: 3px !important;
        box-shadow: 0 0 0 4px rgba(139,124,255,.2) !important;
    }

    .stButton > button:disabled,
    .stDownloadButton > button:disabled {
        opacity: .5 !important;
        cursor: not-allowed !important;
        transform: none !important;
        box-shadow: none !important;
    }

    /* Inputs */
    [data-baseweb="input"],
    [data-baseweb="textarea"],
    [data-baseweb="select"] > div {
        background: rgba(9, 15, 29, .82) !important;
        border-color: rgba(148,163,184,.2) !important;
        border-radius: 12px !important;
        color: #edf2ff !important;
    }

    input, textarea {
        color: #edf2ff !important;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--line);
        border-radius: 15px;
        overflow: hidden;
    }

    [data-testid="stChatMessage"] {
        background: rgba(23, 32, 56, .72);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: .8rem 1rem;
        margin-bottom: .65rem;
    }

    [data-testid="stAlert"] {
        border-radius: 14px;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: rgba(148,163,184,.16);
        border-radius: 16px;
        background: rgba(18,26,48,.42);
    }

    [data-testid="stProgressBar"] > div > div {
        background: linear-gradient(90deg, #6e5be5, #36c8d7);
    }

    hr {
        border-color: var(--line);
    }

    code {
        color: #c8c0ff;
        background: rgba(139,124,255,.12);
        border-radius: 6px;
        padding: .12rem .35rem;
    }

    /* Responsive layout */
    @media (max-width: 900px) {
        [data-testid="stTabs"] [role="tablist"] {
            overflow-x: auto !important;
            justify-content: flex-start !important;
            gap: 8px !important;
        }

        [data-testid="stTabs"] [role="tab"] {
            flex: 0 0 auto !important;
            min-width: 125px !important;
        }
    }

    @media (max-width: 768px) {
        .block-container {
            padding: 1rem 1rem 3rem;
        }

        .hero {
            padding: 1.5rem;
            border-radius: 19px;
        }

        .hero:after {
            right: 2%;
            font-size: 5rem;
        }
    }

    @media (prefers-reduced-motion: reduce) {
        [data-testid="stTabs"] [role="tab"],
        .stButton > button,
        .stDownloadButton > button,
        div[data-testid="stMetric"] {
            transition: none !important;
        }
    }

    /* Remove active tab underline / red indicator */
[data-testid="stTabs"] [role="tablist"]::after,
[data-testid="stTabs"] [role="tab"]::after,
[data-testid="stTabs"] [data-baseweb="tab-highlight"],
[data-testid="stTabs"] [data-baseweb="tab-border"] {
    display: none !important;
    opacity: 0 !important;
    height: 0 !important;
    border: none !important;
    box-shadow: none !important;
}

/* Remove default tab bottom borders */
[data-testid="stTabs"] [role="tab"] {
    border-bottom: none !important;
    box-shadow: none !important;
}

/* Preserve custom selected-tab styling */
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    border-bottom: none !important;
}
    </style>
    """,
    unsafe_allow_html=True,
)



# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
@st.cache_resource
def get_client() -> Optional[genai.Client]:
    """Create and cache a Gemini client."""
    key = os.getenv("GEMINI_API_KEY", "").strip()

    if not key or key == "your_gemini_api_key_here":
        return None

    return genai.Client(api_key=key)


def friendly_error(exc: Exception) -> str:
    """Convert raw API exceptions into readable messages."""
    msg = str(exc)
    low = msg.lower()
    if "api key" in low or "api_key" in low or "401" in low or "403" in low:
        return "Gemini rejected the API key. Check GEMINI_API_KEY in your .env file."
    if "429" in low or "quota" in low or "rate" in low:
        return "Rate limit or quota reached. Wait a minute and try again."
    if "404" in low or "not found" in low:
        return f"Model '{MODEL}' was not found. Set GEMINI_MODEL in .env to a valid model."
    if "timed out" in low or "timeout" in low or "connect" in low:
        return "Network problem while contacting Gemini. Check your connection and retry."
    return f"Gemini request failed: {msg[:300]}"


def generate(client: genai.Client, contents, config, retries: int = 2):
    """Call Gemini with a small retry loop for transient failures."""
    last_exc = None
    for attempt in range(retries + 1):
        try:
            return client.models.generate_content(
                model=MODEL, contents=contents, config=config
            )
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            low = str(exc).lower()
            # Don't retry errors that won't fix themselves
            if any(k in low for k in ("api key", "401", "403", "404", "400")):
                break
            time.sleep(1.5 * (attempt + 1))
    raise last_exc


def decode_upload(raw: bytes) -> str:
    """Decode uploaded bytes into text."""
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="ignore")


def validate_transcript(text: str) -> Optional[str]:
    """Return an error message if the transcript is unusable, else None."""
    stripped = text.strip()
    if len(stripped) < MIN_CHARS:
        return (
            f"The transcript is too short ({len(stripped)} characters). "
            f"Please upload a real meeting transcript (at least {MIN_CHARS} characters)."
        )
    if len(stripped) > MAX_CHARS:
        return (
            f"The transcript is too long ({len(stripped):,} characters). "
            f"The limit for this MVP is {MAX_CHARS:,} characters."
        )
    return None


def analyze_transcript(client: genai.Client, transcript: str) -> dict:
    """Run the structured analysis and return a normalized dict."""
    config = types.GenerateContentConfig(
        system_instruction=ANALYSIS_SYSTEM_PROMPT,
        response_mime_type="application/json",
        response_schema=MeetingAnalysis,
        temperature=0.1,
    )
    response = generate(
        client, f"MEETING TRANSCRIPT:\n\n{transcript}", config
    )
    if not response.text:
        raise ValueError(
            "Gemini returned an empty response (it may have been blocked by safety filters)."
        )

    cleaned = response.text.strip().removeprefix("```json").removesuffix("```").strip()
    data = json.loads(cleaned)

    # Normalize so the UI never breaks on missing keys
    return {
        "meeting_title": data.get("meeting_title"),
        "participants": data.get("participants") or [],
        "summary": data.get("summary") or "",
        "key_discussion_points": data.get("key_discussion_points") or [],
        "decisions": data.get("decisions") or [],
        "action_items": data.get("action_items") or [],
    }


def ask_question(client: genai.Client, transcript: str, history: list[dict]) -> str:
    """Answer the latest user question using the transcript + recent chat."""
    recent = history[-CHAT_HISTORY_LIMIT:]
    contents = [
        types.Content(
            role="user" if m["role"] == "user" else "model",
            parts=[types.Part.from_text(text=m["content"])],
        )
        for m in recent
    ]
    config = types.GenerateContentConfig(
        system_instruction=QA_SYSTEM_PROMPT + transcript,
        temperature=0.2,
    )
    response = generate(client, contents, config)
    return response.text or "I couldn't generate an answer. Please rephrase your question."



def transcribe_audio(client: genai.Client, raw_audio: bytes, mime_type: str) -> str:
    """Transcribe an uploaded or browser-recorded audio file with Gemini."""
    prompt = (
        "Transcribe this meeting audio accurately. Preserve the spoken wording, "
        "use clear punctuation, and label speakers when their voices can be "
        "distinguished. Do not summarize or add information. Return only the transcript."
    )
    response = generate(
        client,
        [
            prompt,
            types.Part.from_bytes(data=raw_audio, mime_type=mime_type),
        ],
        types.GenerateContentConfig(temperature=0.1),
    )
    transcript = (response.text or "").strip()
    if not transcript:
        raise ValueError("Gemini returned an empty transcript. Try a clearer recording.")
    return transcript


def speaker_word_counts(transcript: str) -> pd.DataFrame:
    """Derive word counts from explicit 'Speaker: words' transcript lines."""
    counts = {}
    for line in transcript.splitlines():
        match = re.match(r"^\s*([^:\n]{1,60}):\s*(.+)$", line)
        if match:
            speaker, words = match.groups()
            counts[speaker.strip()] = counts.get(speaker.strip(), 0) + len(words.split())
    return pd.DataFrame(
        [{"Speaker": name, "Words": count} for name, count in counts.items()]
    )


def transcript_timeline(transcript: str, bins: int = 5) -> pd.DataFrame:
    """Create a word-volume timeline from sequential transcript sections."""
    words = transcript.split()
    if not words:
        return pd.DataFrame(columns=["Segment", "Words", "Stage"])
    chunk_size = max(1, (len(words) + bins - 1) // bins)
    rows = []
    chunks = [words[i:i + chunk_size] for i in range(0, len(words), chunk_size)]
    for idx, chunk in enumerate(chunks, start=1):
        rows.append({
            "Segment": f"Part {idx}",
            "Words": len(chunk),
            "Stage": f"{idx}/{len(chunks)}",
        })
    return pd.DataFrame(rows)


def animated_transcript_timeline(transcript: str, bins: int = 6) -> pd.DataFrame:
    """Build real transcript word counts as an animated, cumulative reveal."""
    words = transcript.split()
    if not words:
        return pd.DataFrame(columns=["Segment", "Words so far", "Frame"])
    chunk_size = max(1, (len(words) + bins - 1) // bins)
    chunks = [words[i:i + chunk_size] for i in range(0, len(words), chunk_size)]
    labels = [f"Part {i + 1}" for i in range(len(chunks))]
    counts = [len(chunk) for chunk in chunks]
    rows = []
    for frame_idx in range(len(chunks)):
        running = 0
        for segment_idx, count in enumerate(counts):
            if segment_idx <= frame_idx:
                running += count
                value = running
            else:
                value = 0
            rows.append({
                "Segment": labels[segment_idx],
                "Words so far": value,
                "Frame": f"Through part {frame_idx + 1}",
            })
    return pd.DataFrame(rows)


def normalize_action_items(items: list) -> list:
    """Add local tracking fields without changing Gemini's extracted facts."""
    normalized = []
    for idx, item in enumerate(items):
        row = dict(item)
        row.setdefault("status", "Not started")
        row.setdefault("priority", "Normal")
        row.setdefault("due_date", row.get("deadline") or "")
        row.setdefault("completed", False)
        row.setdefault("id", hashlib.sha1(
            f"{idx}-{row.get('task', '')}".encode("utf-8")
        ).hexdigest()[:10])
        normalized.append(row)
    return normalized


def reset_state_for_new_transcript(text: str, name: str):
    """Store a new transcript and clear stale results/chat."""
    st.session_state.transcript = text
    st.session_state.source_name = name
    st.session_state.analysis = None
    st.session_state.chat = []
    st.session_state.action_tracker = []
    st.session_state.audio_transcript_source = None


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
st.session_state.setdefault("transcript", "")
st.session_state.setdefault("source_name", "")
st.session_state.setdefault("analysis", None)
st.session_state.setdefault("chat", [])
st.session_state.setdefault("last_upload_id", None)
st.session_state.setdefault("action_tracker", [])
st.session_state.setdefault("audio_transcript_source", None)
st.session_state.setdefault("completed_action_ids", set())

client = get_client()

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown(
    """
<div class="hero">
    <h1>✦ Voxora</h1>
    <p>Turn conversations into clarity. Your AI-powered workspace for meeting summaries, decisions, action items and transcript Q&amp;A.</p>
</div>
""",
    unsafe_allow_html=True,
)



# --------------------------------------------------------------------------
# Main-page meeting intake
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Main-page meeting intake - clickable buttons
# --------------------------------------------------------------------------

st.markdown("## Bring a meeting into Voxora")
st.caption(
    "Choose how you want to start: upload a transcript, "
    "upload a recording, or record directly from your browser."
)

st.session_state.setdefault("active_intake", None)

button1, button2, button3 = st.columns(3)

with button1:
    if st.button(
        "📄 Upload Transcript",
        use_container_width=True,
        key="choose_transcript"
    ):
        st.session_state.active_intake = "transcript"

with button2:
    if st.button(
        "🎧 Upload Audio",
        use_container_width=True,
        key="choose_audio"
    ):
        st.session_state.active_intake = "audio"

with button3:
    if st.button(
        "🎙️ Record Meeting",
        use_container_width=True,
        key="choose_record"
    ):
        st.session_state.active_intake = "record"


active_intake = st.session_state.active_intake

if active_intake == "transcript":

    st.markdown("### 📄 Upload your transcript")

    uploaded = st.file_uploader(
        "Choose a transcript file",
        type=["txt", "md", "csv"],
        help="Supported formats: TXT, Markdown and CSV.",
        key="main_transcript_upload",
    )

    if uploaded is not None:
        upload_id = f"{uploaded.name}-{uploaded.size}"

        if upload_id != st.session_state.last_upload_id:
            try:
                reset_state_for_new_transcript(
                    decode_upload(uploaded.getvalue()),
                    uploaded.name
                )

                st.session_state.last_upload_id = upload_id
                st.success(f"Loaded {uploaded.name}")

            except Exception as exc:
                st.error(str(exc))


elif active_intake == "audio":

    st.markdown("### 🎧 Upload your meeting audio")

    audio_file = st.file_uploader(
        "Choose a meeting audio file",
        type=["wav", "mp3", "m4a", "aac", "flac", "ogg"],
        help="Maximum file size: 20 MB.",
        key="main_audio_upload",
    )

    if audio_file is not None:

        st.audio(audio_file)

        if st.button(
            "✨ Transcribe Audio",
            use_container_width=True,
            disabled=client is None,
            key="main_transcribe_uploaded_audio",
        ):

            if audio_file.size > 20 * 1024 * 1024:
                st.error("This MVP accepts audio files up to 20 MB.")

            else:
                with st.spinner("Transcribing your meeting..."):

                    try:
                        text = transcribe_audio(
                            client,
                            audio_file.getvalue(),
                            audio_file.type or "audio/wav"
                        )

                        reset_state_for_new_transcript(
                            text,
                            f"{audio_file.name} · transcript"
                        )

                        st.session_state.audio_transcript_source = audio_file.name

                        st.success("Transcript created.")
                        st.rerun()

                    except Exception as exc:
                        st.error(friendly_error(exc))


elif active_intake == "record":

    st.markdown("### 🎙️ Record your meeting")

    st.caption(
        "Record using your browser microphone. "
        "Stop recording when your meeting is finished."
    )

    mic_audio = st.audio_input(
        "Record meeting audio",
        key="voxora_mic_main"
    )

    if mic_audio is not None:

        st.audio(mic_audio)

        if st.button(
            "✨ Transcribe Recording",
            use_container_width=True,
            disabled=client is None,
            key="main_transcribe_mic_audio",
        ):

            if len(mic_audio.getvalue()) > 20 * 1024 * 1024:
                st.error("This MVP accepts recordings up to 20 MB.")

            else:
                with st.spinner("Transcribing your recording..."):

                    try:
                        text = transcribe_audio(
                            client,
                            mic_audio.getvalue(),
                            "audio/wav"
                        )

                        reset_state_for_new_transcript(
                            text,
                            "Browser microphone recording"
                        )

                        st.session_state.audio_transcript_source = (
                            "Browser microphone"
                        )

                        st.success("Transcript created.")
                        st.rerun()

                    except Exception as exc:
                        st.error(friendly_error(exc))


else:

    st.markdown(
        """
        <div class="card">
            <h4>✦ Get started</h4>
            <p>
                Select one of the three buttons above to upload a transcript,
                upload meeting audio, or record a meeting.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.divider()

# --------------------------------------------------------------------------
# Empty state
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# Empty state
# --------------------------------------------------------------------------

transcript = st.session_state.transcript

if not transcript:
    st.info(
        "Your meeting workspace is ready. "
        "Choose Upload Transcript, Upload Audio, or Record Meeting above."
    )
    st.stop()

# --------------------------------------------------------------------------
# Analyze + workspace header
# --------------------------------------------------------------------------
col_a, col_b = st.columns([3, 1])
with col_a:
    st.markdown(
        f"**Loaded:** `{st.session_state.source_name}` · "
        f"{len(transcript.split()):,} words"
    )
with col_b:
    analyze_clicked = st.button(
        "✨ Analyze meeting",
        type="primary",
        use_container_width=True,
        disabled=client is None,
    )

if analyze_clicked and client is not None:
    with st.spinner("Voxora is extracting meeting intelligence..."):
        try:
            st.session_state.analysis = analyze_transcript(client, transcript)
            st.session_state.action_tracker = normalize_action_items(
                st.session_state.analysis.get("action_items", [])
            )
            st.session_state.completed_action_ids = set()
        except json.JSONDecodeError:
            st.error("Gemini returned malformed JSON. Please analyze again.")
        except Exception as exc:
            st.error(friendly_error(exc))

analysis = st.session_state.analysis

# If this is a session restored with analysis but no tracker, initialize it.
if analysis and not st.session_state.action_tracker:
    st.session_state.action_tracker = normalize_action_items(
        analysis.get("action_items", [])
    )

st.markdown("### Your meeting workspace")
st.caption("One place for the conversation, decisions, owners and next steps.")

if analysis:
    actions = st.session_state.action_tracker
    total_actions = len(actions)
    done_actions = sum(1 for a in actions if a.get("completed"))
    open_actions = total_actions - done_actions
    completion = (done_actions / total_actions * 100) if total_actions else 0

    metric_cols = st.columns(4)
    metric_cols[0].metric("Participants", len(analysis["participants"]))
    metric_cols[1].metric("Discussion points", len(analysis["key_discussion_points"]))
    metric_cols[2].metric("Decisions", len(analysis["decisions"]))
    metric_cols[3].metric("Tasks completed", f"{done_actions}/{total_actions}")

    # Animated progress ring + live task state.
    progress_col, chart_col = st.columns([1, 2])
    with progress_col:
        st.markdown("#### Execution progress")
        fig_ring = go.Figure(go.Pie(
            values=[done_actions, max(total_actions - done_actions, 0)] if total_actions else [0, 1],
            labels=["Completed", "Remaining"],
            hole=.78,
            marker=dict(colors=["#8b7cff", "#25304b"]),
            textinfo="none",
            sort=False,
        ))
        fig_ring.update_layout(
            showlegend=False,
            height=210,
            margin=dict(l=8, r=8, t=8, b=8),
            paper_bgcolor="rgba(0,0,0,0)",
            annotations=[dict(
                text=f"<b>{completion:.0f}%</b><br><span style='font-size:11px'>complete</span>",
                x=.5, y=.5, showarrow=False,
                font=dict(size=25, color="#f3f5ff"),
            )],
        )
        st.plotly_chart(fig_ring, use_container_width=True, config={"displayModeBar": False})
    with chart_col:
        st.markdown("#### Conversation activity")
        speaker_df = speaker_word_counts(transcript)
        if not speaker_df.empty:
            fig_speakers = px.bar(
                speaker_df.sort_values("Words", ascending=True),
                x="Words", y="Speaker", orientation="h",
                color="Words",
                color_continuous_scale=["#3c8ea8", "#8b7cff"],
                title="Words by identified speaker",
            )
            fig_speakers.update_layout(
                height=210, margin=dict(l=8, r=8, t=38, b=8),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#dce5fb",
                coloraxis_showscale=False,
                title_font_size=13,
                xaxis_title=None, yaxis_title=None,
            )
            fig_speakers.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,.12)")
            fig_speakers.update_yaxes(showgrid=False)
            st.plotly_chart(fig_speakers, use_container_width=True, config={"displayModeBar": False})
        else:
            timeline_df = transcript_timeline(transcript)
            fig_timeline = px.bar(
                timeline_df, x="Segment", y="Words",
                animation_frame="Stage",
                range_y=[0, max(1, int(timeline_df["Words"].max() * 1.2))],
                color_discrete_sequence=["#8b7cff"],
                title="Transcript word volume by section",
            )
            fig_timeline.update_layout(
                height=210, margin=dict(l=8, r=8, t=38, b=8),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#dce5fb",
                title_font_size=13,
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
            st.caption("No explicit speaker labels found; showing transcript section volume instead.")

    st.markdown("#### Animated transcript timeline")
    animated_df = animated_transcript_timeline(transcript)
    if not animated_df.empty:
        fig_reveal = px.bar(
            animated_df,
            x="Segment",
            y="Words so far",
            animation_frame="Frame",
            range_y=[0, max(1, len(transcript.split()))],
            color_discrete_sequence=["#36c8d7"],
            title="Cumulative transcript volume",
        )
        fig_reveal.update_layout(
            height=280,
            margin=dict(l=8, r=8, t=42, b=8),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#dce5fb",
            title_font_size=13,
            transition={"duration": 550},
        )
        fig_reveal.update_xaxes(showgrid=False)
        fig_reveal.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,.12)")
        st.plotly_chart(fig_reveal, use_container_width=True)

# --------------------------------------------------------------------------
# Interactive meeting tabs
# --------------------------------------------------------------------------
tab_overview, tab_decisions, tab_actions, tab_qa, tab_transcript, tab_export = st.tabs(
    ["✦ Overview", "✓ Decisions", "◎ Action tracker", "◉ AI Q&A", "▤ Transcript", "↓ Export"]
)

NEEDS_ANALYSIS = "Run **Analyze meeting** to generate meeting intelligence."

with tab_overview:
    if not analysis:
        st.info(NEEDS_ANALYSIS)
    else:
        title = analysis["meeting_title"] or "Meeting summary"
        st.markdown(
            f'<div class="card"><h4>✦ {title}</h4>{analysis["summary"]}</div>',
            unsafe_allow_html=True,
        )
        left, right = st.columns([1.7, 1])
        with left:
            st.markdown("#### Key discussion points")
            for point in analysis["key_discussion_points"]:
                st.markdown(f"- {point}")
            if not analysis["key_discussion_points"]:
                st.caption("No key discussion points were identified.")
        with right:
            st.markdown("#### In the room")
            if analysis["participants"]:
                pills = "".join(
                    f'<span class="pill">{p}</span>' for p in analysis["participants"]
                )
                st.markdown(pills, unsafe_allow_html=True)
            else:
                st.caption("No participants could be identified.")
            st.markdown("#### Meeting signal")
            st.caption(f"{len(transcript.split()):,} words · {len(transcript.splitlines()):,} transcript lines")

with tab_decisions:
    if not analysis:
        st.info(NEEDS_ANALYSIS)
    elif not analysis["decisions"]:
        st.info("No explicit decisions were found in this meeting.")
    else:
        for idx, decision in enumerate(analysis["decisions"], start=1):
            context = decision.get("context")
            with st.expander(f"Decision {idx} · {decision.get('decision', '')}", expanded=True):
                if context:
                    st.write(context)
                else:
                    st.caption("No additional context was stated.")

with tab_actions:
    if not analysis:
        st.info(NEEDS_ANALYSIS)
    else:
        actions = st.session_state.action_tracker
        if not actions:
            st.info("No explicit action items were found in this meeting.")
        else:
            total = len(actions)
            completed = sum(1 for item in actions if item.get("completed"))
            st.progress(completed / total if total else 0, text=f"{completed} of {total} tasks completed")
            st.caption("Update status, priority and due date. Changes are kept in this active session.")
            for idx, item in enumerate(actions):
                with st.container(border=True):
                    top = st.columns([0.08, 0.92])
                    with top[0]:
                        checked = st.checkbox(
                            "Complete",
                            value=bool(item.get("completed")),
                            key=f"task_done_{item['id']}",
                            label_visibility="collapsed",
                        )
                    with top[1]:
                        st.markdown(f"**{item.get('task') or 'Untitled task'}**")
                        evidence = item.get("evidence")
                        if evidence:
                            st.caption(f"Transcript evidence: “{evidence}”")

                    fields = st.columns([1.1, 1, 1, 1.2])
                    with fields[0]:
                        owner = st.text_input(
                            "Owner",
                            value=item.get("owner") or "",
                            placeholder="Unassigned",
                            key=f"task_owner_{item['id']}",
                        )
                    with fields[1]:
                        status = st.selectbox(
                            "Status",
                            ["Not started", "In progress", "Blocked", "Done"],
                            index=["Not started", "In progress", "Blocked", "Done"].index(
                                item.get("status", "Not started")
                                if item.get("status", "Not started") in
                                ["Not started", "In progress", "Blocked", "Done"]
                                else "Not started"
                            ),
                            key=f"task_status_{item['id']}",
                        )
                    with fields[2]:
                        priority = st.selectbox(
                            "Priority",
                            ["Low", "Normal", "High", "Urgent"],
                            index=["Low", "Normal", "High", "Urgent"].index(
                                item.get("priority", "Normal")
                                if item.get("priority", "Normal") in
                                ["Low", "Normal", "High", "Urgent"]
                                else "Normal"
                            ),
                            key=f"task_priority_{item['id']}",
                        )
                    with fields[3]:
                        due = st.text_input(
                            "Due date",
                            value=item.get("due_date") or "",
                            placeholder="Not specified",
                            key=f"task_due_{item['id']}",
                        )

                    item["completed"] = checked or status == "Done"
                    item["owner"] = owner.strip() or None
                    item["status"] = "Done" if item["completed"] else status
                    item["priority"] = priority
                    item["due_date"] = due.strip()

            st.divider()
            if st.button("Reset tracker to AI-extracted values", key="reset_action_tracker"):
                st.session_state.action_tracker = normalize_action_items(
                    analysis.get("action_items", [])
                )
                for item in st.session_state.action_tracker:
                    for suffix in ("done", "owner", "status", "priority", "due"):
                        st.session_state.pop(f"task_{suffix}_{item['id']}", None)
                st.rerun()

with tab_qa:
    st.caption("Ask questions about this meeting. Answers are grounded only in its transcript.")
    if not analysis:
        st.info(NEEDS_ANALYSIS)
    else:
        for msg in st.session_state.chat:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        question = st.chat_input("Ask about decisions, owners, risks or next steps...")
        if question:
            if client is None:
                st.error("Add your GEMINI_API_KEY to use meeting Q&A.")
            else:
                st.session_state.chat.append({"role": "user", "content": question})
                with st.chat_message("user"):
                    st.markdown(question)
                with st.chat_message("assistant"):
                    with st.spinner("Thinking through the transcript..."):
                        try:
                            answer = ask_question(client, transcript, st.session_state.chat)
                        except Exception as exc:
                            answer = f"⚠️ {friendly_error(exc)}"
                    st.markdown(answer)
                st.session_state.chat.append({"role": "assistant", "content": answer})
        if st.session_state.chat and st.button("Clear conversation", key="clear_qa"):
            st.session_state.chat = []
            st.rerun()

with tab_transcript:
    st.caption("Source transcript used for analysis and Q&A.")
    st.text_area(
        "Transcript",
        transcript,
        height=460,
        disabled=True,
        label_visibility="collapsed",
    )

with tab_export:
    if not analysis:
        st.info(NEEDS_ANALYSIS)
    else:
        export_actions = []
        for item in st.session_state.action_tracker:
            export_actions.append({
                "Task": item.get("task"),
                "Owner": item.get("owner") or "Not specified",
                "Status": item.get("status"),
                "Priority": item.get("priority"),
                "Due date": item.get("due_date") or "Not specified",
                "Evidence": item.get("evidence") or "",
            })
        action_df = pd.DataFrame(export_actions)
        left, right = st.columns(2)
        with left:
            st.download_button(
                "↓ Download action tracker (CSV)",
                action_df.to_csv(index=False).encode("utf-8"),
                file_name="voxora_action_tracker.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with right:
            st.download_button(
                "↓ Download meeting analysis (JSON)",
                json.dumps(analysis, indent=2).encode("utf-8"),
                file_name="voxora_meeting_analysis.json",
                mime="application/json",
                use_container_width=True,
            )
        st.markdown("#### Export preview")
        st.dataframe(action_df, use_container_width=True, hide_index=True)

