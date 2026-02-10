    # app.py
import os
import json
import re
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()



DB_PATH = "consultant.db"
CRITERIA_FILE = "Dataset/UniAssistCriteria_clean.txt"  # put cleaned file next to app.py

from fastapi import FastAPI
from contextlib import asynccontextmanager
import os

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---- startup ----
    key = os.getenv("DEEPSEEK_API_KEY")
    url = os.getenv("DEEPSEEK_URL")

    if key and url:
        print("✅ DeepSeek API configured")
        print(f"🔑 Key length: {len(key)}")
        print(f"🌐 URL: {url}")
    else:
        print("⚠️ DeepSeek NOT configured – using rule-based shortlist only")

    yield

    # ---- shutdown ----
    print("🛑 Application shutdown")

app = FastAPI(
    title="Uni Eligibility Finder",
    lifespan=lifespan
)


# Serve frontend
app.mount("/static", StaticFiles(directory="static"), name="static")




# -----------------------------
# Models
# -----------------------------
class StudentInput(BaseModel):
    name: str = Field(..., min_length=1)
    background: str = ""
    paid_pref: str = "no_preference"  # "paid" | "unpaid" | "no_preference"

    gpa: float
    scale_max: float = 4.0
    min_pass: float = 2.0

    # NEW: language restriction
    language_mode: str = "english_only"  # "english_only" | "german_allowed"


# -----------------------------
# GPA conversion (estimate)
# -----------------------------
def convert_to_german_grade(
    grade: float,
    max_grade: float,
    min_passing_grade: float,
    german_best: float = 1.0,
    german_worst_pass: float = 4.0,
) -> float:
    if max_grade <= min_passing_grade:
        raise ValueError("max_grade must be > min_passing_grade")

    grade = max(min_passing_grade, min(grade, max_grade))
    german = german_best + (german_worst_pass - german_best) * (max_grade - grade) / (max_grade - min_passing_grade)
    return round(german, 2)


# -----------------------------
# Parse criteria
# -----------------------------
def load_program_records(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()

    chunks = [c.strip() for c in txt.split("-" * 62) if c.strip()]
    records = []
    for c in chunks:
        if c.startswith("WINTER SEMESTER"):
            continue
        if c.upper().startswith("CRITERIA:"):
            continue

        rec = {}
        for line in c.splitlines():
            m = re.match(r"^([^:]+):\s*(.*)$", line.strip())
            if m:
                rec[m.group(1).strip()] = m.group(2).strip()

        if rec.get("Program") and rec.get("University"):
            records.append(rec)
    return records


def _lang_blob(r: Dict[str, str]) -> str:
    blob = " ".join([
        r.get("Language of instruction", ""),
        r.get("Language proficiency", ""),
        r.get("University info", ""),
        r.get("Notes", ""),
    ])
    return blob.lower()


def _is_clearly_english(blob: str) -> bool:
    return "english" in blob and "german taught" not in blob


def _is_clearly_german_only(blob: str) -> bool:
    # If it explicitly says German, and does not mention English anywhere
    return ("german" in blob) and ("english" not in blob)


def filter_programs(
    records: List[Dict[str, str]],
    german_grade_est: float,
    language_mode: str,
    background_text: str,
) -> List[Dict[str, str]]:
    out = []
    bt = background_text.lower().strip()

    for r in records:
        ok = True

        # Entrance grade (heuristic)
        eg = r.get("Entrance grade", "")
        if eg:
            m = re.search(r"(\d+[.,]\d+|\d+)", eg)
            if m:
                req = float(m.group(1).replace(",", "."))
                if german_grade_est > req:
                    ok = False

        # NEW: language restriction
        blob = _lang_blob(r)
        if language_mode == "english_only":
            # drop clearly German-only programs
            if _is_clearly_german_only(blob):
                ok = False
            # if clearly english -> keep, if unknown -> keep (don’t over-filter)
        elif language_mode == "german_allowed":
            # allow both; no language filter
            pass

        # Light background relevance (not hard reject)
        reqtxt = " ".join([
            r.get("Requirement", ""),
            r.get("Voraussetzung", ""),
            r.get("Admission requirements", ""),
        ]).lower()

        if bt and reqtxt:
            strong_terms = ["electrical", "computer", "engineering", "business", "informatics", "mechanical", "software"]
            if any(t in reqtxt for t in strong_terms):
                # if requirement mentions a strong term, prefer overlap (still not a hard reject)
                if not any(t in bt for t in strong_terms if t in reqtxt):
                    pass

        if ok:
            out.append(r)

    return out


# -----------------------------
# DeepSeek ranking (pluggable)
# -----------------------------
def deepseek_rank(programs: List[Dict[str, str]], student_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    url = os.getenv("DEEPSEEK_URL", "").strip()

    if not api_key or not url:
        top = programs[:10]
        return [{
            "rank": i + 1,
            "program_name": p.get("Program", ""),
            "university": p.get("University", ""),
            "score_0_100": None,
            "reason": "DeepSeek not configured; rule-based shortlist only."
        } for i, p in enumerate(top)]

    # DeepSeek Chat Completions format (OpenAI-compatible)
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    system_msg = (
        "You are an admissions eligibility assistant for German universities. "
        "Score each program 0-100 based on these weighted criteria:\n"
        "1. GPA match (40%): Compare student's German grade estimate with program's entrance grade\n"
        "2. Background relevance (30%): Match between student's background and program requirements\n"
        "3. Language fit (20%): Based on student's language preference\n"
        "4. Paid/unpaid preference (10%): If student has preference\n"
        "Use strict numerical scoring, then rank by score.\n"
        "Return STRICT JSON only."
    )

    user_msg = {
        "student": student_profile,
        "programs": programs[:30],
        "instructions": (
            "Return JSON array ONLY in this exact schema:\n"
            "[{"
            "\"rank\": 1, "
            "\"program_name\": \"...\", "
            "\"university\": \"...\", "
            "\"score_0_100\": 0, "
            "\"reason\": \"1-2 sentences\""
            "}]\n"
            "No markdown, no extra text."
        )
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": json.dumps(user_msg, ensure_ascii=False)}
        ],
        "temperature": 0.2
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=60)

    # Helpful debugging if it fails
    if resp.status_code >= 400:
        raise RuntimeError(f"DeepSeek error {resp.status_code}: {resp.text}")

    data = resp.json()

    # OpenAI-style response parsing
    content = data["choices"][0]["message"]["content"].strip()

    # Extract JSON safely (in case model adds text)
    match = re.search(r"(\[.*\])", content, flags=re.S)
    if not match:
        raise RuntimeError(f"DeepSeek returned non-JSON: {content[:500]}")

    return json.loads(match.group(1))



# -----------------------------
# Storage
# -----------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS consultations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            name TEXT,
            background TEXT,
            gpa REAL,
            scale_max REAL,
            min_pass REAL,
            german_grade_est REAL,
            paid_pref TEXT,
            language_mode TEXT,
            results_json TEXT
        )
    """)
    conn.commit()
    conn.close()


def save_consultation(inp: StudentInput, german_grade_est: float, results: List[Dict[str, Any]]):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO consultations
        (created_at, name, background, gpa, scale_max, min_pass, german_grade_est, paid_pref, language_mode, results_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.utcnow().isoformat(),
        inp.name,
        inp.background,
        inp.gpa,
        inp.scale_max,
        inp.min_pass,
        german_grade_est,
        inp.paid_pref,
        inp.language_mode,
        json.dumps(results, ensure_ascii=False),
    ))
    conn.commit()
    conn.close()


# -----------------------------
# Routes
# -----------------------------
@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/", response_class=HTMLResponse)
def home():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/api/recommend")
def recommend(inp: StudentInput):
    records = load_program_records(CRITERIA_FILE)
    german_est = convert_to_german_grade(inp.gpa, inp.scale_max, inp.min_pass)

    filtered = filter_programs(
        records=records,
        german_grade_est=german_est,
        language_mode=inp.language_mode,
        background_text=inp.background,
    )

    student_profile = {
        "name": inp.name,
        "background": inp.background,
        "gpa": inp.gpa,
        "scale_max": inp.scale_max,
        "min_pass": inp.min_pass,
        "german_grade_est": german_est,
        "paid_pref": inp.paid_pref,
        "language_mode": inp.language_mode,
    }

    ranked = deepseek_rank(filtered, student_profile)
    save_consultation(inp, german_est, ranked[:20])

    return {
        "german_grade_est": german_est,
        "shortlist_count": len(filtered),
        "results": ranked[:10],
    }
