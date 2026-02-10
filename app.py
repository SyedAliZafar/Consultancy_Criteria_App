# app.py
from __future__ import annotations

import json
import os
import re
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import requests
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


# -----------------------------
# Load .env (absolute path)
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

DB_PATH = str(BASE_DIR / "consultant.db")
CRITERIA_FILE = str(BASE_DIR / "Dataset" / "UniAssistCriteria_clean.txt")  # adjust if needed


# -----------------------------
# Helpers: DB
# -----------------------------
def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
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
            study_language TEXT,
            german_level TEXT,
            results_json TEXT
        )
        """
    )
    conn.commit()
    conn.close()


# -----------------------------
# Lifespan (startup/shutdown)
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()

    key = os.getenv("DEEPSEEK_API_KEY")
    url = os.getenv("DEEPSEEK_URL")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

    print(f"📄 Loading .env from: {ENV_PATH} | exists={ENV_PATH.exists()}")
    if key and url:
        print("✅ DeepSeek API configured")
        print(f"🔑 Key length: {len(key)}")
        print(f"🌐 URL: {url}")
        print(f"🤖 Model: {model}")
    else:
        print("⚠️ DeepSeek NOT configured – using rule-based shortlist only")
        print(f"DEEPSEEK_API_KEY loaded? {bool(key)}")
        print(f"DEEPSEEK_URL loaded? {bool(url)}")

    yield
    print("🛑 Application shutdown")


app = FastAPI(title="Uni Eligibility Finder", lifespan=lifespan)

# Serve frontend
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


# -----------------------------
# Models
# -----------------------------
class StudentInput(BaseModel):
    name: str = Field(..., min_length=1)
    background: str = ""

    # paid/unpaid is optional; you can decide later how to use it
    paid_pref: str = "no_preference"  # paid | unpaid | no_preference

    gpa: float
    scale_max: float = 4.0
    min_pass: float = 2.0

    # language choice
    study_language: str = "english"   # english | german | both
    german_level: str = "none"        # none | a2 | b1 | b2 | c1


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
        raise ValueError("scale_max must be > min_pass")

    grade = max(min_passing_grade, min(grade, max_grade))
    german = german_best + (german_worst_pass - german_best) * (max_grade - grade) / (max_grade - min_passing_grade)
    return round(german, 2)


# -----------------------------
# Parse criteria text file
# -----------------------------
def load_program_records(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Criteria file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        txt = f.read()

    chunks = [c.strip() for c in txt.split("-" * 62) if c.strip()]
    records: List[Dict[str, str]] = []

    for c in chunks:
        if c.startswith("WINTER SEMESTER"):
            continue
        if c.upper().startswith("CRITERIA:"):
            continue

        rec: Dict[str, str] = {}
        for line in c.splitlines():
            m = re.match(r"^([^:]+):\s*(.*)$", line.strip())
            if m:
                rec[m.group(1).strip()] = m.group(2).strip()

        if rec.get("Program") and rec.get("University"):
            records.append(rec)

    return records


def _lang_blob(r: Dict[str, str]) -> str:
    blob = " ".join(
        [
            r.get("Language of instruction", ""),
            r.get("Language proficiency", ""),
            r.get("University info", ""),
            r.get("Admission requirements", ""),
            r.get("Notes", ""),
        ]
    )
    return blob.lower()


# -----------------------------
# Language logic + German level logic
# -----------------------------
GER_LEVEL_SCORE = {"none": 0, "a2": 2, "b1": 3, "b2": 4, "c1": 5}

def detect_program_language(blob: str) -> str:
    b = (blob or "").lower()
    has_en = "english" in b
    has_de = ("german" in b) or ("deutsch" in b)
    if has_en and has_de:
        return "both"
    if has_en:
        return "english"
    if has_de:
        return "german"
    return "unknown"


def extract_required_german_level(text: str) -> Optional[str]:
    t = (text or "").lower()
    # very common patterns
    if "c1" in t:
        return "c1"
    if "b2" in t:
        return "b2"
    if "b1" in t:
        return "b1"
    if "a2" in t:
        return "a2"
    return None


def german_level_ok(student_level: str, required_level: Optional[str]) -> bool:
    if not required_level:
        return True
    return GER_LEVEL_SCORE.get(student_level, 0) >= GER_LEVEL_SCORE.get(required_level, 0)


# -----------------------------
# Confidence calibration (FH vs TU heuristic)
# -----------------------------
def estimate_uni_type(university_name: str) -> str:
    u = (university_name or "").lower()
    if "university of applied sciences" in u or "hochschule" in u or "fh" in u:
        return "FH"
    if "technical university" in u or "technische universität" in u or u.startswith("tu "):
        return "TU"
    return "UNI"


def apply_confidence_calibration(score: float, uni_type: str) -> float:
    # small adjustment only (heuristic)
    if uni_type == "FH":
        return min(100.0, score + 5.0)
    if uni_type == "TU":
        return max(0.0, score - 5.0)
    return score


# -----------------------------
# Borderline-aware filtering
# -----------------------------
def filter_programs(
    records: List[Dict[str, str]],
    german_grade_est: float,
    study_language: str,
    german_level: str,
    background_text: str,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    bt = (background_text or "").lower().strip()

    for r in records:
        ok = True
        flags: List[str] = []

        # Entrance grade heuristic
        eg = r.get("Entrance grade", "")
        if eg:
            m = re.search(r"(\d+[.,]\d+|\d+)", eg)
            if m:
                req = float(m.group(1).replace(",", "."))
                if german_grade_est > req:
                    ok = False
            else:
                flags.append("entrance_grade_unparsed")

        blob = _lang_blob(r)
        plang = detect_program_language(blob)

        # user preference: english / german / both
        if study_language == "english":
            if plang == "german":
                ok = False
            elif plang == "unknown":
                flags.append("language_unknown")
        elif study_language == "german":
            if plang == "english":
                ok = False
            elif plang == "unknown":
                flags.append("language_unknown")
        elif study_language == "both":
            # allow everything
            pass

        # German level requirement only if program is German or mixed and user allows German
        if study_language in ("german", "both") and plang in ("german", "both"):
            req_level = extract_required_german_level(
                " ".join(
                    [
                        r.get("Language proficiency", ""),
                        r.get("Admission requirements", ""),
                        r.get("Notes", ""),
                    ]
                )
            )
            if req_level:
                if not german_level_ok(german_level, req_level):
                    ok = False
            else:
                flags.append("german_level_unknown")

        # background relevance = borderline only (not reject)
        reqtxt = " ".join([r.get("Requirement", ""), r.get("Voraussetzung", ""), r.get("Admission requirements", "")]).lower()
        if bt and reqtxt:
            strong_terms = ["electrical", "computer", "engineering", "business", "informatics", "mechanical", "software"]
            if any(t in reqtxt for t in strong_terms):
                if not any(t in bt for t in strong_terms if t in reqtxt):
                    flags.append("background_mismatch_possible")

        if ok:
            r2 = dict(r)
            r2["_flags"] = flags
            r2["_program_language"] = plang
            out.append(r2)

    return out


# -----------------------------
# DeepSeek ranking
# -----------------------------
def deepseek_rank(programs: List[Dict[str, Any]], student_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    url = os.getenv("DEEPSEEK_URL", "").strip()
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    temperature = float(os.getenv("DEEPSEEK_TEMPERATURE", "0.2"))

    # fallback
    if not api_key or not url:
        top = programs[:10]
        return [
            {
                "rank": i + 1,
                "program_name": p.get("Program", ""),
                "university": p.get("University", ""),
                "score_0_100": None,
                "reason": "DeepSeek not configured; rule-based shortlist only.",
                "uni_type": estimate_uni_type(p.get("University", "")),
                "flags": p.get("_flags", []),
            }
            for i, p in enumerate(top)
        ]

    # Pack only what model needs (keep prompt smaller)
    packed_programs = []
    for p in programs[:30]:
        packed_programs.append(
            {
                "program_name": p.get("Program", ""),
                "university": p.get("University", ""),
                "entrance_grade": p.get("Entrance grade", ""),
                "requirements": " ".join([p.get("Requirement", ""), p.get("Voraussetzung", ""), p.get("Admission requirements", "")]).strip(),
                "language_info": " ".join([p.get("Language of instruction", ""), p.get("Language proficiency", "")]).strip(),
                "flags": p.get("_flags", []),
                "uni_type_hint": estimate_uni_type(p.get("University", "")),
            }
        )

    system_msg = (
        "You are an admissions eligibility assistant for German universities.\n"
        "Score each program 0-100 based on these weighted criteria:\n"
        "1) GPA match (40%) comparing student's German grade estimate vs entrance grade.\n"
        "2) Background relevance (30%) vs requirements text.\n"
        "3) Language fit (20%) based on student's study_language and german_level.\n"
        "4) Borderline flags (10%) – reduce score slightly if flags exist.\n"
        "Return STRICT JSON only. No markdown. No explanations outside JSON."
    )

    user_msg = {
        "student": student_profile,
        "programs": packed_programs,
        "schema": [
            {
                "rank": 1,
                "program_name": "...",
                "university": "...",
                "score_0_100": 0,
                "reason": "1-2 sentences",
            }
        ],
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": json.dumps(user_msg, ensure_ascii=False)},
        ],
        "temperature": temperature,
    }

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    resp = requests.post(url, headers=headers, json=payload, timeout=60)

    if resp.status_code >= 400:
        raise RuntimeError(f"DeepSeek error {resp.status_code}: {resp.text}")

    data = resp.json()
    content = data["choices"][0]["message"]["content"].strip()

    match = re.search(r"(\[.*\])", content, flags=re.S)
    if not match:
        raise RuntimeError(f"DeepSeek returned non-JSON: {content[:800]}")

    ranked = json.loads(match.group(1))

    # Apply FH/TU calibration + attach flags
    # Create a quick lookup from (program_name, university) to flags
    flag_map = {}
    for p in programs[:30]:
        flag_map[(p.get("Program", ""), p.get("University", ""))] = p.get("_flags", [])

    for item in ranked:
        uni = item.get("university", "")
        uni_type = estimate_uni_type(uni)
        item["uni_type"] = uni_type
        item["flags"] = flag_map.get((item.get("program_name", ""), uni), [])

        if item.get("score_0_100") is not None:
            try:
                s = float(item["score_0_100"])
                item["score_0_100"] = round(apply_confidence_calibration(s, uni_type), 1)
            except Exception:
                pass

    # Re-rank by calibrated score (if present)
    ranked_sorted = sorted(
        ranked,
        key=lambda x: (x.get("score_0_100") is None, -(x.get("score_0_100") or 0)),
    )
    for i, it in enumerate(ranked_sorted, start=1):
        it["rank"] = i

    return ranked_sorted


# -----------------------------
# Save consultation
# -----------------------------
def save_consultation(inp: StudentInput, german_grade_est: float, results: List[Dict[str, Any]]):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO consultations
        (created_at, name, background, gpa, scale_max, min_pass, german_grade_est, paid_pref, study_language, german_level, results_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.utcnow().isoformat(),
            inp.name,
            inp.background,
            inp.gpa,
            inp.scale_max,
            inp.min_pass,
            german_grade_est,
            inp.paid_pref,
            inp.study_language,
            inp.german_level,
            json.dumps(results, ensure_ascii=False),
        ),
    )
    conn.commit()
    conn.close()


# -----------------------------
# Routes
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def home():
    with open(BASE_DIR / "static" / "index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/api/recommend")
def recommend(inp: StudentInput):
    records = load_program_records(CRITERIA_FILE)
    german_est = convert_to_german_grade(inp.gpa, inp.scale_max, inp.min_pass)

    filtered = filter_programs(
        records=records,
        german_grade_est=german_est,
        study_language=inp.study_language,
        german_level=inp.german_level,
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
        "study_language": inp.study_language,
        "german_level": inp.german_level,
    }

    ranked = deepseek_rank(filtered, student_profile)
    save_consultation(inp, german_est, ranked[:20])

    return {
        "german_grade_est": german_est,
        "shortlist_count": len(filtered),
        "results": ranked[:10],
    }


@app.get("/api/program_details")
def program_details(university: str, program: str):
    # Lightweight “search further” feature: returns search queries + clickable links
    queries = [
        f"{university} {program} admission requirements",
        f"{university} {program} language requirements",
        f"{university} {program} application deadline winter semester",
        f"{university} {program} uni-assist",
        f"{university} {program} entrance grade",
        f"{university} {program} module handbook",
    ]

    return {
        "university": university,
        "program": program,
        "suggested_search_queries": queries,
        "google_links": [f"https://www.google.com/search?q={quote(q)}" for q in queries],
        "note": "Next step: integrate live web lookup + summarization.",
    }
