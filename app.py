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
QS_FILE = str(BASE_DIR / "Data" / "qs_europe_2026.xlsx")


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

    # ✅ LOAD QS AT STARTUP
    app.state.qs_map = load_qs_germany_map(QS_FILE)
    print(f"📊 QS loaded rows: {len(app.state.qs_map)}")

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


def _norm_uni_name(s: str) -> str:
    s = (s or "").lower()
    s = s.replace("&", "and")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # common normalizations
    replacements = {
        "university of applied sciences": "hochschule",
        "technical university": "tu",
        "technische universitat": "tu",
        "technische universität": "tu",
        "universitaet": "universitat",
        "university": "uni",
    }
    for k, v in replacements.items():
        s = s.replace(k, v)

    return s


def load_qs_germany_map(qs_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load QS Europe Excel (Published sheet, multi-row headers),
    filter Germany only, compute selectivity_index (0..1, higher=more selective),
    return mapping: normalized institution name -> metrics dict.
    """
    if not os.path.exists(qs_path):
        print(f"⚠️ QS file not found: {qs_path} (QS calibration disabled)")
        return {}

    raw = pd.read_excel(qs_path, sheet_name="Published", header=None)

    # QS v1.3 pattern: header parts in rows 1 and 3; data starts from row 4
    h_main = raw.iloc[1]
    h_sub = raw.iloc[3]

    def clean(x):
        if pd.isna(x):
            return ""
        return str(x).strip()

    cols = []
    last_main = ""
    for i in range(raw.shape[1]):
        main = clean(h_main[i])
        sub = clean(h_sub[i]).lower()

        if main:
            last_main = main
            if "score" in sub:
                cols.append(f"{main} Score")
            elif "rank" in sub:
                cols.append(main if "rank" in main.lower() else f"{main} Rank")
            else:
                cols.append(main)
        else:
            if "score" in sub:
                cols.append(f"{last_main} Score")
            elif "rank" in sub:
                cols.append(last_main if "rank" in last_main.lower() else f"{last_main} Rank")
            else:
                cols.append(clean(h_sub[i]) or f"col_{i}")

    df = raw.iloc[4:].copy()
    df.columns = cols
    df.reset_index(drop=True, inplace=True)

    # Find country column
    country_col = None
    for c in df.columns:
        if "country" in c.lower():
            country_col = c
            break
    if not country_col:
        print("⚠️ QS country column not found (QS calibration disabled)")
        return {}

    de = df[df[country_col].astype(str).str.lower().eq("germany")].copy()

    # Find institution name column
    inst_col = None
    for c in de.columns:
        if "institution" in c.lower() and "name" in c.lower():
            inst_col = c
            break
    if not inst_col:
        print("⚠️ QS Institution Name column not found (QS calibration disabled)")
        return {}

    # Find 2026 rank and overall score columns
    rank_col = None
    overall_col = None
    for c in de.columns:
        if c.strip().lower() == "2026 rank":
            rank_col = c
        if c.strip().lower() == "overall score":
            overall_col = c
    if not rank_col or not overall_col:
        print("⚠️ QS Rank/Overall columns not found (QS calibration disabled)")
        return {}

    def parse_rank(x):
        s = str(x).replace("=", "").strip()
        return pd.to_numeric(s, errors="coerce")

    de[rank_col] = de[rank_col].apply(parse_rank)
    de[overall_col] = pd.to_numeric(de[overall_col], errors="coerce")

    # rank_pct: best ranks have small pct; invert for selectivity
    de["rank_pct_de"] = de[rank_col].rank(pct=True)
    de["overall_norm"] = de[overall_col] / 100.0

    # selectivity_index in [0..1] roughly: higher = more selective
    de["selectivity_index"] = 0.6 * (1.0 - de["rank_pct_de"]) + 0.4 * (1.0 - de["overall_norm"])

    qs_map: Dict[str, Dict[str, Any]] = {}
    for _, row in de.iterrows():
        name = str(row[inst_col])
        nm = _norm_uni_name(name)
        qs_map[nm] = {
            "qs_rank_2026": None if pd.isna(row[rank_col]) else float(row[rank_col]),
            "qs_overall_score": None if pd.isna(row[overall_col]) else float(row[overall_col]),
            "qs_selectivity": None if pd.isna(row["selectivity_index"]) else float(row["selectivity_index"]),
            "qs_institution_name": name,
        }

    print(f"✅ QS Germany loaded: {len(qs_map)} institutions")
    return qs_map


def match_qs(university_name: str, qs_map: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Best-effort matching:
    1) exact normalized match
    2) contains match (uni name inside qs name or vice versa)
    """
    if not university_name or not qs_map:
        return None

    u = _norm_uni_name(university_name)
    if u in qs_map:
        return qs_map[u]

    for k, v in qs_map.items():
        if u and (u in k or k in u):
            return v

    return None


def apply_qs_selectivity_penalty(score: float, qs_selectivity: Optional[float]) -> float:
    """
    Gentle penalty up to ~8 points for highly selective universities.
    qs_selectivity ~ 0..1 where higher = more selective.
    """
    if qs_selectivity is None:
        return score
    penalty = 8.0 * float(qs_selectivity)  # max 8
    return max(0.0, score - penalty)





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

    qs_map = getattr(app.state, "qs_map", {})  # ✅ loaded in lifespan

    # fallback
    if not api_key or not url:
        top = programs[:10]
        out = []
        for i, p in enumerate(top, start=1):
            qs = match_qs(p.get("University", ""), qs_map) if qs_map else None
            out.append({
                "rank": i,
                "program_name": p.get("Program", ""),
                "university": p.get("University", ""),
                "score_0_100": None,
                "reason": "DeepSeek not configured; rule-based shortlist only.",
                "uni_type": estimate_uni_type(p.get("University", "")),
                "flags": p.get("_flags", []),
                "qs_rank_2026": qs.get("qs_rank_2026") if qs else None,
                "qs_overall_score": qs.get("qs_overall_score") if qs else None,
                "qs_selectivity": qs.get("qs_selectivity") if qs else None,
                "qs_matched_name": qs.get("qs_institution_name") if qs else None,
            })
        return out

    # Pack programs with QS fields (keep prompt compact)
    packed_programs = []
    for p in programs[:30]:
        uni = p.get("University", "")
        qs = match_qs(uni, qs_map) if qs_map else None

        packed_programs.append({
            "program_name": p.get("Program", ""),
            "university": uni,
            "entrance_grade": p.get("Entrance grade", ""),
            "requirements": " ".join([
                p.get("Requirement", ""),
                p.get("Voraussetzung", ""),
                p.get("Admission requirements", "")
            ]).strip(),
            "language_info": " ".join([
                p.get("Language of instruction", ""),
                p.get("Language proficiency", "")
            ]).strip(),
            "flags": p.get("_flags", []),
            "uni_type_hint": estimate_uni_type(uni),
            # ✅ QS metrics (Germany-only)
            "qs_rank_2026": qs.get("qs_rank_2026") if qs else None,
            "qs_overall_score": qs.get("qs_overall_score") if qs else None,
            "qs_selectivity": qs.get("qs_selectivity") if qs else None,
        })

    system_msg = (
        "You are an admissions eligibility assistant for German universities.\n"
        "Score each program 0-100 estimating probability of a positive admission response.\n"
        "Use these criteria:\n"
        "1) GPA match (40%): student's german_grade_est vs entrance_grade.\n"
        "2) Background relevance (30%): background vs requirements.\n"
        "3) Language fit (20%): study_language + german_level vs language_info.\n"
        "4) Borderline flags (5%): if flags exist, slightly reduce score.\n"
        "5) QS selectivity (5%): if qs_selectivity is high, slightly reduce acceptance likelihood.\n"
        "Return STRICT JSON only, no markdown, no extra text."
    )

    user_msg = {
        "student": student_profile,
        "programs": packed_programs,
        "instructions": (
            "Return JSON array ONLY in this schema:\n"
            "[{"
            "\"rank\": 1, "
            "\"program_name\": \"...\", "
            "\"university\": \"...\", "
            "\"score_0_100\": 0, "
            "\"reason\": \"1-2 sentences\""
            "}]\n"
            "No extra keys. No markdown. JSON only."
        ),
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

    ranked: List[Dict[str, Any]] = json.loads(match.group(1))

    # Create lookup to attach flags + QS after model
    meta_map: Dict[tuple, Dict[str, Any]] = {}
    for p in programs[:30]:
        uni = p.get("University", "")
        prog = p.get("Program", "")
        qs = match_qs(uni, qs_map) if qs_map else None
        meta_map[(prog, uni)] = {
            "flags": p.get("_flags", []),
            "uni_type": estimate_uni_type(uni),
            "qs_rank_2026": qs.get("qs_rank_2026") if qs else None,
            "qs_overall_score": qs.get("qs_overall_score") if qs else None,
            "qs_selectivity": qs.get("qs_selectivity") if qs else None,
            "qs_matched_name": qs.get("qs_institution_name") if qs else None,
        }

    # Post-calibration: QS penalty + FH/TU calibration + attach meta
    for item in ranked:
        prog = item.get("program_name", "")
        uni = item.get("university", "")
        meta = meta_map.get((prog, uni), {})

        item["flags"] = meta.get("flags", [])
        item["uni_type"] = meta.get("uni_type", estimate_uni_type(uni))
        item["qs_rank_2026"] = meta.get("qs_rank_2026")
        item["qs_overall_score"] = meta.get("qs_overall_score")
        item["qs_selectivity"] = meta.get("qs_selectivity")
        item["qs_matched_name"] = meta.get("qs_matched_name")

        if item.get("score_0_100") is not None:
            try:
                s = float(item["score_0_100"])

                # ✅ QS penalty
                s = apply_qs_selectivity_penalty(s, item.get("qs_selectivity"))

                # ✅ FH/TU heuristic
                s = apply_confidence_calibration(s, item.get("uni_type", "UNI"))

                item["score_0_100"] = round(s, 1)
            except Exception:
                pass

    # Re-rank by calibrated scores (higher is better)
    ranked_sorted = sorted(
        ranked,
        key=lambda x: (x.get("score_0_100") is None, -(x.get("score_0_100") or 0.0)),
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
