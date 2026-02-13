# app.py
from __future__ import annotations

import json
import os
import re
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import pandas as pd
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

# Data sources
CRITERIA_FILE = str(BASE_DIR / "Dataset" / "UniAssistCriteria_clean.txt")     # Uni-Assist
DAAD_FILE = str(BASE_DIR / "Dataset" / "DaadCriteria.txt")                    # DAAD IT (your new file)
QS_FILE = str(BASE_DIR / "Data" / "qs_europe_2026.xlsx")                      # QS Europe (Germany-only)

# -----------------------------
# DB (migration-safe)
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

    # Add missing cols for older DBs (safe ALTERs)
    cur.execute("PRAGMA table_info(consultations)")
    existing = {row[1] for row in cur.fetchall()}

    def add_col(col: str, typ: str) -> None:
        if col not in existing:
            cur.execute(f"ALTER TABLE consultations ADD COLUMN {col} {typ}")

    add_col("study_language", "TEXT")
    add_col("german_level", "TEXT")

    conn.commit()
    conn.close()


# -----------------------------
# QS parsing + matching
# -----------------------------
def _norm_uni_name(s: str) -> str:
    s = (s or "").lower()
    s = s.replace("&", "and")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

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

    cols: List[str] = []
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
    country_col = next((c for c in df.columns if "country" in c.lower()), None)
    if not country_col:
        print("⚠️ QS country column not found (QS calibration disabled)")
        return {}

    de = df[df[country_col].astype(str).str.lower().eq("germany")].copy()

    inst_col = next((c for c in de.columns if "institution" in c.lower() and "name" in c.lower()), None)
    if not inst_col:
        print("⚠️ QS Institution Name column not found (QS calibration disabled)")
        return {}

    rank_col = next((c for c in de.columns if c.strip().lower() == "2026 rank"), None)
    overall_col = next((c for c in de.columns if c.strip().lower() == "overall score"), None)
    if not rank_col or not overall_col:
        print("⚠️ QS Rank/Overall columns not found (QS calibration disabled)")
        return {}

    def parse_rank(x):
        s = str(x).replace("=", "").strip()
        return pd.to_numeric(s, errors="coerce")

    de[rank_col] = de[rank_col].apply(parse_rank)
    de[overall_col] = pd.to_numeric(de[overall_col], errors="coerce")

    de["rank_pct_de"] = de[rank_col].rank(pct=True)
    de["overall_norm"] = de[overall_col] / 100.0

    # Higher selectivity = tougher admissions
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
    if qs_selectivity is None:
        return score
    penalty = 8.0 * float(qs_selectivity)  # max ~8 points
    return max(0.0, score - penalty)


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

    # ✅ Load QS at startup
    app.state.qs_map = load_qs_germany_map(QS_FILE)

    yield
    print("🛑 Application shutdown")


app = FastAPI(title="Uni Eligibility Finder", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


# -----------------------------
# Models
# -----------------------------
class StudentInput(BaseModel):
    name: str = Field(..., min_length=1)
    background: str = ""

    paid_pref: str = "no_preference"  # paid | unpaid | no_preference

    gpa: float
    scale_max: float = 4.0
    min_pass: float = 2.0

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
# Uni-Assist parsing
# -----------------------------
def load_uniassist_records(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        print(f"⚠️ Uni-Assist criteria file not found: {path}")
        return []

    txt = Path(path).read_text(encoding="utf-8", errors="ignore")
    chunks = [c.strip() for c in txt.split("-" * 62) if c.strip()]

    records: List[Dict[str, Any]] = []
    for c in chunks:
        if c.startswith("WINTER SEMESTER"):
            continue
        if c.upper().startswith("CRITERIA:"):
            continue

        rec: Dict[str, Any] = {"source": "uniassist"}
        for line in c.splitlines():
            m = re.match(r"^([^:]+):\s*(.*)$", line.strip())
            if m:
                rec[m.group(1).strip()] = m.group(2).strip()

        if rec.get("Program") and rec.get("University"):
            records.append(rec)

    return records


# -----------------------------
# DAAD parsing (IT file)
# -----------------------------
_DAAD_TITLE_RE = re.compile(r"\n\n([A-Z][^\n]{3,120})\n([^\n]*international course[^\n]*)", re.IGNORECASE)

_DAAD_KEYS = {
    "Degree",
    "Standard period of study (amount)",
    "Location",
    "Deadlines",
    "Study Type",
    "Admission semester",
    "Area of study",
    "Focus",
    "Target group",
    "Admission modus",
    "Admission requirements",
    "More information regarding admission requirements",
    "Languages of instruction",
    "Main language",
    "Further languages",
    "Tuition fees",
    "Tuition fee",
    "Fees in EUR",
    "Total fees",
    "More information regarding tuition fees",
    "Application deadlines",
    "Lecture period",
    "Annotation",
}


def _is_key_line(line: str) -> bool:
    return line.strip() in _DAAD_KEYS


def _extract_grade_requirement_from_text(text: str) -> Optional[float]:
    """
    Try to extract a German-grade-like requirement from DAAD admission requirement text.
    Examples: 'at least 2.5', 'overall grade 2.3', 'grade <= 2.7' etc.
    """
    t = (text or "").lower()
    # common patterns
    m = re.search(r"(?:grade|overall grade|final grade|note)\D{0,30}(\d[.,]\d)", t)
    if m:
        return float(m.group(1).replace(",", "."))
    m = re.search(r"(?:at least|minimum)\D{0,20}(\d[.,]\d)", t)
    if m:
        return float(m.group(1).replace(",", "."))
    return None


def load_daad_records(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        print(f"⚠️ DAAD file not found: {path}")
        return []

    txt = Path(path).read_text(encoding="utf-8", errors="ignore")

    iters = list(_DAAD_TITLE_RE.finditer(txt))
    if not iters:
        print("⚠️ DAAD blocks not detected (parser pattern mismatch)")
        return []

    records: List[Dict[str, Any]] = []
    for i, m in enumerate(iters):
        start = m.start(1) - 2
        end = (iters[i + 1].start(1) - 2) if i + 1 < len(iters) else len(txt)
        block = txt[start:end].strip()

        title = m.group(1).strip()
        rec: Dict[str, Any] = {
            "Program": title,
            "University": "",  # Often missing in DAAD text → keep empty
            "source": "daad",
        }

        lines = [ln.rstrip() for ln in block.splitlines()]
        current_key: Optional[str] = None
        buf: Dict[str, List[str]] = {}

        # Skip first two lines (title + descriptor) when scanning keys/values
        for ln in lines[2:]:
            s = ln.strip()
            if not s:
                continue

            if _is_key_line(s):
                current_key = s
                buf.setdefault(current_key, [])
                continue

            if current_key:
                buf[current_key].append(s)

        # Map DAAD keys into a common schema
        rec["Degree"] = " ".join(buf.get("Degree", [])).strip()
        rec["Location"] = " ".join(buf.get("Location", [])).strip()

        # Deadlines + Application deadlines combined
        deadlines = []
        deadlines += buf.get("Deadlines", [])
        deadlines += buf.get("Application deadlines", [])
        rec["Deadlines"] = "\n".join(deadlines).strip()

        # Language info
        main_lang = " ".join(buf.get("Main language", [])).strip()
        further_lang = " ".join(buf.get("Further languages", [])).strip()
        if main_lang or further_lang:
            rec["Language of instruction"] = " / ".join([x for x in [main_lang, further_lang] if x]).strip()
        else:
            rec["Language of instruction"] = ""

        # Admission requirements
        adm_req = "\n".join(buf.get("Admission requirements", [])).strip()
        more_adm = "\n".join(buf.get("More information regarding admission requirements", [])).strip()
        full_adm = "\n".join([x for x in [adm_req, more_adm] if x]).strip()
        rec["Admission requirements"] = full_adm

        # Admission modus
        rec["Admission modus"] = " ".join(buf.get("Admission modus", [])).strip()

        # Tuition
        tuition = []
        tuition += buf.get("Tuition fee", [])
        tuition += buf.get("Tuition fees", [])
        tuition += buf.get("Fees in EUR", [])
        tuition += buf.get("Total fees", [])
        tuition += buf.get("More information regarding tuition fees", [])
        rec["Tuition fee"] = " ".join(tuition).strip()

        # Use extracted grade as "Entrance grade" equivalent if possible
        grade_req = _extract_grade_requirement_from_text(full_adm)
        if grade_req is not None:
            rec["Entrance grade"] = str(grade_req)  # keep as string, same as uniassist style
        else:
            rec["Entrance grade"] = ""

        records.append(rec)

    print(f"✅ DAAD records loaded: {len(records)} programs")
    return records


# -----------------------------
# Merge records (Uni-Assist + DAAD)
# -----------------------------
def _norm_program_name(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def merge_records(ua: List[Dict[str, Any]], daad: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Merge duplicates where possible:
      - primary key: (Program, University) when University exists
      - if DAAD university missing, keep as separate record (still valuable)
    """
    out: List[Dict[str, Any]] = []
    index: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def key_of(r: Dict[str, Any]) -> Optional[Tuple[str, str]]:
        prog = _norm_program_name(r.get("Program", ""))
        uni = _norm_uni_name(r.get("University", "")) if r.get("University") else ""
        if prog and uni:
            return (prog, uni)
        return None

    for r in ua:
        k = key_of(r)
        if k:
            index[k] = r
        out.append(r)

    for r in daad:
        k = key_of(r)
        if k and k in index:
            # merge into existing uniassist record (prefer filled fields)
            base = index[k]
            base["source"] = "both"
            for field in [
                "Deadlines", "Tuition fee", "Admission modus",
                "Admission requirements", "Language of instruction"
            ]:
                if (not base.get(field)) and r.get(field):
                    base[field] = r[field]
            if (not base.get("Entrance grade")) and r.get("Entrance grade"):
                base["Entrance grade"] = r["Entrance grade"]
        else:
            out.append(r)

    return out


# -----------------------------
# Language logic + German level logic
# -----------------------------
GER_LEVEL_SCORE = {"none": 0, "a2": 2, "b1": 3, "b2": 4, "c1": 5}


def _lang_blob(r: Dict[str, Any]) -> str:
    blob = " ".join([
        str(r.get("Language of instruction", "")),
        str(r.get("Language proficiency", "")),
        str(r.get("Admission requirements", "")),
        str(r.get("University info", "")),
        str(r.get("Notes", "")),
    ])
    return blob.lower()


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
# FH vs TU heuristic
# -----------------------------
def estimate_uni_type(university_name: str) -> str:
    u = (university_name or "").lower()
    if "university of applied sciences" in u or "hochschule" in u or "fh" in u:
        return "FH"
    if "technical university" in u or "technische universität" in u or u.startswith("tu "):
        return "TU"
    return "UNI"


def apply_confidence_calibration(score: float, uni_type: str) -> float:
    if uni_type == "FH":
        return min(100.0, score + 5.0)
    if uni_type == "TU":
        return max(0.0, score - 5.0)
    return score


# -----------------------------
# Filtering (borderline-aware)
# -----------------------------
def filter_programs(
    records: List[Dict[str, Any]],
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

        # Entrance grade (if parseable)
        eg = str(r.get("Entrance grade", "") or "").strip()
        if eg:
            m = re.search(r"(\d+[.,]\d+|\d+)", eg)
            if m:
                req = float(m.group(1).replace(",", "."))
                if german_grade_est > req:
                    ok = False
            else:
                flags.append("entrance_grade_unparsed")
        else:
            # not a rejection; but important to flag
            flags.append("entrance_grade_missing")

        blob = _lang_blob(r)
        plang = detect_program_language(blob)

        # language preference filter
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
            if plang == "unknown":
                flags.append("language_unknown")

        # German level requirement
        if study_language in ("german", "both") and plang in ("german", "both"):
            req_level = extract_required_german_level(blob)
            if req_level:
                if not german_level_ok(german_level, req_level):
                    ok = False
            else:
                flags.append("german_level_unknown")

        # Requirements missing
        reqtxt = " ".join([
            str(r.get("Requirement", "")),
            str(r.get("Voraussetzung", "")),
            str(r.get("Admission requirements", "")),
        ]).strip().lower()
        if not reqtxt:
            flags.append("requirements_missing")

        # background mismatch possible (soft)
        if bt and reqtxt:
            strong_terms = ["electrical", "computer", "engineering", "business", "informatics", "mechanical", "software", "information technology", "it"]
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
# DeepSeek ranking (with QS + post-calibration)
# -----------------------------
def deepseek_rank(programs: List[Dict[str, Any]], student_profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    url = os.getenv("DEEPSEEK_URL", "").strip()
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    temperature = float(os.getenv("DEEPSEEK_TEMPERATURE", "0.2"))

    qs_map = getattr(app.state, "qs_map", {})

    # fallback if not configured
    if not api_key or not url:
        top = programs[:10]
        out: List[Dict[str, Any]] = []
        for i, p in enumerate(top, start=1):
            uni = p.get("University", "")
            qs = match_qs(uni, qs_map) if uni else None
            out.append({
                "rank": i,
                "program_name": p.get("Program", ""),
                "university": uni or "(institution not provided)",
                "score_0_100": None,
                "reason": "DeepSeek not configured; rule-based shortlist only.",
                "uni_type": estimate_uni_type(uni),
                "flags": p.get("_flags", []),
                "source": p.get("source", ""),
                "qs_rank_2026": qs.get("qs_rank_2026") if qs else None,
                "qs_overall_score": qs.get("qs_overall_score") if qs else None,
                "qs_selectivity": qs.get("qs_selectivity") if qs else None,
                "qs_matched_name": qs.get("qs_institution_name") if qs else None,
            })
        return out

    packed_programs = []
    for p in programs[:30]:
        uni = p.get("University", "")
        qs = match_qs(uni, qs_map) if uni else None
        packed_programs.append({
            "program_name": p.get("Program", ""),
            "university": uni or "",
            "location": p.get("Location", ""),
            "entrance_grade": p.get("Entrance grade", ""),
            "requirements": " ".join([
                str(p.get("Requirement", "")),
                str(p.get("Voraussetzung", "")),
                str(p.get("Admission requirements", "")),
            ]).strip(),
            "language_info": str(p.get("Language of instruction", "")).strip(),
            "deadlines": str(p.get("Deadlines", "")).strip(),
            "tuition_fee": str(p.get("Tuition fee", "")).strip(),
            "admission_modus": str(p.get("Admission modus", "")).strip(),
            "flags": p.get("_flags", []),
            "source": p.get("source", ""),
            "uni_type_hint": estimate_uni_type(uni),
            "qs_rank_2026": qs.get("qs_rank_2026") if qs else None,
            "qs_overall_score": qs.get("qs_overall_score") if qs else None,
            "qs_selectivity": qs.get("qs_selectivity") if qs else None,
        })

    system_msg = (
        "You are an admissions eligibility assistant for German university programs.\n"
        "Score each program 0-100 estimating probability of a positive admission response.\n"
        "Use these criteria:\n"
        "1) GPA match (40%): student's german_grade_est vs entrance_grade.\n"
        "2) Background relevance (30%): background vs requirements.\n"
        "3) Language fit (20%): study_language + german_level vs language_info.\n"
        "4) Borderline flags (5%): if flags exist, reduce score slightly.\n"
        "5) QS selectivity (5%): if qs_selectivity is high, reduce acceptance likelihood slightly.\n"
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
            "JSON only."
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

    m = re.search(r"(\[.*\])", content, flags=re.S)
    if not m:
        raise RuntimeError(f"DeepSeek returned non-JSON: {content[:800]}")

    ranked: List[Dict[str, Any]] = json.loads(m.group(1))

    # Attach meta + post-calibration
    meta_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for p in programs[:30]:
        prog = p.get("Program", "")
        uni = p.get("University", "") or ""
        qs = match_qs(uni, qs_map) if uni else None
        meta_map[(prog, uni)] = {
            "flags": p.get("_flags", []),
            "uni_type": estimate_uni_type(uni),
            "source": p.get("source", ""),
            "location": p.get("Location", ""),
            "deadlines": p.get("Deadlines", ""),
            "qs_rank_2026": qs.get("qs_rank_2026") if qs else None,
            "qs_overall_score": qs.get("qs_overall_score") if qs else None,
            "qs_selectivity": qs.get("qs_selectivity") if qs else None,
            "qs_matched_name": qs.get("qs_institution_name") if qs else None,
        }

    for item in ranked:
        prog = item.get("program_name", "")
        uni = item.get("university", "") or ""
        meta = meta_map.get((prog, uni), {})

        item["flags"] = meta.get("flags", [])
        item["uni_type"] = meta.get("uni_type", estimate_uni_type(uni))
        item["source"] = meta.get("source", "")
        item["location"] = meta.get("location", "")
        item["deadlines"] = meta.get("deadlines", "")

        item["qs_rank_2026"] = meta.get("qs_rank_2026")
        item["qs_overall_score"] = meta.get("qs_overall_score")
        item["qs_selectivity"] = meta.get("qs_selectivity")
        item["qs_matched_name"] = meta.get("qs_matched_name")

        # score post-calibration
        if item.get("score_0_100") is not None:
            try:
                s = float(item["score_0_100"])
                s = apply_qs_selectivity_penalty(s, item.get("qs_selectivity"))
                s = apply_confidence_calibration(s, item.get("uni_type", "UNI"))
                item["score_0_100"] = round(s, 1)
            except Exception:
                pass

    ranked_sorted = sorted(
        ranked,
        key=lambda x: (x.get("score_0_100") is None, -(x.get("score_0_100") or 0.0)),
    )
    for i, it in enumerate(ranked_sorted, start=1):
        it["rank"] = i

    return ranked_sorted


# -----------------------------
# Storage
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
    ua = load_uniassist_records(CRITERIA_FILE)
    daad = load_daad_records(DAAD_FILE)
    records = merge_records(ua, daad)

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
