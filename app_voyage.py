# app.py
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import time
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


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_PATH, override=True)

DB_PATH = str(BASE_DIR / "consultant.db")

# Data sources
CRITERIA_FILE = str(BASE_DIR / "Dataset" / "UniAssistCriteria_clean.txt")  # Uni-Assist
DAAD_FILE = str(BASE_DIR / "Dataset" / "DaadCriteria.txt")                 # DAAD
QS_FILE = str(BASE_DIR / "Data" / "qs_europe_2026.xlsx")                   # QS Europe (Germany-only)

# Retrieval knobs
RETRIEVE_TOP_K = int(os.getenv("RETRIEVE_TOP_K", "250"))
LLM_PACK_TOP_N = int(os.getenv("LLM_PACK_TOP_N", "30"))
RETURN_TOP_N = int(os.getenv("RETURN_TOP_N", "10"))

PROMPT_VERSION = os.getenv("PROMPT_VERSION", "v4-deepseek-chat-voyage-embeddings")

# DeepSeek chat config
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_URL = os.getenv("DEEPSEEK_URL", "").strip()  # https://api.deepseek.com/v1/chat/completions
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_TEMPERATURE = float(os.getenv("DEEPSEEK_TEMPERATURE", "0.2"))

# Embeddings provider: Voyage
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY", "").strip()
VOYAGE_EMBED_URL = os.getenv("VOYAGE_EMBED_URL", "https://api.voyageai.com/v1/embeddings").strip()
VOYAGE_EMBED_MODEL = os.getenv("VOYAGE_EMBED_MODEL", "voyage-3-large").strip()
VOYAGE_INPUT_TYPE_DOC = os.getenv("VOYAGE_INPUT_TYPE_DOC", "document").strip()
VOYAGE_INPUT_TYPE_QUERY = os.getenv("VOYAGE_INPUT_TYPE_QUERY", "query").strip()


# ============================================================
# DB
# ============================================================

def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        '''
        CREATE TABLE IF NOT EXISTS consultations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            name TEXT,
            background TEXT,
            gpa REAL,
            scale_max REAL,
            min_pass REAL,
            scale_direction TEXT,
            german_grade_est REAL,
            paid_pref TEXT,
            study_language TEXT,
            german_level TEXT,
            dataset_hash_ua TEXT,
            dataset_hash_daad TEXT,
            prompt_version TEXT,
            results_json TEXT
        )
        '''
    )

    cur.execute(
        '''
        CREATE TABLE IF NOT EXISTS llm_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT,
            model TEXT,
            temperature REAL,
            prompt_version TEXT,
            dataset_hash_ua TEXT,
            dataset_hash_daad TEXT,
            student_json TEXT,
            packed_programs_json TEXT,
            system_prompt TEXT,
            user_prompt TEXT,
            raw_response TEXT,
            parsed_json TEXT,
            status TEXT,
            error TEXT
        )
        '''
    )

    cur.execute(
        '''
        CREATE TABLE IF NOT EXISTS response_cache (
            cache_key TEXT PRIMARY KEY,
            created_at TEXT,
            response_json TEXT
        )
        '''
    )

    cur.execute(
        '''
        CREATE TABLE IF NOT EXISTS embeddings_cache (
            text_hash TEXT PRIMARY KEY,
            created_at TEXT,
            provider TEXT,
            model TEXT,
            vector_json TEXT
        )
        '''
    )

    conn.commit()
    conn.close()


# ============================================================
# UTIL
# ============================================================

def file_sha256(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha_text(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()


def make_cache_key(student_profile: Dict[str, Any], dataset_hash_ua: str, dataset_hash_daad: str, prompt_version: str) -> str:
    payload = {
        "student": student_profile,
        "dataset_hash_ua": dataset_hash_ua,
        "dataset_hash_daad": dataset_hash_daad,
        "prompt_version": prompt_version,
        "retrieve_top_k": RETRIEVE_TOP_K,
        "llm_pack_top_n": LLM_PACK_TOP_N,
        "embed_provider": "voyage" if VOYAGE_API_KEY else "disabled",
        "embed_model": VOYAGE_EMBED_MODEL if VOYAGE_API_KEY else "",
    }
    return hashlib.sha256(stable_json_dumps(payload).encode("utf-8")).hexdigest()


def cache_get(cache_key: str) -> Optional[Dict[str, Any]]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT response_json FROM response_cache WHERE cache_key = ?", (cache_key,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    try:
        return json.loads(row[0])
    except Exception:
        return None


def cache_set(cache_key: str, response: Dict[str, Any]) -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO response_cache (cache_key, created_at, response_json) VALUES (?, ?, ?)",
        (cache_key, datetime.utcnow().isoformat(), json.dumps(response, ensure_ascii=False)),
    )
    conn.commit()
    conn.close()


def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(a: List[float]) -> float:
    return math.sqrt(max(1e-12, _dot(a, a)))


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    a2 = a[:n]
    b2 = b[:n]
    return float(_dot(a2, b2) / (_norm(a2) * _norm(b2)))


# ============================================================
# QS parsing + matching
# ============================================================

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
    penalty = 8.0 * float(qs_selectivity)
    return max(0.0, score - penalty)


# ============================================================
# INPUT MODELS
# ============================================================

class StudentInput(BaseModel):
    name: str = Field(..., min_length=1)
    background: str = ""

    paid_pref: str = "no_preference"

    gpa: float
    scale_max: float = 4.0
    min_pass: float = 2.0

    # Pakistan typical: 4.0 is best
    scale_direction: str = "higher_is_better"  # higher_is_better | lower_is_better

    study_language: str = "english"
    german_level: str = "none"


# ============================================================
# GPA conversion
# ============================================================

def convert_to_german_grade(
    grade: float,
    max_grade: float,
    min_passing_grade: float,
    scale_direction: str,
    german_best: float = 1.0,
    german_worst_pass: float = 4.0,
) -> float:
    if max_grade <= min_passing_grade:
        raise ValueError("scale_max must be > min_pass")

    grade = max(min_passing_grade, min(grade, max_grade))

    if (scale_direction or "").lower() == "lower_is_better":
        grade_eff = (max_grade + min_passing_grade) - grade
    else:
        grade_eff = grade

    german = german_best + (german_worst_pass - german_best) * (max_grade - grade_eff) / (max_grade - min_passing_grade)
    return round(float(german), 2)


# ============================================================
# PARSERS: Uni-Assist + DAAD
# ============================================================

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
    t = (text or "").lower()
    m = re.search(r"(?:grade|overall grade|final grade|note)\D{0,30}(\d[.,]\d)", t)
    if m:
        return float(m.group(1).replace(",", "."))
    m = re.search(r"(?:at least|minimum)\D{0,20}(\d[.,]\d)", t)
    if m:
        return float(m.group(1).replace(",", "."))
    return None


def _extract_university_hint_from_block(block: str) -> str:
    b = (block or "")
    patterns = [
        r"\bRWTH\s+[A-Z][A-Za-zÄÖÜäöüß\- ]{2,80}\b",
        r"\bTU\s+[A-Z][A-Za-zÄÖÜäöüß\- ]{2,80}\b",
        r"\bTechnische\s+Universit[aä]t\s+[A-Z][A-Za-zÄÖÜäöüß\- ]{2,80}\b",
        r"\bUniversit[aä]t\s+[A-Z][A-Za-zÄÖÜäöüß\- ]{2,80}\b",
        r"\bUniversity\s+of\s+[A-Z][A-Za-z\- ]{2,80}\b",
        r"\bHochschule\s+[A-Z][A-Za-zÄÖÜäöüß\- ]{2,80}\b",
    ]
    for pat in patterns:
        m = re.search(pat, b)
        if m:
            return m.group(0).strip()
    return ""


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
        rec: Dict[str, Any] = {"Program": title, "University": "", "source": "daad"}

        lines = [ln.rstrip() for ln in block.splitlines()]
        current_key: Optional[str] = None
        buf: Dict[str, List[str]] = {}

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

        rec["Degree"] = " ".join(buf.get("Degree", [])).strip()
        rec["Location"] = " ".join(buf.get("Location", [])).strip()

        deadlines = []
        deadlines += buf.get("Deadlines", [])
        deadlines += buf.get("Application deadlines", [])
        rec["Deadlines"] = "\n".join(deadlines).strip()

        main_lang = " ".join(buf.get("Main language", [])).strip()
        further_lang = " ".join(buf.get("Further languages", [])).strip()
        rec["Language of instruction"] = " / ".join([x for x in [main_lang, further_lang] if x]).strip()

        adm_req = "\n".join(buf.get("Admission requirements", [])).strip()
        more_adm = "\n".join(buf.get("More information regarding admission requirements", [])).strip()
        full_adm = "\n".join([x for x in [adm_req, more_adm] if x]).strip()
        rec["Admission requirements"] = full_adm

        rec["Admission modus"] = " ".join(buf.get("Admission modus", [])).strip()

        tuition = []
        tuition += buf.get("Tuition fee", [])
        tuition += buf.get("Tuition fees", [])
        tuition += buf.get("Fees in EUR", [])
        tuition += buf.get("Total fees", [])
        tuition += buf.get("More information regarding tuition fees", [])
        rec["Tuition fee"] = " ".join(tuition).strip()

        grade_req = _extract_grade_requirement_from_text(full_adm)
        rec["Entrance grade"] = str(grade_req) if grade_req is not None else ""

        rec["University"] = _extract_university_hint_from_block(block)

        records.append(rec)

    print(f"✅ DAAD records loaded: {len(records)} programs")
    return records


def _norm_program_name(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def merge_records(ua: List[Dict[str, Any]], daad: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
            base = index[k]
            base["source"] = "both"
            for field in ["Deadlines", "Tuition fee", "Admission modus", "Admission requirements", "Language of instruction"]:
                if (not base.get(field)) and r.get(field):
                    base[field] = r[field]
            if (not base.get("Entrance grade")) and r.get("Entrance grade"):
                base["Entrance grade"] = r["Entrance grade"]
        else:
            out.append(r)

    return out


# ============================================================
# Language logic
# ============================================================

GER_LEVEL_SCORE = {"none": 0, "a2": 2, "b1": 3, "b2": 4, "c1": 5}


def _lang_blob(r: Dict[str, Any]) -> str:
    blob = " ".join(
        [
            str(r.get("Language of instruction", "")),
            str(r.get("Language proficiency", "")),
            str(r.get("Admission requirements", "")),
            str(r.get("University info", "")),
            str(r.get("Notes", "")),
        ]
    )
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


# ============================================================
# FH vs TU
# ============================================================

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


# ============================================================
# FILTERING
# ============================================================

def filter_programs(records: List[Dict[str, Any]], german_grade_est: float, study_language: str, german_level: str, background_text: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    bt = (background_text or "").lower().strip()

    for r in records:
        ok = True
        flags: List[str] = []

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
            flags.append("entrance_grade_missing")

        blob = _lang_blob(r)
        plang = detect_program_language(blob)

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

        if study_language in ("german", "both") and plang in ("german", "both"):
            req_level = extract_required_german_level(blob)
            if req_level and (not german_level_ok(german_level, req_level)):
                ok = False

        reqtxt = " ".join([str(r.get("Requirement", "")), str(r.get("Voraussetzung", "")), str(r.get("Admission requirements", ""))]).strip().lower()
        if not reqtxt:
            flags.append("requirements_missing")

        # soft flag only
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


# ============================================================
# VOYAGE EMBEDDINGS
# ============================================================

def program_embedding_text(r: Dict[str, Any]) -> str:
    parts = [
        str(r.get("Program", "")),
        str(r.get("University", "")),
        str(r.get("Focus", "")),
        str(r.get("Specialisation", "")),
        str(r.get("Requirement", "")),
        str(r.get("Voraussetzung", "")),
        str(r.get("Admission requirements", "")),
        str(r.get("Area of study", "")),
        str(r.get("Target group", "")),
        str(r.get("Annotation", "")),
    ]
    text = " | ".join([p.strip() for p in parts if p and str(p).strip()])
    return text[:5000]


def embeddings_cache_get_many(hashes: List[str], provider: str, model: str) -> Dict[str, List[float]]:
    if not hashes:
        return {}
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    qmarks = ",".join(["?"] * len(hashes))
    cur.execute(
        f"SELECT text_hash, vector_json FROM embeddings_cache WHERE provider = ? AND model = ? AND text_hash IN ({qmarks})",
        (provider, model, *hashes),
    )
    rows = cur.fetchall()
    conn.close()
    out: Dict[str, List[float]] = {}
    for h, vj in rows:
        try:
            out[h] = json.loads(vj)
        except Exception:
            continue
    return out


def embeddings_cache_set_many(items: Dict[str, List[float]], provider: str, model: str) -> None:
    if not items:
        return
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    now = datetime.utcnow().isoformat()
    for h, vec in items.items():
        cur.execute(
            "INSERT OR REPLACE INTO embeddings_cache (text_hash, created_at, provider, model, vector_json) VALUES (?, ?, ?, ?, ?)",
            (h, now, provider, model, json.dumps(vec)),
        )
    conn.commit()
    conn.close()


def voyage_embed(texts: List[str], input_type: str, max_retries: int = 2) -> List[List[float]]:
    if not VOYAGE_API_KEY:
        raise RuntimeError("VOYAGE_API_KEY missing (required for Voyage embeddings).")

    headers = {"Authorization": f"Bearer {VOYAGE_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": VOYAGE_EMBED_MODEL,
        "input": texts if len(texts) > 1 else texts[0],
        "input_type": input_type,
    }

    last_err = None
    for attempt in range(max_retries + 1):
        try:
            r = requests.post(VOYAGE_EMBED_URL, headers=headers, json=payload, timeout=60)
            if r.status_code >= 400:
                raise RuntimeError(f"Embeddings error {r.status_code}: {r.text[:1200]}")
            data = r.json()
            items = data.get("data", [])
            if not items or "embedding" not in items[0]:
                raise RuntimeError(f"Unexpected embeddings response: {json.dumps(data)[:1200]}")
            items_sorted = sorted(items, key=lambda x: x.get("index", 0))
            return [it["embedding"] for it in items_sorted]
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(0.8 * (attempt + 1))
                continue
            raise RuntimeError(str(last_err))


def ensure_program_embeddings(texts: List[str]) -> List[List[float]]:
    provider = "voyage"
    model = VOYAGE_EMBED_MODEL

    hashes = [sha_text(t) for t in texts]
    cached = embeddings_cache_get_many(hashes, provider=provider, model=model)

    missing: List[Tuple[int, str, str]] = []
    for i, (h, t) in enumerate(zip(hashes, texts)):
        if h not in cached:
            missing.append((i, h, t))

    if missing:
        batch = int(os.getenv("EMBED_BATCH", "64"))
        new_vecs: Dict[str, List[float]] = {}
        for start in range(0, len(missing), batch):
            chunk = missing[start:start + batch]
            chunk_texts = [t for _, _, t in chunk]
            vecs = voyage_embed(chunk_texts, input_type=VOYAGE_INPUT_TYPE_DOC)
            if len(vecs) != len(chunk):
                raise RuntimeError("Embeddings size mismatch.")
            for (idx, h, _), v in zip(chunk, vecs):
                new_vecs[h] = v
        embeddings_cache_set_many(new_vecs, provider=provider, model=model)
        cached.update(new_vecs)

    return [cached[h] for h in hashes]


def retrieve_by_background(records: List[Dict[str, Any]], background: str, top_k: int = RETRIEVE_TOP_K) -> List[Tuple[Dict[str, Any], float]]:
    bg = (background or "").strip()
    if not bg:
        return [(r, 0.0) for r in records[:top_k]]

    prog_texts = [program_embedding_text(r) for r in records]
    prog_vecs = ensure_program_embeddings(prog_texts)
    bg_vec = voyage_embed([bg], input_type=VOYAGE_INPUT_TYPE_QUERY)[0]

    pairs: List[Tuple[Dict[str, Any], float]] = []
    for r, v in zip(records, prog_vecs):
        pairs.append((r, cosine_similarity(bg_vec, v)))

    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[: max(1, top_k)]


# ============================================================
# DeepSeek ranking
# ============================================================

def log_llm_run(model: str, temperature: float, dataset_hash_ua: str, dataset_hash_daad: str, student_json: Dict[str, Any],
                packed_programs: List[Dict[str, Any]], system_prompt: str, user_prompt: str, raw_response: str,
                parsed_json: Optional[Any], status: str, error: str = "") -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        '''
        INSERT INTO llm_runs
        (created_at, model, temperature, prompt_version, dataset_hash_ua, dataset_hash_daad,
         student_json, packed_programs_json, system_prompt, user_prompt, raw_response, parsed_json,
         status, error)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''',
        (
            datetime.utcnow().isoformat(),
            model,
            float(temperature),
            PROMPT_VERSION,
            dataset_hash_ua,
            dataset_hash_daad,
            json.dumps(student_json, ensure_ascii=False),
            json.dumps(packed_programs, ensure_ascii=False),
            system_prompt,
            user_prompt,
            raw_response,
            json.dumps(parsed_json, ensure_ascii=False) if parsed_json is not None else "",
            status,
            error,
        ),
    )
    conn.commit()
    conn.close()


def deepseek_rank(programs: List[Dict[str, Any]], student_profile: Dict[str, Any],
                  dataset_hash_ua: str, dataset_hash_daad: str, qs_map: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not DEEPSEEK_API_KEY or not DEEPSEEK_URL:
        return [
            {
                "rank": i,
                "program_name": p.get("Program", ""),
                "university": p.get("University", "") or "(institution not provided)",
                "score_0_100": None,
                "reason": "DeepSeek not configured; deterministic shortlist only.",
                "background_similarity": p.get("_background_similarity", None),
            }
            for i, p in enumerate(programs[:RETURN_TOP_N], start=1)
        ]

    packed_programs = []
    for p in programs[:LLM_PACK_TOP_N]:
        uni = p.get("University", "")
        qs = match_qs(uni, qs_map) if uni else None
        packed_programs.append(
            {
                "program_name": p.get("Program", ""),
                "university": uni or "",
                "location": p.get("Location", ""),
                "entrance_grade": p.get("Entrance grade", ""),
                "requirements": " ".join([str(p.get("Requirement", "")), str(p.get("Voraussetzung", "")), str(p.get("Admission requirements", ""))]).strip(),
                "language_info": str(p.get("Language of instruction", "")).strip(),
                "deadlines": str(p.get("Deadlines", "")).strip(),
                "tuition_fee": str(p.get("Tuition fee", "")).strip(),
                "admission_modus": str(p.get("Admission modus", "")).strip(),
                "flags": p.get("_flags", []),
                "source": p.get("source", ""),
                "uni_type_hint": estimate_uni_type(uni),
                "background_similarity": p.get("_background_similarity", None),
                "qs_selectivity": qs.get("qs_selectivity") if qs else None,
            }
        )

    system_msg = (
        "You are an admissions eligibility assistant for German university programs.\n"
        "Score each program 0-100 estimating probability of a positive admission response.\n"
        "Use these criteria:\n"
        "1) GPA match (40%): student's german_grade_est vs entrance_grade.\n"
        "2) Background relevance (30%): student's background vs requirements.\n"
        "   - IMPORTANT: you are also given background_similarity (0..1). Use it as a strong signal.\n"
        "3) Language fit (20%): study_language + german_level vs language_info.\n"
        "4) Borderline flags (5%): if flags exist, reduce score slightly.\n"
        "5) QS selectivity (5%): if qs_selectivity is high, reduce acceptance likelihood slightly.\n"
        "Return STRICT JSON only, no markdown, no extra text."
    )

    user_payload = {
        "student": student_profile,
        "programs": packed_programs,
        "instructions": (
            "Return a JSON array ONLY in this schema:\n"
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
        "model": DEEPSEEK_MODEL,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        "temperature": float(DEEPSEEK_TEMPERATURE),
    }
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}

    raw_text = ""
    ranked_items: List[Dict[str, Any]] = []
    try:
        resp = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=60)
        raw_text = resp.text
        if resp.status_code >= 400:
            raise RuntimeError(f"DeepSeek error {resp.status_code}: {resp.text[:2000]}")
        data = resp.json()
        content = data["choices"][0]["message"]["content"].strip()
        m = re.search(r"(\[.*\])", content, flags=re.S)
        if not m:
            raise RuntimeError(f"DeepSeek returned non-JSON: {content[:800]}")
        ranked_items = json.loads(m.group(1))
        log_llm_run(DEEPSEEK_MODEL, DEEPSEEK_TEMPERATURE, dataset_hash_ua, dataset_hash_daad,
                    student_profile, packed_programs, system_msg, json.dumps(user_payload, ensure_ascii=False),
                    content, ranked_items, "ok")
    except Exception as e:
        log_llm_run(DEEPSEEK_MODEL, DEEPSEEK_TEMPERATURE, dataset_hash_ua, dataset_hash_daad,
                    student_profile, packed_programs, system_msg, json.dumps(user_payload, ensure_ascii=False),
                    raw_text, None, "error", str(e))
        fallback = sorted(programs[:LLM_PACK_TOP_N], key=lambda p: float(p.get("_background_similarity") or 0.0), reverse=True)
        ranked_items = [
            {
                "rank": i,
                "program_name": p.get("Program", ""),
                "university": p.get("University", "") or "(institution not provided)",
                "score_0_100": 50.0,
                "reason": "LLM unavailable/error; fallback ranking using background similarity + rules.",
            }
            for i, p in enumerate(fallback[:RETURN_TOP_N], start=1)
        ]

    return ranked_items


# ============================================================
# Storage
# ============================================================

def save_consultation(inp: StudentInput, german_grade_est: float, results: List[Dict[str, Any]], dataset_hash_ua: str, dataset_hash_daad: str) -> None:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        '''
        INSERT INTO consultations
        (created_at, name, background, gpa, scale_max, min_pass, scale_direction, german_grade_est,
         paid_pref, study_language, german_level, dataset_hash_ua, dataset_hash_daad, prompt_version, results_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''',
        (
            datetime.utcnow().isoformat(),
            inp.name,
            inp.background,
            inp.gpa,
            inp.scale_max,
            inp.min_pass,
            inp.scale_direction,
            german_grade_est,
            inp.paid_pref,
            inp.study_language,
            inp.german_level,
            dataset_hash_ua,
            dataset_hash_daad,
            PROMPT_VERSION,
            json.dumps(results, ensure_ascii=False),
        ),
    )
    conn.commit()
    conn.close()


# ============================================================
# Program detail helper
# ============================================================

def program_details_queries(university: str, program: str) -> Dict[str, Any]:
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


# ============================================================
# Lifespan
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()

    print(f"📄 Loading .env from: {ENV_PATH} | exists={ENV_PATH.exists()}")
    if DEEPSEEK_API_KEY and DEEPSEEK_URL:
        print("✅ DeepSeek chat configured")
        print(f"🤖 Model: {DEEPSEEK_MODEL} | temp={DEEPSEEK_TEMPERATURE}")
    else:
        print("⚠️ DeepSeek chat NOT configured – using deterministic shortlist only")

    if VOYAGE_API_KEY:
        print(f"✅ Voyage embeddings configured | model={VOYAGE_EMBED_MODEL}")
        print(f"🌐 Voyage embeddings URL: {VOYAGE_EMBED_URL}")
    else:
        print("⚠️ Voyage embeddings NOT configured – set VOYAGE_API_KEY")

    ua = load_uniassist_records(CRITERIA_FILE)
    daad = load_daad_records(DAAD_FILE)
    records = merge_records(ua, daad)

    app.state.records = records
    app.state.dataset_hash_ua = file_sha256(CRITERIA_FILE)
    app.state.dataset_hash_daad = file_sha256(DAAD_FILE)
    app.state.qs_map = load_qs_germany_map(QS_FILE)

    print(f"✅ Records loaded: {len(records)}")
    yield
    print("🛑 Application shutdown")


# ============================================================
# APP + ROUTES
# ============================================================

app = FastAPI(title="Uni Eligibility Finder", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
def home():
    with open(BASE_DIR / "static" / "index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/api/recommend")
def recommend(inp: StudentInput):
    records: List[Dict[str, Any]] = getattr(app.state, "records", [])
    dataset_hash_ua: str = getattr(app.state, "dataset_hash_ua", "")
    dataset_hash_daad: str = getattr(app.state, "dataset_hash_daad", "")
    qs_map: Dict[str, Dict[str, Any]] = getattr(app.state, "qs_map", {})

    german_est = convert_to_german_grade(inp.gpa, inp.scale_max, inp.min_pass, inp.scale_direction)

    student_profile = {
        "name": inp.name,
        "background": inp.background,
        "gpa": inp.gpa,
        "scale_max": inp.scale_max,
        "min_pass": inp.min_pass,
        "scale_direction": inp.scale_direction,
        "german_grade_est": german_est,
        "paid_pref": inp.paid_pref,
        "study_language": inp.study_language,
        "german_level": inp.german_level,
    }

    cache_key = make_cache_key(student_profile, dataset_hash_ua, dataset_hash_daad, PROMPT_VERSION)
    cached = cache_get(cache_key)
    if cached:
        return cached

    try:
        retrieved = retrieve_by_background(records, inp.background, top_k=RETRIEVE_TOP_K)
    except Exception as e:
        print(f"⚠️ Semantic retrieval disabled: {e}")
        retrieved = [(r, 0.0) for r in records[:RETRIEVE_TOP_K]]

    candidates: List[Dict[str, Any]] = []
    for r, sim in retrieved:
        r2 = dict(r)
        r2["_background_similarity"] = round(float(sim), 4)
        candidates.append(r2)

    filtered = filter_programs(candidates, german_est, inp.study_language, inp.german_level, inp.background)
    filtered.sort(key=lambda x: float(x.get("_background_similarity") or 0.0), reverse=True)

    ranked = deepseek_rank(filtered, student_profile, dataset_hash_ua, dataset_hash_daad, qs_map)

    save_consultation(inp, german_est, ranked[:20], dataset_hash_ua, dataset_hash_daad)

    response = {
        "prompt_version": PROMPT_VERSION,
        "dataset_hash_ua": dataset_hash_ua,
        "dataset_hash_daad": dataset_hash_daad,
        "german_grade_est": german_est,
        "candidate_count": len(candidates),
        "shortlist_count": len(filtered),
        "results": ranked[:RETURN_TOP_N],
        "notes": {
            "embeddings": {
                "provider": "voyage" if VOYAGE_API_KEY else "disabled",
                "model": VOYAGE_EMBED_MODEL if VOYAGE_API_KEY else "(not set)",
            }
        },
    }

    cache_set(cache_key, response)
    return response


@app.get("/api/program_details")
def program_details(university: str, program: str):
    return program_details_queries(university=university, program=program)
