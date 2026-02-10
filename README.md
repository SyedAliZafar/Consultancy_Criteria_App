# Uni Eligibility Finder (Germany) — Consultant Tool

A small FastAPI web app that helps a consultant shortlist German university programs and estimate the probability of a positive admission response based on:

- **Hard eligibility rules** (grade thresholds, language fit, basic background fit)
- **Borderline flags** (unknown/missing data → reduce confidence, don’t hard-reject)
- **University competitiveness prior (QS Europe 2026, Germany-only)** used as a light penalty for highly selective universities
- Optional AI ranking using **DeepSeek** (OpenAI-style chat/completions endpoint)

---

## Project structure

```txt
ConsultancyAlgorithm/
  app.py
  .env
  requirements.txt
  consultant.db                  # auto-created by app
  Dataset/
    UniAssistCriteria_clean.txt  # parsed program criteria
  Data/
    qs_europe_2026.xlsx          # QS Europe file (Germany rows used)
  static/
    index.html                   # simple frontend


2) Create .env in project root

Example:

DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx
DEEPSEEK_URL=https://api.deepseek.com/v1/chat/completions
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_TEMPERATURE=0.2



3) Place input files

Put your criteria file here:

Dataset/UniAssistCriteria_clean.txt

Put your QS file here and rename it:

Data/qs_europe_2026.xlsx

The QS file is used only as a “competitiveness” adjustment and only Germany rows are extracted.

4) Run
uvicorn app:app --reload --port 8000


Open:

http://127.0.0.1:8000

How the app decides eligibility and ranking

The system runs in two stages:

Stage A — Rule-based shortlisting (hard filters + borderline flags)

This stage is deterministic Python logic and is intended to prevent AI hallucinations.

A1) Convert GPA to an estimated German grade

The app converts your GPA (Pakistan/other scale) into an estimated German grade using a common linear conversion:

German grade range: 1.0 (best) to 4.0 (worst pass)

Input: gpa, scale_max, min_pass

Output:

german_grade_est (example: 2.65)

A2) Entrance grade check (hard constraint)

If the program has a numeric entrance grade requirement (e.g., 2.5):

If german_grade_est is worse than required grade → program is excluded

If entrance grade is missing/unparseable → add flag:

entrance_grade_unparsed

A3) Language preference filter (hard constraint)

Student selects:

study_language = english | german | both

german_level = none | a2 | b1 | b2 | c1

Program language is inferred from text fields (language of instruction / proficiency / notes):

If student chooses English and program is clearly German-only → exclude

If student chooses German and program is clearly English-only → exclude

If language is unknown → keep but add:

language_unknown

A4) German level requirement check (hard constraint when German is involved)

If the program is German or mixed and student allows German:

Extract required level (A2/B1/B2/C1) from text

If student level < required → exclude

If no level can be extracted → keep but add:

german_level_unknown

A5) Background relevance (soft signal)

The system does not hard-reject on background mismatch (because program requirements text is often incomplete).

If keywords in requirements suggest a strong field mismatch → keep but add:

background_mismatch_possible

Result of Stage A

A shortlist of programs with:

Core fields (Program, University, etc.)

Borderline flags (used later for scoring)

Stage B — Ranking + calibration

After shortlisting, the system ranks results using:

DeepSeek AI scoring (if configured)

Post-calibration adjustments (QS selectivity + FH/TU heuristic)

B1) DeepSeek AI scoring (0–100)

DeepSeek is asked to score each program 0–100 based on weighted criteria:

GPA match (40%)

Background relevance (30%)

Language fit (20%)

Borderline flags (5%)

QS selectivity (5%) (only if QS match exists)

[
  {
    "rank": 1,
    "program_name": "...",
    "university": "...",
    "score_0_100": 78,
    "reason": "..."
  }
]


If DeepSeek is not configured, the app returns a rule-based shortlist only (no AI scoring).

B2) QS selectivity post-penalty (light adjustment)

The QS Europe 2026 Excel is parsed and filtered to Germany. For each German university, a selectivity_index is computed:

High selectivity = tougher admissions (small penalty)

Range: ~0..1

Penalty applied after DeepSeek scoring:

score -= 8 * selectivity_index (max ~8 points)

QS values attached to results:

qs_rank_2026

qs_overall_score

qs_selectivity

qs_matched_name

If no QS match is found, no penalty is applied.

B3) FH/TU/UNI calibration (small heuristic)

Heuristic:

FH / Hochschule often slightly less selective for many applied programs → +5

TU often more selective → -5

Regular UNI → no change

This is intentionally small and should not override hard constraints.

B4) Final re-ranking

After post-calibration, results are re-sorted by score_0_100 (descending).

“Borderline” philosophy (important)

This app is designed for consultant workflows.

Instead of rejecting when data is missing, it:

keeps programs in the shortlist

adds flags to explain uncertainty

reduces score slightly

Typical borderline flags:

language_unknown

german_level_unknown

entrance_grade_unparsed

background_mismatch_possible

This prevents losing good options just because Uni-Assist text is incomplete.

Program Details feature

The UI includes “More details” which calls:

GET /api/program_details?university=...&program=...

It returns:

search queries

clickable Google links

This is a lightweight way to quickly research:

admission requirements

deadlines

language requirements

module handbook

uni-assist details

Notes / limitations

GPA → German grade is an estimate; universities may use different official conversion rules.

QS is not an admissions database; it is used only as a competitiveness prior.

Name matching between your dataset and QS is best-effort; some universities may not match due to naming differences.

Next recommended improvements

Add an alias map for QS name matching (manual overrides)

Store DeepSeek prompt/response for audit (consultant traceability)

Add an endpoint to view saved consultations (/api/history)

Improve program language extraction (more robust patterns for “Deutsch”, “Englisch”, “TestDaF/DSH”)


If DeepSeek is not configured, the app returns a rule-based shortlist only (no AI scoring).

B2) QS selectivity post-penalty (light adjustment)

The QS Europe 2026 Excel is parsed and filtered to Germany. For each German university, a selectivity_index is computed:

High selectivity = tougher admissions (small penalty)

Range: ~0..1

Penalty applied after DeepSeek scoring:

score -= 8 * selectivity_index (max ~8 points)

QS values attached to results:

qs_rank_2026

qs_overall_score

qs_selectivity

qs_matched_name

If no QS match is found, no penalty is applied.

B3) FH/TU/UNI calibration (small heuristic)

Heuristic:

FH / Hochschule often slightly less selective for many applied programs → +5

TU often more selective → -5

Regular UNI → no change

This is intentionally small and should not override hard constraints.

B4) Final re-ranking

After post-calibration, results are re-sorted by score_0_100 (descending).

“Borderline” philosophy (important)

This app is designed for consultant workflows.

Instead of rejecting when data is missing, it:

keeps programs in the shortlist

adds flags to explain uncertainty

reduces score slightly

Typical borderline flags:

language_unknown

german_level_unknown

entrance_grade_unparsed

background_mismatch_possible

This prevents losing good options just because Uni-Assist text is incomplete.

Program Details feature

The UI includes “More details” which calls:

GET /api/program_details?university=...&program=...

It returns:

search queries

clickable Google links

This is a lightweight way to quickly research:

admission requirements

deadlines

language requirements

module handbook

uni-assist details

Notes / limitations

GPA → German grade is an estimate; universities may use different official conversion rules.

QS is not an admissions database; it is used only as a competitiveness prior.

Name matching between your dataset and QS is best-effort; some universities may not match due to naming differences.

