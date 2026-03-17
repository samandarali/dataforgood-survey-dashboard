# Survey Analysis Dashboard

## Folder Structure

```
survey_dashboard/
│
├── app.py            ← Streamlit UI entry-point
├── analytics.py      ← All data logic (loading, cleaning, stats, plots)
├── requirements.txt  ← Python dependencies
├── README.md         ← This file
└── data/
    ├── surveys.csv
    ├── questions.csv
    ├── responses.csv
    └── concepts.csv
```

## How to Run

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Mac / Linux
.venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place your CSVs in the data/ folder, then launch
python -m streamlit run app.py
```

---

## Dashboard Layout

### Sidebar
| Control | Purpose |
|---|---|
| **Survey Type** | Dropdown built from `main_dataset['survey_type'].unique()` — e.g. LEA, PAR, MEN … |
| **Survey Version** | Auto-filters to versions available for the chosen type (e.g. v1) |
| **School** (optional) | Further filter by `school_id` |

### Fact Cards  *(always visible)*
| Card | Calculation |
|---|---|
| PRE responses | `response_id.nunique()` where `survey_phase == 'PRE'` |
| POST responses | `response_id.nunique()` where `survey_phase == 'POST'` |
| Common PRE/POST Qs | Likert questions appearing in **both** phases |
| POST-only Qs | Likert questions appearing in POST **only** |
| Schools | `school_id.nunique()` |
| PRE sessions | unique `survey_session_id` where phase == PRE |
| POST sessions | unique `survey_session_id` where phase == POST |

---

### Tab 1 — Statistical Analysis (All Schools)

#### ① PRE vs POST — Mann-Whitney U
- Applied to **common Likert questions** (present in both phases).
- One-sided test: H₁ = POST > PRE.
- Benjamini-Hochberg FDR correction across all questions.
- Visualisation: diverging bar (median shift) + dot plot (PRE vs POST medians).

#### ② POST-only — Wilcoxon vs Neutral
- Applied to **POST-only Likert questions**.
- One-sample signed-rank test: H₁ = median > 3 (neutral midpoint).
- BH correction applied.
- Visualisation: median dot + IQR bar, teal = significant.

---

### Tab 2 — Cross-Session Views

Select a question type to view per-session breakdowns:

| Mode | Description |
|---|---|
| **Likert Text** | Stacked bar + pie per `survey_session_id` |
| **Likert Numeric** | Same, using `response_encoded` |
| **Categorical** | Same for categorical questions |
| **Common PRE/POST (bar + mean)** | PRE sessions followed by POST sessions, mean line overlay, PRE/POST divider |
| **POST-only (bar + mean)** | POST sessions only, mean line overlay |

---

## Key Engineered Variable: `survey_session_id`

Format: `{TYPE}{version}_{PHASE}_{YYMMDD}_{AM|PM}`  
Example: `MENv1_PRE_241015_AM`

Built in `analytics.create_survey_session_id()` — uniquely identifies each
data-collection event (school × date × AM/PM × phase).

---

## analytics.py — Function Reference

| Function | Purpose |
|---|---|
| `load_data()` | Merge surveys / questions / responses / concepts CSVs |
| `create_survey_session_id()` | Engineer `survey_session_id` column |
| `apply_filters()` | Filter dataframe by any combination of dimensions |
| `get_common_pre_post_questions()` | Likert questions in both PRE and POST |
| `get_post_only_questions()` | Likert questions in POST only |
| `compute_fact_cards()` | Dict of all metric values for the fact-card row |
| `run_mann_whitney()` | Mann-Whitney U, BH correction, returns results DataFrame |
| `run_wilcoxon_vs_neutral()` | Wilcoxon signed-rank vs midpoint, BH correction |
| `plot_mann_whitney_shift()` | Diverging bar + PRE/POST dot-plot figure |
| `plot_wilcoxon_neutral()` | Median + IQR dot plot for POST-only questions |
| `plot_pre_post_bar_with_mean()` | Stacked bars per session (common Qs) + mean line |
| `plot_post_bar_with_mean()` | Stacked bars per session (POST-only Qs) + mean line |
| `plot_likert_by_session()` | Stacked bar + pie per session (text or numeric) |
| `plot_categorical_by_session()` | Stacked bar + pie per session (categorical) |
| `generate_summary()` | Dynamic summary table with optional grouping |
