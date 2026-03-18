# Survey Analysis Dashboard

<<<<<<< HEAD
## Overview

This project is a Streamlit-based interactive dashboard for analyzing
survey data.

It supports:

-   Filtering by survey type and survey phase
-   Dynamic summary tables
-   Visualization of Likert-scale questions
-   Workshop-level grouping of responses

The application merges multiple datasets (surveys, questions, responses,
and concepts) and provides flexible analytical views.

------------------------------------------------------------------------

# Project Structure

## Project Structure

```text
project/
│
├── app.py              # Streamlit application
├── data_utils.py       # Data processing & plotting logic
├── requirements.txt    # Required Python libraries
├── README.md           # Project documentation
=======
## Folder Structure

```
survey_dashboard/
│
├── app.py            ← Streamlit UI entry-point
├── analytics.py      ← All data logic (loading, cleaning, stats, plots)
├── requirements.txt  ← Python dependencies
├── README.md         ← This file
>>>>>>> dashboard-statistical-enhancement
└── data/
    ├── surveys.csv
    ├── questions.csv
    ├── responses.csv
    └── concepts.csv
```

<<<<<<< HEAD



------------------------------------------------------------------------

# Data Processing Pipeline

## Data Loading

The application loads and merges:

-   surveys.csv
-   questions.csv
-   responses.csv
-   concepts.csv

Merging keys:

-   survey_key
-   question_key
-   concept_key

The merged dataframe includes:

-   survey_type
-   survey_phase
-   concept_key
-   response_id
-   timestamp
-   scale_type
-   response
-   response_encoded

------------------------------------------------------------------------

# Key Design Decision: Definition of workshop_id

The most important engineered variable in this project is workshop_id.

It is created inside:

create_workshop_id(df)

## Why workshop_id is needed

Survey responses are recorded at the question-response level. However,
visualization requires grouping responses by workshop session.

A workshop session is defined as:

-   Same school_id
-   Same calendar date
-   Same time session (AM or PM)
-   Same survey_phase (PRE or POST)

------------------------------------------------------------------------

## Step-by-step Construction

1)  Convert timestamp

df\['timestamp'\] = pd.to_datetime(df\['timestamp'\])

2)  Extract date

df\['date'\] = df\['timestamp'\].dt.date

3)  Define session (AM / PM)

df\['session'\] = np.where(df\['timestamp'\].dt.hour \< 12, 'AM', 'PM')

This separates morning and afternoon sessions occurring on the same day.

4)  Create base workshop index

df\['base_id'\] = df.groupby(\['school_id', 'date'\]).ngroup()

This assigns a unique integer for each (school_id, date) combination.

5)  Final workshop_id construction

df\['workshop_id'\] = ( df\['survey_phase'\].str.upper() + "*" +
df\['base_id'\].astype(str) + "*" + df\['session'\].str.upper() )

Final format examples:

PRE_3\_AM POST_3\_AM PRE_7\_PM POST_7\_PM

Structure:

{SURVEY_PHASE}*{BASE_ID}*{SESSION}

This ensures:

-   PRE and POST are separated
-   Multiple sessions per day are separated
-   Each workshop instance is uniquely identified

Without this variable, session-level visualization would be inaccurate.

------------------------------------------------------------------------

# Summary Table Logic

The dashboard supports dynamic summary tables using:

generate_summary(...)

Available grouping variables:

-   survey_type
-   survey_phase

Available metrics:

-   Number of questions (concept_key, nunique)
-   Number of observations (response_id, nunique)

If no grouping is selected, overall totals are returned.

------------------------------------------------------------------------

# Visualization Logic

Visualization mode:

-   Requires a selected survey phase
-   Supports Likert text and Likert numeric questions

Each visualization includes:

-   Stacked bar charts by workshop
-   Pie chart of total distribution across workshops

Plot routing is handled by:

run_likert_plot(...)

------------------------------------------------------------------------

# <font color="red"># How to Run the App</font>

1)  Create virtual environment

python -m venv .venv

Activate:

Windows: .venv`\Scripts`{=tex}`\activate`{=tex}

Mac/Linux: source .venv/bin/activate

2)  Install dependencies

pip install -r requirements.txt

3)  Run Streamlit

python -m streamlit run app.py

------------------------------------------------------------------------

# Required Libraries (example requirements.txt)

streamlit pandas numpy matplotlib

------------------------------------------------------------------------

# Architecture Design

-   app.py handles UI and interaction logic.
-   data_utils.py contains data transformation and plotting functions.

This separation keeps the application modular, maintainable, and
scalable.


Use of AI Tools

AI tools were used to support and accelerate development of this project.

## ChatGPT

ChatGPT was used to:

- Accelerate coding workflows

- Refactor and optimize functions

- Improve code structure and modular design

- Debug logic and resolve errors

- Design dynamic summary and plotting architecture

- Generate and refine documentation

- Improve clarity of explanations and comments

The AI was used as a development assistant, not as a replacement for understanding. All architectural decisions (such as the definition of workshop_id, session logic, and dynamic summary structure) were reviewed and validated before implementation.

## Cursor

Cursor was used as an AI-assisted development environment to:

- Build and organize the Streamlit application

- Navigate and refactor code across files

- Improve productivity while maintaining control over logic

AI tools were used to enhance productivity and code quality while ensuring full understanding of the analytical and structural decisions behind the implementation.
=======
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
>>>>>>> dashboard-statistical-enhancement
