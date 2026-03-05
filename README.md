# Survey Analysis Dashboard

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
└── data/
    ├── surveys.csv
    ├── questions.csv
    ├── responses.csv
    └── concepts.csv
```




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