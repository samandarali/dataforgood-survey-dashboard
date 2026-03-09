"""
Survey Analysis Dashboard — Streamlit app.
Uses data_utils for filtering, dynamic summary, and Likert plots.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.io

from data_utils import (
    create_survey_session_id,
    apply_filters,
    generate_summary,
    SUMMARY_GROUPBY_OPTIONS,
    SUMMARY_METRIC_OPTIONS,
    SUMMARY_METRIC_LABELS,
    run_likert_plot,
)


# -----------------------------
# Data loading
# -----------------------------
@st.cache_data
def load_data() -> pd.DataFrame:
    """Load and merge survey data. Returns a dataframe with survey_type, survey_phase, concept_key, response_id, etc."""
    base_dir = Path.cwd()
    data_path = base_dir / "data"

    surveys = pd.read_csv(data_path / "surveys.csv")
    questions = pd.read_csv(data_path / "questions.csv")
    responses = pd.read_csv(data_path / "responses.csv")
    concepts = pd.read_csv(data_path / "concepts.csv")

    responses["timestamp"] = pd.to_datetime(responses["timestamp"], errors="coerce")

    df = (
        questions.merge(surveys, on="survey_key")
        .merge(responses, on="question_key")
        .merge(concepts, on="concept_key")
    )
    return df


# -----------------------------
# App
# -----------------------------
st.set_page_config(page_title="Survey Dashboard", layout="wide")
st.title("Survey Dashboard")

df = load_data()

# Ensure workshop_id exists for plotting (needed in Visualization mode)
if "workshop_id" not in df.columns:
    df = create_survey_session_id(df)

# -----------------------------
# 1️⃣ Sample selection (top-level filter)
# -----------------------------
st.sidebar.header("1️⃣ Sample selection")

survey_types = ["All"] + sorted(df["survey_type"].dropna().unique().tolist())
survey_phases = ["All"] + sorted(df["survey_phase"].dropna().unique().tolist())

selected_survey_type = st.sidebar.selectbox(
    "Survey type",
    options=survey_types,
    index=0,
)
selected_survey_phase = st.sidebar.selectbox(
    "Survey phase",
    options=survey_phases,
    index=0,
)

try:
    df_filtered = apply_filters(
        df,
        survey_type=selected_survey_type,
        survey_phase=selected_survey_phase,
    )
except KeyError as e:
    st.error(f"Data error: {e}")
    st.stop()

st.caption(f"Filtered rows: {len(df_filtered):,}")

# -----------------------------
# Mode: Summary table vs Visualization
# -----------------------------
st.sidebar.header("View mode")
mode = st.sidebar.radio(
    "Mode",
    options=["Summary table", "Visualization"],
    index=0,
)

# -----------------------------
# 2️⃣ Summary table (dynamic)
# -----------------------------
if mode == "Summary table":
    st.sidebar.subheader("Summary options")

    groupby_options = st.sidebar.multiselect(
        "Group by",
        options=SUMMARY_GROUPBY_OPTIONS,
        default=[],
        help="Select one or more variables to group by. Leave empty for overall totals.",
    )

    selected_metrics = st.sidebar.multiselect(
        "Metrics",
        options=list(SUMMARY_METRIC_OPTIONS.keys()),
        default=list(SUMMARY_METRIC_OPTIONS.keys()),
        format_func=lambda k: SUMMARY_METRIC_LABELS[k],
        help="Select one or more metrics to include in the summary.",
    )

    if st.sidebar.button("Show Summary"):
        if not selected_metrics:
            st.warning("Select at least one metric.")
        else:
            try:
                summary_df = generate_summary(
                    df_filtered,
                    groupby_columns=groupby_options,
                    metrics=selected_metrics,
                )
                st.subheader("Summary")
                st.dataframe(summary_df, use_container_width=True)
                st.download_button(
                    label="Download CSV",
                    data=summary_df.to_csv(index=False).encode("utf-8"),
                    file_name="summary.csv",
                    mime="text/csv",
                )
            except (KeyError, ValueError) as e:
                st.error(str(e))

# -----------------------------
# 3️⃣ Visualization (phase required, then question type)
# -----------------------------
else:
    st.sidebar.subheader("Visualization options")

    if selected_survey_phase == "All":
        st.warning(
            "Please select a specific **Survey phase** (e.g. PRE or POST) to view visualizations. "
            "Phase is required for plots."
        )
    else:
        likert_type = st.sidebar.selectbox(
            "Question type",
            options=["Likert text", "Likert numeric"],
            index=0,
        )

        plot_title = f"{selected_survey_type} | {selected_survey_phase} | {likert_type}"

        st.subheader("Plots")
        try:
            likert_kind = "text" if likert_type == "Likert text" else "numeric"
            figs = run_likert_plot(
                df_filtered,
                likert_kind=likert_kind,
                title=plot_title,
            )
            if figs is not None:
                for i, fig in enumerate(figs):
                    st.plotly_chart(fig, use_container_width=True)
                    # Add more spacing between plots (except after the last one)
                    if i < len(figs) - 1:
                        st.markdown("<br><br>", unsafe_allow_html=True)
                        st.markdown("---")  # Add a horizontal divider
                        st.markdown("<br>", unsafe_allow_html=True)
        except (KeyError, ValueError) as e:
            st.error(str(e))
