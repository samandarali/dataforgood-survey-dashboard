"""
Survey Analysis Dashboard — Streamlit app.
Uses data_utils for filtering, dynamic summary, and Likert plots.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

from data_utils import (
    create_survey_session_id,
    apply_filters,
    generate_summary,
    SUMMARY_GROUPBY_OPTIONS,
    SUMMARY_METRIC_OPTIONS,
    SUMMARY_METRIC_LABELS,
    run_likert_plot,
    run_categorical_plot,
    explore_semantic_text,
)
from semantic_exploration import (
    clean_responses,
    compute_umap,
    cluster_responses_bertopic,
    extract_topics_bertopic,
    summarize_clusters_bertopic,
    summarize_small_dataset,
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

# Initialize page in session state
if "page" not in st.session_state:
    st.session_state.page = "Main Dashboard"

# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    options=["Main Dashboard", "Semantic Exploration"],
    index=0 if st.session_state.page == "Main Dashboard" else 1,
    key="page_selector"
)
st.session_state.page = page

df = load_data()

# Ensure workshop_id exists for plotting (needed in Visualization mode)
if "workshop_id" not in df.columns:
    df = create_survey_session_id(df)

# -----------------------------
# Semantic Exploration Page
# -----------------------------
if page == "Semantic Exploration":
    st.title("Semantic Exploration Dashboard")
    
    # Load semantic data
    @st.cache_data
    def load_semantic_data(_version: str = "v2"):
        """Load data for semantic exploration."""
        df_sub = explore_semantic_text(df)
        return df_sub
    
    # Bump _version to refresh cache when we change which columns are included
    df_sub = load_semantic_data("v2")

    # ---------------------------------
    # Semantic-level filters (survey_type only)
    # ---------------------------------
    st.sidebar.subheader("Semantic filters")
    # Prefer survey types present in the open-ended subset; fall back to full df.
    available_types = []
    if "survey_type" in df_sub.columns:
        available_types = sorted(df_sub["survey_type"].dropna().unique().tolist())
    if not available_types and "survey_type" in df.columns:
        available_types = sorted(df["survey_type"].dropna().unique().tolist())
    sem_survey_types = ["All"] + available_types if available_types else ["All"]

    selected_sem_survey_type = st.sidebar.selectbox(
        "Survey type (semantic)",
        options=sem_survey_types,
        index=0,
    )
    if selected_sem_survey_type == "All":
        st.sidebar.caption("Pick a specific survey type to run BERTopic per type.")

    df_sub_filtered = df_sub.copy()
    # Only filter by survey_type; survey_phase is always POST in current data.
    if "survey_type" in df_sub_filtered.columns and selected_sem_survey_type != "All":
        df_sub_filtered = df_sub_filtered[df_sub_filtered["survey_type"] == selected_sem_survey_type]
    
    # Missing responses analysis
    st.header(" Missing Responses Analysis")
    
    def calc_missing_pct(x):
        """Calculate percentage of missing responses."""
        if len(x) == 0:
            return 0
        missing = x.isna().sum()
        empty = (x.astype(str).str.strip() == "").sum()
        return (missing + empty) / len(x) * 100
    
    # Group by concept_key + survey_type to see patterns
    group_cols = ["concept_key"]
    if "survey_type" in df_sub_filtered.columns:
        group_cols.append("survey_type")

    missing_stats = (
        df_sub_filtered.groupby(group_cols)["response"]
        .apply(calc_missing_pct)
        .reset_index(name="missing_pct")
    )
    
    total_counts = (
        df_sub_filtered.groupby(group_cols)["response"]
        .count()
        .reset_index(name="total_responses")
    )
    
    missing_stats = missing_stats.merge(total_counts, on=group_cols)
    missing_stats = missing_stats.sort_values("missing_pct", ascending=False)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Missing Response Percentages")
        st.dataframe(
            missing_stats.style.format({"missing_pct": "{:.1f}%"}),
            use_container_width=True,
        )
    
    with col2:
        st.subheader("Missing Responses Chart")
        fig_missing = px.bar(
            missing_stats,
            x="concept_key",
            y="missing_pct",
            color="survey_type" if "survey_type" in missing_stats.columns else None,
            barmode="group",
            title="Percentage of Missing Responses by Concept / Survey type",
            labels={
                "missing_pct": "Missing %",
                "concept_key": "Concept Key",
                "survey_type": "Survey type",
            },
        )
        fig_missing.update_xaxes(tickangle=45)
        st.plotly_chart(fig_missing, use_container_width=True)
    
    # Apply same filters for semantic analysis
    df_open = clean_responses(df_sub_filtered)
    
    # Semantic Analysis per Concept
    st.header("Semantic Analysis by Concept")
    st.caption("Using **BERTopic** for topic modeling (minimum topic size = 10 responses per topic).")

    # Require a specific survey_type for semantic analysis so clusters are per type.
    if selected_sem_survey_type == "All":
        st.info("Select a specific **Survey type (semantic)** in the left sidebar to run BERTopic analysis.")
    
    concept_options = sorted(df_open["concept_key"].dropna().unique().tolist())
    selected_concept_analysis = st.selectbox(
        "Select Concept Key for Semantic Analysis",
        options=concept_options,
        index=0,
    )
    
    run_disabled = selected_sem_survey_type == "All"
    if st.button("Run Semantic Analysis", type="primary", disabled=run_disabled):
        with st.spinner("Computing embeddings and clustering responses..."):
            df_q = df_open[df_open["concept_key"] == selected_concept_analysis].copy()
            
            if len(df_q) < 3:
                st.warning(
                    f"Not enough responses for concept '{selected_concept_analysis}'. Need at least 3 responses."
                )
            elif len(df_q) == 5:
                # Special-case tiny datasets: treat each response as its own 'topic'
                summary = summarize_small_dataset(df_q)
                st.session_state[f"df_q_{selected_concept_analysis}"] = df_q
                st.session_state[f"summary_{selected_concept_analysis}"] = summary
                st.session_state[f"topics_dict_{selected_concept_analysis}"] = {}
                st.session_state[f"topic_model_{selected_concept_analysis}"] = None
                st.session_state[f"topic_info_{selected_concept_analysis}"] = None
                st.session_state[f"umap_{selected_concept_analysis}"] = None
            else:
                try:
                    df_q, topic_model, topic_info, embeddings = cluster_responses_bertopic(df_q)
                    topics_dict = extract_topics_bertopic(topic_model, df_q)
                    summary = summarize_clusters_bertopic(
                        df_q,
                        topics_dict,
                        topic_info,
                        topic_model,
                        total_responses=len(df_q),
                    )
                    
                    umap_embeddings = None
                    try:
                        if hasattr(topic_model, "umap_model"):
                            umap_embeddings = topic_model.umap_model.transform(embeddings)
                        else:
                            umap_embeddings = compute_umap(embeddings)
                    except Exception:
                        umap_embeddings = compute_umap(embeddings)
                    
                    st.session_state[f"df_q_{selected_concept_analysis}"] = df_q
                    st.session_state[f"summary_{selected_concept_analysis}"] = summary
                    st.session_state[f"topics_dict_{selected_concept_analysis}"] = topics_dict
                    st.session_state[f"topic_model_{selected_concept_analysis}"] = topic_model
                    st.session_state[f"topic_info_{selected_concept_analysis}"] = topic_info
                    st.session_state[f"umap_{selected_concept_analysis}"] = umap_embeddings
                except Exception as e:
                    st.error(f"Error with BERTopic: {str(e)}")
    
    # Display results if available
    if f"df_q_{selected_concept_analysis}" in st.session_state:
        df_q = st.session_state[f"df_q_{selected_concept_analysis}"]
        summary = st.session_state[f"summary_{selected_concept_analysis}"]
        topics_dict = st.session_state.get(f"topics_dict_{selected_concept_analysis}", {})
        topic_model = st.session_state.get(f"topic_model_{selected_concept_analysis}", None)
        umap_embeddings = st.session_state.get(f"umap_{selected_concept_analysis}", None)
        
        # Show question text
        question_text_example = None
        try:
            question_text_example = (
                df_sub.loc[df_sub["concept_key"] == selected_concept_analysis, "question_text"]
                .dropna()
                .iloc[0]
            )
        except Exception:
            question_text_example = None
        
        if question_text_example:
            st.markdown(f"## **Question text:** {question_text_example}")
        
        # Summary table
        st.subheader("Topic Summary")
        if len(summary) > 0:
            summary_display = summary.copy()
            if "percentage" in summary_display.columns:
                summary_display["percentage"] = summary_display["percentage"].apply(
                    lambda x: f"{x:.1f}%"
                )
            st.dataframe(
                summary_display[
                    ["cluster", "size", "percentage", "keywords", "example_responses"]
                ],
                use_container_width=True,
                height=320,
            )
        else:
            st.info("No topics found for this concept.")
        
        # Visualizations
        if len(summary) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Topic Distribution (Pie Chart)")
                fig_pie = px.pie(
                    summary,
                    values="percentage",
                    names="topic_name",
                    title=f"Topic distribution for {selected_concept_analysis}",
                    hover_data=["size"],
                )
                fig_pie.update_traces(textposition="inside", textinfo="percent+label")
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                st.subheader("Topic Percentages (Bar Chart)")
                fig_bar = px.bar(
                    summary.sort_values("percentage", ascending=False),
                    x="topic_name",
                    y="percentage",
                    title="Percentage distribution by topic",
                    labels={"percentage": "Percentage (%)", "topic_name": "Topic"},
                    text="percentage",
                )
                fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig_bar.update_xaxes(tickangle=45)
                st.plotly_chart(fig_bar, use_container_width=True)
            
            # UMAP visualization
            if umap_embeddings is not None:
                st.subheader("UMAP Visualization")
                
                min_len = min(len(umap_embeddings), len(df_q))
                umap_df = pd.DataFrame({
                    "x": umap_embeddings[:min_len, 0],
                    "y": umap_embeddings[:min_len, 1],
                    "cluster": df_q["cluster"].values[:min_len],
                    "response": df_q["response"].values[:min_len],
                })
                
                topic_name_map = dict(zip(summary["cluster"], summary["topic_name"]))
                topic_name_map[-1] = "Noise"
                umap_df["topic_name"] = umap_df["cluster"].map(topic_name_map).fillna("Unknown")
                
                fig_umap = px.scatter(
                    umap_df,
                    x="x",
                    y="y",
                    color="topic_name",
                    hover_data=["response"],
                    title=f"UMAP visualization of responses for {selected_concept_analysis}",
                    labels={"x": "UMAP 1", "y": "UMAP 2"},
                )
                st.plotly_chart(fig_umap, use_container_width=True)
                
                # Topic similarity matrix
                if topic_model is not None:
                    st.subheader("Topic Similarity Matrix")
                    try:
                        fig_similarity = topic_model.visualize_heatmap()
                        st.plotly_chart(fig_similarity, use_container_width=True)
                    except Exception:
                        st.info("Topic similarity matrix not available for this concept.")
    
    st.stop()

# -----------------------------
# Main Dashboard Page
# -----------------------------
st.title("Survey Dashboard")

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
        question_type = st.sidebar.selectbox(
            "Question type",
            options=["Likert text", "Likert numeric", "Categorical"],
            index=0,
        )

        plot_title = f"{selected_survey_type} | {selected_survey_phase} | {question_type}"

        st.subheader("Plots")
        try:
            if question_type == "Categorical":
                figs = run_categorical_plot(
                    df_filtered,
                    title=plot_title,
                )
            else:
                likert_kind = "text" if question_type == "Likert text" else "numeric"
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
