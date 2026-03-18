"""
Semantic Exploration Dashboard — Streamlit app for analyzing open-ended text responses.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from app import load_data
from data_utils import explore_semantic_text, create_survey_session_id
from semantic_exploration import (
    clean_responses,
    compute_umap,
    cluster_responses_bertopic,
    extract_topics_bertopic,
    summarize_clusters_bertopic,
    semantic_analysis_per_question_bertopic,
    run_semantic_pipeline_bertopic,
)

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Semantic Exploration", layout="wide")
st.title("Semantic Exploration Dashboard")

# -----------------------------
# Data loading
# -----------------------------
@st.cache_data
def load_semantic_data():
    """Load data for semantic exploration."""
    df = load_data()
    df_sub = explore_semantic_text(df)
    return df, df_sub

df, df_sub = load_semantic_data()

# -----------------------------
# Missing responses analysis
# -----------------------------
st.header("1️⃣ Missing Responses Analysis")

# Calculate missing response percentages per concept_key
def calc_missing_pct(x):
    """Calculate percentage of missing responses."""
    if len(x) == 0:
        return 0
    missing = x.isna().sum()
    # Also count empty strings (after converting to string)
    empty = (x.astype(str).str.strip() == "").sum()
    return (missing + empty) / len(x) * 100

missing_stats = (
    df_sub.groupby("concept_key")["response"]
    .apply(calc_missing_pct)
    .reset_index(name="missing_pct")
)

total_counts = (
    df_sub.groupby("concept_key")["response"]
    .count()
    .reset_index(name="total_responses")
)

missing_stats = missing_stats.merge(total_counts, on="concept_key")
missing_stats = missing_stats.sort_values("missing_pct", ascending=False)

# Display missing percentages
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
        title="Percentage of Missing Responses by Concept",
        labels={"missing_pct": "Missing %", "concept_key": "Concept Key"},
    )
    fig_missing.update_xaxes(tickangle=45)
    st.plotly_chart(fig_missing, use_container_width=True)

df_open = clean_responses(df_sub)



# -----------------------------
# Semantic Analysis per Concept
# -----------------------------
st.header("3️⃣ Semantic Analysis by Concept")

# Inform about method and defaults
st.caption("Using **BERTopic** for topic modeling (minimum topic size = 10 responses per topic).")

# Select concept for detailed analysis
concept_options = sorted(df_open["concept_key"].dropna().unique().tolist())
selected_concept_analysis = st.selectbox(
    "Select Concept Key for Semantic Analysis",
    options=concept_options,
    index=0,
)

if st.button("Run Semantic Analysis", type="primary"):
    with st.spinner("Computing embeddings and clustering responses..."):
        # Filter data for selected concept
        df_q = df_open[df_open["concept_key"] == selected_concept_analysis].copy()

        if len(df_q) < 3:
            st.warning(
                f"Not enough responses for concept '{selected_concept_analysis}'. Need at least 3 responses."
            )
        else:
            # Always use BERTopic
            try:
                df_q, topic_model, topic_info, embeddings = cluster_responses_bertopic(
                    df_q
                )
                topics_dict = extract_topics_bertopic(topic_model, df_q)
                summary = summarize_clusters_bertopic(
                    df_q,
                    topics_dict,
                    topic_info,
                    topic_model,
                    total_responses=len(df_q),
                )

                # Get UMAP embeddings from BERTopic's UMAP model
                umap_embeddings = None
                try:
                    if hasattr(topic_model, "umap_model"):
                        umap_embeddings = topic_model.umap_model.transform(embeddings)
                    else:
                        umap_embeddings = compute_umap(embeddings)
                except Exception:
                    # Fallback to manual UMAP if BERTopic's UMAP fails
                    umap_embeddings = compute_umap(embeddings)

                # Store in session state
                st.session_state[f"df_q_{selected_concept_analysis}"] = df_q
                st.session_state[f"summary_{selected_concept_analysis}"] = summary
                st.session_state[f"topics_dict_{selected_concept_analysis}"] = topics_dict
                st.session_state[f"topic_model_{selected_concept_analysis}"] = topic_model
                st.session_state[f"topic_info_{selected_concept_analysis}"] = topic_info
                st.session_state[f"embeddings_{selected_concept_analysis}"] = embeddings
                st.session_state[f"umap_{selected_concept_analysis}"] = umap_embeddings
            except Exception as e:
                st.error(f"Error with BERTopic: {str(e)}")

# Display results if available
if f"df_q_{selected_concept_analysis}" in st.session_state:
    df_q = st.session_state[f"df_q_{selected_concept_analysis}"]
    summary = st.session_state[f"summary_{selected_concept_analysis}"]
    topics_dict = st.session_state.get(f"topics_dict_{selected_concept_analysis}", {})
    topic_model = st.session_state.get(f"topic_model_{selected_concept_analysis}", None)
    topic_info = st.session_state.get(f"topic_info_{selected_concept_analysis}", None)

    # Get UMAP if available
    umap_embeddings = st.session_state.get(f"umap_{selected_concept_analysis}", None)

    # Show question text for the selected concept
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
        st.markdown(f"**Question text:** {question_text_example}")

    # Summary table per concept_key
<<<<<<< HEAD
    st.subheader("Topic Summary")
=======
    st.subheader("📊 Topic Summary")
>>>>>>> dashboard-statistical-enhancement
    if len(summary) > 0:
        # Format summary for display
        summary_display = summary.copy()
        if "percentage" in summary_display.columns:
            summary_display["percentage"] = summary_display["percentage"].apply(
                lambda x: f"{x:.1f}%"
            )
        # Show cluster id, size, percentage, keywords and example responses
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
        # Cluster distribution with percentages
        col1, col2 = st.columns(2)

        with col1:
<<<<<<< HEAD
            st.subheader("Topic Distribution (Pie Chart)")
=======
            st.subheader("📈 Topic Distribution (Pie Chart)")
>>>>>>> dashboard-statistical-enhancement
            # Use summary data with names and percentages
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
<<<<<<< HEAD
            st.subheader("Topic Percentages (Bar Chart)")
=======
            st.subheader("📊 Topic Percentages (Bar Chart)")
>>>>>>> dashboard-statistical-enhancement
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
            st.subheader("🗺️ UMAP Visualization")

            # Prepare data for UMAP plot
            # Ensure lengths match
            min_len = min(len(umap_embeddings), len(df_q))
            umap_df = pd.DataFrame(
                {
                    "x": umap_embeddings[:min_len, 0],
                    "y": umap_embeddings[:min_len, 1],
                    "cluster": df_q["cluster"].values[:min_len],
                    "response": df_q["response"].values[:min_len],
                }
            )

            # Map cluster IDs to topic names
            topic_name_map = dict(zip(summary["cluster"], summary["topic_name"]))
            topic_name_map[-1] = "Noise"
            umap_df["topic_name"] = (
                umap_df["cluster"].map(topic_name_map).fillna("Unknown")
            )

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

            # Topic similarity matrix (heatmap)
            if topic_model is not None:
                st.subheader("🔢 Topic Similarity Matrix")
                try:
                    fig_similarity = topic_model.visualize_heatmap()
                    st.plotly_chart(fig_similarity, use_container_width=True)
                except Exception:
                    st.info("Topic similarity matrix not available for this concept.")
<<<<<<< HEAD
=======


# -----------------------------
# Batch Analysis
# -----------------------------
# (Batch / cross‑concept analysis removed to keep UI focused on per‑concept exploration.)
>>>>>>> dashboard-statistical-enhancement
