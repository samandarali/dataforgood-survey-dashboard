"""
app.py — Survey Analysis Dashboard
Streamlit entry-point.  All data logic lives in analytics.py.

Layout
------
Sidebar  : survey_type selector (dropdown from data)
Main     : Fact cards  → Two tabs
              Tab 1 – Statistical Analysis  (Mann-Whitney + Wilcoxon)
              Tab 2 – Cross-Session Views   (stacked bar + pie per session_id)
"""

import streamlit as st
import pandas as pd
from pathlib import Path

import analytics as an

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Survey Analysis Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — clean card-style UI
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── fact card ── */
    .fact-card {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 18px 20px 12px;
        text-align: center;
        margin-bottom: 8px;
    }
    .fact-card .value {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1D9E75;
        line-height: 1.1;
    }
    .fact-card .label {
        font-size: 0.82rem;
        color: #6c757d;
        margin-top: 4px;
    }
    /* ── section title ── */
    .section-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #343a40;
        margin: 18px 0 6px;
        padding-bottom: 4px;
        border-bottom: 2px solid #1D9E75;
    }
    /* ── divider ── */
    hr.thin { border: none; border-top: 1px solid #dee2e6; margin: 20px 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading survey data…")
def load() -> pd.DataFrame:
    df = an.load_data()
    df = an.create_survey_session_id(df)
    return df


df_all = load()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — survey_type selector
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title("📋 Survey Dashboard")
st.sidebar.markdown("---")

survey_types = sorted(df_all["survey_type"].dropna().unique().tolist())
selected_type = st.sidebar.selectbox(
    "Select Survey Type",
    options=survey_types,
    index=0,
    help="Each survey type has PRE and POST phases collected across schools.",
)

# Derive available versions for selected type
versions = sorted(
    df_all.loc[df_all["survey_type"] == selected_type, "survey_version"]
    .dropna().unique().tolist()
)
selected_version = st.sidebar.selectbox(
    "Survey Version",
    options=versions,
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Optional filters**")

schools = ["All"] + sorted(df_all["school_id"].dropna().unique().astype(str).tolist())
selected_school = st.sidebar.selectbox("School", schools, index=0)

# ─────────────────────────────────────────────────────────────────────────────
# Filter data to selected survey type + version
# ─────────────────────────────────────────────────────────────────────────────
df = df_all[
    (df_all["survey_type"]    == selected_type) &
    (df_all["survey_version"] == selected_version)
].copy()

if selected_school != "All":
    df = df[df["school_id"].astype(str) == selected_school]

survey_label = f"{selected_type}_{selected_version}"

# ─────────────────────────────────────────────────────────────────────────────
# Page title
# ─────────────────────────────────────────────────────────────────────────────
st.title(f"📊 Survey Analysis — {survey_label}")
if selected_school != "All":
    st.caption(f"Filtered to school: **{selected_school}**")

# ─────────────────────────────────────────────────────────────────────────────
# Fact cards
# ─────────────────────────────────────────────────────────────────────────────
facts = an.compute_fact_cards(df)
common_qs   = an.get_common_pre_post_questions(df)
post_only_qs = an.get_post_only_questions(df)

st.markdown('<p class="section-title">📌 Key Facts</p>', unsafe_allow_html=True)

c1, c2, c3, c4, c5, c6, c7 = st.columns(7)

def _card(col, value, label):
    col.markdown(
        f'<div class="fact-card"><div class="value">{value}</div>'
        f'<div class="label">{label}</div></div>',
        unsafe_allow_html=True,
    )

_card(c1, f"{facts['pre_observations']:,}",  "PRE responses")
_card(c2, f"{facts['post_observations']:,}", "POST responses")
_card(c3, facts["common_questions"],         "Common PRE/POST Qs")
_card(c4, facts["post_only_questions"],      "POST-only Qs")
_card(c5, facts["n_schools"],                "Schools")
_card(c6, facts["pre_sessions"],             "PRE sessions")
_card(c7, facts["post_sessions"],            "POST sessions")

st.markdown('<hr class="thin">', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs([
    "📈 Statistical Analysis (All Schools)",
    "🔄 Cross-Session Views",
])

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 – Statistical Analysis
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown(
        "Statistical tests comparing PRE ↔ POST responses across all schools "
        f"for **{survey_label}**."
    )

    if not an.STATS_AVAILABLE:
        st.error(
            "📦 `scipy` and/or `statsmodels` are not installed. "
            "Install them with `pip install scipy statsmodels` to enable statistical analysis."
        )
    elif len(common_qs) == 0 and len(post_only_qs) == 0:
        st.warning("No Likert questions found for this survey type/version.")
    else:
        # ── PRE vs POST: Mann-Whitney ────────────────────────────────────────
        st.markdown('<p class="section-title">① PRE vs POST — Mann-Whitney U (common questions)</p>',
                    unsafe_allow_html=True)

        if len(common_qs) == 0:
            st.info("No common PRE/POST Likert questions found.")
        else:
            with st.spinner("Running Mann-Whitney U tests…"):
                likert_df = df[
                    df["scale_type"].isin(["Likert Scale Numeric", "Likert Scale Text"])
                ].copy()
                likert_df["response_encoded"] = pd.to_numeric(
                    likert_df["response_encoded"], errors="coerce"
                )
                mw_results = an.run_mann_whitney(likert_df, common_qs)

            if not mw_results.empty:
                # results table (collapsible)
                with st.expander("📄 Show results table", expanded=False):
                    display_df = mw_results[[
                        "question", "n_pre", "n_post",
                        "pre_median", "post_median", "median_shift",
                        "p_value", "p_adj_BH", "significant",
                    ]].copy()
                    display_df["significant"] = display_df["significant"].map(
                        {True: "✅ Yes", False: "No", None: "—"}
                    ).fillna("—")

                    def _color_row(row):
                        sig   = row["significant"] == "✅ Yes"
                        shift = row["median_shift"]
                        # zero shift (regardless of significance) → red
                        if shift == 0:
                            return ["background-color: #fde8e8; color: #7f1d1d"] * len(row)
                        # not significant → orange
                        if not sig:
                            return ["background-color: #fff3e0; color: #7c4700"] * len(row)
                        # positive + significant → green
                        if shift > 0:
                            return ["background-color: #d4f5e9; color: #0f5132"] * len(row)
                        # negative + significant → purple
                        return ["background-color: #ede9fe; color: #4c1d95"] * len(row)

                    st.dataframe(
                        display_df.style
                        .format({
                            "pre_median":   "{:.2f}",
                            "post_median":  "{:.2f}",
                            "median_shift": "{:.2f}",
                            "p_value":      "{:.4f}",
                            "p_adj_BH":     "{:.4f}",
                        })
                        .apply(_color_row, axis=1),
                        use_container_width=True,
                    )
                    st.download_button(
                        "⬇ Download CSV",
                        data=mw_results.to_csv(index=False).encode(),
                        file_name=f"{survey_label}_mann_whitney.csv",
                        mime="text/csv",
                    )

                # Plot — left column chart, right column guide
                st.markdown("**Visualisation — Median shift & PRE vs POST medians**")
                col_plot, col_guide = st.columns([3, 1])
                with col_plot:
                    fig_shift = an.plot_mann_whitney_shift(mw_results, survey_label)
                    st.plotly_chart(fig_shift, use_container_width=True)
                with col_guide:
                    st.markdown("##### How to read these charts")
                    st.markdown(
                        "- **Left (shift):** dark colour = statistically significant (★), "
                        "light = not significant.\n"
                        "- Teal = positive shift (improvement); red-orange = negative shift.\n"
                        "- **Right (medians):** circle = PRE, diamond = POST.\n"
                        "- Solid line = significant shift; dashed = not significant.\n"
                        "- Significance criterion: BH-adjusted p < 0.05."
                    )

        st.markdown('<hr class="thin">', unsafe_allow_html=True)

        # ── POST-only: Wilcoxon vs neutral ───────────────────────────────────
        st.markdown(
            '<p class="section-title">② POST-only questions — Wilcoxon vs neutral (midpoint = 3)</p>',
            unsafe_allow_html=True,
        )

        if len(post_only_qs) == 0:
            st.info("No POST-only Likert questions found.")
        else:
            df_post = df[
                (df["survey_phase"] == "POST") &
                (df["scale_type"].isin(["Likert Scale Numeric", "Likert Scale Text"]))
            ].copy()

            with st.spinner("Running Wilcoxon signed-rank tests…"):
                wlcx_results = an.run_wilcoxon_vs_neutral(df_post, post_only_qs)

            if not wlcx_results.empty:
                with st.expander("📄 Show results table", expanded=False):
                    show_cols = [c for c in [
                        "question", "scale_type", "n", "median", "pct_above",
                        "W_stat", "p_value", "p_adj_BH", "significant", "note",
                    ] if c in wlcx_results.columns]

                    wlcx_display = wlcx_results[show_cols].copy()
                    wlcx_display["significant"] = wlcx_display["significant"].map(
                        {True: "✅ Yes", False: "No", None: "—"}
                    ).fillna("—")

                    def _color_row_wlcx(row):
                        sig    = row["significant"] == "✅ Yes"
                        # shift relative to neutral midpoint (3)
                        shift  = row["median"] - 3 if "median" in row.index else 0
                        if shift == 0:
                            return ["background-color: #fde8e8; color: #7f1d1d"] * len(row)
                        if not sig:
                            return ["background-color: #fff3e0; color: #7c4700"] * len(row)
                        if shift > 0:
                            return ["background-color: #d4f5e9; color: #0f5132"] * len(row)
                        return ["background-color: #ede9fe; color: #4c1d95"] * len(row)

                    fmt = {c: "{:.2f}" for c in ["median", "pct_above", "W_stat", "p_value", "p_adj_BH"]
                           if c in wlcx_display.columns}
                    st.dataframe(
                        wlcx_display.style.format(fmt).apply(_color_row_wlcx, axis=1),
                        use_container_width=True,
                    )
                    st.download_button(
                        "⬇ Download CSV",
                        data=wlcx_results.to_csv(index=False).encode(),
                        file_name=f"{survey_label}_wilcoxon.csv",
                        mime="text/csv",
                    )

                col_left2, col_right2 = st.columns(2)
                with col_left2:
                    fig_wlcx = an.plot_wilcoxon_neutral(wlcx_results, df_post)
                    st.plotly_chart(fig_wlcx, use_container_width=True)
                with col_right2:
                    st.markdown(
                        "##### How to read this chart\n"
                        "- Each row is a **POST-only** question.\n"
                        "- The dot shows the **median** response; the horizontal bar "
                        "spans the **interquartile range (Q1–Q3)**.\n"
                        "- **Teal** = significantly above neutral (BH-adjusted p < 0.05); "
                        "**grey** = not significant.\n"
                        "- ★ and percentage label shows how many respondents scored "
                        "above the neutral midpoint (3).\n"
                        "- Dashed vertical line = neutral midpoint (3)."
                    )


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 – Cross-Session Views
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown(
        f"Response distributions broken down by **survey_session_id** for **{survey_label}**. "
        "Each session ID encodes survey type, version, phase, date and AM/PM slot."
    )

    # Optional phase filter for cross-session tab
    phase_options = ["All"] + sorted(df["survey_phase"].dropna().unique().tolist())
    selected_phase = st.selectbox(
        "Filter by phase (optional)",
        options=phase_options,
        index=0,
        key="tab2_phase",
    )

    question_type = st.radio(
        "Question type to display",
        options=["Likert Text", "Likert Numeric", "Categorical",
                 "Common PRE/POST (bar + mean)", "POST-only (bar + mean)"],
        horizontal=True,
        key="tab2_qtype",
    )

    df_tab2 = df.copy()
    if selected_phase != "All":
        df_tab2 = df_tab2[df_tab2["survey_phase"] == selected_phase]

    tab2_title = f"{survey_label} | {selected_phase} | {question_type}"

    if df_tab2.empty:
        st.warning("No data for selected filters.")
    else:
        with st.spinner("Building plots…"):
            if question_type == "Likert Text":
                figs = an.plot_likert_by_session(df_tab2, likert_kind="text",  title=tab2_title)
            elif question_type == "Likert Numeric":
                figs = an.plot_likert_by_session(df_tab2, likert_kind="numeric", title=tab2_title)
            elif question_type == "Categorical":
                figs = an.plot_categorical_by_session(df_tab2, title=tab2_title)
            elif question_type == "Common PRE/POST (bar + mean)":
                figs = an.plot_pre_post_bar_with_mean(df_tab2, common_qs)
            else:  # POST-only
                df_post_tab2 = df_tab2[df_tab2["survey_phase"] == "POST"]
                figs = an.plot_post_bar_with_mean(df_post_tab2, post_only_qs)

        if not figs:
            st.info("No plots to display — the selected question type may not exist for this survey.")
        else:
            st.caption(f"{len(figs)} question(s) displayed.")
            for i, fig in enumerate(figs):
                st.plotly_chart(fig, use_container_width=True)
                if i < len(figs) - 1:
                    st.markdown('<hr class="thin">', unsafe_allow_html=True)