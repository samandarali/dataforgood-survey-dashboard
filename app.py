"""
app.py — Survey Analytics Platform

Two-level analytics application:
  1) Landing page (portal): Compare Surveys vs Deep Survey Analysis
  2) Compare Surveys page: cross-survey KPIs + 3 executive plots
  3) Deep Survey Analysis page: your existing dashboard (unchanged logic)
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import plotly.express as px

import analytics as an


from semantic_exploration import (
    clean_responses,
    compute_umap,
    cluster_responses_bertopic,
    extract_topics_bertopic,
    summarize_clusters_bertopic,
    summarize_small_dataset,
)

def explore_semantic_text(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df["scale_type"].isin(_CLOSED_SCALE_TYPES)].copy()

st.set_page_config(
    page_title="Survey Analytics Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown("""
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

/* ── portal (landing) cards ── */
.portal-wrap {
  max-width: 980px;
  margin: 0 auto;
}
.portal-title {
  font-size: 44px;
  font-weight: 800;
  text-align: center;
  margin-top: 8px;
  margin-bottom: 4px;
  color: #1f2937;
}
.portal-subtitle {
  font-size: 16px;
  text-align: center;
  color: #6b7280;
  margin-bottom: 26px;
}
.nav-card {
  background: #ffffff;
  border: 1px solid #e5e7eb;
  border-radius: 16px;
  padding: 22px 22px 18px;
  box-shadow: 0 10px 20px rgba(17, 24, 39, 0.06);
}
.nav-card h3 {
  margin: 0 0 6px 0;
  font-size: 22px;
  font-weight: 750;
  color: #111827;
}
.nav-card p {
  margin: 0 0 12px 0;
  color: #6b7280;
  font-size: 14px;
}
.nav-card ul {
  margin: 0 0 14px 18px;
  color: #6b7280;
  font-size: 13px;
}
.kpi-strip {
  background: #f3faf6;
  border: 1px solid #d1fae5;
  border-radius: 14px;
  padding: 14px 14px 6px;
  margin-top: 18px;
}

/* ── section title ── */
.section-title {
    font-size: 28px !important;
    font-weight: 700 !important;
    color: #343a40;
    margin: 18px 0 8px;
    padding-bottom: 4px;
    border-bottom: 2px solid #1D9E75;
}

/* ── expander title ── */
details summary p,
[data-testid="stExpander"] summary p {
    font-size: 22px !important;
    font-weight: 600 !important;
    color: #1f4e79;          
}

/* ── guide headings ── */
h5 {
    font-size: 20px !important;
    font-weight: 700 !important;
         
}

/* ── divider ── */
hr.thin {
    border: none;
    border-top: 1px solid #dee2e6;
    margin: 20px 0;
}
            
[data-testid="stExpander"] summary {
    background-color: #f8f9fa;
    border: 1px solid #dee2e6;
    border-radius: 8px;
    padding: 8px 12px;
}    
                    
button[data-baseweb="tab"] {
    font-size: 24px !important;
    font-weight: 700 !important;
    padding: 10px 18px !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #1D9E75 !important;
}
            
/* ── tab titles ── */
button[data-baseweb="tab"] p {
    font-size: 1.30rem;
    font-weight: 600;
}
            
/* ── spacing above tab bar ── */
div[data-testid="stTabs"] {
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner="Loading survey data…")
def load() -> pd.DataFrame:
    df = an.load_data()
    df = an.create_survey_session_id(df)
    return df


df_all = load()


_CLOSED_SCALE_TYPES = {"Likert Scale Numeric", "Likert Scale Text", "Categorical"}

def get_open_ended_df(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df["scale_type"].isin(_CLOSED_SCALE_TYPES)].copy()

def _ensure_page_state():
    if "page" not in st.session_state:
        st.session_state.page = "landing"
    if "top_nav" not in st.session_state:
        st.session_state.top_nav = "Deep Survey Analysis"

def _goto(page: str):
    st.session_state.page = page
    st.rerun()

def _render_top_nav(active: str):
    """
    Executive-style top nav for switching between the two dashboards.
    This is intentionally simple (Streamlit-native) and keeps navigation available
    after entering either dashboard.
    """
    st.markdown(
        """
        <style>
          .top-tabs { max-width: 980px; margin: 0 auto 0.5rem auto; }
          .top-tabs [data-testid="stHorizontalBlock"] { gap: 0.4rem; }
          .top-tabs button[kind="secondary"] {
            border-radius: 999px !important;
            border: 1px solid #e5e7eb !important;
            background: #ffffff !important;
            color: #374151 !important;
            font-weight: 700 !important;
            padding: 0.55rem 0.9rem !important;
          }
          .top-tabs .active button[kind="secondary"] {
            border: 1px solid #10b981 !important;
            background: #ecfdf5 !important;
            color: #065f46 !important;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.container():
        st.markdown('<div class="top-tabs">', unsafe_allow_html=True)
        c1, c2 = st.columns(2)

        # "Tab" 1: Compare
        with c1:
            st.markdown(
                '<div class="active">' if active == "Compare Surveys" else "<div>",
                unsafe_allow_html=True,
            )
            if st.button("Compare Surveys", key="top_tab_compare", use_container_width=True):
                if active != "Compare Surveys":
                    st.session_state.top_nav = "Compare Surveys"
                    _goto("compare")
            st.markdown("</div>", unsafe_allow_html=True)

        # "Tab" 2: Deep
        with c2:
            st.markdown(
                '<div class="active">' if active == "Deep Survey Analysis" else "<div>",
                unsafe_allow_html=True,
            )
            if st.button("Deep Survey Analysis", key="top_tab_deep", use_container_width=True):
                if active != "Deep Survey Analysis":
                    st.session_state.top_nav = "Deep Survey Analysis"
                    _goto("deep")
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

def _card(col, value, label, bg_color="#f8f9fa"):
    col.markdown(
        f'<div class="fact-card" style="background:{bg_color};">'
        f'<div class="value">{value}</div>'
        f'<div class="label">{label}</div></div>',
        unsafe_allow_html=True,
    )

def render_landing(df_compare_base: pd.DataFrame):
    # Hide sidebar on landing for centered portal feel
    st.markdown(
        "<style>[data-testid='stSidebar']{display:none;} .block-container{padding-top: 2rem;}</style>",
        unsafe_allow_html=True,
    )

    kpis = an.compute_compare_kpis(df_compare_base)

    st.markdown('<div class="portal-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="portal-title">Survey Analytics Platform</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="portal-subtitle">Explore workshops across survey types or analyze one survey in detail</div>',
        unsafe_allow_html=True,
    )

    c_left, c_right = st.columns(2, gap="large")

    with c_left:
        st.markdown(
            """
            <div class="nav-card">
              <h3>Compare Surveys</h3>
              <p>Compare survey programs across:</p>
              <ul>
                <li>number of workshops</li>
                <li>response volume</li>
                <li>PRE vs POST balance</li>
                <li>trends over time</li>
                <li>school coverage</li>
              </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Open Comparison Dashboard", use_container_width=True):
            st.session_state.top_nav = "Compare Surveys"
            _goto("compare")

    with c_right:
        st.markdown(
            """
            <div class="nav-card">
              <h3>Deep Survey Analysis</h3>
              <p>Detailed statistical analysis for one survey:</p>
              <ul>
                <li>question-level significance</li>
                <li>median shifts</li>
                <li>response distributions</li>
                <li>semantic analysis</li>
              </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Open Detailed Dashboard", type="primary", use_container_width=True):
            st.session_state.top_nav = "Deep Survey Analysis"
            _goto("deep")

    st.markdown('<div class="kpi-strip">', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    _card(k1, f"{kpis['total_survey_types']:,}", "Survey Types", bg_color="#f3faf6")
    _card(k2, f"{kpis['total_workshops']:,}", "Workshops", bg_color="#f3faf6")
    _card(k3, f"{kpis['total_schools']:,}", "Schools Covered", bg_color="#f3faf6")
    _card(k4, f"{kpis['total_responses']:,}", "Total Responses", bg_color="#f3faf6")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

def render_compare(df_all: pd.DataFrame):
    _render_top_nav("Compare Surveys")

    st.sidebar.title("📊 Compare Surveys")
    st.sidebar.markdown("---")
    if st.sidebar.button("← Back to landing"):
        st.session_state.top_nav = "Deep Survey Analysis"
        _goto("landing")

    schools = ["All"] + sorted(df_all["school_id"].dropna().astype(str).unique().tolist())
    selected_school = st.sidebar.selectbox("School (optional)", schools, index=0)

    df_compare = df_all.copy()
    if selected_school != "All":
        df_compare = df_compare[df_compare["school_id"].astype(str) == selected_school]

    st.title("📊 Compare Surveys")
    st.caption("Cross-survey comparison across all survey types (PRE vs POST).")
    if selected_school != "All":
        st.caption(f"Filtered to school: **{selected_school}**")

    kpis = an.compute_compare_kpis(df_compare)

    k1, k2, k3, k4 = st.columns(4)
    _card(k1, f"{kpis['total_survey_types']:,}", "Total survey types", bg_color="#f3faf6")
    _card(k2, f"{kpis['total_workshops']:,}", "Total workshops", bg_color="#f3faf6")
    _card(k3, f"{kpis['total_schools']:,}", "Total schools", bg_color="#f3faf6")
    _card(k4, f"{kpis['total_responses']:,}", "Total responses", bg_color="#f3faf6")

    st.markdown('<hr class="thin">', unsafe_allow_html=True)

    left, right = st.columns(2, gap="large")
    with left:
        fig1 = an.plot_sessions_by_survey_type_phase(df_compare)
        st.plotly_chart(fig1, use_container_width=True)
    with right:
        fig2 = an.plot_sessions_by_school_phase(df_compare)
        st.plotly_chart(fig2, use_container_width=True)

    left2, right2 = st.columns(2, gap="large")
    with left2:
        fig3 = an.plot_sessions_over_time_phase_daily(df_compare)
        st.plotly_chart(fig3, use_container_width=True)
    with right2:
        st.markdown(
            """
            <div class="nav-card">
              <h3>Notes</h3>
              <p>This page counts <b>workshops</b> using <code>survey_session_id</code> (unique session).</p>
              <p>The timeline uses <b>daily</b> aggregated counts and compares PRE vs POST.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

def render_deep(df_all: pd.DataFrame):
    _render_top_nav("Deep Survey Analysis")

    # ─────────────────────────────────────────────────────────────────────────────
    # Sidebar — survey_type selector  (existing logic)
    # ─────────────────────────────────────────────────────────────────────────────
    st.sidebar.title("📋 Survey Dashboard")
    st.sidebar.markdown("---")
    if st.sidebar.button("← Back to landing"):
        st.session_state.top_nav = "Deep Survey Analysis"
        _goto("landing")

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
    # Filter data to selected survey type + version  (existing logic)
    # ─────────────────────────────────────────────────────────────────────────────
    df = df_all[
        (df_all["survey_type"]    == selected_type) &
        (df_all["survey_version"] == selected_version)
    ].copy()

    if selected_school != "All":
        df = df[df["school_id"].astype(str) == selected_school]

    survey_label = f"{selected_type}_{selected_version}"
    common_qs = an.get_common_pre_post_questions(df)
    post_only_qs = an.get_post_only_questions(df)

    # ─────────────────────────────────────────────────────────────────────────────
    # Page title  (existing logic)
    # ─────────────────────────────────────────────────────────────────────────────
    st.title(f"📊 Survey Analysis — {survey_label}")
    if selected_school != "All":
        st.caption(f"Filtered to school: **{selected_school}**")

    # ─────────────────────────────────────────────────────────────────────────────
    # Fact cards  (existing logic)
    # ─────────────────────────────────────────────────────────────────────────────
    facts = an.compute_fact_cards(df)

    st.markdown("""
    <style>
    .big-title {
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="big-title">📌 Key Facts</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)

    # row 1
    _card(c1, f"{facts['pre_observations']:,}", "PRE observations")
    _card(c2, f"{facts['post_observations']:,}", "POST observations")
    _card(c3, facts["pre_questions_total"], "PRE survey questions")
    _card(c4, facts["post_questions_total"], "POST survey questions")
    _card(c5, facts["n_schools"], "Schools")
    _card(c6, facts["pre_sessions"], "PRE data collection sessions")
    _card(c7, facts["post_sessions"], "POST data collection sessions")

    # Parent row with two large containers
    left_col, right_col = st.columns(2)

    # LEFT: Common PRE/POST
    with left_col:
        st.markdown("### 🔁 Common PRE/POST Questions by Response Type")

        common_items = list(facts["common_questions_by_type"].items())
        common_cols = st.columns(len(common_items)) if common_items else []
        for col, (rtype, count) in zip(common_cols, common_items):
            short_label = f"Common Qs ({rtype})"
            _card(col, count, short_label, bg_color="#fafde8")

    # RIGHT: POST-only
    with right_col:
        st.markdown("### ➕ POST-only Questions by Response Type")

        post_only_items = list(facts["post_only_questions_by_type"].items())
        post_only_cols = st.columns(len(post_only_items)) if post_only_items else []

        for col, (rtype, count) in zip(post_only_cols, post_only_items):
            short_label = f"POST-only Qs ({rtype})"
            _card(col, count, short_label, bg_color="#77f1e7")

    # ─────────────────────────────────────────────────────────────────────────────
    # TABS  (existing logic)
    # ─────────────────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "📈 Statistical Analysis (All Schools)",
        "🔄 Cross-Session Views",
        "💬 Open-Ended Response Analysis",
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
                    st.caption("Expand below to inspect question-level statistical results.")
                    with st.expander("📂 Click to expand results table", expanded=False):
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
                    st.caption("Expand below to inspect question-level statistical results.")
                    with st.expander("📂 Click to expand results table", expanded=False):
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

        # 1) Choose question type first
        question_type = st.radio(
            "Question type",
            [
                "Likert Text",
                "Likert Numeric",
                "Categorical",
                "Common PRE/POST",
                "POST-only"
            ],
            horizontal=True,
            key="tab2_question_type",
        )

        # 2) Show phase filter only when relevant
        selected_phase = "All"
        if question_type in ["Likert Text", "Likert Numeric", "Categorical"]:
            phase_options = ["All"] + sorted(df["survey_phase"].dropna().unique().tolist())
            selected_phase = st.selectbox(
                "Survey phase",
                options=phase_options,
                index=0,
                key="tab2_phase",
            )

        # 3) Apply filters
        df_tab2 = df.copy()
        if question_type in ["Likert Text", "Likert Numeric", "Categorical"]:
            if selected_phase != "All":
                df_tab2 = df_tab2[df_tab2["survey_phase"] == selected_phase]

        tab2_title = f"{survey_label} | {selected_phase} | {question_type}"

        if df_tab2.empty:
            st.warning("No data for selected filters.")
        else:
            with st.spinner("Building plots…"):
                if question_type == "Likert Text":
                    figs = an.plot_likert_by_session(
                        df_tab2, likert_kind="text", title=tab2_title
                    )
                elif question_type == "Likert Numeric":
                    figs = an.plot_likert_by_session(
                        df_tab2, likert_kind="numeric", title=tab2_title
                    )
                elif question_type == "Categorical":
                    figs = an.plot_categorical_by_session(df_tab2, title=tab2_title)
                elif question_type == "Common PRE/POST":
                    figs = an.plot_pre_post_bar_with_mean(df_tab2, common_qs)
                else:  # POST-only
                    df_post_tab2 = df.copy()
                    df_post_tab2 = df_post_tab2[df_post_tab2["survey_phase"] == "POST"]
                    figs = an.plot_post_bar_with_mean(df_post_tab2, post_only_qs)

            if not figs:
                st.info("No plots to display — the selected question type may not exist for this survey.")
            else:
                st.caption(f"{len(figs)} question(s) displayed.")
                for i, fig in enumerate(figs):
                    st.plotly_chart(fig, use_container_width=True)
                    if i < len(figs) - 1:
                        st.markdown('<hr class="thin">', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════════
    #  TAB 3 – Open-Ended / Semantic Analysis
    # ══════════════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown(
            f"Topic modelling of open-ended text responses for **{survey_label}** "
            "using **BERTopic** (sentence embeddings + HDBSCAN clustering)."
        )

        # ── Isolate open-ended rows for the currently selected survey ─────────────
        df_sub = explore_semantic_text(df)

        if df_sub.empty:
            st.info(
                "No open-ended questions found for this survey type / version. "
                "Open-ended questions must have a `scale_type` that is not "
                "'Likert Scale Numeric', 'Likert Scale Text', or 'Categorical'."
            )
        else:
            # ── Section 1 — Missing response overview ─────────────────────────────
            st.markdown('<p class="section-title">① Missing Responses by Concept</p>',
                        unsafe_allow_html=True)

            def _calc_missing_pct(x):
                if len(x) == 0:
                    return 0.0
                missing = x.isna().sum()
                empty   = (x.astype(str).str.strip() == "").sum()
                return (missing + empty) / len(x) * 100

            missing_stats = (
                df_sub.groupby("concept_key")["response"]
                .apply(_calc_missing_pct)
                .reset_index(name="missing_pct")
            )
            total_counts = (
                df_sub.groupby("concept_key")["response"]
                .count()
                .reset_index(name="total_responses")
            )
            missing_stats = (
                missing_stats.merge(total_counts, on="concept_key")
                .sort_values("missing_pct", ascending=False)
            )

            col_tbl, col_chart = st.columns(2)
            with col_tbl:
                st.dataframe(
                    missing_stats.style.format({"missing_pct": "{:.1f}%"}),
                    use_container_width=True,
                )
            with col_chart:
                fig_missing = px.bar(
                    missing_stats,
                    x="concept_key",
                    y="missing_pct",
                    title="% Missing Responses by Concept",
                    labels={"missing_pct": "Missing %", "concept_key": "Concept"},
                )
                fig_missing.update_xaxes(tickangle=45)
                st.plotly_chart(fig_missing, use_container_width=True)

            # ── Section 2 — Per-concept semantic analysis ─────────────────────────
            st.markdown('<hr class="thin">', unsafe_allow_html=True)
            st.markdown('<p class="section-title">② Semantic Analysis by Concept</p>',
                        unsafe_allow_html=True)
            st.caption(
                "Minimum topic size adapts automatically to dataset size. "
                "Results are cached in session — re-select a concept to view previous runs."
            )

            df_open = clean_responses(df_sub)
            concept_map = (
                df_open[["concept_key", "concept_name"]]
                .dropna(subset=["concept_key"])
                .drop_duplicates()
                .set_index("concept_key")["concept_name"]
                .to_dict()
            )
            concept_options = sorted(df_open["concept_key"].dropna().unique().tolist())
            selected_concept_analysis = st.selectbox(
                "Select Concept for Semantic Analysis",
                options=concept_options,
                format_func=lambda k: concept_map.get(k, k),  # shows concept_name, stores concept_key
                index=0,
            )

            if not concept_options:
                st.warning("All open-ended responses were empty or too short to analyse.")
            else:
                selected_concept = st.selectbox(
                    "Select Concept Key",
                    options=concept_options,
                    index=0,
                    key="tab3_concept",
                )

                # Show question text immediately
                try:
                    q_text = (
                        df_sub.loc[
                            df_sub["concept_key"] == selected_concept, "question_text"
                        ].dropna().iloc[0]
                    )
                    st.markdown(f"**Question:** {q_text}")
                except (IndexError, KeyError):
                    pass

                n_responses = len(df_open[df_open["concept_key"] == selected_concept])
                st.caption(f"{n_responses} valid responses available for this concept.")

                if st.button("▶ Run Semantic Analysis", type="primary", key="tab3_run"):
                    with st.spinner("Computing embeddings and clustering responses…"):
                        df_q = df_open[df_open["concept_key"] == selected_concept].copy()

                        if len(df_q) < 3:
                            st.warning(
                                f"Not enough responses for **{selected_concept}** "
                                "(minimum 3 required)."
                            )
                        elif len(df_q) <= 5:
                            # Tiny dataset — skip BERTopic, use simple fallback
                            summary = summarize_small_dataset(df_q)
                            st.session_state[f"tab3_df_q_{selected_concept}"]      = df_q
                            st.session_state[f"tab3_summary_{selected_concept}"]   = summary
                            st.session_state[f"tab3_model_{selected_concept}"]     = None
                            st.session_state[f"tab3_umap_{selected_concept}"]      = None
                        else:
                            try:
                                df_q, topic_model, topic_info, embeddings = \
                                    cluster_responses_bertopic(df_q)
                                topics_dict = extract_topics_bertopic(topic_model, df_q)
                                summary = summarize_clusters_bertopic(
                                    df_q, topics_dict, topic_info, topic_model,
                                    total_responses=len(df_q),
                                )

                                # UMAP for scatter plot
                                umap_embeddings = None
                                try:
                                    if hasattr(topic_model, "umap_model"):
                                        umap_embeddings = topic_model.umap_model.transform(
                                            embeddings
                                        )
                                    else:
                                        umap_embeddings = compute_umap(embeddings)
                                except Exception:
                                    try:
                                        umap_embeddings = compute_umap(embeddings)
                                    except Exception:
                                        umap_embeddings = None

                                st.session_state[f"tab3_df_q_{selected_concept}"]    = df_q
                                st.session_state[f"tab3_summary_{selected_concept}"] = summary
                                st.session_state[f"tab3_model_{selected_concept}"]   = topic_model
                                st.session_state[f"tab3_umap_{selected_concept}"]    = umap_embeddings
                            except Exception as e:
                                st.error(f"BERTopic error: {e}")

                # ── Display cached results ─────────────────────────────────────────
                key_prefix = f"tab3_df_q_{selected_concept}"
                if key_prefix in st.session_state:
                    df_q          = st.session_state[f"tab3_df_q_{selected_concept}"]
                    summary        = st.session_state[f"tab3_summary_{selected_concept}"]
                    topic_model    = st.session_state.get(f"tab3_model_{selected_concept}")
                    umap_embeddings = st.session_state.get(f"tab3_umap_{selected_concept}")

                    # ── Topic summary table ────────────────────────────────────────
                    st.markdown("#### 📊 Topic Summary")
                    if len(summary) == 0:
                        st.info("No distinct topics found — all responses may be too similar "
                                "or the dataset is too small.")
                    else:
                        summary_display = summary.copy()
                        summary_display["percentage"] = summary_display["percentage"].apply(
                            lambda x: f"{x:.1f}%"
                        )
                        st.dataframe(
                            summary_display[
                                ["cluster", "size", "percentage",
                                 "keywords", "example_responses"]
                            ],
                            use_container_width=True,
                            height=320,
                        )
                        st.download_button(
                            "⬇ Download topic summary CSV",
                            data=summary.to_csv(index=False).encode(),
                            file_name=f"{survey_label}_{selected_concept}_topics.csv",
                            mime="text/csv",
                            key="tab3_dl",
                        )

                        # ── Charts ────────────────────────────────────────────────
                        col_pie, col_bar = st.columns(2)

                        with col_pie:
                            st.markdown("#### 📈 Topic Distribution")
                            fig_pie = px.pie(
                                summary,
                                values="percentage",
                                names="topic_name",
                                title=f"Topic distribution — {selected_concept}",
                                hover_data=["size"],
                            )
                            fig_pie.update_traces(
                                textposition="inside", textinfo="percent+label"
                            )
                            st.plotly_chart(fig_pie, use_container_width=True)

                        with col_bar:
                            st.markdown("#### 📊 Topic Percentages")
                            fig_bar = px.bar(
                                summary.sort_values("percentage", ascending=False),
                                x="topic_name",
                                y="percentage",
                                title="Percentage by topic",
                                labels={
                                    "percentage": "Percentage (%)",
                                    "topic_name": "Topic",
                                },
                                text="percentage",
                            )
                            fig_bar.update_traces(
                                texttemplate="%{text:.1f}%", textposition="outside"
                            )
                            fig_bar.update_xaxes(tickangle=45)
                            st.plotly_chart(fig_bar, use_container_width=True)

                        # ── UMAP scatter ──────────────────────────────────────────
                        if umap_embeddings is not None:
                            st.markdown("#### 🗺️ UMAP Response Map")
                            st.caption(
                                "Each dot is one response. Responses with similar meaning "
                                "cluster together. Colour = topic. Hover to read the response."
                            )
                            min_len = min(len(umap_embeddings), len(df_q))
                            topic_name_map = dict(
                                zip(summary["cluster"], summary["topic_name"])
                            )
                            topic_name_map[-1] = "Noise / Unassigned"
                            umap_df = pd.DataFrame({
                                "x":        umap_embeddings[:min_len, 0],
                                "y":        umap_embeddings[:min_len, 1],
                                "cluster":  df_q["cluster"].values[:min_len],
                                "response": df_q["response"].values[:min_len],
                            })
                            umap_df["topic_name"] = (
                                umap_df["cluster"].map(topic_name_map).fillna("Unknown")
                            )
                            fig_umap = px.scatter(
                                umap_df,
                                x="x", y="y",
                                color="topic_name",
                                hover_data=["response"],
                                title=f"UMAP — {selected_concept}",
                                labels={"x": "UMAP 1", "y": "UMAP 2"},
                            )
                            fig_umap.update_traces(marker=dict(size=7, opacity=0.8))
                            st.plotly_chart(fig_umap, use_container_width=True)

                        # ── Topic similarity heatmap ──────────────────────────────
                        if topic_model is not None:
                            st.markdown("#### 🔢 Topic Similarity Matrix")
                            try:
                                fig_heat = topic_model.visualize_heatmap()
                                st.plotly_chart(fig_heat, use_container_width=True)
                            except Exception:
                                st.info(
                                    "Similarity matrix requires ≥ 2 topics — "
                                    "not available for this concept."
                                )


_ensure_page_state()

# df_compare_base: minimal filtering (keep all survey types)
df_compare_base = df_all.copy()

if st.session_state.page == "landing":
    render_landing(df_compare_base)
elif st.session_state.page == "compare":
    render_compare(df_all)
else:
    render_deep(df_all)