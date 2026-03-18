"""
app.py — Survey Analytics Platform

Two-level analytics application:
  1) Landing page (portal): Compare Surveys vs Deep Survey Analysis vs Semantic Analysis
  2) Compare Surveys page: cross-survey KPIs + 3 executive plots
  3) Deep Survey Analysis page: your existing dashboard (unchanged logic)
  4) Semantic Analysis page: Semantic Analysis of open-ended responses using BERTopic
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
    color: #3182bd;
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
  padding: 10px 22px 18px;
  box-shadow: 0 10px 20px rgba(17, 24, 39, 0.06);
  min-height: 220px;
  height: 220px;
  box-sizing: border-box;
}

.nav-card-blue {
  background: #eff6ff;
  border: 1.5px solid #93c5fd;
}
.nav-card h3 {
  margin: 0 0 2px 0;
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
    border-bottom: 2px solid #3182bd;
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
    color: #3182bd !important;
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
    """Top portal cards (3 columns): Compare / Deep / Semantic."""

    # Default (inactive) colours
    compare_bg, compare_border, compare_color = "#d1fae5", "#6ee7b7", "#065f46"
    compare_hover_bg, compare_hover_border = "#a7f3d0", "#34d399"

    deep_bg, deep_border, deep_color = "#dbeafe", "#93c5fd", "#1e3a5f"
    deep_hover_bg, deep_hover_border = "#bfdbfe", "#60a5fa"

    sem_bg, sem_border, sem_color = deep_bg, deep_border, deep_color
    sem_hover_bg, sem_hover_border = deep_hover_bg, deep_hover_border

    # When active, highlight that card's button in red.
    if active == "Compare Surveys":
        compare_bg, compare_border, compare_color = "#e53e3e", "#c53030", "#ffffff"
        compare_hover_bg, compare_hover_border = "#c53030", "#9b2c2c"
    elif active == "Deep Survey Analysis":
        deep_bg, deep_border, deep_color = "#e53e3e", "#c53030", "#ffffff"
        deep_hover_bg, deep_hover_border = "#c53030", "#9b2c2c"
    elif active == "Semantic Exploration":
        sem_bg, sem_border, sem_color = "#e53e3e", "#c53030", "#ffffff"
        sem_hover_bg, sem_hover_border = "#c53030", "#9b2c2c"

    st.markdown(
        f"""
        <style>
          /* Portal buttons: consistent light-blue styling for all 3 buttons */
          [data-testid="stHorizontalBlock"] > div .stButton > button {{
            background: #dbeafe !important;
            border: 1.5px solid #93c5fd !important;
            color: #1e3a5f !important;
            border-radius: 10px !important;
            font-weight: 700 !important;
            padding: 0.65rem 1rem !important;
          }}
          [data-testid="stHorizontalBlock"] > div .stButton > button:hover {{
            background: #bfdbfe !important;
            border-color: #60a5fa !important;
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3, gap="large")

    with c1:
        st.markdown(
            """
            <div class="nav-card nav-card-blue">
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
        if st.button("Open Comparison Dashboard", key="top_tab_compare", use_container_width=True):
            st.session_state.top_nav = "Compare Surveys"
            _goto("compare")

    with c2:
        st.markdown(
            """
            <div class="nav-card nav-card-blue">
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
        if st.button("Open Detailed Dashboard", key="top_tab_deep", use_container_width=True):
            st.session_state.top_nav = "Deep Survey Analysis"
            _goto("deep")

    with c3:
        st.markdown(
            """
            <div class="nav-card nav-card-blue">
              <h3>Semantic Analysis</h3>
              <p>BERTopic clustering for open-ended responses.</p>
              <ul>
                <li>missing-response stats</li>
                <li>topic summaries + keywords</li>
                <li>UMAP + similarity matrix</li>
              </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Open Semantic Analysis", key="top_tab_semantic_nav", use_container_width=True):
            st.session_state.top_nav = "Semantic Exploration"
            _goto("semantic")

    st.markdown('<hr class="thin">', unsafe_allow_html=True)
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
    # Landing-page button styling (scoped to portal)
    st.markdown(
        """
        <style>
          /* Portal buttons (all columns) — consistent light blue */
          .portal-wrap [data-testid="stHorizontalBlock"] > div .stButton > button {
            background: #dbeafe !important;
            border: 1.5px solid #93c5fd !important;
            color: #1e3a5f !important;
            border-radius: 10px !important;
            font-weight: 700 !important;
            padding: 0.65rem 1rem !important;
          }
          .portal-wrap [data-testid="stHorizontalBlock"] > div .stButton > button:hover {
            background: #bfdbfe !important;
            border-color: #60a5fa !important;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="portal-title">Survey Analytics Platform</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="portal-subtitle">Explore workshops across survey types or analyze one survey in detail</div>',
        unsafe_allow_html=True,
    )

    c_left, c_right, c_sem = st.columns(3, gap="large")

    with c_left:
        st.markdown(
            """
            <div class="nav-card nav-card-blue">
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
            <div class="nav-card nav-card-blue">
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
        if st.button("Open Detailed Dashboard", use_container_width=True):
            st.session_state.top_nav = "Deep Survey Analysis"
            _goto("deep")

    with c_sem:
        st.markdown(
            """
            <div class="nav-card nav-card-blue">
              <h3>Semantic Exploration</h3>
              <p>Topic modelling for open-ended responses (BERTopic).</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(
            "Open Semantic Analysis",
            key="top_tab_semantic_landing",
            use_container_width=True,
        ):
            st.session_state.top_nav = "Semantic Exploration"
            _goto("semantic")
    k1, k2, k3, k4 = st.columns(4)
    _card(k1, f"{kpis['total_survey_types']:,}", "Survey Types", bg_color="#f3faf6")
    _card(k2, f"{kpis['total_workshops']:,}", "Workshops", bg_color="#f3faf6")
    _card(k3, f"{kpis['total_schools']:,}", "Schools Covered", bg_color="#f3faf6")
    _card(k4, f"{kpis['total_responses']:,}", "Total Responses", bg_color="#f3faf6")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


    st.markdown('<hr class="thin">', unsafe_allow_html=True)
    

def render_compare(df_all: pd.DataFrame):
    _render_top_nav("Compare Surveys")

    st.sidebar.title( "Compare Surveys")
    st.sidebar.markdown("---")
    if st.sidebar.button("← Back to landing"):
        st.session_state.top_nav = "Deep Survey Analysis"
        _goto("landing")

    schools = ["All"] + sorted(df_all["school_id"].dropna().astype(str).unique().tolist())
    selected_school = st.sidebar.selectbox("School (optional)", schools, index=0)

    df_compare = df_all.copy()
    if selected_school != "All":
        df_compare = df_compare[df_compare["school_id"].astype(str) == selected_school]
    st.title("Compare Surveys")
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
    st.sidebar.title("Survey Dashboard")
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
    st.title(f"Survey Analysis — {survey_label}")
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

    st.markdown('<div class="big-title">Key Facts</div>', unsafe_allow_html=True)

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
        st.markdown("### Common PRE/POST Questions by Response Type")

        common_items = list(facts["common_questions_by_type"].items())
        common_cols = st.columns(len(common_items)) if common_items else []
        for col, (rtype, count) in zip(common_cols, common_items):
            short_label = f"{rtype}"
            _card(col, count, short_label, bg_color="#d1fae5")

    # RIGHT: POST-only
    with right_col:
        st.markdown("### POST-only Questions by Response Type")

        post_only_items = list(facts["post_only_questions_by_type"].items())
        post_only_cols = st.columns(len(post_only_items)) if post_only_items else []

        for col, (rtype, count) in zip(post_only_cols, post_only_items):
            short_label = f"{rtype}"
            _card(col, count, short_label, bg_color="#d1fae5")

    # ─────────────────────────────────────────────────────────────────────────────
    # TABS  (existing logic)
    # ─────────────────────────────────────────────────────────────────────────────
    tab1, tab2 = st.tabs([
        "Statistical Analysis (All Schools)",
        "Cross-Session Views",
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
                    with st.expander("Click to expand results table", expanded=False):
                        display_df = mw_results[[
                            "question", "n_pre", "n_post",
                            "pre_median", "post_median", "median_shift",
                            "p_value", "p_adj_BH", "significant",
                        ]].copy()
                        display_df["significant"] = display_df["significant"].map(
                            {True: "Yes", False: "No", None: "—"}
                        ).fillna("—")

                        def _color_row(row):
                            # Keep styling consistent with a single blue palette.
                            sig = row["significant"] == "Yes"
                            shift = row["median_shift"]

                            not_sig_bg, not_sig_color = "#f1f5f9", "#64748b"
                            pos_bg, pos_color = "#dbeafe", "#1e3a5f"
                            neg_bg, neg_color = "#93c5fd", "#0f172a"

                            if (not sig) or shift == 0:
                                bg, color = not_sig_bg, not_sig_color
                            elif shift > 0:
                                bg, color = pos_bg, pos_color
                            else:
                                bg, color = neg_bg, neg_color

                            return [f"background-color: {bg}; color: {color}"] * len(row)

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
                            "- **Left (shift):** darker = statistically significant (★), light = not significant.\n"
                            "- Blue shades show direction: lighter = positive, darker = negative.\n"
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
                    with st.expander("Click to expand results table", expanded=False):
                        show_cols = [c for c in [
                            "question", "scale_type", "n", "median", "pct_above",
                            "W_stat", "p_value", "p_adj_BH", "significant", "note",
                        ] if c in wlcx_results.columns]

                        wlcx_display = wlcx_results[show_cols].copy()
                        wlcx_display["significant"] = wlcx_display["significant"].map(
                            {True: "Yes", False: "No", None: "—"}
                        ).fillna("—")

                        def _color_row_wlcx(row):
                            # Same blue-only palette as Mann–Whitney table.
                            sig = row["significant"] == "Yes"
                            shift = row["median"] - 3 if "median" in row.index else 0

                            not_sig_bg, not_sig_color = "#f1f5f9", "#64748b"
                            pos_bg, pos_color = "#dbeafe", "#1e3a5f"
                            neg_bg, neg_color = "#93c5fd", "#0f172a"

                            if (not sig) or shift == 0:
                                bg, color = not_sig_bg, not_sig_color
                            elif shift > 0:
                                bg, color = pos_bg, pos_color
                            else:
                                bg, color = neg_bg, neg_color

                            return [f"background-color: {bg}; color: {color}"] * len(row)

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
                            "- **Blue shades** = significant direction (lighter = above, darker = below); "
                            "**grey/neutral** = not significant.\n"
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

def render_semantic(df_all: pd.DataFrame):
    st.title("Semantic Exploration Dashboard")

    # Sidebar
    st.sidebar.subheader("Semantic filters")
    if st.sidebar.button("← Back to landing"):
        st.session_state.top_nav = "Semantic Exploration"
        _goto("landing")

    @st.cache_data(show_spinner="Loading open-ended responses…")
    def load_semantic_data() -> pd.DataFrame:
        return explore_semantic_text(df_all)

    df_sub = load_semantic_data()

    # Prefer survey types present in the open-ended subset; fall back to full df.
    available_types = []
    if "survey_type" in df_sub.columns:
        available_types = sorted(df_sub["survey_type"].dropna().unique().tolist())
    if not available_types and "survey_type" in df_all.columns:
        available_types = sorted(df_all["survey_type"].dropna().unique().tolist())

    sem_survey_types = ["All"] + available_types if available_types else ["All"]

    selected_sem_survey_type = st.sidebar.selectbox(
        "Survey type (semantic)",
        options=sem_survey_types,
        index=0,
    )
    if selected_sem_survey_type == "All":
        st.sidebar.caption("Pick a specific survey type to run BERTopic per type.")

    df_sub_filtered = df_sub.copy()
    if "survey_type" in df_sub_filtered.columns and selected_sem_survey_type != "All":
        df_sub_filtered = df_sub_filtered[df_sub_filtered["survey_type"] == selected_sem_survey_type]

    if df_sub_filtered.empty:
        st.info("No open-ended responses found for the selected filters.")
        st.stop()

    # ── Missing Responses Analysis ──────────────────────────────────────────
    st.header("Missing Responses Analysis")

    def calc_missing_pct(x: pd.Series) -> float:
        if len(x) == 0:
            return 0.0
        missing = x.isna().sum()
        empty = (x.astype(str).str.strip() == "").sum()
        return (missing + empty) / len(x) * 100

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

    # Prevent Pandas Styler formatting crashes if any values are None/non-numeric.
    missing_stats["missing_pct"] = (
        pd.to_numeric(missing_stats["missing_pct"], errors="coerce").fillna(0.0)
    )

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

    # ── Semantic Analysis by Concept ────────────────────────────────────────
    st.header("Semantic Analysis by Concept")
    st.caption("Using BERTopic for topic modeling (small-dataset fallback supported).")

    df_open = clean_responses(df_sub_filtered)
    concept_options = sorted(df_open["concept_key"].dropna().unique().tolist())

    if not concept_options:
        st.warning("No concepts available for semantic analysis.")
        st.stop()

    selected_concept_analysis = st.selectbox(
        "Select Concept Key for Semantic Analysis",
        options=concept_options,
        index=0,
        key="semantic_concept_key",
    )

    run_disabled = selected_sem_survey_type == "All"

    def _safe_key(x) -> str:
        return str(x).replace("/", "_").replace(" ", "_")

    state_prefix = f"semantic_{_safe_key(selected_sem_survey_type)}_{_safe_key(selected_concept_analysis)}"

    if st.button("Run Semantic Analysis", type="primary", disabled=run_disabled, key="semantic_run_button"):
        with st.spinner("Computing embeddings and clustering responses…"):
            df_q = df_open[df_open["concept_key"] == selected_concept_analysis].copy()

            if len(df_q) < 3:
                st.warning(
                    f"Not enough responses for concept '{selected_concept_analysis}'. Need at least 3 responses."
                )
            elif len(df_q) == 5:
                summary = summarize_small_dataset(df_q)
                st.session_state[f"df_q_{state_prefix}"] = df_q
                st.session_state[f"summary_{state_prefix}"] = summary
                st.session_state[f"topic_model_{state_prefix}"] = None
                st.session_state[f"umap_{state_prefix}"] = None
                st.session_state[f"topics_dict_{state_prefix}"] = {}
            else:
                try:
                    df_q2, topic_model, topic_info, embeddings = cluster_responses_bertopic(df_q)
                    topics_dict = extract_topics_bertopic(topic_model, df_q2)
                    summary = summarize_clusters_bertopic(
                        df_q2,
                        topics_dict,
                        topic_info,
                        topic_model,
                        total_responses=len(df_q2),
                    )

                    umap_embeddings = None
                    try:
                        if hasattr(topic_model, "umap_model"):
                            umap_embeddings = topic_model.umap_model.transform(embeddings)
                        else:
                            umap_embeddings = compute_umap(embeddings)
                    except Exception:
                        umap_embeddings = compute_umap(embeddings)

                    st.session_state[f"df_q_{state_prefix}"] = df_q2
                    st.session_state[f"summary_{state_prefix}"] = summary
                    st.session_state[f"topic_model_{state_prefix}"] = topic_model
                    st.session_state[f"umap_{state_prefix}"] = umap_embeddings
                    st.session_state[f"topics_dict_{state_prefix}"] = topics_dict
                except Exception as e:
                    st.error(f"Error with BERTopic: {str(e)}")

    # ── Display Results ────────────────────────────────────────────────────
    summary_key = f"summary_{state_prefix}"
    if summary_key in st.session_state:
        df_q = st.session_state.get(f"df_q_{state_prefix}")
        summary = st.session_state.get(f"summary_{state_prefix}")
        topic_model = st.session_state.get(f"topic_model_{state_prefix}")
        umap_embeddings = st.session_state.get(f"umap_{state_prefix}")

        # Question text
        try:
            question_text_example = (
                df_sub.loc[df_sub["concept_key"] == selected_concept_analysis, "question_text"]
                .dropna()
                .iloc[0]
            )
            if question_text_example:
                st.markdown(f"## **Question text:** {question_text_example}")
        except Exception:
            pass

        st.subheader("Topic Summary")
        if summary is None or len(summary) == 0:
            st.info("No topics found for this concept.")
            return

        summary_display = summary.copy()
        if "percentage" in summary_display.columns:
            summary_display["percentage"] = summary_display["percentage"].apply(lambda x: f"{x:.1f}%")

        # Table + download
        st.dataframe(
            summary_display[["cluster", "size", "percentage", "keywords", "example_responses"]],
            use_container_width=True,
            height=320,
        )
        st.download_button(
            "⬇ Download topic summary CSV",
            data=summary.to_csv(index=False).encode(),
            file_name=f"semantic_{_safe_key(selected_sem_survey_type)}_{_safe_key(selected_concept_analysis)}_topics.csv",
            mime="text/csv",
        )

        # Visualizations
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

        if topic_model is not None:
            st.subheader("Topic Similarity Matrix")
            try:
                fig_similarity = topic_model.visualize_heatmap()
                st.plotly_chart(fig_similarity, use_container_width=True)
            except Exception:
                st.info("Topic similarity matrix not available for this concept.")

    st.stop()


_ensure_page_state()

# df_compare_base: minimal filtering (keep all survey types)
df_compare_base = df_all.copy()

if st.session_state.page == "landing":
    render_landing(df_compare_base)
elif st.session_state.page == "compare":
    render_compare(df_all)
elif st.session_state.page == "semantic":
    render_semantic(df_all)
else:
    render_deep(df_all)
