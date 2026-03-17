"""
analytics.py — Single source of truth for the Survey Analysis Dashboard.

Contains:
  • Data loading & merging
  • Session ID construction (survey_session_id)
  • Fact-card computations
  • Statistical tests  (Mann-Whitney PRE vs POST, Wilcoxon vs neutral)
  • All Plotly visualisation functions
      - plot_mann_whitney_shift()   ← PRE/POST median shift + dot plot
      - plot_wilcoxon_neutral()     ← POST-only vs midpoint dot plot
      - plot_pre_post_bar_with_mean()   ← stacked bar per session (common Qs)
      - plot_post_bar_with_mean()       ← stacked bar per session (post-only Qs)
      - plot_likert_by_session()    ← existing session-level stacked bar + pie
      - plot_categorical_by_session()   ← categorical version
"""

from __future__ import annotations

# ── stdlib ────────────────────────────────────────────────────────────────────
import ast
import hashlib
import math
import re
import textwrap
from pathlib import Path

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# optional stats
try:
    from scipy.stats import mannwhitneyu, wilcoxon as scipy_wilcoxon
    from statsmodels.stats.multitest import multipletests
    STATS_AVAILABLE = True
except ImportError:
    STATS_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATA LOADING & MERGING
# ══════════════════════════════════════════════════════════════════════════════

def load_data(data_path: str | Path | None = None) -> pd.DataFrame:
    """
    Load and merge the four CSV files into one long-format dataframe.

    Expected files:  surveys.csv, questions.csv, responses.csv, concepts.csv
    Merge keys:      survey_key, question_key, concept_key

    Returns a dataframe with (at minimum):
        survey_type, survey_phase, survey_version, concept_key, question_text,
        response_id, timestamp, scale_type, response, response_encoded,
        school_id, possible_responses, measurement_level
    """
    if data_path is None:
        data_path = Path.cwd() / "data"
    data_path = Path(data_path)

    surveys   = pd.read_csv(data_path / "surveys.csv")
    questions = pd.read_csv(data_path / "questions.csv")
    responses = pd.read_csv(data_path / "responses.csv")
    concepts  = pd.read_csv(data_path / "concepts.csv")

    responses["timestamp"] = pd.to_datetime(responses["timestamp"], errors="coerce")

    df = (
        questions
        .merge(surveys,   on="survey_key",   how="left")
        .merge(responses, on="question_key",  how="left")
        .merge(concepts,  on="concept_key",   how="left")
    )
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2.  SESSION-ID CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def create_survey_session_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer `survey_session_id` that uniquely identifies one collection event.

    Format:  {SURVEY_TYPE}{survey_version}_{PHASE}_{YYMMDD}_{AM|PM}
    Example: MENv1_PRE_241015_AM

    Also adds helper columns: date, hour, session, date_id.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["date"]      = df["timestamp"].dt.date
    df["hour"]      = df["timestamp"].dt.hour
    df["session"]   = np.where(df["timestamp"].dt.hour < 12, "AM", "PM")
    df["date_id"]   = df["timestamp"].dt.strftime("%y%m%d").astype("Int64")

    df["survey_session_id"] = (
        df["survey_type"].str.upper()
        + df["survey_version"].str.lower()
        + "_"
        + df["survey_phase"].str.upper()
        + "_"
        + df["date_id"].astype(str)
        + "_"
        + df["session"].str.upper()
    )
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3.  FILTERING
# ══════════════════════════════════════════════════════════════════════════════

def apply_filters(
    df: pd.DataFrame,
    survey_type: str = "All",
    survey_phase: str = "All",
    survey_session_id: str = "All",
    school_id: str = "All",
    scale_type: str = "All",
) -> pd.DataFrame:
    """Return a filtered copy of *df*.  Pass 'All' to skip a dimension."""
    out = df.copy()
    if survey_type       and survey_type       != "All":
        out = out[out["survey_type"]       == survey_type]
    if survey_phase      and survey_phase      != "All":
        out = out[out["survey_phase"]      == survey_phase]
    if survey_session_id and survey_session_id != "All":
        out = out[out["survey_session_id"] == survey_session_id]
    if school_id         and school_id         != "All":
        out = out[out["school_id"]         == school_id]
    if scale_type        and scale_type        != "All":
        out = out[out["scale_type"]        == scale_type]
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 4.  QUESTION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_likert_questions(df: pd.DataFrame) -> np.ndarray:
    return df.loc[
        df["scale_type"].isin(["Likert Scale Numeric", "Likert Scale Text"]),
        "concept_key",
    ].dropna().unique()


def get_common_pre_post_questions(df: pd.DataFrame) -> list[str]:
    """Question texts that appear in BOTH PRE and POST phases (Likert only)."""
    likert = df[df["scale_type"].isin(["Likert Scale Numeric", "Likert Scale Text"])]
    pre  = set(likert.loc[likert["survey_phase"] == "PRE",  "question_text"].unique())
    post = set(likert.loc[likert["survey_phase"] == "POST", "question_text"].unique())
    return sorted(pre & post)


def get_post_only_questions(df: pd.DataFrame) -> list[str]:
    """Question texts that appear ONLY in POST phase (Likert only)."""
    likert = df[df["scale_type"].isin(["Likert Scale Numeric", "Likert Scale Text"])]
    pre  = set(likert.loc[likert["survey_phase"] == "PRE",  "question_text"].unique())
    post = set(likert.loc[likert["survey_phase"] == "POST", "question_text"].unique())
    return sorted(post - pre)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  FACT-CARD COMPUTATIONS
# ══════════════════════════════════════════════════════════════════════════════
def get_question_counts_by_phase(df: pd.DataFrame, id_col: str = "concept_key") -> dict:
    pre = df[df["survey_phase"] == "PRE"]
    post = df[df["survey_phase"] == "POST"]

    return {
        "pre_questions_total": pre[id_col].dropna().nunique() if id_col in df.columns else pre["question_text"].dropna().nunique(),
        "post_questions_total": post[id_col].dropna().nunique() if id_col in df.columns else post["question_text"].dropna().nunique(),
    }


def get_common_pre_post_questions_by_type(df: pd.DataFrame, id_col: str = "concept_key") -> dict:
    out = {}

    for scale_type, g in df.groupby("scale_type", dropna=False):
        pre_ids = set(
            g.loc[g["survey_phase"] == "PRE", id_col].dropna().unique()
        ) if id_col in g.columns else set(
            g.loc[g["survey_phase"] == "PRE", "question_text"].dropna().unique()
        )

        post_ids = set(
            g.loc[g["survey_phase"] == "POST", id_col].dropna().unique()
        ) if id_col in g.columns else set(
            g.loc[g["survey_phase"] == "POST", "question_text"].dropna().unique()
        )

        out[str(scale_type)] = len(pre_ids & post_ids)

    return out


def get_post_only_questions_by_type(df: pd.DataFrame, id_col: str = "concept_key") -> dict:
    out = {}

    for scale_type, g in df.groupby("scale_type", dropna=False):
        pre_ids = set(
            g.loc[g["survey_phase"] == "PRE", id_col].dropna().unique()
        ) if id_col in g.columns else set(
            g.loc[g["survey_phase"] == "PRE", "question_text"].dropna().unique()
        )

        post_ids = set(
            g.loc[g["survey_phase"] == "POST", id_col].dropna().unique()
        ) if id_col in g.columns else set(
            g.loc[g["survey_phase"] == "POST", "question_text"].dropna().unique()
        )

        out[str(scale_type)] = len(post_ids - pre_ids)

    return out


def compute_fact_cards(df: pd.DataFrame) -> dict:
    """
    Return a dict of key metrics for the chosen survey_type slice.

    Keys
    ----
    pre_observations      int
    post_observations     int
    common_questions      int  – Likert questions in both PRE and POST
    post_only_questions   int  – Likert questions in POST only
    n_schools             int
    pre_sessions          int  – unique survey_session_id where phase == PRE
    post_sessions         int  – unique survey_session_id where phase == POST
    """
    pre = df[df["survey_phase"] == "PRE"].copy()
    post = df[df["survey_phase"] == "POST"].copy()

    question_counts = get_question_counts_by_phase(df, id_col="concept_key")
    common_by_type = get_common_pre_post_questions_by_type(df, id_col="concept_key")
    post_only_by_type = get_post_only_questions_by_type(df, id_col="concept_key")

    pre_sessions = pre["survey_session_id"].nunique() if "survey_session_id" in df.columns else 0
    post_sessions = post["survey_session_id"].nunique() if "survey_session_id" in df.columns else 0

    return {
        "pre_observations": pre["response_id"].nunique() if "response_id" in df.columns else len(pre),
        "post_observations": post["response_id"].nunique() if "response_id" in df.columns else len(post),
        "pre_questions_total": question_counts["pre_questions_total"],
        "post_questions_total": question_counts["post_questions_total"],
        "common_questions_by_type": common_by_type,
        "post_only_questions_by_type": post_only_by_type,
        "n_schools": df["school_id"].nunique() if "school_id" in df.columns else 0,
        "pre_sessions": pre_sessions,
        "post_sessions": post_sessions,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6.  STATISTICAL TESTS
# ══════════════════════════════════════════════════════════════════════════════

def run_mann_whitney(
    df: pd.DataFrame,
    common_questions: list[str],
    question_col: str = "question_text",
    phase_col: str = "survey_phase",
    response_col: str = "response_encoded",
) -> pd.DataFrame:
    """
    Mann-Whitney U test (one-sided: POST > PRE) for each common Likert question.
    Applies Benjamini-Hochberg FDR correction.

    Returns a DataFrame with columns:
        question, n_pre, n_post, pre_median, post_median,
        median_shift, U_stat, p_value, p_adj_BH, significant
    """
    if not STATS_AVAILABLE:
        raise ImportError("scipy and statsmodels are required for statistical tests.")

    df = df.copy()
    df[response_col] = pd.to_numeric(df[response_col], errors="coerce")
    df = df.dropna(subset=[response_col])
    df = df[df[question_col].isin(common_questions)]

    results = []
    for q in common_questions:
        tmp  = df[df[question_col] == q]
        pre  = tmp[tmp[phase_col] == "PRE"][response_col]
        post = tmp[tmp[phase_col] == "POST"][response_col]
        if len(pre) == 0 or len(post) == 0:
            continue
        stat, p = mannwhitneyu(pre, post, alternative="less")
        results.append({
            "question":     q,
            "n_pre":        len(pre),
            "n_post":       len(post),
            "pre_median":   round(pre.median(),  2),
            "post_median":  round(post.median(), 2),
            "median_shift": round(post.median() - pre.median(), 2),
            "U_stat":       round(stat, 2),
            "p_value":      round(p, 4),
        })

    results_df = pd.DataFrame(results)
    if results_df.empty:
        return results_df

    reject, p_adj, _, _ = multipletests(results_df["p_value"], method="fdr_bh")
    results_df["p_adj_BH"]    = p_adj.round(4)
    results_df["significant"] = reject
    return results_df.sort_values("p_value")


def run_wilcoxon_vs_neutral(
    df_post: pd.DataFrame,
    post_only_questions: list[str],
    midpoint: float = 3.0,
    question_col: str = "question_text",
    response_col: str = "response_encoded",
    scale_type_col: str = "scale_type",
) -> pd.DataFrame:
    """
    One-sample Wilcoxon signed-rank test: H1 median > midpoint (3).
    Applied to POST-only Likert questions.
    Applies BH correction.
    """
    if not STATS_AVAILABLE:
        raise ImportError("scipy and statsmodels are required for statistical tests.")

    likert = df_post[
        df_post[scale_type_col].isin(["Likert Scale Numeric", "Likert Scale Text"])
    ].copy()
    likert[response_col] = pd.to_numeric(likert[response_col], errors="coerce")
    likert = likert.dropna(subset=[response_col])

    results = []
    for q in post_only_questions:
        sub       = likert[likert[question_col] == q]
        responses = sub[response_col]
        scale     = sub[scale_type_col].iloc[0] if not sub.empty else "Unknown"
        if len(responses) == 0:
            continue

        diffs   = responses - midpoint
        nonzero = diffs[diffs != 0]

        base = {
            "question":   q,
            "scale_type": scale,
            "n":          len(responses),
            "n_nonzero":  len(nonzero),
            "median":     round(responses.median(), 2),
            "mean":       round(responses.mean(),   2),
            "pct_above":  round((responses > midpoint).mean() * 100, 1),
        }

        if len(nonzero) < 10:
            results.append({**base, "W_stat": None, "p_value": None,
                             "significant": None,
                             "note": f"Skipped – only {len(nonzero)} non-tied obs"})
            continue

        stat, p = scipy_wilcoxon(nonzero, alternative="greater")
        results.append({**base, "W_stat": round(stat, 2), "p_value": round(p, 4),
                        "significant": None, "note": ""})

    results_df = pd.DataFrame(results)
    if results_df.empty:
        return results_df

    testable = results_df["p_value"].notna()
    if testable.sum() > 0:
        reject, p_adj, _, _ = multipletests(results_df.loc[testable, "p_value"], method="fdr_bh")
        results_df.loc[testable, "p_adj_BH"]    = p_adj.round(4)
        results_df.loc[testable, "significant"] = reject
    return results_df.sort_values("p_value")


# ══════════════════════════════════════════════════════════════════════════════
# 7.  COLOUR HELPERS
# ══════════════════════════════════════════════════════════════════════════════

LIKERT_COLORS = {
    1: "#b2182b",
    2: "#ef8a62",
    3: "#bdbdbd",
    4: "#a6dba0",
    5: "#1b7837",
}
LIKERT_TEXT_COLORS = {
    "Strongly Disagree":          "#b2182b",
    "Disagree":                   "#ef8a62",
    "Neither Agree or Disagree":  "#bdbdbd",
    "Agree":                      "#a6dba0",
    "Strongly Agree":             "#1b7837",
}
LIKERT_TEXT_ORDER = [
    "Strongly Disagree",
    "Disagree",
    "Neither Agree or Disagree",
    "Agree",
    "Strongly Agree",
]

PRE_COLOR  = "#378ADD"
POST_COLOR = "#1D9E75"
SIG_COLOR  = "#1D9E75"
NS_COLOR   = "#B4B2A9"


def _get_categorical_color(response_text: str) -> str:
    """Map categorical response text to a Likert-style colour heuristic."""
    r = str(response_text).lower().strip()

    def has_word(w):
        return bool(re.search(r"\b" + re.escape(w) + r"\b", r))

    if "still unsure" in r:
        return "#fdb863"
    if any(w in r for w in ["always", "often", "very familiar", "advanced", "yes"]):
        if any(w in r for w in ["strongly", "very", "advanced", "always"]):
            return "#1b7837"
        return "#a6dba0"
    if has_word("no") or has_word("never") or has_word("rarely") or "new to this topic" in r:
        if "strongly" in r or "never" in r:
            return "#b2182b"
        return "#ef8a62"
    if "not sure" in r or "unsure" in r or "sometimes" in r:
        return "#bdbdbd"

    fallback = ["#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
    h = int(hashlib.md5(str(response_text).encode()).hexdigest(), 16)
    return fallback[h % len(fallback)]


def _wrap(text: str, width: int = 55) -> str:
    return "<br>".join(textwrap.wrap(str(text), width))


# ══════════════════════════════════════════════════════════════════════════════
# 8.  STATISTICAL VISUALISATION  (Tab 1 – Statistical Analysis)
# ══════════════════════════════════════════════════════════════════════════════

def plot_mann_whitney_shift(
    results_df: pd.DataFrame,
    survey_label: str = "",
) -> go.Figure:
    """
    Two-panel figure matching MEN_v1_Mann_Whitney_test.png exactly:
      LEFT  : horizontal diverging bar  — POST−PRE median shift
              bar colour encodes phase × significance (4 categories)
      RIGHT : connected dot plot        — PRE circle + POST diamond
              line/dot colour encodes significance
      LEGEND: 4-item horizontal bar at bottom (PRE sig, POST sig, PRE ns, POST ns)
    """
    if results_df.empty:
        return go.Figure().update_layout(title="No data available")

    # ── colours matching the target image ────────────────────────────────
    C_PRE_SIG  = "#1a5fa8"   # dark blue  – PRE significant
    C_POST_SIG = "#1D9E75"   # dark teal  – POST significant
    C_PRE_NS   = "#92b8dc"   # light blue – PRE not significant
    C_POST_NS  = "#9FE1CB"   # light teal – POST not significant

    def bar_color(shift, sig):
        if sig:
            return C_POST_SIG if shift >= 0 else C_PRE_SIG
        else:
            return C_POST_NS  if shift >= 0 else C_PRE_NS

    def dot_color(sig, phase):
        if phase == "PRE":
            return C_PRE_SIG  if sig else C_PRE_NS
        else:
            return C_POST_SIG if sig else C_POST_NS

    plot_df = results_df.sort_values("pre_median").reset_index(drop=True)

    # Wrap labels — same text used on BOTH y-axes so rows align perfectly
    labels = [_wrap(q, 28) for q in plot_df["question"]]

    # ── subplots: shared y-axis categories keep rows aligned ─────────────
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Median shift per question", "PRE vs POST medians"],
        horizontal_spacing=0.20,
        column_widths=[0.45, 0.55],
    )

    # ── LEFT: one Bar trace per row so each bar gets its own colour ───────
    for i, (_, r) in enumerate(plot_df.iterrows()):
        sig   = bool(r["significant"])
        shift = float(r["median_shift"])
        color = bar_color(shift, sig)
        fig.add_trace(
            go.Bar(
                y=[labels[i]],
                x=[shift],
                orientation="h",
                marker_color=color,
                showlegend=False,
                hovertemplate=f"<b>{labels[i]}</b><br>Shift: {shift:+.2f}<extra></extra>",
            ),
            row=1, col=1,
        )

    # ★ annotations — placed just beyond the bar tip, never inside
    for i, (_, r) in enumerate(plot_df.iterrows()):
        if bool(r["significant"]):
            shift = float(r["median_shift"])
            # offset 0.05 units beyond the bar end
            ax_offset = 0.07 if shift >= 0 else -0.07
            fig.add_annotation(
                x=shift + ax_offset,
                y=labels[i],
                text="★",
                showarrow=False,
                font=dict(size=12, color="#333"),
                xref="x", yref="y",
                xanchor="left" if shift >= 0 else "right",
            )

    # zero reference line
    fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="#888888",
                  row=1, col=1)

    # ── RIGHT: connector lines ────────────────────────────────────────────
    for i, (_, r) in enumerate(plot_df.iterrows()):
        sig   = bool(r["significant"])
        color = C_POST_SIG if sig else C_PRE_NS
        dash  = "solid" if sig else "dot"
        fig.add_trace(
            go.Scatter(
                x=[r["pre_median"], r["post_median"]],
                y=[labels[i], labels[i]],
                mode="lines",
                line=dict(color=color, width=2, dash=dash),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1, col=2,
        )

    # ── RIGHT: PRE dots (circle) — 4 legend entries via legendgroup ──────
    # We emit 4 "dummy" legend traces first, then the real data without legend
    legend_traces = [
        ("PRE (significant)",     C_PRE_SIG,  "circle",  True),
        ("POST (significant)",    C_POST_SIG, "diamond", True),
        ("PRE (not significant)", C_PRE_NS,   "circle",  False),
        ("POST (not significant)",C_POST_NS,  "diamond", False),
    ]
    for name, color, symbol, _sig in legend_traces:
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode="markers",
                name=name,
                marker=dict(size=10, color=color, symbol=symbol,
                            opacity=1.0 if _sig else 0.5),
                legendgroup=name,
            ),
            row=1, col=2,
        )

    # Real PRE dots
    for i, (_, r) in enumerate(plot_df.iterrows()):
        sig   = bool(r["significant"])
        color = C_PRE_SIG if sig else C_PRE_NS
        lgrp  = "PRE (significant)" if sig else "PRE (not significant)"
        fig.add_trace(
            go.Scatter(
                x=[r["pre_median"]],
                y=[labels[i]],
                mode="markers",
                marker=dict(size=10, color=color, symbol="circle",
                            opacity=1.0 if sig else 0.5),
                showlegend=False,
                legendgroup=lgrp,
                hovertemplate=f"<b>{labels[i]}</b><br>PRE median: {r['pre_median']}<extra></extra>",
            ),
            row=1, col=2,
        )

    # Real POST dots (diamond)
    for i, (_, r) in enumerate(plot_df.iterrows()):
        sig   = bool(r["significant"])
        color = C_POST_SIG if sig else C_POST_NS
        lgrp  = "POST (significant)" if sig else "POST (not significant)"
        fig.add_trace(
            go.Scatter(
                x=[r["post_median"]],
                y=[labels[i]],
                mode="markers",
                marker=dict(size=10, color=color, symbol="diamond"),
                showlegend=False,
                legendgroup=lgrp,
                hovertemplate=f"<b>{labels[i]}</b><br>POST median: {r['post_median']}<extra></extra>",
            ),
            row=1, col=2,
        )

    # ── Layout ────────────────────────────────────────────────────────────
    h = max(500, len(plot_df) * 80 + 180)

    fig.update_layout(
        title=dict(
            text=f"{survey_label}<br>Where do participants show signals of change?",
            x=0.5,
            font=dict(size=14, color="#333"),
        ),
        height=h,
        barmode="overlay",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=20, r=60, t=120, b=160),   # b=160 guarantees legend fits
        legend=dict(
            orientation="h",
            x=0.5, xanchor="center",
            y=-0.15, yanchor="top",               # sits below x-axis labels
            font=dict(size=11),
            traceorder="normal",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#dee2e6",
            borderwidth=1,
        ),
    )

    # Left panel axes
    fig.update_xaxes(
        title_text="POST − PRE (median)",
        zeroline=False,
        showgrid=True,
        gridcolor="rgba(200,200,200,0.4)",
        row=1, col=1,
    )
    fig.update_yaxes(
        tickfont=dict(size=10),
        automargin=True,
        ticklabelposition="outside left",
        row=1, col=1,
    )

    # Right panel axes
    fig.update_xaxes(
        title_text="Median response (1–5 scale)",
        range=[0.5, 5.5],
        tickvals=[1, 2, 3, 4, 5],
        showgrid=True,
        gridcolor="rgba(200,200,200,0.4)",
        row=1, col=2,
    )
    fig.update_yaxes(
        tickfont=dict(size=10),
        automargin=True,
        ticklabelposition="outside left",
        row=1, col=2,
    )

    return fig


def plot_wilcoxon_neutral(
    wilcoxon_results: pd.DataFrame,
    df_post: pd.DataFrame,
    question_col: str = "question_text",
    response_col: str = "response_encoded",
) -> go.Figure:
    """
    Dot plot: POST-only questions vs neutral midpoint (3).
    Median dot + IQR bar, coloured by significance.
    """
    if wilcoxon_results.empty:
        return go.Figure().update_layout(title="No POST-only Wilcoxon data")

    df = df_post.copy()
    df[response_col] = pd.to_numeric(df[response_col], errors="coerce")
    df = df.dropna(subset=[response_col])

    stats = []
    for _, row in wilcoxon_results.iterrows():
        q   = row["question"]
        sub = df[df[question_col] == q][response_col]
        if len(sub) == 0:
            continue
        stats.append({
            "question":    q,
            "median":      sub.median(),
            "q1":          sub.quantile(0.25),
            "q3":          sub.quantile(0.75),
            "n":           int(row["n"]),
            "pct_above":   float(row["pct_above"]),
            "significant": bool(row["significant"]) if pd.notna(row.get("significant")) else False,
        })

    plot_df = pd.DataFrame(stats).sort_values("median").reset_index(drop=True)
    labels  = [_wrap(q, 55) for q in plot_df["question"]]

    fig = go.Figure()

    # IQR bars
    for i, r in plot_df.iterrows():
        color = SIG_COLOR if r["significant"] else NS_COLOR
        fig.add_trace(go.Scatter(
            x=[r["median"]],
            y=[labels[i]],
            mode="markers",
            marker=dict(symbol="line-ew", size=18, color=color,
                        line=dict(color=color, width=3)),
            error_x=dict(
                type="data", symmetric=False,
                array=[r["q3"] - r["median"]],
                arrayminus=[r["median"] - r["q1"]],
                color=color, thickness=3, width=0,
            ),
            showlegend=False, hoverinfo="skip",
        ))

    # Median dots
    for sig_val, group in plot_df.groupby("significant", sort=False):
        idx    = group.index.tolist()
        color  = SIG_COLOR if sig_val else NS_COLOR
        border = "#0F6E56"  if sig_val else "#888780"
        label  = "Significant (p_adj < 0.05)" if sig_val else "Not significant"
        fig.add_trace(go.Scatter(
            x=group["median"],
            y=[labels[i] for i in idx],
            mode="markers+text",
            name=label,
            marker=dict(size=12, color=color,
                        line=dict(color=border, width=2)),
            text=[f"  {r['pct_above']:.0f}% above neutral" for _, r in group.iterrows()],
            textposition="middle right",
            textfont=dict(size=10, color=color),
        ))
        # star for significant
        if sig_val:
            fig.add_trace(go.Scatter(
                x=group["median"],
                y=[labels[i] for i in idx],
                mode="markers",
                marker=dict(symbol="star", size=14, color="#FFD700",
                            line=dict(color=border, width=1)),
                showlegend=False, hoverinfo="skip",
            ))

    fig.add_vline(x=3, line_dash="dash", line_color="#888", line_width=1)
    fig.add_annotation(x=3, y=1.02, yref="paper", text="Neutral (3)",
                       showarrow=False, font=dict(size=10, color="#888"))

    h = max(400, len(plot_df) * 55)
    fig.update_layout(
        title=dict(text="POST-only questions — are responses above neutral?",
                   x=0.5, font=dict(size=14)),
        xaxis=dict(title="Median (1–5 scale)", range=[0.5, 5.5],
                   tickvals=[1, 2, 3, 4, 5]),
        height=h,
        margin=dict(l=40, r=200, t=80, b=60),
        legend=dict(orientation="h", y=-0.12),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 9.  SESSION-LEVEL VISUALISATION  (Tab 2 – Cross-Session)
# ══════════════════════════════════════════════════════════════════════════════

def _stacked_bar_pie_layout(num_sessions: int, wrapped_title: str, max_side: int = 10):
    """Return (fig, bar_row, bar_col, pie_row, pie_col) for a stacked-bar+pie layout."""
    if num_sessions <= max_side:
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"secondary_y": True}, {"type": "domain"}]],
            column_widths=[0.60, 0.40],
            horizontal_spacing=0.05,
            subplot_titles=[wrapped_title, ""],
        )
        return fig, 1, 1, 1, 2
    else:
        fig = make_subplots(
            rows=2, cols=1,
            specs=[[{"secondary_y": True}], [{"type": "domain"}]],
            vertical_spacing=0.20,
            subplot_titles=[wrapped_title, ""],
        )
        return fig, 1, 1, 2, 1


def plot_pre_post_bar_with_mean(
    df: pd.DataFrame,
    common_questions: list[str],
    question_col: str = "question_text",
    phase_col: str = "survey_phase",
    response_col: str = "response_encoded",
    session_col: str = "survey_session_id",
) -> list[go.Figure]:
    """
    For each common PRE/POST question:
      Stacked bars per session (PRE sessions first, then POST),
      mean line on secondary y-axis, overall pie.
    """
    df = df.copy()
    df[response_col] = pd.to_numeric(df[response_col], errors="coerce")
    df = df.dropna(subset=[response_col])
    df[response_col] = df[response_col].astype(int)
    df = df[df[question_col].isin(common_questions)]

    figs = []
    for q in common_questions:
        sub = df[df[question_col] == q].copy()
        if sub.empty:
            continue

        pre_sessions  = sorted(sub[sub[phase_col] == "PRE" ][session_col].unique())
        post_sessions = sorted(sub[sub[phase_col] == "POST"][session_col].unique())
        all_sessions  = pre_sessions + post_sessions

        counts = (
            sub.groupby([session_col, phase_col, response_col])
            .size().reset_index(name="count")
        )
        means = (
            sub.groupby([session_col, phase_col])[response_col]
            .agg(mean="mean", n="count").reset_index()
        )
        pivot = (
            counts.pivot_table(index=session_col, columns=response_col,
                               values="count", aggfunc="sum")
            .reindex(index=all_sessions, columns=[1, 2, 3, 4, 5], fill_value=0)
            .fillna(0)
        )
        total_series = (
            sub[response_col].value_counts()
            .reindex([1, 2, 3, 4, 5], fill_value=0)
        )

        wrapped = _wrap(q, 70)
        fig, br, bc, pr, pc = _stacked_bar_pie_layout(len(all_sessions), wrapped)

        # stacked bars
        for rating in [1, 2, 3, 4, 5]:
            vals = [pivot.loc[s, rating] if s in pivot.index else 0 for s in all_sessions]
            bar_colors_list = [
                PRE_COLOR if s in pre_sessions else POST_COLOR
                for s in all_sessions
            ]
            fig.add_trace(
                go.Bar(
                    name=str(rating),
                    x=all_sessions,
                    y=vals,
                    marker=dict(
                        color=LIKERT_COLORS[rating],
                        line=dict(
                            color=bar_colors_list,
                            width=3,
                        ),
                    ),
                    text=vals,
                    textposition="inside",
                ),
                row=br, col=bc, secondary_y=False,
            )

        # mean line
        means_sorted = means.set_index(session_col).reindex(all_sessions)
        fig.add_trace(
            go.Scatter(
                x=all_sessions,
                y=means_sorted["mean"],
                mode="lines+markers",
                name="Mean",
                line=dict(color="#333", width=2),
                marker=dict(size=6),
                yaxis="y2",
            ),
            row=br, col=bc, secondary_y=True,
        )

        # vertical divider PRE / POST
        if pre_sessions and post_sessions:
            fig.add_vline(
                x=len(pre_sessions) - 0.5,
                line_dash="dash", line_color="#666", line_width=1.5,
                row=br, col=bc,
            )

        # pie
        fig.add_trace(
            go.Pie(
                labels=[str(r) for r in [1, 2, 3, 4, 5]],
                values=total_series.values,
                marker=dict(colors=[LIKERT_COLORS[r] for r in [1, 2, 3, 4, 5]]),
                textinfo="percent",
                hole=0.45,
                sort=False,
                showlegend=False,
            ),
            row=pr, col=pc,
        )

        h = max(500, len(all_sessions) * 35 + 200)
        fig.update_layout(
            barmode="stack",
            height=h,
            margin=dict(l=40, r=40, t=140, b=80),
            legend=dict(orientation="h", y=1.15, x=0.5, xanchor="center"),
        )
        fig.update_xaxes(tickangle=45, automargin=True, row=br, col=bc)
        fig.update_yaxes(title="Responses", row=br, col=bc, secondary_y=False)
        fig.update_yaxes(title="Mean (1–5)", range=[0.5, 5.5],
                         tickvals=[1, 2, 3, 4, 5], row=br, col=bc, secondary_y=True)
        figs.append(fig)

    return figs


def plot_post_bar_with_mean(
    df_post: pd.DataFrame,
    post_only_questions: list[str],
    question_col: str = "question_text",
    response_col: str = "response_encoded",
    session_col: str = "survey_session_id",
) -> list[go.Figure]:
    """
    For each POST-only question:
      Stacked bars per POST session + mean line + overall pie.
    """
    df = df_post.copy()
    df[response_col] = pd.to_numeric(df[response_col], errors="coerce")
    df = df.dropna(subset=[response_col])
    df[response_col] = df[response_col].astype(int)
    df = df[df[question_col].isin(post_only_questions)]

    figs = []
    for q in post_only_questions:
        sub = df[df[question_col] == q].copy()
        if sub.empty:
            continue

        all_sessions = sorted(sub[session_col].unique())
        counts = (
            sub.groupby([session_col, response_col])
            .size().reset_index(name="count")
        )
        means = (
            sub.groupby(session_col)[response_col]
            .agg(mean="mean", n="count").reset_index().sort_values(session_col)
        )
        pivot = (
            counts.pivot_table(index=session_col, columns=response_col,
                               values="count", aggfunc="sum")
            .reindex(index=all_sessions, columns=[1, 2, 3, 4, 5], fill_value=0)
            .fillna(0)
        )
        total_series = (
            sub[response_col].value_counts()
            .reindex([1, 2, 3, 4, 5], fill_value=0)
        )

        wrapped = _wrap(q, 70)
        fig, br, bc, pr, pc = _stacked_bar_pie_layout(len(all_sessions), wrapped)

        for rating in [1, 2, 3, 4, 5]:
            vals = [pivot.loc[s, rating] if s in pivot.index else 0 for s in all_sessions]
            fig.add_trace(
                go.Bar(
                    name=str(rating),
                    x=all_sessions,
                    y=vals,
                    marker=dict(color=LIKERT_COLORS[rating],
                                line=dict(color=POST_COLOR, width=2)),
                    text=vals,
                    textposition="inside",
                ),
                row=br, col=bc, secondary_y=False,
            )

        means_sorted = means.set_index(session_col).reindex(all_sessions)
        fig.add_trace(
            go.Scatter(
                x=all_sessions,
                y=means_sorted["mean"],
                mode="lines+markers",
                name="Mean",
                line=dict(color="#333", width=2),
                marker=dict(size=6),
                yaxis="y2",
            ),
            row=br, col=bc, secondary_y=True,
        )

        fig.add_trace(
            go.Pie(
                labels=[str(r) for r in [1, 2, 3, 4, 5]],
                values=total_series.values,
                marker=dict(colors=[LIKERT_COLORS[r] for r in [1, 2, 3, 4, 5]]),
                textinfo="percent",
                hole=0.45,
                sort=False,
                showlegend=False,
            ),
            row=pr, col=pc,
        )

        h = max(500, len(all_sessions) * 35 + 200)
        fig.update_layout(
            barmode="stack",
            height=h,
            margin=dict(l=40, r=40, t=140, b=80),
            legend=dict(orientation="h", y=1.15, x=0.5, xanchor="center"),
        )
        fig.update_xaxes(tickangle=45, automargin=True, row=br, col=bc)
        fig.update_yaxes(title="Responses", row=br, col=bc, secondary_y=False)
        fig.update_yaxes(title="Mean (1–5)", range=[0.5, 5.5],
                         tickvals=[1, 2, 3, 4, 5], row=br, col=bc, secondary_y=True)
        figs.append(fig)

    return figs


def plot_likert_by_session(
    df: pd.DataFrame,
    likert_kind: str = "text",          # "text" | "numeric"
    question_text_col: str = "question_text",
    session_col: str = "survey_session_id",
    title: str | None = None,
) -> list[go.Figure]:
    """
    Stacked bar + pie per concept_key, grouped by survey_session_id.
    Mirrors the original data_utils QuestionTypeAccessor plots.
    """
    if likert_kind == "text":
        scale_filter = "Likert Scale Text"
        response_col = "response"
        order  = LIKERT_TEXT_ORDER
        colors = LIKERT_TEXT_COLORS
        if title is None:
            title = "Likert Scale Text — by Session"
    else:
        scale_filter = "Likert Scale Numeric"
        response_col = "response_encoded"
        order  = [1, 2, 3, 4, 5]
        colors = LIKERT_COLORS
        if title is None:
            title = "Likert Scale Numeric — by Session"

    df_sub = df[df["scale_type"] == scale_filter].copy()
    if response_col == "response_encoded":
        df_sub[response_col] = pd.to_numeric(df_sub[response_col], errors="coerce")
        df_sub = df_sub.dropna(subset=[response_col])
        df_sub[response_col] = df_sub[response_col].astype(int)

    if df_sub.empty:
        return []

    concept_labels = (
        df_sub[["concept_key", question_text_col]]
        .dropna().drop_duplicates()
        .set_index("concept_key")[question_text_col].to_dict()
        if question_text_col in df_sub.columns else {}
    )

    figs = []
    for concept in df_sub["concept_key"].dropna().unique():
        sub = df_sub[df_sub["concept_key"] == concept]

        counts_ws = (
            sub.groupby([session_col, response_col])
            .size().reset_index(name="count")
        )
        counts_total = (
            sub.groupby(response_col).size().reset_index(name="count")
        )

        pivot_ws = (
            counts_ws.pivot(index=session_col, columns=response_col, values="count")
            .fillna(0).reindex(columns=order, fill_value=0).sort_index()
        )
        total_series = (
            counts_total.set_index(response_col)["count"]
            .reindex(order, fill_value=0)
        )

        sessions  = pivot_ws.index.tolist()
        num_ws    = len(sessions)
        label_str = concept_labels.get(concept, str(concept))
        wrapped   = _wrap(f"{concept}: {label_str}", 160)

        if num_ws <= 10:
            fig = make_subplots(
                rows=1, cols=2,
                specs=[[{"type": "bar"}, {"type": "domain"}]],
                column_widths=[0.55, 0.45],
                subplot_titles=[wrapped, ""],
                horizontal_spacing=0.05,
            )
            br, pr, pc = 1, 1, 2
        else:
            fig = make_subplots(
                rows=2, cols=1,
                specs=[[{"type": "bar"}], [{"type": "domain"}]],
                subplot_titles=[wrapped, ""],
                vertical_spacing=0.20,
            )
            br, pr, pc = 1, 2, 1

        for resp in order:
            vals = pivot_ws[resp]
            fig.add_trace(
                go.Bar(
                    x=sessions, y=vals,
                    marker_color=colors[resp],
                    name=str(resp),
                    text=vals, textposition="inside",
                ),
                row=br, col=1,
            )

        totals = pivot_ws.sum(axis=1)
        fig.add_trace(
            go.Scatter(x=sessions, y=totals, mode="text", text=totals,
                       textposition="top center", showlegend=False),
            row=br, col=1,
        )

        fig.add_trace(
            go.Pie(
                labels=[str(x) for x in order],
                values=total_series.values,
                marker=dict(colors=[colors[x] for x in order]),
                textinfo="percent", hole=0.45, sort=False, showlegend=False,
            ),
            row=pr, col=pc,
        )

        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=16), pad=dict(t=100, b=10)),
            barmode="stack",
            height=600 if num_ws <= 10 else 800,
            margin=dict(l=40, r=40, t=180, b=40),
        )
        fig.update_xaxes(tickangle=45, automargin=True, row=br, col=1)
        fig.update_yaxes(title="Responses", showgrid=True,
                         gridcolor="rgba(200,200,200,0.3)", row=br, col=1)
        figs.append(fig)

    return figs


def plot_categorical_by_session(
    df: pd.DataFrame,
    question_text_col: str = "question_text",
    session_col: str = "survey_session_id",
    title: str | None = None,
) -> list[go.Figure]:
    """Categorical questions — stacked bar + pie by survey_session_id."""
    if title is None:
        title = "Categorical Questions — by Session"

    df_sub = df[df["scale_type"] == "Categorical"].copy()
    if df_sub.empty:
        return []

    concept_labels = (
        df_sub[["concept_key", question_text_col]]
        .dropna().drop_duplicates()
        .set_index("concept_key")[question_text_col].to_dict()
        if question_text_col in df_sub.columns else {}
    )

    figs = []
    for concept in df_sub["concept_key"].dropna().unique():
        tmp = df_sub[df_sub["concept_key"] == concept].copy()

        response_order = None
        if "possible_responses" in tmp.columns and tmp["possible_responses"].notna().any():
            pr_str = str(tmp["possible_responses"].dropna().iloc[0])
            try:
                pr = ast.literal_eval(pr_str)
                if isinstance(pr, (list, tuple)):
                    response_order = [str(x) for x in pr]
            except Exception:
                pass
        if response_order is None:
            response_order = sorted(tmp["response"].dropna().astype(str).unique())

        counts_ws = (
            tmp.groupby([session_col, "response"]).size().reset_index(name="count")
        )
        counts_total = (
            tmp.groupby("response").size().reset_index(name="count")
        )

        pivot_ws = (
            counts_ws.pivot(index=session_col, columns="response", values="count")
            .fillna(0).reindex(columns=response_order, fill_value=0).sort_index()
        )
        total_series = (
            counts_total.set_index("response")["count"]
            .reindex(response_order, fill_value=0)
        )

        sessions = pivot_ws.index.tolist()
        num_ws   = len(sessions)
        label_str = concept_labels.get(concept, str(concept))
        wrapped   = _wrap(f"{concept}: {label_str}", 160)
        resp_colors = {r: _get_categorical_color(r) for r in response_order}

        if num_ws <= 10:
            fig = make_subplots(
                rows=1, cols=2,
                specs=[[{"type": "bar"}, {"type": "domain"}]],
                column_widths=[0.55, 0.45],
                subplot_titles=[wrapped, ""],
                horizontal_spacing=0.05,
            )
            br, pr, pc = 1, 1, 2
        else:
            fig = make_subplots(
                rows=2, cols=1,
                specs=[[{"type": "bar"}], [{"type": "domain"}]],
                subplot_titles=[wrapped, ""],
                vertical_spacing=0.20,
            )
            br, pr, pc = 1, 2, 1

        for resp in response_order:
            vals = pivot_ws[resp]
            fig.add_trace(
                go.Bar(
                    x=sessions, y=vals,
                    marker_color=resp_colors[resp],
                    name=str(resp),
                    text=vals, textposition="inside",
                ),
                row=br, col=1,
            )

        totals = pivot_ws.sum(axis=1)
        fig.add_trace(
            go.Scatter(x=sessions, y=totals, mode="text", text=totals,
                       textposition="top center", showlegend=False),
            row=br, col=1,
        )
        fig.add_trace(
            go.Pie(
                labels=[str(x) for x in response_order],
                values=total_series.values,
                marker=dict(colors=[resp_colors[x] for x in response_order]),
                textinfo="percent", hole=0.45, sort=False, showlegend=False,
            ),
            row=pr, col=pc,
        )

        fig.update_layout(
            title=dict(text=title, x=0.5, font=dict(size=16), pad=dict(t=100, b=10)),
            barmode="stack",
            height=600 if num_ws <= 10 else 800,
            margin=dict(l=40, r=40, t=180, b=40),
            legend=dict(orientation="h", y=1.31, x=0.5, xanchor="center", yanchor="bottom"),
        )
        fig.update_xaxes(tickangle=45, automargin=True, row=br, col=1)
        fig.update_yaxes(title="Responses", showgrid=True,
                         gridcolor="rgba(200,200,200,0.3)", row=br, col=1)
        figs.append(fig)

    return figs


# ══════════════════════════════════════════════════════════════════════════════
# 10.  SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════

SUMMARY_GROUPBY_OPTIONS = [
    "survey_type", "survey_phase", "survey_session_id",
    "school_id", "scale_type", "date", "hour",
]
SUMMARY_METRIC_OPTIONS = {
    "num_questions":    ("concept_key",  "nunique"),
    "num_observations": ("response_id",  "nunique"),
}
SUMMARY_METRIC_LABELS = {
    "num_questions":    "Number of questions (concept_key)",
    "num_observations": "Number of observations (response_id)",
}


def generate_summary(
    df: pd.DataFrame,
    groupby_columns: list[str] | None = None,
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """Dynamic summary table with optional grouping."""
    if groupby_columns is None:
        groupby_columns = []
    if metrics is None or len(metrics) == 0:
        metrics = list(SUMMARY_METRIC_OPTIONS.keys())

    agg_dict = {
        name: (SUMMARY_METRIC_OPTIONS[name][0], SUMMARY_METRIC_OPTIONS[name][1])
        for name in metrics
    }
    if not groupby_columns:
        return pd.DataFrame({
            name: [getattr(df[col], agg)()]
            for name, (col, agg) in agg_dict.items()
        })

    return (
        df.groupby(groupby_columns, dropna=False)
        .agg(**{name: (col, agg) for name, (col, agg) in agg_dict.items()})
        .reset_index()
    )



# comparing surveys
import plotly.express as px
import plotly.graph_objects as go

def compute_compare_kpis(df_compare: pd.DataFrame) -> dict:
    """
    KPIs for cross-survey comparison.

    Required columns:
      - survey_type
      - survey_session_id (unique workshop/session)
      - school_id
      - response_id (unique response)
    """
    dfc = df_compare.copy()
    return {
        "total_survey_types": int(dfc["survey_type"].dropna().nunique()) if "survey_type" in dfc.columns else 0,
        "total_workshops": int(dfc["survey_session_id"].dropna().nunique()) if "survey_session_id" in dfc.columns else 0,
        "total_schools": int(dfc["school_id"].dropna().nunique()) if "school_id" in dfc.columns else 0,
        "total_responses": int(dfc["response_id"].dropna().nunique()) if "response_id" in dfc.columns else len(dfc),
    }


def plot_sessions_by_survey_type_phase(df: pd.DataFrame) -> go.Figure:
    """
    Bar chart:
    unique workshops/sessions by survey_type, split by PRE/POST.
    """
    tmp = (
        df.dropna(subset=["survey_type", "survey_phase", "survey_session_id"])
          .groupby(["survey_type", "survey_phase"])["survey_session_id"]
          .nunique()
          .reset_index(name="n_sessions")
    )

    if tmp.empty:
        return go.Figure().update_layout(title="No data available")

    fig = px.bar(
        tmp,
        x="survey_type",
        y="n_sessions",
        color="survey_phase",
        barmode="group",
        category_orders={"survey_phase": ["PRE", "POST"]},
        title="Number of workshops by survey type and phase",
        labels={
            "survey_type": "Survey Type",
            "n_sessions": "Number of workshops",
            "survey_phase": "Survey Phase",
        },
    )

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=40, t=70, b=50),
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
    )
    fig.update_yaxes(showgrid=True, gridcolor="rgba(200,200,200,0.3)")
    return fig


def plot_sessions_by_school_phase(df: pd.DataFrame) -> go.Figure:
    """
    Grouped bar chart:
    unique workshops/sessions by school_id, split by PRE/POST.
    """
    tmp = (
        df.dropna(subset=["school_id", "survey_phase", "survey_session_id"])
          .assign(school_id=lambda d: d["school_id"].astype(str))
          .groupby(["school_id", "survey_phase"])["survey_session_id"]
          .nunique()
          .reset_index(name="n_sessions")
    )

    if tmp.empty:
        return go.Figure().update_layout(title="No data available")

    # Executive palette (greens)
    color_map = {"PRE": "#9FE1CB", "POST": "#1D9E75"}

    fig = px.bar(
        tmp,
        x="school_id",
        y="n_sessions",
        color="survey_phase",
        barmode="group",
        category_orders={"survey_phase": ["PRE", "POST"]},
        color_discrete_map=color_map,
        title="Number of workshops by school and phase",
        labels={
            "school_id": "School",
            "n_sessions": "Number of workshops",
            "survey_phase": "Survey Phase",
        },
    )

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=40, t=70, b=60),
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
    )
    fig.update_xaxes(tickangle=35)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(200,200,200,0.3)")
    return fig


def plot_responses_by_survey_type(df: pd.DataFrame) -> go.Figure:
    """
    Bar chart:
    number of responses by survey_type.
    """
    tmp = (
        df.dropna(subset=["survey_type", "response_id"])
          .groupby("survey_type")["response_id"]
          .nunique()
          .reset_index(name="n_responses")
          .sort_values("n_responses", ascending=False)
    )

    if tmp.empty:
        return go.Figure().update_layout(title="No data available")

    fig = px.bar(
        tmp,
        x="survey_type",
        y="n_responses",
        title="Number of responses by survey type",
        labels={
            "survey_type": "Survey Type",
            "n_responses": "Number of responses",
        },
        text="n_responses",
    )

    fig.update_traces(textposition="outside")
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=40, t=70, b=50),
        showlegend=False,
    )
    fig.update_yaxes(showgrid=True, gridcolor="rgba(200,200,200,0.3)")
    return fig


def plot_sessions_over_time_phase(df: pd.DataFrame) -> go.Figure:
    """
    Scatter plot:
    number of workshops over time by PRE/POST.
    Counts unique survey_session_id per date and phase.
    """
    tmp = df.copy()
    tmp["timestamp"] = pd.to_datetime(tmp["timestamp"], errors="coerce")
    tmp["date"] = tmp["timestamp"].dt.date

    agg = (
        tmp.dropna(subset=["date", "survey_phase", "survey_session_id"])
           .groupby(["date", "survey_phase"])["survey_session_id"]
           .nunique()
           .reset_index(name="n_sessions")
           .sort_values("date")
    )

    if agg.empty:
        return go.Figure().update_layout(title="No data available")

    fig = px.scatter(
        agg,
        x="date",
        y="n_sessions",
        color="survey_phase",
        category_orders={"survey_phase": ["PRE", "POST"]},
        title="Number of workshops over time by phase",
        labels={
            "date": "Date",
            "n_sessions": "Number of workshops",
            "survey_phase": "Survey Phase",
        },
    )

    fig.update_traces(marker=dict(size=9))
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=40, t=70, b=50),
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
    )
    fig.update_yaxes(showgrid=True, gridcolor="rgba(200,200,200,0.3)")
    return fig


def plot_sessions_over_time_phase_daily(df: pd.DataFrame) -> go.Figure:
    """
    Timeline plot (daily aggregation):
    unique workshops/sessions per day, split by PRE/POST.
    """
    tmp = df.copy()
    tmp["timestamp"] = pd.to_datetime(tmp["timestamp"], errors="coerce")
    tmp = tmp.dropna(subset=["timestamp", "survey_phase", "survey_session_id"])
    tmp["date"] = tmp["timestamp"].dt.floor("D")

    agg = (
        tmp.groupby(["date", "survey_phase"])["survey_session_id"]
           .nunique()
           .reset_index(name="n_sessions")
           .sort_values("date")
    )

    if agg.empty:
        return go.Figure().update_layout(title="No data available")

    color_map = {"PRE": "#9FE1CB", "POST": "#1D9E75"}

    fig = px.line(
        agg,
        x="date",
        y="n_sessions",
        color="survey_phase",
        markers=True,
        category_orders={"survey_phase": ["PRE", "POST"]},
        color_discrete_map=color_map,
        title="Number of workshops over time (daily) by phase",
        labels={
            "date": "Date",
            "n_sessions": "Number of workshops",
            "survey_phase": "Survey Phase",
        },
    )

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=40, t=70, b=50),
        legend=dict(orientation="h", y=1.08, x=0.5, xanchor="center"),
    )
    fig.update_yaxes(showgrid=True, gridcolor="rgba(200,200,200,0.3)")
    return fig


def plot_sessions_over_time_by_survey_type_phase(df: pd.DataFrame) -> go.Figure:
    """
    Scatter plot:
    number of workshops over time by survey_type and PRE/POST.
    """
    tmp = df.copy()
    tmp["timestamp"] = pd.to_datetime(tmp["timestamp"], errors="coerce")
    tmp["date"] = tmp["timestamp"].dt.date

    agg = (
        tmp.dropna(subset=["date", "survey_type", "survey_phase", "survey_session_id"])
           .groupby(["date", "survey_type", "survey_phase"])["survey_session_id"]
           .nunique()
           .reset_index(name="n_sessions")
           .sort_values("date")
    )

    if agg.empty:
        return go.Figure().update_layout(title="No data available")

    agg["series"] = agg["survey_type"].astype(str) + " | " + agg["survey_phase"].astype(str)

    fig = px.scatter(
        agg,
        x="date",
        y="n_sessions",
        color="survey_type",
        symbol="survey_phase",
        title="Number of workshops over time by survey type and phase",
        labels={
            "date": "Date",
            "n_sessions": "Number of workshops",
            "survey_type": "Survey Type",
            "survey_phase": "Survey Phase",
        },
        hover_data=["survey_type", "survey_phase"],
    )

    fig.update_traces(marker=dict(size=8, opacity=0.85))
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=40, r=40, t=70, b=50),
        legend=dict(orientation="h", y=1.12, x=0.5, xanchor="center"),
    )
    fig.update_yaxes(showgrid=True, gridcolor="rgba(200,200,200,0.3)")
    return fig