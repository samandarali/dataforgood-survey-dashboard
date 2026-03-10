import pandas as pd
import numpy as np
import math
import textwrap

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import textwrap
import ast
import hashlib
# function to filter by  survay type and version

def filter_survey_df(df, filter_survey, filter_survey_version):
    # Filter the data to show only survey type and version
    filtered_df = df[
        (df['survey_type'] == filter_survey) & # Filter the data to show only survey type
        (df['survey_version'] == filter_survey_version)] # Filter the data for v# of the survey
    return filtered_df


# function for providing summary table



def create_survey_session_id(df):
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    df['hour'] = df['timestamp'].dt.hour
    

    # session variable before noon vs after noon
    df['session'] = np.where(df['timestamp'].dt.hour < 12,
                                'AM',
                                'PM')
    df['date_id'] = (
        df['timestamp'].dt.strftime('%y%m%d').astype(int)
    )

    df['survey_session_id'] = (
        df['survey_type'].str.upper()
        + df['survey_version'].str.lower()
        + "_"
        + df['survey_phase'].str.upper()
        + "_"
        + df['date_id'].astype(str)
        + "_"
        + df['session'].str.upper()
    )
    print(
        df[['school_id', 'date', 'hour', 'survey_phase', 'survey_session_id']]
        .drop_duplicates()
        .sort_values(by=['school_id', 'date', 'hour'])
    )

    return df

def split_by_phase(df, phase_col="survey_phase"):
    """
    Split a long survey dataframe into PRE and POST subsets.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing survey data in long format.
    phase_col : str, default="survey_phase"
        Column name indicating survey phase (e.g., "PRE", "POST").

    Returns
    -------
    df_pre : pandas.DataFrame
        Subset of the dataframe where phase_col == "PRE".
    df_post : pandas.DataFrame
        Subset of the dataframe where phase_col == "POST".

    Example
    -------
    >>> df_pre, df_post = split_by_phase(long_ACA_v1)

    Notes
    -----
    The function returns two dataframes and must be unpacked
    into two variables.
    """
    
    df_pre = df[df[phase_col] == "PRE"].copy()
    df_post = df[df[phase_col] == "POST"].copy()
    
    print(f"Pre sample size: {df_pre.shape}")
    print(f"Post sample size: {df_post.shape}")
    
    return df_pre, df_post



# Custom Accessor to for filtering survey questions by response type
@pd.api.extensions.register_dataframe_accessor("qtype")
class QuestionTypeAccessor:
    """
    DataFrame accessor for question-type utilities + plotting.

    Required columns (minimum):
      - scale_type
      - concept_key
      - survey_session_id
      - response_id (optional, not used in plotting below)
      - response (Likert text)
      - response_encoded (Likert numeric)

    Common usage:
      df.qtype.likert_text_questions()
      df.qtype.likert_numeric_questions()
      df.qtype.open_ended_questions()

      df.qtype.plot_LikText_q(...)
      df.qtype.plot_LikNum_q(...)
    """

    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    # ---------- question lists ----------
    def likert_text_questions(self):
        return self._obj.loc[self._obj["scale_type"] == "Likert Scale Text", "concept_key"].dropna().unique()

    def likert_numeric_questions(self):
        return self._obj.loc[self._obj["scale_type"] == "Likert Scale Numeric", "concept_key"].dropna().unique()

    def open_ended_questions(self):
        return self._obj.loc[self._obj["scale_type"] == "Open-Ended", "concept_key"].dropna().unique()

    def categorical_questions(self):
        return self._obj.loc[self._obj["scale_type"] == "Categorical", "concept_key"].dropna().unique()
    def open_ended_text_questions(self):
        return self._obj.loc[self._obj["scale_type"] == "Open-Ended", "concept_key"].dropna().unique()

    # ---------- internal helpers ----------
    def _concept_labels(self, question_text_col="question_text"):
        cols = ["concept_key", question_text_col]
        if question_text_col not in self._obj.columns:
            # no labels available
            return {}
        return (
            self._obj[cols]
            .dropna(subset=["concept_key"])
            .drop_duplicates()
            .set_index("concept_key")[question_text_col]
            .to_dict()
        )

    def _plot_by_workshop_and_total(
        self,
        df_sub,
        response_col,
        likert_order,
        likert_colors,
        title,
        question_text_col="question_text",
        wrap_width=160,
        max_side_bars=10
    ):


        counts_ws = (
            df_sub.groupby(["concept_key", "survey_session_id", response_col])
            .size()
            .reset_index(name="count")
        )

        counts_total = (
            df_sub.groupby(["concept_key", response_col])
            .size()
            .reset_index(name="count")
        )

        concept_labels = self._concept_labels(question_text_col)

        figs = []

        for concept in df_sub["concept_key"].dropna().unique():

            sub_ws = counts_ws[counts_ws["concept_key"] == concept]

            pivot_ws = (
                sub_ws.pivot(index="survey_session_id", columns=response_col, values="count")
                .fillna(0)
                .reindex(columns=likert_order, fill_value=0)
                .sort_index()
            )

            workshops = pivot_ws.index.tolist()
            num_ws = len(workshops)

            sub_tot = counts_total[counts_total["concept_key"] == concept]

            total_series = (
                sub_tot.set_index(response_col)["count"]
                .reindex(likert_order, fill_value=0)
            )

            label = concept_labels.get(concept, str(concept))
            wrapped = "<br>".join(textwrap.wrap(f"{concept}: {label}", wrap_width))

            # ---------- Layout decision ----------

            if num_ws <= max_side_bars:

                fig = make_subplots(
                    rows=1,
                    cols=2,
                    specs=[[{"type": "bar"}, {"type": "domain"}]],
                    column_widths=[0.55, 0.45],
                    subplot_titles=[wrapped, ""],
                    horizontal_spacing=0.05,
                )

                bar_row, pie_row = 1, 1
                pie_col = 2

            else:

                fig = make_subplots(
                    rows=2,
                    cols=1,
                    specs=[[{"type": "bar"}], [{"type": "domain"}]],
                    subplot_titles=[wrapped, ""],
                    vertical_spacing=0.20,
                )

                bar_row, pie_row = 1, 2
                pie_col = 1

            # ---------- Bars ----------

            for resp in likert_order:

                vals = pivot_ws[resp]

                fig.add_trace(
                    go.Bar(
                        x=workshops,
                        y=vals,
                        marker_color=likert_colors[resp],
                        name=str(resp),
                        text=vals,
                        textposition="inside",
                    ),
                    row=bar_row,
                    col=1,
                )

            totals = pivot_ws.sum(axis=1)

            fig.add_trace(
                go.Scatter(
                    x=workshops,
                    y=totals,
                    mode="text",
                    text=totals,
                    textposition="top center",
                    showlegend=False,
                ),
                row=bar_row,
                col=1,
            )

            # ---------- Pie ----------

            # Pie (hidden from legend to avoid duplicating bar entries)
            fig.add_trace(
                go.Pie(
                    labels=[str(x) for x in likert_order],
                    values=total_series.values,
                    marker=dict(colors=[likert_colors[x] for x in likert_order]),
                    textinfo="percent",
                    hole=0.45,
                    sort=False,
                    showlegend=False,
                ),
                row=pie_row,
                col=pie_col,
            )

            fig.update_layout(
                title=dict(text=title, x=0.5, font=dict(size=16), pad=dict(t=100, b=10)),
                barmode="stack",
                height=600 if num_ws <= max_side_bars else 800,
                margin=dict(l=40, r=40, t=180, b=40),
                
            )

            fig.update_xaxes(
                tickangle=45,
                automargin=True,
                row=bar_row,
                col=1,
            )

            fig.update_yaxes(
                title="Responses",
                showgrid=True,
                gridcolor="rgba(200,200,200,0.3)",
                row=bar_row,
                col=1,
            )

            figs.append(fig)

        return figs
    
    # ---------- public plotting methods ----------
    def plot_LikText_q(
        self,
        question_text_col="question_text",
        likert_order=None,
        likert_colors=None,
        title=None,
        wrap_width=160,
    ):
        """
        Plot Likert Text questions using Plotly.

        Produces:
        - stacked bars by workshop
        - pie distribution of responses
        """

        df = self._obj

        concepts = self.likert_text_questions()

        cols = ["concept_key", "survey_session_id", "response"]
        if question_text_col in df.columns:
            cols.append(question_text_col)

        df_sub = df.loc[df["concept_key"].isin(concepts), cols].copy()

        if df_sub.empty:
            raise ValueError("No Likert text data available for plotting.")

        if likert_order is None:
            likert_order = [
                "Strongly Disagree",
                "Disagree",
                "Neither Agree or Disagree",
                "Agree",
                "Strongly Agree",
            ]

        if likert_colors is None:
            likert_colors = {
                "Strongly Disagree": "#b2182b",
                "Disagree": "#ef8a62",
                "Neither Agree or Disagree": "#bdbdbd",
                "Agree": "#a6dba0",
                "Strongly Agree": "#1b7837",
            }

        if title is None:
            title = "Likert Scale Text — by Workshop and Total"

        fig = self._plot_by_workshop_and_total(
            df_sub=df_sub,
            response_col="response",
            likert_order=likert_order,
            likert_colors=likert_colors,
            title=title,
            question_text_col=question_text_col,
            wrap_width=wrap_width,
        )

        return fig

    def plot_LikNum_q(
        self,
        question_text_col="question_text",
        likert_order=None,
        likert_colors=None,
        title=None,
        wrap_width=160,
    ):
        """
        Plot Likert Numeric questions using Plotly.

        Produces:
        - stacked bars by workshop
        - pie distribution of responses
        """

        df = self._obj

        concepts = self.likert_numeric_questions()

        cols = ["concept_key", "survey_session_id", "response_encoded"]
        if question_text_col in df.columns:
            cols.append(question_text_col)

        df_sub = df.loc[df["concept_key"].isin(concepts), cols].copy()

        if df_sub.empty:
            raise ValueError("No Likert numeric data available for plotting.")

        # clean numeric responses
        df_sub["response_encoded"] = pd.to_numeric(
            df_sub["response_encoded"],
            errors="coerce"
        )

        df_sub = df_sub.dropna(subset=["response_encoded"])
        df_sub["response_encoded"] = df_sub["response_encoded"].astype(int)

        if likert_order is None:
            likert_order = [1, 2, 3, 4, 5]

        if likert_colors is None:
            likert_colors = {
                1: "#b2182b",
                2: "#ef8a62",
                3: "#bdbdbd",
                4: "#a6dba0",
                5: "#1b7837",
            }

        if title is None:
            title = "Likert Scale Numeric — by Workshop and Total"

        fig = self._plot_by_workshop_and_total(
            df_sub=df_sub,
            response_col="response_encoded",
            likert_order=likert_order,
            likert_colors=likert_colors,
            title=title,
            question_text_col=question_text_col,
            wrap_width=wrap_width,
        )

        return fig

    def plot_Categorical_q(
        self,
        question_text_col: str = "question_text",
        title: str | None = None,
        wrap_width: int = 160,
    ):
        """
        Plot Categorical questions using Plotly.

        - Uses `possible_responses` (when available) to determine the set
          and order of response options.
        - Handles multiple‐response selections like "Yes, Not sure" by
          splitting into individual options and counting each separately.
        """

        df = self._obj

        concepts = self.categorical_questions()

        cols = ["concept_key", "survey_session_id", "response"]
        if "possible_responses" in df.columns:
            cols.append("possible_responses")
        if question_text_col in df.columns:
            cols.append(question_text_col)

        df_sub = df.loc[df["concept_key"].isin(concepts), cols].copy()

        if df_sub.empty:
            raise ValueError("No Categorical data available for plotting.")

        concept_labels = self._concept_labels(question_text_col)

        figs: list[go.Figure] = []

        for concept in df_sub["concept_key"].dropna().unique():
            tmp = df_sub[df_sub["concept_key"] == concept].copy()

            # -------------------------------------------------
            # 1) Determine response options and their order
            # -------------------------------------------------
            response_order: list[str] | None = None

            if "possible_responses" in tmp.columns and tmp["possible_responses"].notna().any():
                pr_str = str(tmp["possible_responses"].dropna().iloc[0])
                try:
                    pr = ast.literal_eval(pr_str)
                    if isinstance(pr, (list, tuple)):
                        response_order = [str(x) for x in pr]
                except Exception:
                    response_order = None

            # If no explicit ordering from possible_responses, infer from data
            if response_order is None:
                response_order = sorted(tmp["response"].dropna().astype(str).unique())

            # -------------------------------------------------
            # 2) Aggregate counts by workshop and overall
            # -------------------------------------------------
            counts_ws = (
                tmp.groupby(["concept_key", "survey_session_id", "response"])
                .size()
                .reset_index(name="count")
            )

            counts_total = (
                tmp.groupby(["concept_key", "response"])
                .size()
                .reset_index(name="count")
            )

            sub_ws = counts_ws[counts_ws["concept_key"] == concept]

            pivot_ws = (
                sub_ws.pivot(index="survey_session_id", columns="response", values="count")
                .fillna(0)
                .reindex(columns=response_order, fill_value=0)
                .sort_index()
            )

            workshops = pivot_ws.index.tolist()
            num_ws = len(workshops)

            sub_tot = counts_total[counts_total["concept_key"] == concept]

            total_series = (
                sub_tot.set_index("response")["count"]
                .reindex(response_order, fill_value=0)
            )

            label = concept_labels.get(concept, str(concept))
            wrapped = "<br>".join(textwrap.wrap(f"{concept}: {label}", wrap_width))

            # Color mapping per response option, using Likert-style helper
            response_colors = {resp: get_categorical_color(resp) for resp in response_order}

            # -------------------------------------------------
            # 3) Build figure: bars + pie
            # -------------------------------------------------
            max_side_bars = 10
            if num_ws <= max_side_bars:
                fig = make_subplots(
                    rows=1,
                    cols=2,
                    specs=[[{"type": "bar"}, {"type": "domain"}]],
                    column_widths=[0.55, 0.45],
                    subplot_titles=[wrapped, ""],
                    horizontal_spacing=0.05,
                )
                bar_row, pie_row = 1, 1
                pie_col = 2
            else:
                fig = make_subplots(
                    rows=2,
                    cols=1,
                    specs=[[{"type": "bar"}], [{"type": "domain"}]],
                    subplot_titles=[wrapped, ""],
                    vertical_spacing=0.20,
                )
                bar_row, pie_row = 1, 2
                pie_col = 1

            # Bars
            for resp in response_order:
                vals = pivot_ws[resp]
                fig.add_trace(
                    go.Bar(
                        x=workshops,
                        y=vals,
                        marker_color=response_colors[resp],
                        name=str(resp),
                        text=vals,
                        textposition="inside",
                    ),
                    row=bar_row,
                    col=1,
                )

            totals = pivot_ws.sum(axis=1)
            fig.add_trace(
                go.Scatter(
                    x=workshops,
                    y=totals,
                    mode="text",
                    text=totals,
                    textposition="top center",
                    showlegend=False,
                ),
                row=bar_row,
                col=1,
            )

            # Pie (hidden from legend to avoid duplication)
            fig.add_trace(
                go.Pie(
                    labels=[str(x) for x in response_order],
                    values=total_series.values,
                    marker=dict(colors=[response_colors[x] for x in response_order]),
                    textinfo="percent",
                    hole=0.45,
                    sort=False,
                    showlegend=False,
                ),
                row=pie_row,
                col=pie_col,
            )

            plot_title = title if title else "Categorical Questions — by Workshop and Total"
            fig.update_layout(
                title=dict(text=plot_title, x=0.5, font=dict(size=16), pad=dict(t=100, b=10)),
                barmode="stack",
                height=600 if num_ws <= max_side_bars else 800,
                margin=dict(l=40, r=40, t=180, b=40),
                legend=dict(
                    orientation="h",
                    y=1.31,
                    x=0.5,
                    xanchor="center",
                    yanchor="bottom",
                ),
            )

            fig.update_xaxes(
                tickangle=45,
                automargin=True,
                row=bar_row,
                col=1,
            )

            fig.update_yaxes(
                title="Responses",
                showgrid=True,
                gridcolor="rgba(200,200,200,0.3)",
                row=bar_row,
                col=1,
            )

            figs.append(fig)

        return figs


# Helper function to map categorical responses to Likert-style colors
def get_categorical_color(response_text):
    """Map categorical responses to Likert-style colors."""
    response_lower = str(response_text).lower().strip()
    
    # Helper for whole-word matching (so "no" doesn't match "not")
    import re
    def has_word(word: str) -> bool:
        return re.search(r"\b" + re.escape(word) + r"\b", response_lower) is not None

    # 1) Explicit "still unsure" (separate from plain "not sure" and "no")
    if "still unsure" in response_lower:
        # Amber-ish color distinct from both red "No" and grey "Not sure"
        return "#fdb863"

    # 2) Strong positive / positive -> Green shades (like Agree / Strongly Agree)
    if any(word in response_lower for word in ["always", "often", "very familiar", "advanced", "yes"]):
        if "strongly" in response_lower or "very" in response_lower or "advanced" in response_lower or "always" in response_lower:
            return "#1b7837"  # Dark green (Strongly positive)
        elif "can recognize racism but i'm not sure what to do" in response_lower:
            return "#fdb863" 
        else:
            return "#a6dba0"  # Light green (Positive)

    # 3) Pure negative -> Red shades (like Disagree)
    if has_word("no") or has_word("never") or has_word("rarely") or "new to this topic" in response_lower:
        if "strongly" in response_lower or "never" in response_lower:
            return "#b2182b"  # Dark red (Strongly negative)
        else:
            return "#ef8a62"  # Light orange/red (Negative)

    # 4) Neutral / uncertain -> Grey (like Neither / Not sure)
    if "not sure" in response_lower or "unsure" in response_lower or "sometimes" in response_lower or "somewhat knowledgeable" in response_lower:
        return "#bdbdbd"  # Grey
    
    # Default fallback colors for other responses
    # Use a neutral palette for other responses
    fallback_colors = ['#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    # Hash the response text to get a consistent color
    
    hash_val = int(hashlib.md5(str(response_text).encode()).hexdigest(), 16)
    return fallback_colors[hash_val % len(fallback_colors)]


def explore_semantic_text(df: pd.DataFrame):
    """
    Return a subset of the dataframe for open‑ended text questions,
    with the columns needed for semantic exploration.

    If `survey_session_id` is missing (e.g. when `load_data` is used
    outside the Streamlit app), this function will create it using
    `create_survey_session_id`.
    """

    # Ensure survey_session_id exists when called outside app.py
    if "survey_session_id" not in df.columns:
        df = create_survey_session_id(df)

    concepts = df.qtype.open_ended_text_questions()
    cols = ["concept_key", "question_number", "question_text", "survey_session_id", "response"]

    # Guard against missing columns to give a clearer error
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns for semantic exploration: {missing}")

    df_sub = df.loc[df["concept_key"].isin(concepts), cols].copy()
    return df_sub


# --- added function for filtering data


def apply_filters(
    df: pd.DataFrame,
    survey_type: str = "All",
    survey_phase: str = "All",
    survey_session_id: str = "All",
    school_id: str = "All",
    scale_type: str = "All",
    date: str = "All",
    hour: str = "All",

) -> pd.DataFrame:
    """
    Filter the dataframe by survey_type, survey_phase, and survey_session_id.
    Use "All" to leave a dimension unfiltered.
    """
    required = {"survey_type", "survey_phase", "survey_session_id"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns for filtering: {missing}")

    out = df.copy()

    if survey_type and survey_type != "All":
        out = out[out["survey_type"] == survey_type]

    if survey_phase and survey_phase != "All":
        out = out[out["survey_phase"] == survey_phase]
        
    if survey_session_id and survey_session_id != "All":
        out = out[out["survey_session_id"] == survey_session_id]

    if school_id and school_id != "All":
        out = out[out["school_id"] == school_id]

    if scale_type and scale_type != "All":
        out = out[out["scale_type"] == scale_type]

    if date and date != "All":
        out = out[out["date"] == date]

    if hour and hour != "All":
        out = out[out["hour"] == hour]                
    return out


# Allowed grouping columns for the dynamic summary (subset of df columns).
SUMMARY_GROUPBY_OPTIONS = [
    "survey_type",
    "survey_phase",
    "survey_session_id",
    "school_id",
    "scale_type",
    "date",
    'hour'

]

# Metric key -> (source_column, agg_func). Keys are used in generate_summary(metrics=...).
SUMMARY_METRIC_OPTIONS = {
    "num_questions": ("concept_key", "nunique"),
    "num_observations": ("response_id", "nunique"),
}

# Human-readable labels for the UI (key -> label).
SUMMARY_METRIC_LABELS = {
    "num_questions": "Number of questions (concept_key)",
    "num_observations": "Number of observations (response_id)",
}


def generate_summary(
    df_filtered: pd.DataFrame,
    groupby_columns: list[str] | None = None,
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build a summary table with selected metrics, optionally grouped.

    groupby_columns: list of column names to group by (e.g. [] or ["survey_type"]).
    Only columns in SUMMARY_GROUPBY_OPTIONS are allowed. If None or empty, returns one row with overall totals.

    metrics: list of metric keys to compute (e.g. ["num_questions"], ["num_observations"], or both).
    Allowed keys: list(SUMMARY_METRIC_OPTIONS.keys()). If None or empty, all metrics are computed.
    """
    if groupby_columns is None:
        groupby_columns = []

    if metrics is None or len(metrics) == 0:
        metrics = list(SUMMARY_METRIC_OPTIONS.keys())

    invalid_metrics = set(metrics) - set(SUMMARY_METRIC_OPTIONS.keys())
    if invalid_metrics:
        raise ValueError(
            f"Invalid metrics: {invalid_metrics}. Allowed: {list(SUMMARY_METRIC_OPTIONS.keys())}"
        )

    required_columns = {SUMMARY_METRIC_OPTIONS[k][0] for k in metrics}
    missing = required_columns - set(df_filtered.columns)
    if missing:
        raise KeyError(f"Missing required columns for selected metrics: {missing}")

    invalid = set(groupby_columns) - set(SUMMARY_GROUPBY_OPTIONS)
    if invalid:
        raise ValueError(f"Invalid groupby columns: {invalid}. Allowed: {SUMMARY_GROUPBY_OPTIONS}")

    missing_groupby = set(groupby_columns) - set(df_filtered.columns)
    if missing_groupby:
        raise KeyError(f"DataFrame missing groupby columns: {missing_groupby}")

    agg_dict = {
        name: (SUMMARY_METRIC_OPTIONS[name][0], SUMMARY_METRIC_OPTIONS[name][1])
        for name in metrics
    }

    if not groupby_columns:
        out = pd.DataFrame(
            {
                name: [getattr(df_filtered[col], agg)()]
                for name, (col, agg) in agg_dict.items()
            }
        )
        return out

    out = (
        df_filtered.groupby(groupby_columns, dropna=False)
        .agg(**{name: (col, agg) for name, (col, agg) in agg_dict.items()})
        .reset_index()
    )
    return out


def run_likert_plot(
    df_filtered: pd.DataFrame,
    likert_kind: str,  # "text" or "numeric"
    title: str | None = None,
    question_text_col: str = "question_text",
):
    """
    Route to accessor plots: plot_LikText_q or plot_LikNum_q.
    Returns the Plotly figure(s) for display in Streamlit.
    Requires survey_session_id to exist on the dataframe.
    """
    required_base = {"concept_key", "survey_session_id", "scale_type"}
    missing = required_base - set(df_filtered.columns)
    if missing:
        raise KeyError(f"Missing required columns for plotting: {missing}")

    if likert_kind == "text":
        if "response" not in df_filtered.columns:
            raise KeyError("Likert text plotting needs column: 'response'")
        return df_filtered.qtype.plot_LikText_q(
            question_text_col=question_text_col, title=title
        )

    if likert_kind == "numeric":
        if "response_encoded" not in df_filtered.columns:
            raise KeyError("Likert numeric plotting needs column: 'response_encoded'")
        return df_filtered.qtype.plot_LikNum_q(
            question_text_col=question_text_col, title=title
        )

    raise ValueError("likert_kind must be 'text' or 'numeric'")


def run_categorical_plot(
    df_filtered: pd.DataFrame,
    title: str | None = None,
    question_text_col: str = "question_text",
):
    """
    Route to accessor plots: plot_Categorical_q.
    Returns the Plotly figure(s) for display in Streamlit.
    Requires survey_session_id to exist on the dataframe.
    """
    required_base = {"concept_key", "survey_session_id", "scale_type"}
    missing = required_base - set(df_filtered.columns)
    if missing:
        raise KeyError(f"Missing required columns for plotting: {missing}")

    if "response" not in df_filtered.columns:
        raise KeyError("Categorical plotting needs column: 'response'")
    
    return df_filtered.qtype.plot_Categorical_q(
        question_text_col=question_text_col, title=title
    )




