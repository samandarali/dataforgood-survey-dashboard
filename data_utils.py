import pandas as pd
import numpy as np
import math
import textwrap
import matplotlib.pyplot as plt


# function to filter by  survay type and version

def filter_survey_df(df, filter_survey, filter_survey_version):
    # Filter the data to show only survey type and version
    filtered_df = df[
        (df['survey_type'] == filter_survey) & # Filter the data to show only survey type
        (df['survey_version'] == filter_survey_version)] # Filter the data for v# of the survey
    return filtered_df


# function for providing summary table



def create_workshop_id(df):
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

    df['workshop_id'] = (
        df['survey_type'].str.upper()
        + "_"
        + df['survey_phase'].str.upper()
        + "_"
        + df['date_id'].astype(str)
        + "_"
        + df['session'].str.upper()
    )
    print(
        df[['school_id', 'date', 'hour', 'survey_phase', 'workshop_id']]
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
      - workshop_id
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
        figsize_width=18,
        row_height=4.2,
    ):
        # counts
        counts_ws = (
            df_sub.groupby(["concept_key", "workshop_id", response_col])
                  .size()
                  .reset_index(name="count")
        )
        counts_total = (
            df_sub.groupby(["concept_key", response_col])
                  .size()
                  .reset_index(name="count")
        )

        concepts = df_sub["concept_key"].dropna().unique()
        n = len(concepts)
        if n == 0:
            raise ValueError("No concepts to plot (empty after filtering).")

        concept_labels = self._concept_labels(question_text_col=question_text_col)

        fig_height = row_height * n
        fig, axes = plt.subplots(
            nrows=n, ncols=2,
            figsize=(figsize_width, row_height * n),
            gridspec_kw={"width_ratios": [1.6, 1.0]}
        )



        # --- header band: adjust for small n ---
        if n == 1:
            axes = np.array([axes])

        # --- Dynamic header positioning ---
        # Header band shrinks as n grows because each row takes a smaller fraction of fig height
        # Reserve a fixed pixel amount (~1.2 inches) for the header band regardless of n
        header_inches = 1.2
        header_fraction = header_inches / fig_height

        AX_TOP = 1.0 - header_fraction          # plots start here (in figure fraction)
        SUPTITLE_Y = 1.0 - (header_inches * 0.10) / fig_height   # very close to top
        COLHDR_Y   = 1.0 - (header_inches * 0.55) / fig_height   # midway in header band

        # Clamp to sane values
        AX_TOP     = max(0.60, min(0.95, AX_TOP))
        SUPTITLE_Y = max(AX_TOP + 0.02, min(0.99, SUPTITLE_Y))
        COLHDR_Y   = max(AX_TOP + 0.005, min(SUPTITLE_Y - 0.01, COLHDR_Y))

        fig.subplots_adjust(top=AX_TOP, wspace=0.06, hspace=0.75)

        fig.suptitle(title, y=SUPTITLE_Y, fontsize=14, fontweight="bold")
        fig.text(0.30, COLHDR_Y, "By workshop", ha="center", fontsize=12, fontweight="bold")
        fig.text(0.77, COLHDR_Y, "Total (all workshops)", ha="center", fontsize=12, fontweight="bold")


        for i, c_key in enumerate(concepts):
            ax_ws = axes[i, 0]
            ax_tot = axes[i, 1]

            # ---- left: stacked bar ----
            sub_ws = counts_ws[counts_ws["concept_key"] == c_key]
            pivot_ws = (
                sub_ws.pivot(index="workshop_id", columns=response_col, values="count")
                      .fillna(0)
                      .reindex(columns=likert_order, fill_value=0)
                      .sort_index()
            )

            pivot_ws.plot(
                kind="bar",
                stacked=True,
                ax=ax_ws,
                legend=False,
                color=[likert_colors[c] for c in pivot_ws.columns],
            )

            ax_ws.set_xlabel("workshop_id")
            ax_ws.set_ylabel("count")
            ax_ws.tick_params(axis="x", labelrotation=90, labelsize=8)

            # ---- right: pie (total) ----
            sub_tot = counts_total[counts_total["concept_key"] == c_key]
            total_series = (
                sub_tot.set_index(response_col)["count"]
                      .reindex(likert_order, fill_value=0)
            )

            pie_colors = [likert_colors[c] for c in likert_order]
            wedges, texts, autotexts = ax_tot.pie(
                total_series,
                labels=likert_order,
                colors=pie_colors,
                autopct="%1.1f%%",
                startangle=90,
                labeldistance=1.12,
                pctdistance=0.70,
                textprops={"fontsize": 8},
                radius=1.15,
            )
            for t in autotexts:
                t.set_fontsize(7)

            ax_tot.axis("equal")
            ax_tot.set_anchor("W")

            # ---- row title centered above both ----
            question_text = concept_labels.get(c_key, str(c_key))
            row_title = "\n".join(textwrap.wrap(f"{c_key}: {question_text}", wrap_width))

            pos_left = ax_ws.get_position()
            pos_right = ax_tot.get_position()
            x_center = (pos_left.x0 + pos_right.x1) / 2
            y_top = max(pos_left.y1, pos_right.y1)

            fig.text(
                x_center, y_top + 0.009,
                row_title,
                ha="center", va="bottom",
                fontsize=10, fontweight="bold"
            )

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, title="Response", loc="upper right")
        return fig

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
        Plot Likert Scale Text questions:
          - stacked bars by workshop
          - pie chart total across workshops
        """
        df = self._obj.copy()
        concepts = self.likert_text_questions()

        df_sub = df.loc[
            df["concept_key"].isin(concepts),
            ["concept_key", "workshop_id", "response"] + ([question_text_col] if question_text_col in df.columns else [])
        ].copy()

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

        return self._plot_by_workshop_and_total(
            df_sub=df_sub,
            response_col="response",
            likert_order=likert_order,
            likert_colors=likert_colors,
            title=title,
            question_text_col=question_text_col,
            wrap_width=wrap_width,
        )

    def plot_LikNum_q(
        self,
        question_text_col="question_text",
        likert_order=None,
        likert_colors=None,
        title=None,
        wrap_width=160,
    ):
        """
        Plot Likert Scale Numeric questions:
          - stacked bars by workshop
          - pie chart total across workshops

        Expects response_encoded to be convertible to int 1..5 (or your scale).
        """
        df = self._obj.copy()
        concepts = self.likert_numeric_questions()

        df_sub = df.loc[
            df["concept_key"].isin(concepts),
            ["concept_key", "workshop_id", "response_encoded"] + ([question_text_col] if question_text_col in df.columns else [])
        ].copy()

        df_sub["response_encoded"] = pd.to_numeric(df_sub["response_encoded"], errors="coerce")
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

        return self._plot_by_workshop_and_total(
            df_sub=df_sub,
            response_col="response_encoded",
            likert_order=likert_order,
            likert_colors=likert_colors,
            title=title,
            question_text_col=question_text_col,
            wrap_width=wrap_width,
        )


    

# --- added function for filtering data


def apply_filters(
    df: pd.DataFrame,
    survey_type: str = "All",
    survey_phase: str = "All",
    workshop_id: str = "All",
    school_id: str = "All",
    scale_type: str = "All",
    date: str = "All",
    hour: str = "All",

) -> pd.DataFrame:
    """
    Filter the dataframe by survey_type, survey_phase, and workshop_id.
    Use "All" to leave a dimension unfiltered.
    """
    required = {"survey_type", "survey_phase", "workshop_id"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns for filtering: {missing}")

    out = df.copy()

    if survey_type and survey_type != "All":
        out = out[out["survey_type"] == survey_type]

    if survey_phase and survey_phase != "All":
        out = out[out["survey_phase"] == survey_phase]
        
    if workshop_id and workshop_id != "All":
        out = out[out["workshop_id"] == workshop_id]

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
    "workshop_id",
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
    Returns the matplotlib figure for display in Streamlit.
    Requires workshop_id to exist on the dataframe.
    """
    required_base = {"concept_key", "workshop_id", "scale_type"}
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



