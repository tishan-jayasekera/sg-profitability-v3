import altair as alt
import pandas as pd
import streamlit as st

from lib.constants import (
    COL_ALLOCATED_REVENUE,
    COL_COST,
    COL_HOURS,
    COL_JOB_KEY,
    COL_MONTH_KEY,
    COL_QUOTED_AMOUNT,
    COL_QUOTED_TIME,
    COL_TASK_KEY,
    ROW_QUOTE_ONLY,
)
from lib.data_loader import load_data
from lib.metrics import (
    add_task_month_metrics,
    build_task_summary,
    compute_monthly_metrics,
    compute_quote_by_task,
    safe_divide,
)
from lib.qa import render_data_integrity
from lib.semantic import add_row_type, build_dim_job_month, build_dim_job_task_quote
from lib.ui import apply_filters, render_sidebar


def _fmt_currency(value):
    if value is None or pd.isna(value):
        return "N/A"
    return f"${value:,.0f}"


def _fmt_rate(value):
    if value is None or pd.isna(value):
        return "N/A"
    return f"${value:,.2f}/hr"


def _fmt_hours(value):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:,.1f}h"


def _fmt_pct(value):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.1%}"


st.set_page_config(page_title="Job Profitability", layout="wide")
st.title("Job Summary")
st.caption("Lifetime view by default with safe aggregation and QA diagnostics.")

df = add_row_type(load_data())
filters = render_sidebar(df)

job_key = filters["job_key"]
df_job = df[df[COL_JOB_KEY] == job_key]
df_job = apply_filters(df_job, filters)

start = pd.to_datetime(filters["start_date"])
end = pd.to_datetime(filters["end_date"])
is_lifetime = (filters["start_date"] == filters["lifetime_start"]) and (
    filters["end_date"] == filters["lifetime_end"]
)

actuals_window = df_job[
    df_job[COL_MONTH_KEY].notna()
    & (df_job[COL_MONTH_KEY] >= start)
    & (df_job[COL_MONTH_KEY] <= end)
]
if actuals_window.empty:
    st.warning(
        "No actuals in the selected window. Showing quote-only data where available. "
        "Try resetting filters or choosing the Lifetime preset for actuals."
    )

task_month = actuals_window[actuals_window[COL_TASK_KEY] != "__UNALLOCATED__"].copy()
unallocated = actuals_window[actuals_window[COL_TASK_KEY] == "__UNALLOCATED__"].copy()
task_month = add_task_month_metrics(task_month)

task_month_lifetime = df_job[
    df_job[COL_MONTH_KEY].notna() & (df_job[COL_TASK_KEY] != "__UNALLOCATED__")
].copy()
task_month_lifetime = add_task_month_metrics(task_month_lifetime)

dim_job_month = build_dim_job_month(df_job[df_job[COL_MONTH_KEY].notna()])
dim_job_month_window = dim_job_month[
    (dim_job_month[COL_MONTH_KEY] >= start) & (dim_job_month[COL_MONTH_KEY] <= end)
]

dim_job_task_quote = build_dim_job_task_quote(df_job)

quote_mode = "Lifetime Quote"
if not is_lifetime:
    st.warning(
        "You're viewing period actuals against lifetime quote. Choose quote alignment."
    )
    quote_mode = st.radio(
        "Quote Mode",
        options=["Lifetime Quote", "Earned Quote Proxy"],
        horizontal=True,
        key="quote_mode",
    )

quote_by_task = compute_quote_by_task(
    quote_mode, task_month, task_month_lifetime, dim_job_task_quote
)

actual_revenue = task_month[COL_ALLOCATED_REVENUE].sum()
if filters["include_unallocated"]:
    actual_revenue += unallocated[COL_ALLOCATED_REVENUE].sum()
actual_cost = task_month[COL_COST].sum()
actual_hours = task_month[COL_HOURS].sum()
actual_profit = actual_revenue - actual_cost
actual_margin = safe_divide(actual_profit, actual_revenue)

quoted_amount = quote_by_task["Quoted_Amount_Mode"].sum(min_count=1)
quoted_hours = quote_by_task["Quoted_Time_Mode"].sum(min_count=1)
quoted_rate = safe_divide(quoted_amount, quoted_hours)

realized_rate = safe_divide(actual_revenue, actual_hours)
base_cost_rate = safe_divide(actual_cost, actual_hours)

ghost_work_hours = task_month.loc[task_month["Ghost_Work_Month"], COL_HOURS].sum()
unallocated_total = unallocated[COL_ALLOCATED_REVENUE].sum()
unallocated_months = (
    unallocated[[COL_JOB_KEY, COL_MONTH_KEY]].drop_duplicates().shape[0]
    if not unallocated.empty
    else 0
)

st.subheader("KPIs")
row1 = st.columns(4)
row1[0].metric(
    "Quoted Amount",
    _fmt_currency(quoted_amount),
    help="Sum of Quoted Amount over unique Job_Key x Task_Key (mode dependent).",
)
row1[1].metric(
    "Actual Revenue",
    _fmt_currency(actual_revenue),
    help="Sum of Allocated_Revenue across task-months plus optional UNALLOCATED.",
)
row1[2].metric(
    "Actual Cost",
    _fmt_currency(actual_cost),
    help="Sum of Cost across task-months.",
)
row1[3].metric(
    "Actual Profit",
    _fmt_currency(actual_profit),
    help="Actual Revenue minus Actual Cost.",
)

row2 = st.columns(4)
row2[0].metric(
    "Actual Margin",
    _fmt_pct(actual_margin),
    help="Actual Profit divided by Actual Revenue.",
)
row2[1].metric(
    "Quoted Rate",
    _fmt_rate(quoted_rate),
    help="Quoted Amount divided by Quoted Time (mode dependent).",
)
row2[2].metric(
    "Realized Rate",
    _fmt_rate(realized_rate),
    help="Actual Revenue divided by Actual Hours.",
)
row2[3].metric(
    "Base Cost Rate",
    _fmt_rate(base_cost_rate),
    help="Actual Cost divided by Actual Hours.",
)

row3 = st.columns(4)
row3[0].metric(
    "Actual Hours",
    _fmt_hours(actual_hours),
    help="Sum of Hours across task-months.",
)
row3[1].metric(
    "Quoted Hours",
    _fmt_hours(quoted_hours),
    help="Sum of Quoted Time over unique Job_Key x Task_Key (mode dependent).",
)
row3[2].metric(
    f"UNALLOCATED Revenue ({unallocated_months} months)",
    _fmt_currency(unallocated_total),
    help="Allocated_Revenue for Task_Key=__UNALLOCATED__.",
)
row3[3].metric(
    "Ghost Work Hours",
    _fmt_hours(ghost_work_hours),
    help="Hours where Hours>0 and Allocated_Revenue==0.",
)

st.subheader("Monthly Performance")
monthly = compute_monthly_metrics(task_month, unallocated, filters["include_unallocated"])
if monthly.empty:
    st.info("No monthly actuals to chart for this window.")
else:
    metric_map = {
        "Revenue": "Revenue",
        "Cost": "Cost",
        "Unallocated_Revenue": "Unallocated Revenue",
    }
    rev_cols = ["Revenue", "Cost"]
    if filters["include_unallocated"]:
        rev_cols.append("Unallocated_Revenue")
    rev_long = monthly.melt(
        id_vars=[COL_MONTH_KEY],
        value_vars=rev_cols,
        var_name="Metric",
        value_name="Value",
    )
    rev_long["Metric"] = rev_long["Metric"].map(metric_map).fillna(rev_long["Metric"])

    rev_chart = (
        alt.Chart(rev_long)
        .mark_line(point=True)
        .encode(
            x=alt.X(f"{COL_MONTH_KEY}:T", title="Month"),
            y=alt.Y("Value:Q", title="Amount"),
            color="Metric:N",
            tooltip=[COL_MONTH_KEY, "Metric", "Value"],
        )
        .properties(height=260)
    )

    hours_chart = (
        alt.Chart(monthly)
        .mark_line(point=True, color="#2b7c85")
        .encode(
            x=alt.X(f"{COL_MONTH_KEY}:T", title="Month"),
            y=alt.Y("Hours:Q", title="Hours"),
            tooltip=[COL_MONTH_KEY, "Hours"],
        )
        .properties(height=260)
    )

    st.altair_chart(rev_chart, width="stretch")
    st.altair_chart(hours_chart, width="stretch")

st.subheader("Rate Convergence")
exclude_negative = st.toggle(
    "Exclude negative revenue months (rate charts only)",
    value=False,
    help="Applies to rate chart only when revenue is <= 0.",
)
rate_df = monthly.copy()
if rate_df.empty:
    st.info("No rate series available for this window.")
else:
    if exclude_negative:
        rate_df = rate_df[rate_df["Revenue"] > 0]
    rate_df["Quoted_Rate"] = quoted_rate
    rate_cols = ["Realized_Rate", "Base_Cost_Rate", "Billable_Rate_Wtd", "Quoted_Rate"]
    rate_long = rate_df.melt(
        id_vars=[COL_MONTH_KEY],
        value_vars=[c for c in rate_cols if c in rate_df.columns],
        var_name="Rate",
        value_name="Value",
    )
    rate_chart = (
        alt.Chart(rate_long)
        .mark_line(point=True)
        .encode(
            x=alt.X(f"{COL_MONTH_KEY}:T", title="Month"),
            y=alt.Y("Value:Q", title="Rate"),
            color="Rate:N",
            tooltip=[COL_MONTH_KEY, "Rate", "Value"],
        )
        .properties(height=260)
    )
    st.altair_chart(rate_chart, width="stretch")

st.subheader("Task Contribution (Ranked by Profit)")
task_summary = build_task_summary(
    task_month, dim_job_task_quote, filters["include_quote_only"]
)
task_summary = task_summary.merge(
    quote_by_task, on=[COL_JOB_KEY, COL_TASK_KEY], how="left"
)
task_summary["Quoted_Amount_Mode"] = task_summary["Quoted_Amount_Mode"].where(
    task_summary["Quoted_Amount_Mode"].notna(), task_summary[COL_QUOTED_AMOUNT]
)
task_summary["Quoted_Time_Mode"] = task_summary["Quoted_Time_Mode"].where(
    task_summary["Quoted_Time_Mode"].notna(), task_summary[COL_QUOTED_TIME]
)
task_summary["Quoted_Rate_Mode"] = safe_divide(
    task_summary["Quoted_Amount_Mode"], task_summary["Quoted_Time_Mode"]
)

task_label_col = next(
    (c for c in ["[Job Task] Name", "Task Name", "Task"] if c in task_summary.columns),
    COL_TASK_KEY,
)

task_view = task_summary.copy()
task_view["Profit"] = task_view["Actual_Profit"]
task_view = task_view.sort_values("Profit", ascending=False)
columns = [
    COL_TASK_KEY,
    task_label_col,
    "Actual_Hours_Total",
    "Actual_Revenue",
    "Actual_Cost",
    "Actual_Profit",
    "Realized_Rate",
    "Base_Cost_Rate",
    "Actual_Margin",
    "Below_Cost_Hours_Pct",
    "Ghost_Work_Hours",
    "Unquoted_Work",
    "Scope_Overrun",
]
columns = [c for c in columns if c in task_view.columns]
st.dataframe(task_view[columns], width="stretch", height=340)

st.subheader("Exceptions")
tab1, tab2, tab3, tab4 = st.tabs(
    ["Ghost Work", "Below Cost", "Unquoted Work", "Quote-only"]
)
with tab1:
    ghost = task_summary[
        (task_summary["Actual_Hours_Total"] > 0)
        & (task_summary["Actual_Revenue"] == 0)
    ]
    st.dataframe(ghost[columns], width="stretch", height=240)
with tab2:
    below = task_summary[
        task_summary["Realized_Rate"] < task_summary["Base_Cost_Rate"]
    ]
    st.dataframe(below[columns], width="stretch", height=240)
with tab3:
    unquoted = task_summary[task_summary["Unquoted_Work"]]
    st.dataframe(unquoted[columns], width="stretch", height=240)
with tab4:
    quote_only = task_summary[
        (task_summary["Actual_Hours_Total"].fillna(0) == 0)
        & (task_summary[COL_QUOTED_TIME] > 0)
    ]
    st.dataframe(quote_only[columns], width="stretch", height=240)

render_data_integrity(
    task_month,
    unallocated,
    dim_job_month_window,
    dim_job_task_quote,
    include_unallocated=filters["include_unallocated"],
    quote_mode=quote_mode,
    is_lifetime=is_lifetime,
)
