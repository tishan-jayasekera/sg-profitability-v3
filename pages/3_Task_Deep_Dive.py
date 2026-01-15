import altair as alt
import pandas as pd
import streamlit as st

from lib.constants import (
    COL_ALLOCATED_REVENUE,
    COL_BASE_RATE,
    COL_BASE_RATE_QUOTE,
    COL_COST,
    COL_HOURS,
    COL_JOB_KEY,
    COL_MONTH_KEY,
    COL_QUOTED_AMOUNT,
    COL_QUOTED_TIME,
    COL_TASK_KEY,
)
from lib.data_loader import load_data
from lib.metrics import add_task_month_metrics, compute_quote_by_task, safe_divide
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


def _job_label_map(df: pd.DataFrame) -> dict[str, str]:
    label = None
    for candidate in ["[Job] Name", "[Job] Job No.", "Job Number"]:
        if candidate in df.columns:
            label = candidate
            break
    if label is None:
        return {k: k for k in df[COL_JOB_KEY].dropna().astype(str).unique()}
    job_map = (
        df[[COL_JOB_KEY, label]]
        .dropna()
        .drop_duplicates()
        .astype({COL_JOB_KEY: str, label: str})
    )
    return {row[COL_JOB_KEY]: f"{row[COL_JOB_KEY]} | {row[label]}" for _, row in job_map.iterrows()}


st.set_page_config(page_title="Task Deep Dive", layout="wide")
st.title("Task Deep Dive")
st.caption("Investigate task profitability, rate evolution, and diagnostics.")

df = add_row_type(load_data())
filters = render_sidebar(df)
df_filtered = apply_filters(df, filters)

job_map = _job_label_map(df_filtered)
job_keys = sorted(job_map.keys())
if not job_keys:
    st.info("No jobs available with the current filters.")
    st.stop()

job_key = st.selectbox(
    "Job",
    options=job_keys,
    format_func=lambda k: job_map.get(k, k),
)
df_job = df_filtered[df_filtered[COL_JOB_KEY] == job_key]

start = pd.to_datetime(filters["start_date"])
end = pd.to_datetime(filters["end_date"])
job_lifetime = df_job[df_job[COL_MONTH_KEY].notna()][COL_MONTH_KEY]
if job_lifetime.empty:
    job_start = filters["lifetime_start"]
    job_end = filters["lifetime_end"]
else:
    job_start = job_lifetime.min().date()
    job_end = job_lifetime.max().date()
is_lifetime = (filters["start_date"] == job_start) and (filters["end_date"] == job_end)

actuals_window = df_job[
    df_job[COL_MONTH_KEY].notna()
    & (df_job[COL_MONTH_KEY] >= start)
    & (df_job[COL_MONTH_KEY] <= end)
]
if actuals_window.empty:
    st.warning(
        "No actuals in the selected window. Showing quote-only data where available."
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

task_label_col = next(
    (c for c in ["[Job Task] Name", "Task Name", "Task"] if c in dim_job_task_quote.columns),
    COL_TASK_KEY,
)
task_map = (
    dim_job_task_quote[[COL_TASK_KEY, task_label_col]]
    .drop_duplicates()
    .dropna()
    .astype(str)
)
task_labels = {row[COL_TASK_KEY]: f"{row[COL_TASK_KEY]} | {row[task_label_col]}" for _, row in task_map.iterrows()}
task_keys = sorted(task_labels.keys())

if not task_keys:
    st.info("No tasks available for this job with the current filters.")
    st.stop()

selected_tasks = st.multiselect(
    "Task",
    options=task_keys,
    default=task_keys[:1],
    format_func=lambda k: task_labels.get(k, k),
)
if not selected_tasks:
    st.stop()

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

task_summary = (
    task_month.groupby([COL_JOB_KEY, COL_TASK_KEY], as_index=False)
    .agg(
        Actual_Hours_Total=(COL_HOURS, "sum"),
        Actual_Revenue=(COL_ALLOCATED_REVENUE, "sum"),
        Actual_Cost=(COL_COST, "sum"),
    )
    .merge(dim_job_task_quote, on=[COL_JOB_KEY, COL_TASK_KEY], how="outer")
    .merge(quote_by_task, on=[COL_JOB_KEY, COL_TASK_KEY], how="left")
)
task_summary["Quoted_Amount_Mode"] = task_summary["Quoted_Amount_Mode"].where(
    task_summary["Quoted_Amount_Mode"].notna(), task_summary[COL_QUOTED_AMOUNT]
)
task_summary["Quoted_Time_Mode"] = task_summary["Quoted_Time_Mode"].where(
    task_summary["Quoted_Time_Mode"].notna(), task_summary[COL_QUOTED_TIME]
)

task_summary["Realized_Rate"] = safe_divide(
    task_summary["Actual_Revenue"], task_summary["Actual_Hours_Total"]
)
task_summary["Base_Rate_Eff"] = task_summary[COL_BASE_RATE]
task_summary["Base_Rate_Eff"] = task_summary["Base_Rate_Eff"].where(
    task_summary["Base_Rate_Eff"].notna(), task_summary[COL_BASE_RATE_QUOTE]
)
task_summary["Base_Rate_Eff"] = task_summary["Base_Rate_Eff"].where(
    task_summary["Base_Rate_Eff"].notna(),
    safe_divide(task_summary["Actual_Cost"], task_summary["Actual_Hours_Total"]),
)
task_summary["Planned_Cost"] = task_summary["Quoted_Time_Mode"] * task_summary["Base_Rate_Eff"]

selected_summary = task_summary[task_summary[COL_TASK_KEY].isin(selected_tasks)]

actual_hours = selected_summary["Actual_Hours_Total"].sum(min_count=1)
actual_revenue = selected_summary["Actual_Revenue"].sum(min_count=1)
actual_cost = selected_summary["Actual_Cost"].sum(min_count=1)
actual_profit = actual_revenue - actual_cost
actual_margin = safe_divide(actual_profit, actual_revenue)

quoted_hours = selected_summary["Quoted_Time_Mode"].sum(min_count=1)
quoted_amount = selected_summary["Quoted_Amount_Mode"].sum(min_count=1)
quoted_rate = safe_divide(quoted_amount, quoted_hours)

hours_variance = actual_hours - quoted_hours if quoted_hours is not None else None
revenue_variance = actual_revenue - quoted_amount if quoted_amount is not None else None
cost_variance = actual_cost - selected_summary["Planned_Cost"].sum(min_count=1)

st.subheader("Task KPIs")
row1 = st.columns(4)
row1[0].metric("Actual Revenue", _fmt_currency(actual_revenue))
row1[1].metric("Actual Cost", _fmt_currency(actual_cost))
row1[2].metric("Actual Profit", _fmt_currency(actual_profit))
row1[3].metric(
    "Actual Margin",
    f"{actual_margin:.1%}" if actual_margin is not None and not pd.isna(actual_margin) else "N/A",
)

row2 = st.columns(4)
row2[0].metric("Actual Hours", _fmt_hours(actual_hours))
row2[1].metric("Quoted Hours", _fmt_hours(quoted_hours))
row2[2].metric("Quoted Amount", _fmt_currency(quoted_amount))
row2[3].metric("Quoted Rate", _fmt_rate(quoted_rate))

row3 = st.columns(3)
row3[0].metric("Hours Variance", _fmt_hours(hours_variance))
row3[1].metric("Revenue Variance", _fmt_currency(revenue_variance))
row3[2].metric("Cost Variance", _fmt_currency(cost_variance))

st.subheader("Monthly P&L")
task_month_sel = task_month[task_month[COL_TASK_KEY].isin(selected_tasks)]
if task_month_sel.empty:
    st.info("No actuals for the selected tasks in this window.")
else:
    monthly = (
        task_month_sel.groupby(COL_MONTH_KEY, as_index=False)
        .agg(Revenue=(COL_ALLOCATED_REVENUE, "sum"), Cost=(COL_COST, "sum"), Hours=(COL_HOURS, "sum"))
        .sort_values(COL_MONTH_KEY)
    )
    monthly["Profit"] = monthly["Revenue"] - monthly["Cost"]

    pnl_long = monthly.melt(
        id_vars=[COL_MONTH_KEY],
        value_vars=["Revenue", "Cost", "Profit"],
        var_name="Metric",
        value_name="Value",
    )
    pnl_chart = (
        alt.Chart(pnl_long)
        .mark_line(point=True)
        .encode(
            x=alt.X(f"{COL_MONTH_KEY}:T", title="Month"),
            y=alt.Y("Value:Q", title="Amount"),
            color="Metric:N",
            tooltip=[COL_MONTH_KEY, "Metric", "Value"],
        )
        .properties(height=260)
    )
    st.altair_chart(pnl_chart, width="stretch")

    st.subheader("Rate Convergence")
    monthly["Realized_Rate"] = safe_divide(monthly["Revenue"], monthly["Hours"])
    monthly["Base_Cost_Rate"] = safe_divide(monthly["Cost"], monthly["Hours"])
    billable = (
        task_month_sel.groupby(COL_MONTH_KEY)[["Billable_Rate_Eff", COL_HOURS]]
        .apply(
            lambda g: (g["Billable_Rate_Eff"] * g[COL_HOURS]).sum() / g[COL_HOURS].sum()
            if g[COL_HOURS].sum() > 0
            else None
        )
        .reset_index(name="Billable_Rate_Wtd")
    )
    monthly = monthly.merge(billable, on=COL_MONTH_KEY, how="left")
    monthly["Quoted_Rate"] = quoted_rate
    rate_long = monthly.melt(
        id_vars=[COL_MONTH_KEY],
        value_vars=["Realized_Rate", "Base_Cost_Rate", "Billable_Rate_Wtd", "Quoted_Rate"],
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

st.subheader("Diagnostics")
ghost = task_month_sel[task_month_sel["Ghost_Work_Month"]]
below = task_month_sel[task_month_sel["Below_Cost_Month"]]

diag_cols = st.columns(2)
with diag_cols[0]:
    st.markdown("**Ghost Work Months**")
    if not ghost.empty:
        st.dataframe(
            ghost[[COL_MONTH_KEY, COL_HOURS, COL_ALLOCATED_REVENUE]],
            width="stretch",
            height=200,
        )
    else:
        st.write("No ghost work months in selection.")
with diag_cols[1]:
    st.markdown("**Below-Cost Months**")
    if not below.empty:
        st.dataframe(
            below[[COL_MONTH_KEY, COL_ALLOCATED_REVENUE, COL_COST, "Realized_Rate", "Base_Cost_Rate"]],
            width="stretch",
            height=200,
        )
    else:
        st.write("No below-cost months in selection.")

st.subheader("Staff Involvement (Descriptive)")
if "[Staff] Name" in task_month_sel.columns:
    staff = (
        task_month_sel.groupby(COL_MONTH_KEY)["[Staff] Name"]
        .apply(lambda s: ", ".join(sorted(set(s.dropna().astype(str)))))
        .reset_index(name="Staff")
    )
    st.dataframe(staff, width="stretch", height=240)
else:
    st.write("No staff data available in this dataset.")

render_data_integrity(
    task_month,
    unallocated,
    dim_job_month_window,
    dim_job_task_quote,
    include_unallocated=filters["include_unallocated"],
    quote_mode=quote_mode,
    is_lifetime=is_lifetime,
)
