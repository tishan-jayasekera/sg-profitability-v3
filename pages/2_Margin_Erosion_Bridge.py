import altair as alt
import pandas as pd
import streamlit as st

from lib.constants import (
    COL_ALLOCATED_REVENUE,
    COL_BASE_RATE,
    COL_BASE_RATE_QUOTE,
    COL_BILLABLE_RATE,
    COL_BILLABLE_RATE_QUOTE,
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


def _waterfall_data(potential, discounting, overrun, leakage, actual):
    steps = [
        ("Potential Profit", potential),
        ("Discounting", -discounting),
        ("Delivery Overrun", -overrun),
        ("Revenue Leakage", -leakage),
        ("Actual Profit", actual),
    ]
    data = []
    running = 0.0
    for i, (label, value) in enumerate(steps):
        if label == "Actual Profit":
            data.append(
                {"Step": label, "Start": 0.0, "End": value, "Value": value, "Total": True}
            )
            continue
        start = running
        end = running + value
        data.append({"Step": label, "Start": start, "End": end, "Value": value, "Total": False})
        running = end
    return pd.DataFrame(data)


st.set_page_config(page_title="Margin Erosion Bridge", layout="wide")
st.title("Margin Erosion Bridge")
st.caption("Decompose margin erosion into discounting, delivery overrun, and revenue leakage.")

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
task_summary["Quoted_Rate_Mode"] = safe_divide(
    task_summary["Quoted_Amount_Mode"], task_summary["Quoted_Time_Mode"]
)

task_summary["Billable_Rate_Eff"] = task_summary[COL_BILLABLE_RATE]
task_summary["Billable_Rate_Eff"] = task_summary["Billable_Rate_Eff"].where(
    task_summary["Billable_Rate_Eff"].notna(), task_summary[COL_BILLABLE_RATE_QUOTE]
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
task_summary["RateCard_Revenue"] = (
    task_summary["Quoted_Time_Mode"] * task_summary["Billable_Rate_Eff"]
)

task_summary["Discounting_Erosion_Task"] = (
    task_summary["Billable_Rate_Eff"] - task_summary["Quoted_Rate_Mode"]
) * task_summary["Quoted_Time_Mode"]
task_summary["Delivery_Overrun_Task"] = task_summary["Actual_Cost"] - task_summary["Planned_Cost"]
task_summary["Leakage_Task"] = task_summary["Quoted_Amount_Mode"] - task_summary["Actual_Revenue"]

quoted_amount = task_summary["Quoted_Amount_Mode"].sum(min_count=1)
quoted_hours = task_summary["Quoted_Time_Mode"].sum(min_count=1)
planned_cost = task_summary["Planned_Cost"].sum(min_count=1)
ratecard_revenue = task_summary["RateCard_Revenue"].sum(min_count=1)

actual_revenue = task_month[COL_ALLOCATED_REVENUE].sum()
if filters["include_unallocated"]:
    actual_revenue += unallocated[COL_ALLOCATED_REVENUE].sum()
actual_cost = task_month[COL_COST].sum()
actual_profit = actual_revenue - actual_cost

potential_profit = ratecard_revenue - planned_cost
discounting = ratecard_revenue - quoted_amount
delivery_overrun = actual_cost - planned_cost
revenue_leakage = quoted_amount - actual_revenue

closure = potential_profit - discounting - delivery_overrun - revenue_leakage
closure_delta = closure - actual_profit

st.subheader("Bridge Summary")
cols = st.columns(5)
cols[0].metric("Quoted Amount", _fmt_currency(quoted_amount))
cols[1].metric("Actual Revenue", _fmt_currency(actual_revenue))
cols[2].metric("Actual Cost", _fmt_currency(actual_cost))
cols[3].metric("Actual Profit", _fmt_currency(actual_profit))
cols[4].metric(
    "Closure Delta",
    _fmt_currency(closure_delta),
    help="Potential Profit minus erosion components vs Actual Profit (should be ~0).",
)

waterfall_df = _waterfall_data(
    potential_profit, discounting, delivery_overrun, revenue_leakage, actual_profit
)
waterfall_chart = (
    alt.Chart(waterfall_df)
    .mark_bar()
    .encode(
        x=alt.X("Step:N", title=""),
        y=alt.Y("End:Q", title="Profit"),
        y2="Start:Q",
        color=alt.condition("datum.Total", alt.value("#2b7c85"), alt.value("#5a6b7a")),
        tooltip=["Step", "Value"],
    )
    .properties(height=320)
)
st.altair_chart(waterfall_chart, use_container_width=True)

st.subheader("Drivers")
driver_cols = st.columns(3)
top_discount = task_summary.sort_values("Discounting_Erosion_Task", ascending=False).head(15)
top_overrun = task_summary.sort_values("Delivery_Overrun_Task", ascending=False).head(15)
top_leakage = task_summary.sort_values("Leakage_Task", ascending=False).head(15)

task_label_col = next(
    (c for c in ["[Job Task] Name", "Task Name", "Task"] if c in task_summary.columns),
    COL_TASK_KEY,
)

with driver_cols[0]:
    st.markdown("**Discounting Erosion by Task**")
    st.dataframe(
        top_discount[[COL_TASK_KEY, task_label_col, "Discounting_Erosion_Task"]],
        use_container_width=True,
        height=260,
    )
with driver_cols[1]:
    st.markdown("**Delivery Overrun by Task**")
    st.dataframe(
        top_overrun[[COL_TASK_KEY, task_label_col, "Delivery_Overrun_Task"]],
        use_container_width=True,
        height=260,
    )
with driver_cols[2]:
    st.markdown("**Revenue Leakage by Task**")
    st.dataframe(
        top_leakage[[COL_TASK_KEY, task_label_col, "Leakage_Task"]],
        use_container_width=True,
        height=260,
    )

st.subheader("Root-Cause Scatter")
scatter = task_summary.copy()
scatter["Pricing_Eff"] = safe_divide(scatter["Quoted_Rate_Mode"], scatter["Billable_Rate_Eff"])
scatter["Delivery_Eff"] = safe_divide(scatter["Realized_Rate"], scatter["Quoted_Rate_Mode"])
scatter["Actual_Margin"] = safe_divide(
    scatter["Actual_Revenue"] - scatter["Actual_Cost"], scatter["Actual_Revenue"]
)
scatter_chart = (
    alt.Chart(scatter)
    .mark_circle(size=80, opacity=0.7)
    .encode(
        x=alt.X("Pricing_Eff:Q", title="Pricing Efficiency (Quoted / Billable)"),
        y=alt.Y("Delivery_Eff:Q", title="Delivery Efficiency (Realized / Quoted)"),
        size=alt.Size("Actual_Hours_Total:Q", title="Actual Hours"),
        color=alt.Color("Actual_Margin:Q", title="Actual Margin", scale=alt.Scale(scheme="tealblues")),
        tooltip=[COL_TASK_KEY, task_label_col, "Pricing_Eff", "Delivery_Eff", "Actual_Margin"],
    )
    .properties(height=320)
)
st.altair_chart(scatter_chart, use_container_width=True)

st.subheader("Coverage & Integrity")
billable_coverage = safe_divide(
    task_summary.loc[task_summary["Billable_Rate_Eff"].notna(), "Quoted_Time_Mode"].sum(min_count=1),
    quoted_hours,
)
base_coverage = safe_divide(
    task_summary.loc[task_summary["Base_Rate_Eff"].notna(), "Quoted_Time_Mode"].sum(min_count=1),
    quoted_hours,
)
unallocated_pct = safe_divide(unallocated[COL_ALLOCATED_REVENUE].sum(), actual_revenue)

unquoted_tasks = task_summary[
    (task_summary["Actual_Hours_Total"] > 0)
    & (task_summary["Quoted_Time_Mode"].isna() | (task_summary["Quoted_Time_Mode"] == 0))
].shape[0]
quote_only_tasks = task_summary[
    (task_summary["Actual_Hours_Total"].fillna(0) == 0)
    & (task_summary[COL_QUOTED_TIME] > 0)
].shape[0]

cov_cols = st.columns(4)
cov_cols[0].metric(
    "Ratecard Coverage",
    f"{billable_coverage:.1%}"
    if billable_coverage is not None and not pd.isna(billable_coverage)
    else "N/A",
)
cov_cols[1].metric(
    "Base Rate Coverage",
    f"{base_coverage:.1%}" if base_coverage is not None and not pd.isna(base_coverage) else "N/A",
)
cov_cols[2].metric(
    "Unallocated Revenue %",
    f"{unallocated_pct:.1%}" if unallocated_pct is not None and not pd.isna(unallocated_pct) else "N/A",
)
cov_cols[3].metric("Unquoted / Quote-only Tasks", f"{unquoted_tasks} / {quote_only_tasks}")

render_data_integrity(
    task_month,
    unallocated,
    dim_job_month_window,
    dim_job_task_quote,
    include_unallocated=filters["include_unallocated"],
    quote_mode=quote_mode,
    is_lifetime=is_lifetime,
)
