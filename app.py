import altair as alt
import pandas as pd
import streamlit as st

from lib.constants import (
    COL_ALLOCATED_REVENUE,
    COL_AMOUNT,
    COL_COST,
    COL_HOURS,
    COL_JOB_KEY,
    COL_MONTH_KEY,
    COL_QUOTED_AMOUNT,
    COL_QUOTED_TIME,
    COL_TASK_KEY,
)
from lib.data_loader import load_data_enriched
from lib.metrics import (
    add_earned_quote_task_month,
    build_job_task_rates,
    build_task_summary,
    compute_quote_by_task,
    safe_divide,
)
from lib.qa import render_data_integrity
from lib.semantic import build_dim_job_month, build_dim_job_task_quote
from lib.ui import apply_filters, render_sidebar

MARGIN_TARGET = 0.35

st.set_page_config(page_title="Profitability Story", layout="wide")


def _fmt_currency(value):
    if value is None or pd.isna(value):
        return "N/A"
    abs_value = abs(value)
    if abs_value >= 1_000_000:
        return f"${value/1_000_000:,.1f}M"
    if abs_value >= 1_000:
        return f"${value/1_000:,.1f}K"
    return f"${value:,.0f}"


def _fmt_rate(value):
    if value is None or pd.isna(value):
        return "N/A"
    return f"${value:,.0f}/hr"


def _fmt_hours(value):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:,.0f}h"


def _fmt_pct(value):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.1%}"


def _status_label(value, good, watch):
    if value is None or pd.isna(value):
        return "N/A"
    if value >= good:
        return "Good"
    if value >= watch:
        return "Watch"
    return "Risk"


def _render_hero(title: str, model: str, pills: list[str]) -> None:
    pills_html = "".join([f"<span class='pill'>{pill}</span>" for pill in pills])
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-title">{title}</div>
            <div class="hero-model">{model}</div>
            <div class="hero-pills">{pills_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _resolve_product_key(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    for col in ["Product", "Deliverable", "Business Unit", "Function"]:
        if col in df.columns:
            df = df.copy()
            df["Product_Key"] = df[col].fillna("Unspecified")
            return df, col
    df = df.copy()
    df["Product_Key"] = "Unspecified"
    return df, "Unspecified"


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


def _group_metrics(task_month: pd.DataFrame, earned_task_month: pd.DataFrame, group_col: str) -> pd.DataFrame:
    actual = (
        task_month.groupby(group_col, as_index=False)
        .agg(
            Actual_Cost=(COL_COST, "sum"),
            Actual_Hours=(COL_HOURS, "sum"),
            Job_Count=(COL_JOB_KEY, "nunique"),
        )
    )
    earned = (
        earned_task_month.groupby(group_col, as_index=False)
        .agg(
            Quoted_Revenue=("Earned_Quoted_Amount", "sum"),
            Quoted_Hours=("Earned_Quoted_Hours", "sum"),
            Benchmark=("Benchmark_Revenue", "sum"),
            Planned_Cost=("Planned_Cost", "sum"),
        )
    )
    summary = actual.merge(earned, on=group_col, how="outer").fillna(0)
    summary["Profit"] = summary["Quoted_Revenue"] - summary["Actual_Cost"]
    summary["Margin"] = safe_divide(summary["Profit"], summary["Quoted_Revenue"])
    summary["Quote_Gap"] = summary["Benchmark"] - summary["Quoted_Revenue"]
    summary["Quote_Gap_Pct"] = safe_divide(summary["Quote_Gap"], summary["Benchmark"])
    summary["Overrun"] = summary["Actual_Cost"] - summary["Planned_Cost"]
    summary["Hours_Variance"] = summary["Actual_Hours"] - summary["Quoted_Hours"]
    summary["Hours_Variance_Pct"] = safe_divide(summary["Hours_Variance"], summary["Quoted_Hours"])
    summary["Margin_Status"] = summary["Margin"].apply(lambda v: _status_label(v, MARGIN_TARGET, 0.2))
    return summary


st.markdown(
    """
    <style>
    .hero {padding: 18px 22px; background: linear-gradient(135deg, #0f1f2b 0%, #1a394a 100%); border-radius: 14px; margin-bottom: 18px;}
    .hero-title {font-family: 'Sora', sans-serif; font-size: 28px; color: #f7fbff; font-weight: 700;}
    .hero-model {font-family: 'Sora', sans-serif; font-size: 14px; color: #c6d7e1; margin-top: 6px;}
    .hero-pills {margin-top: 10px;}
    .pill {display: inline-block; padding: 4px 10px; border-radius: 999px; background: #2b586e; color: #eaf3f8; font-size: 12px; margin-right: 6px;}
    </style>
    """,
    unsafe_allow_html=True,
)

_render_hero(
    "Profitability Story",
    "Revenue (Quoted) - Benchmark (Quoted x Standard Rate) - Cost (Actual)",
    ["Pricing Discipline", "Scope Control", "Margin Health", "Delivery Efficiency"],
)

st.caption("Executive view first, then drill to jobs and tasks.")

with st.expander("Understanding this dashboard"):
    st.write(
        "Revenue = Quoted Amount. Benchmark = Quoted Hours x Standard Billable Rate. "
        "Cost = Actual Hours x Cost Rate. Margin = Revenue - Cost. "
        "Pricing failure = underquoting vs benchmark. Delivery failure = cost or hours overrun."
    )

raw = load_data_enriched()
filters = render_sidebar(raw)
filtered = apply_filters(raw, filters)
if filtered.empty:
    st.error("Current filters exclude all data. Use Reset Filters in the sidebar.")
    st.stop()

filtered, product_source = _resolve_product_key(filtered)

fy_labels = filters.get("fy_labels", [])
actuals_window = filtered[
    filtered[COL_MONTH_KEY].notna() & filtered["FY_Label"].isin(fy_labels)
]
if actuals_window.empty:
    actuals_window = filtered[filtered[COL_MONTH_KEY].notna()]
    if not actuals_window.empty:
        st.warning("No actuals for the selected FYs; showing all available FYs instead.")

if actuals_window.empty:
    st.warning("No actuals available in the dataset.")
    st.stop()

scope_fys = sorted(actuals_window["FY_Label"].dropna().astype(str).unique().tolist())

if filters.get("quote_mode") == "Lifetime Quote" and len(scope_fys) > 1:
    st.warning("Lifetime quote selected while viewing a period window.")


task_month = actuals_window[actuals_window[COL_TASK_KEY] != "__UNALLOCATED__"].copy()
unallocated = actuals_window[actuals_window[COL_TASK_KEY] == "__UNALLOCATED__"].copy()

task_month_lifetime = filtered[
    filtered[COL_MONTH_KEY].notna() & (filtered[COL_TASK_KEY] != "__UNALLOCATED__")
].copy()

dim_job_month = build_dim_job_month(filtered[filtered[COL_MONTH_KEY].notna()])
dim_job_task_quote = build_dim_job_task_quote(filtered)
job_task_rates = build_job_task_rates(task_month_lifetime, dim_job_task_quote)

earned_task_month = add_earned_quote_task_month(
    task_month, task_month_lifetime, dim_job_task_quote, job_task_rates
)

quote_rates = dim_job_task_quote.merge(
    job_task_rates, on=[COL_JOB_KEY, COL_TASK_KEY], how="left"
)

quoted_revenue_earned = earned_task_month["Earned_Quoted_Amount"].sum(min_count=1)
quoted_hours_earned = earned_task_month["Earned_Quoted_Hours"].sum(min_count=1)
benchmark_earned = earned_task_month["Benchmark_Revenue"].sum(min_count=1)
planned_cost_earned = earned_task_month["Planned_Cost"].sum(min_count=1)

quoted_revenue_lifetime = quote_rates[COL_QUOTED_AMOUNT].sum(min_count=1)
quoted_hours_lifetime = quote_rates[COL_QUOTED_TIME].sum(min_count=1)
benchmark_lifetime = (quote_rates[COL_QUOTED_TIME] * quote_rates["Billable_Rate_Eff"]).sum(
    min_count=1
)
planned_cost_lifetime = (quote_rates[COL_QUOTED_TIME] * quote_rates["Base_Rate_Eff"]).sum(
    min_count=1
)

if filters.get("quote_mode") == "Lifetime Quote":
    quoted_revenue = quoted_revenue_lifetime
    quoted_hours = quoted_hours_lifetime
    benchmark_revenue = benchmark_lifetime
    planned_cost = planned_cost_lifetime
else:
    quoted_revenue = quoted_revenue_earned
    quoted_hours = quoted_hours_earned
    benchmark_revenue = benchmark_earned
    planned_cost = planned_cost_earned

recognized_revenue = dim_job_month[COL_AMOUNT].sum(min_count=1)
actual_cost = task_month[COL_COST].sum()
actual_hours = task_month[COL_HOURS].sum()
profit = quoted_revenue - actual_cost
margin = safe_divide(profit, quoted_revenue)
quote_gap = benchmark_revenue - quoted_revenue
quote_gap_pct = safe_divide(quote_gap, benchmark_revenue)

hours_overrun = actual_hours - quoted_hours
hours_overrun_pct = safe_divide(hours_overrun, quoted_hours)

unallocated_revenue = unallocated[COL_ALLOCATED_REVENUE].sum()
total_allocated = task_month[COL_ALLOCATED_REVENUE].sum() + unallocated_revenue
unallocated_pct = safe_divide(unallocated_revenue, total_allocated)

st.caption(
    f"FYs: {', '.join(scope_fys)} | Quote mode: {filters.get('quote_mode')} | Product key: {product_source}"
)

st.markdown(
    f"**Headline:** {_status_label(margin, MARGIN_TARGET, 0.2)} margin at {_fmt_pct(margin)}; "
    f"quote gap {_status_label(quote_gap_pct, 0.05, 0.02)} at {_fmt_pct(quote_gap_pct)}."
)

st.subheader("1) Performance at a glance")
row1 = st.columns(4)
row1[0].metric("Quoted Revenue", _fmt_currency(quoted_revenue))
row1[1].metric("Benchmark", _fmt_currency(benchmark_revenue))
row1[2].metric("Cost", _fmt_currency(actual_cost))
row1[3].metric("Profit", _fmt_currency(profit))

row2 = st.columns(4)
row2[0].metric("Margin %", _fmt_pct(margin))
row2[1].metric("Quote Gap $", _fmt_currency(quote_gap))
row2[2].metric("Quote Gap %", _fmt_pct(quote_gap_pct))
row2[3].metric("Recognized Revenue", _fmt_currency(recognized_revenue))

row3 = st.columns(4)
row3[0].metric("Actual Hours", _fmt_hours(actual_hours))
row3[1].metric("Quoted Hours", _fmt_hours(quoted_hours))
row3[2].metric("Hours Overrun", _fmt_hours(hours_overrun))
row3[3].metric("Overrun %", _fmt_pct(hours_overrun_pct))

row4 = st.columns(4)
row4[0].metric("Effective Rate", _fmt_rate(safe_divide(quoted_revenue, actual_hours)))
row4[1].metric("Cost Rate", _fmt_rate(safe_divide(actual_cost, actual_hours)))
row4[2].metric("Unallocated Revenue", _fmt_currency(unallocated_revenue))
row4[3].metric("Unallocated %", _fmt_pct(unallocated_pct))

st.subheader("2) What changed over time")
trend_df = earned_task_month.copy()
if trend_df.empty:
    st.info("No trend data for the selected FYs.")
else:
    trend_df["FY_Label"] = trend_df["FY_Label"].astype(str)
    fy_trend = (
        trend_df.groupby("FY_Label", as_index=False)
        .agg(
            Revenue=("Earned_Quoted_Amount", "sum"),
            Benchmark=("Benchmark_Revenue", "sum"),
            Planned_Cost=("Planned_Cost", "sum"),
        )
        .sort_values("FY_Label")
    )
    cost_by_fy = (
        task_month.groupby("FY_Label", as_index=False)[COL_COST]
        .sum()
        .rename(columns={COL_COST: "Cost"})
    )
    fy_trend = fy_trend.merge(cost_by_fy, on="FY_Label", how="left")
    fy_trend["Profit"] = fy_trend["Revenue"] - fy_trend["Cost"]
    fy_trend["Margin"] = safe_divide(fy_trend["Profit"], fy_trend["Revenue"])
    fy_trend["Quote_Gap_Pct"] = safe_divide(
        fy_trend["Benchmark"] - fy_trend["Revenue"], fy_trend["Benchmark"]
    )

    chart = (
        alt.Chart(fy_trend)
        .mark_bar(color="#2b586e")
        .encode(x="FY_Label:N", y="Revenue:Q", tooltip=["FY_Label", "Revenue", "Profit"])
        .properties(height=260)
    )
    line_margin = (
        alt.Chart(fy_trend)
        .mark_line(color="#f2b544", point=True)
        .encode(x="FY_Label:N", y=alt.Y("Margin:Q", axis=alt.Axis(format="%")))
    )
    line_gap = (
        alt.Chart(fy_trend)
        .mark_line(color="#cc5f5f", point=True)
        .encode(x="FY_Label:N", y=alt.Y("Quote_Gap_Pct:Q", axis=alt.Axis(format="%")))
    )
    st.altair_chart(chart + line_margin + line_gap, width="stretch")

st.subheader("3) Where is it happening")
if "Department_Eff" in task_month.columns:
    dept_summary = _group_metrics(task_month, earned_task_month, "Department_Eff")
    dept_summary = dept_summary.sort_values("Profit", ascending=False)
    st.dataframe(dept_summary, width="stretch", height=260)
else:
    st.info("No Department data available.")

if "Department_Eff" in task_month.columns:
    dept_options = sorted(task_month["Department_Eff"].dropna().unique())
    if dept_options:
        selected_dept = st.selectbox("Drill to Department", dept_options)
    else:
        selected_dept = None
else:
    selected_dept = None

if selected_dept:
    dept_task_month = task_month[task_month["Department_Eff"] == selected_dept]
    dept_earned = earned_task_month[earned_task_month["Department_Eff"] == selected_dept]
    product_summary = _group_metrics(dept_task_month, dept_earned, "Product_Key")
    product_summary = product_summary.sort_values("Profit", ascending=False)
    st.dataframe(product_summary, width="stretch", height=240)

st.subheader("4) Drill to jobs and tasks")
job_summary = _group_metrics(task_month, earned_task_month, COL_JOB_KEY)
job_summary = job_summary.sort_values("Profit", ascending=False)

job_map = _job_label_map(filtered)
job_keys = job_summary[COL_JOB_KEY].astype(str).tolist()
selected_job = None
if job_keys:
    selected_job = st.selectbox(
        "Select a job",
        options=job_keys,
        format_func=lambda k: job_map.get(k, k),
    )
    job_view = job_summary[job_summary[COL_JOB_KEY] == selected_job]
    st.dataframe(job_view, width="stretch", height=120)
else:
    st.info("No jobs available.")

if selected_job:
    job_task_month = task_month[task_month[COL_JOB_KEY] == selected_job]
    job_task_lifetime = task_month_lifetime[task_month_lifetime[COL_JOB_KEY] == selected_job]
    job_dim_quote = dim_job_task_quote[dim_job_task_quote[COL_JOB_KEY] == selected_job]
    task_summary = build_task_summary(
        job_task_month, job_dim_quote, filters.get("include_quote_only", True)
    )
    quote_mode = filters.get("quote_mode", "Earned Quote Proxy")
    aligned_quote = compute_quote_by_task(
        quote_mode, job_task_month, job_task_lifetime, job_dim_quote
    )
    task_summary = task_summary.merge(
        aligned_quote, on=[COL_JOB_KEY, COL_TASK_KEY], how="left"
    )
    job_rates = job_task_rates[job_task_rates[COL_JOB_KEY] == selected_job]
    task_summary = task_summary.merge(job_rates, on=[COL_JOB_KEY, COL_TASK_KEY], how="left")
    task_summary["Hours_Variance"] = (
        task_summary["Actual_Hours_Total"] - task_summary["Quoted_Time_Mode"]
    )
    task_summary["Underquoted"] = task_summary["Quoted_Rate"] < task_summary["Billable_Rate_Eff"]
    task_summary["Overrun"] = task_summary["Hours_Variance"] > 0

    show_cols = [
        COL_TASK_KEY,
        "Actual_Hours_Total",
        "Quoted_Time_Mode",
        "Hours_Variance",
        "Actual_Revenue",
        "Quoted_Amount_Mode",
        "Actual_Cost",
        "Actual_Margin",
        "Underquoted",
        "Overrun",
        "Scope_Overrun",
        "Unquoted_Work",
    ]
    available_cols = [col for col in show_cols if col in task_summary.columns]
    st.dataframe(task_summary[available_cols], width="stretch", height=360)

render_data_integrity(
    task_month,
    unallocated,
    dim_job_month,
    dim_job_task_quote,
    include_unallocated=filters.get("include_unallocated", True),
    quote_mode=filters.get("quote_mode", "Earned Quote Proxy"),
    is_lifetime=False,
)
