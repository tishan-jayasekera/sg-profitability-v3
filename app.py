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
    safe_divide,
)
from lib.qa import render_data_integrity
from lib.semantic import build_dim_job_month, build_dim_job_task_quote
from lib.ui import apply_filters, render_sidebar

MARGIN_TARGET = 0.35
MARGIN_WATCH = 0.2
QUOTE_GAP_WATCH = 0.05
QUOTE_GAP_RISK = 0.1


st.set_page_config(page_title="Profitability Cockpit", layout="wide")


@st.cache_data(show_spinner=False)
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


def _status_margin(value):
    if value is None or pd.isna(value):
        return "N/A"
    if value >= MARGIN_TARGET:
        return "Good"
    if value >= MARGIN_WATCH:
        return "Watch"
    return "Risk"


def _status_quote_gap(value):
    if value is None or pd.isna(value):
        return "N/A"
    if value >= QUOTE_GAP_RISK:
        return "Risk"
    if value >= QUOTE_GAP_WATCH:
        return "Watch"
    return "Good"


def _render_hero(title: str, model: str, pills: list[str]) -> None:
    pills_html = "".join(
        [
            f"<span class='pill'>{pill}</span>"
            for pill in pills
        ]
    )
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


def _render_callout(text: str) -> None:
    st.info(text)


def _resolve_product_key(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    for col in ["Product", "Deliverable", "Business Unit", "Function"]:
        if col in df.columns:
            df = df.copy()
            df["Product_Key"] = df[col].fillna("Unspecified")
            return df, col
    df = df.copy()
    df["Product_Key"] = "Unspecified"
    return df, "Unspecified"


def _group_metrics(
    task_month: pd.DataFrame,
    earned_task_month: pd.DataFrame,
    group_col: str,
) -> pd.DataFrame:
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
            Benchmark_Revenue=("Benchmark_Revenue", "sum"),
            Planned_Cost=("Planned_Cost", "sum"),
        )
    )
    summary = actual.merge(earned, on=group_col, how="outer")
    summary["Quoted_Revenue"] = summary["Quoted_Revenue"].fillna(0)
    summary["Actual_Cost"] = summary["Actual_Cost"].fillna(0)
    summary["Actual_Hours"] = summary["Actual_Hours"].fillna(0)
    summary["Benchmark_Revenue"] = summary["Benchmark_Revenue"].fillna(0)
    summary["Planned_Cost"] = summary["Planned_Cost"].fillna(0)

    summary["Profit"] = summary["Quoted_Revenue"] - summary["Actual_Cost"]
    summary["Margin"] = safe_divide(summary["Profit"], summary["Quoted_Revenue"])
    summary["Quote_Gap"] = summary["Benchmark_Revenue"] - summary["Quoted_Revenue"]
    summary["Quote_Gap_Pct"] = safe_divide(
        summary["Quote_Gap"], summary["Benchmark_Revenue"]
    )
    summary["Overrun"] = summary["Actual_Cost"] - summary["Planned_Cost"]
    summary["Effective_Rate"] = safe_divide(
        summary["Quoted_Revenue"], summary["Actual_Hours"]
    )
    summary["Cost_Rate"] = safe_divide(
        summary["Actual_Cost"], summary["Actual_Hours"]
    )
    summary["Margin_Status"] = summary["Margin"].apply(_status_margin)
    summary["Quote_Status"] = summary["Quote_Gap_Pct"].apply(_status_quote_gap)
    return summary


def _scope_info(filtered: pd.DataFrame, filters: dict) -> dict:
    scope_level = filters.get("scope_level", "Company")
    selections = {
        "Department": st.session_state.get("drill_department"),
        "Product": st.session_state.get("drill_product"),
        "Job": st.session_state.get("drill_job"),
    }
    breadcrumb = ["Company"]
    if selections["Department"]:
        breadcrumb.append(selections["Department"])
    if selections["Product"]:
        breadcrumb.append(selections["Product"])
    if selections["Job"]:
        breadcrumb.append(selections["Job"])

    scope_job_keys = None
    scope_label = "Company"
    warning = None
    if scope_level == "Department":
        dept = selections["Department"]
        if "Department_Eff" not in filtered.columns:
            warning = "Department field not available for this dataset."
        elif dept:
            scope_label = dept
            scope_job_keys = set(
                filtered.loc[filtered["Department_Eff"] == dept, COL_JOB_KEY].astype(str)
            )
        else:
            warning = "Pick a Department in the Drill-Down tab to activate Department scope."
    elif scope_level == "Product":
        product = selections["Product"]
        if "Product_Key" not in filtered.columns:
            warning = "Product field not available for this dataset."
        elif product:
            scope_label = product
            scope_job_keys = set(
                filtered.loc[filtered["Product_Key"] == product, COL_JOB_KEY].astype(str)
            )
        else:
            warning = "Pick a Product in the Drill-Down tab to activate Product scope."
    elif scope_level == "Job":
        job = selections["Job"]
        if job:
            scope_label = job
            scope_job_keys = {str(job)}
        else:
            warning = "Pick a Job in Drill-Down or Job Diagnosis to activate Job scope."

    return {
        "level": scope_level,
        "label": scope_label,
        "job_keys": scope_job_keys,
        "breadcrumb": " \u2192 ".join(breadcrumb),
        "warning": warning,
    }


def _filter_by_job(df: pd.DataFrame, job_keys: set[str] | None) -> pd.DataFrame:
    if job_keys is None or COL_JOB_KEY not in df.columns:
        return df
    return df[df[COL_JOB_KEY].astype(str).isin(job_keys)].copy()


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
    "Profitability Cockpit",
    "Revenue (Quoted) - Benchmark (Quoted x Standard Rate) - Cost (Actual)",
    ["Pricing Discipline", "Scope Control", "Margin Health", "Delivery Efficiency"],
)

st.caption("Company to task, executive-first, FY anchored.")

with st.expander("Understanding this dashboard"):
    st.write(
        "Revenue = Quoted Amount. Benchmark = Quoted Hours x Standard Billable Rate. "
        "Cost = Actual Hours x Cost Rate. Margin = Revenue - Cost. "
        "Pricing failure = underquoting vs benchmark. Delivery failure = cost overrun vs plan."
    )

with st.expander("Metric definitions"):
    st.write(
        "Quoted revenue is deduped at Job x Task. Benchmarks and costs are aggregated from "
        "task-months using weighted rates. All ratios are computed from sums, never from averages."
    )


# Load and filter data
raw = load_data_enriched()
filters = render_sidebar(raw)
filtered = apply_filters(raw, filters)
if filtered.empty:
    st.error("Current filters exclude all data. Use Reset Filters in the sidebar.")
    st.stop()

filtered, product_source = _resolve_product_key(filtered)

if st.sidebar.button("Clear Drill Path"):
    st.session_state.pop("drill_department", None)
    st.session_state.pop("drill_product", None)
    st.session_state.pop("drill_job", None)

filters_version = st.session_state.get("filters_version", 0)
cache = st.session_state.get("portfolio_cache")
cache_version = st.session_state.get("portfolio_cache_version")

if cache and cache_version == filters_version:
    data = cache
else:
    fy_labels = filters.get("fy_labels", [])
    actuals_window = filtered[
        filtered[COL_MONTH_KEY].notna() & filtered["FY_Label"].isin(fy_labels)
    ]
    fy_labels_effective = fy_labels
    fallback_all_fy = False
    if actuals_window.empty:
        actuals_all = filtered[filtered[COL_MONTH_KEY].notna()]
        if not actuals_all.empty:
            fallback_all_fy = True
            actuals_window = actuals_all
            fy_labels_effective = (
                actuals_all["FY_Label"].dropna().astype(str).unique().tolist()
            )
    task_month = actuals_window[actuals_window[COL_TASK_KEY] != "__UNALLOCATED__"].copy()
    unallocated = actuals_window[actuals_window[COL_TASK_KEY] == "__UNALLOCATED__"].copy()

    task_month_lifetime = filtered[
        filtered[COL_MONTH_KEY].notna() & (filtered[COL_TASK_KEY] != "__UNALLOCATED__")
    ].copy()

    dim_job_month = build_dim_job_month(filtered[filtered[COL_MONTH_KEY].notna()])
    dim_job_month_window = dim_job_month[dim_job_month["FY_Label"].isin(fy_labels_effective)]
    dim_job_task_quote = build_dim_job_task_quote(filtered)

    job_task_rates = build_job_task_rates(task_month_lifetime, dim_job_task_quote)
    earned_task_month = add_earned_quote_task_month(
        task_month, task_month_lifetime, dim_job_task_quote, job_task_rates
    )

    data = {
        "fy_labels_effective": fy_labels_effective,
        "fallback_all_fy": fallback_all_fy,
        "task_month": task_month,
        "unallocated": unallocated,
        "task_month_lifetime": task_month_lifetime,
        "dim_job_month_window": dim_job_month_window,
        "dim_job_task_quote": dim_job_task_quote,
        "job_task_rates": job_task_rates,
        "earned_task_month": earned_task_month,
    }
    st.session_state["portfolio_cache"] = data
    st.session_state["portfolio_cache_version"] = filters_version

scope = _scope_info(filtered, filters)
scope_level = scope["level"]
scope_job_keys = scope["job_keys"]
scope_data = {
    "task_month": _filter_by_job(data["task_month"], scope_job_keys),
    "unallocated": _filter_by_job(data["unallocated"], scope_job_keys),
    "task_month_lifetime": _filter_by_job(data["task_month_lifetime"], scope_job_keys),
    "dim_job_month_window": _filter_by_job(data["dim_job_month_window"], scope_job_keys),
    "dim_job_task_quote": _filter_by_job(data["dim_job_task_quote"], scope_job_keys),
    "job_task_rates": _filter_by_job(data["job_task_rates"], scope_job_keys),
    "earned_task_month": _filter_by_job(data["earned_task_month"], scope_job_keys),
}
filtered_scope = _filter_by_job(filtered, scope_job_keys)

fy_labels_effective = data["fy_labels_effective"]
scope_fy_labels = (
    sorted(scope_data["task_month"]["FY_Label"].dropna().astype(str).unique().tolist())
    if not scope_data["task_month"].empty
    else fy_labels_effective
)
if data["fallback_all_fy"]:
    st.warning(
        "No actuals for the selected FYs; showing all available FYs instead. "
        f"Available FYs: {', '.join(sorted(set(fy_labels_effective)))}"
    )
if scope["warning"]:
    st.info(scope["warning"])

st.caption(
    f"Scope: {scope['level']} ({scope['label']}) | "
    f"FYs: {', '.join(scope_fy_labels)} | "
    f"Quote mode: {filters.get('quote_mode', 'Earned Quote Proxy')} | "
    f"Product key: {product_source}"
)
st.caption(f"Breadcrumbs: {scope['breadcrumb']}")

if scope_data["task_month"].empty:
    st.warning("No actuals in the selected window for this scope. Try All FYs.")

# Compute portfolio KPIs
quote_mode = filters.get("quote_mode", "Earned Quote Proxy")
quote_rates = scope_data["dim_job_task_quote"].merge(
    scope_data["job_task_rates"], on=[COL_JOB_KEY, COL_TASK_KEY], how="left"
)

quoted_revenue_earned = scope_data["earned_task_month"]["Earned_Quoted_Amount"].sum(min_count=1)
quoted_hours_earned = scope_data["earned_task_month"]["Earned_Quoted_Hours"].sum(min_count=1)
benchmark_earned = scope_data["earned_task_month"]["Benchmark_Revenue"].sum(min_count=1)
planned_cost_earned = scope_data["earned_task_month"]["Planned_Cost"].sum(min_count=1)

quoted_revenue_lifetime = quote_rates[COL_QUOTED_AMOUNT].sum(min_count=1)
quoted_hours_lifetime = quote_rates[COL_QUOTED_TIME].sum(min_count=1)
benchmark_lifetime = (quote_rates[COL_QUOTED_TIME] * quote_rates["Billable_Rate_Eff"]).sum(
    min_count=1
)
planned_cost_lifetime = (quote_rates[COL_QUOTED_TIME] * quote_rates["Base_Rate_Eff"]).sum(
    min_count=1
)

if quote_mode == "Lifetime Quote":
    quoted_revenue = quoted_revenue_lifetime
    quoted_hours = quoted_hours_lifetime
    benchmark_revenue = benchmark_lifetime
    planned_cost = planned_cost_lifetime
    if scope_data["task_month"].empty is False:
        st.warning("Lifetime quote is not aligned with period actuals.")
else:
    quoted_revenue = quoted_revenue_earned
    quoted_hours = quoted_hours_earned
    benchmark_revenue = benchmark_earned
    planned_cost = planned_cost_earned

recognized_revenue = scope_data["dim_job_month_window"][COL_AMOUNT].sum(min_count=1)
actual_cost = scope_data["task_month"][COL_COST].sum()
actual_hours = scope_data["task_month"][COL_HOURS].sum()
profit = quoted_revenue - actual_cost
margin = safe_divide(profit, quoted_revenue)
quote_gap = benchmark_revenue - quoted_revenue
quote_gap_pct = safe_divide(quote_gap, benchmark_revenue)

unallocated_revenue = scope_data["unallocated"][COL_ALLOCATED_REVENUE].sum()
total_allocated = (
    scope_data["task_month"][COL_ALLOCATED_REVENUE].sum() + unallocated_revenue
)
unallocated_pct = safe_divide(unallocated_revenue, total_allocated)
billable_coverage = safe_divide(
    scope_data["earned_task_month"]
    .loc[scope_data["earned_task_month"]["Billable_Rate_Eff"].notna(), "Earned_Quoted_Hours"]
    .sum(min_count=1),
    quoted_hours_earned,
)
base_rate_coverage = safe_divide(
    scope_data["earned_task_month"]
    .loc[scope_data["earned_task_month"]["Base_Rate_Eff"].notna(), "Earned_Quoted_Hours"]
    .sum(min_count=1),
    quoted_hours_earned,
)
job_count = scope_data["task_month"][COL_JOB_KEY].nunique()

# Precompute summaries used across tabs
if scope_data["task_month"].empty:
    job_summary = pd.DataFrame()
else:
    job_summary = _group_metrics(scope_data["task_month"], scope_data["earned_task_month"], COL_JOB_KEY)

# Tabs
TAB_TITLES = [
    "Executive Summary",
    "Trends",
    "Drill-Down",
    "Insights",
    "Job Diagnosis",
    "Profitability Drivers",
    "Reconciliation",
    "Smart Quote Builder",
]

(tab_exec, tab_trends, tab_drill, tab_insights, tab_job, tab_drivers, tab_recon, tab_quote) = st.tabs(TAB_TITLES)

with tab_exec:
    _render_callout("How to read: start with the KPI strip, then the FY trend, then the flags.")
    st.markdown(
        f"**Headline:** {_status_margin(margin)} margin at {_fmt_pct(margin)}; "
        f"quote gap {_status_quote_gap(quote_gap_pct)} at {_fmt_pct(quote_gap_pct)}."
    )
    st.write(
        "- What happened: Margin and quote gap summarize pricing vs delivery health.\n"
        "- Why: Compare benchmark to quoted revenue and planned vs actual cost.\n"
        "- Where: Use flags to isolate loss, underquoted, and overrun jobs."
    )
    row1 = st.columns(4)
    row1[0].metric("Quoted Revenue", _fmt_currency(quoted_revenue))
    row1[1].metric("Benchmark", _fmt_currency(benchmark_revenue))
    row1[2].metric("Cost", _fmt_currency(actual_cost))
    row1[3].metric("Margin $", _fmt_currency(profit))

    row2 = st.columns(4)
    row2[0].metric("Margin %", _fmt_pct(margin))
    row2[1].metric("Quote Gap $", _fmt_currency(quote_gap))
    row2[2].metric("Quote Gap %", _fmt_pct(quote_gap_pct))
    row2[3].metric("Effective Rate", _fmt_rate(safe_divide(quoted_revenue, actual_hours)))

    row3 = st.columns(4)
    row3[0].metric("Cost Rate", _fmt_rate(safe_divide(actual_cost, actual_hours)))
    row3[1].metric("Quoted Hours", _fmt_hours(quoted_hours))
    row3[2].metric("Unallocated Revenue", _fmt_currency(unallocated_revenue))
    row3[3].metric("Recognized Revenue", _fmt_currency(recognized_revenue))

    row4 = st.columns(4)
    row4[0].metric("Unallocated %", _fmt_pct(unallocated_pct))
    row4[1].metric("Billable Coverage", _fmt_pct(billable_coverage))
    row4[2].metric("Base Rate Coverage", _fmt_pct(base_rate_coverage))
    row4[3].metric("Job Count", f"{job_count:,}")

    st.subheader("FY Trend")
    trend_df = scope_data["earned_task_month"].copy()
    if not trend_df.empty:
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
            scope_data["task_month"].groupby("FY_Label", as_index=False)[COL_COST]
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
    else:
        st.info("No trend data for the selected FYs.")

    st.subheader("Performance Flags")
    if job_summary.empty:
        st.info("No jobs available for this scope.")
    else:
        loss_jobs = job_summary[job_summary["Profit"] < 0].head(10)
        underquoted_jobs = job_summary[job_summary["Quote_Gap"] > 0].head(10)
        overrun_jobs = job_summary[job_summary["Overrun"] > 0].head(10)

        col_flags = st.columns(3)
        col_flags[0].dataframe(loss_jobs, width="stretch", height=240)
        col_flags[1].dataframe(underquoted_jobs, width="stretch", height=240)
        col_flags[2].dataframe(overrun_jobs, width="stretch", height=240)

with tab_trends:
    _render_callout("How to read: pick a metric, then look for breaks in FY and month.")
    metric = st.selectbox(
        "Metric",
        ["Margin %", "Quote Gap %", "Revenue", "Effective Rate", "Hours Variance %"],
    )
    grain = filters.get("trend_grain", "FY")
    trend_df = scope_data["earned_task_month"].copy()
    if trend_df.empty:
        st.info("No trend data.")
    else:
        if grain == "FY":
            group_col = "FY_Label"
        else:
            group_col = COL_MONTH_KEY

        trend = (
            trend_df.groupby(group_col, as_index=False)
            .agg(
                Revenue=("Earned_Quoted_Amount", "sum"),
                Benchmark=("Benchmark_Revenue", "sum"),
                Quoted_Hours=("Earned_Quoted_Hours", "sum"),
            )
        )
        actual = (
            scope_data["task_month"].groupby(group_col, as_index=False)
            .agg(Actual_Cost=(COL_COST, "sum"), Actual_Hours=(COL_HOURS, "sum"))
        )
        trend = trend.merge(actual, on=group_col, how="left")
        trend["Profit"] = trend["Revenue"] - trend["Actual_Cost"]
        trend["Margin"] = safe_divide(trend["Profit"], trend["Revenue"])
        trend["Quote_Gap_Pct"] = safe_divide(
            trend["Benchmark"] - trend["Revenue"], trend["Benchmark"]
        )
        trend["Effective_Rate"] = safe_divide(trend["Revenue"], trend["Actual_Hours"])
        trend["Hours_Variance_Pct"] = safe_divide(
            trend["Actual_Hours"] - trend["Quoted_Hours"], trend["Quoted_Hours"]
        )

        if metric == "Margin %":
            y = "Margin"
            ref = MARGIN_TARGET
        elif metric == "Quote Gap %":
            y = "Quote_Gap_Pct"
            ref = 0
        elif metric == "Revenue":
            y = "Revenue"
            ref = None
        elif metric == "Effective Rate":
            y = "Effective_Rate"
            ref = None
        else:
            y = "Hours_Variance_Pct"
            ref = 0

        chart = (
            alt.Chart(trend)
            .mark_line(point=True, color="#2b586e")
            .encode(
                x=alt.X(f"{group_col}:T" if group_col == COL_MONTH_KEY else f"{group_col}:N"),
                y=alt.Y(f"{y}:Q"),
                tooltip=[group_col, y],
            )
            .properties(height=300)
        )
        if ref is not None:
            ref_line = alt.Chart(pd.DataFrame({"ref": [ref]})).mark_rule(color="#cc5f5f").encode(y="ref:Q")
            st.altair_chart(chart + ref_line, width="stretch")
        else:
            st.altair_chart(chart, width="stretch")

with tab_drill:
    _render_callout("How to read: start at department, then move to product and job.")
    dept_available = "Department_Eff" in filtered.columns
    product_available = "Product_Key" in filtered.columns

    if scope_level == "Company":
        if dept_available and not scope_data["task_month"].empty:
            dept_summary = _group_metrics(
                scope_data["task_month"], scope_data["earned_task_month"], "Department_Eff"
            ).sort_values("Profit", ascending=False)
            st.subheader("Department Scoreboard")
            scatter = (
                alt.Chart(dept_summary)
                .mark_circle(size=120, opacity=0.7)
                .encode(
                    x=alt.X("Quote_Gap_Pct:Q", title="Quote Gap %"),
                    y=alt.Y("Margin:Q", title="Margin %"),
                    size=alt.Size("Quoted_Revenue:Q", title="Quoted Revenue"),
                    color=alt.Color("Margin_Status:N"),
                    tooltip=["Department_Eff", "Margin", "Quote_Gap_Pct", "Quoted_Revenue"],
                )
                .properties(height=320)
            )
            st.altair_chart(scatter, width="stretch")
            st.dataframe(dept_summary, width="stretch", height=260)
        else:
            st.info("No Department data available.")

    st.subheader("Product within Department")
    selected_dept = None
    if dept_available:
        dept_options = sorted(filtered["Department_Eff"].dropna().unique())
        if dept_options:
            with st.form("dept_select"):
                selected_dept = st.selectbox("Department", dept_options)
                apply_dept = st.form_submit_button("Apply Department")
            if apply_dept:
                st.session_state["drill_department"] = selected_dept
            selected_dept = st.session_state.get("drill_department", selected_dept)
        else:
            st.info("No departments available.")
    if selected_dept and product_available:
        dept_task_month = scope_data["task_month"][
            scope_data["task_month"]["Department_Eff"] == selected_dept
        ]
        dept_earned = scope_data["earned_task_month"][
            scope_data["earned_task_month"]["Department_Eff"] == selected_dept
        ]
        product_summary = _group_metrics(dept_task_month, dept_earned, "Product_Key").sort_values(
            "Profit", ascending=False
        )
        st.dataframe(product_summary, width="stretch", height=260)
    elif product_available:
        st.info("Pick a Department to view product performance.")
    else:
        st.info("No Product data available.")

    st.subheader("Product Selection")
    if product_available:
        product_options = sorted(filtered["Product_Key"].dropna().unique())
        if product_options:
            with st.form("product_select"):
                selected_product = st.selectbox("Product", product_options)
                apply_product = st.form_submit_button("Apply Product")
            if apply_product:
                st.session_state["drill_product"] = selected_product
            selected_product = st.session_state.get("drill_product", selected_product)
        else:
            st.info("No products available.")
            selected_product = None
    else:
        selected_product = None

    st.subheader("Job Portfolio")
    if job_summary.empty:
        st.info("No jobs available for this scope.")
    else:
        job_view = job_summary.copy()
        if selected_product:
            job_keys = set(
                filtered.loc[filtered["Product_Key"] == selected_product, COL_JOB_KEY].astype(str)
            )
            job_view = job_view[job_view[COL_JOB_KEY].astype(str).isin(job_keys)]
        with st.form("job_filters"):
            loss_only = st.checkbox("Loss-only")
            underquoted_only = st.checkbox("Underquoted-only")
            overrun_only = st.checkbox("Overrun-only")
            apply_job_filters = st.form_submit_button("Apply Job Filters")
        if apply_job_filters:
            st.session_state["job_filters"] = (loss_only, underquoted_only, overrun_only)
        loss_only, underquoted_only, overrun_only = st.session_state.get(
            "job_filters", (False, False, False)
        )
        if loss_only:
            job_view = job_view[job_view["Profit"] < 0]
        if underquoted_only:
            job_view = job_view[job_view["Quote_Gap"] > 0]
        if overrun_only:
            job_view = job_view[job_view["Overrun"] > 0]
        st.dataframe(job_view.head(25), width="stretch", height=320)

    st.subheader("Task Breakdown")
    job_map = _job_label_map(filtered_scope)
    job_keys = job_summary[COL_JOB_KEY].astype(str).tolist() if not job_summary.empty else []
    if job_keys:
        with st.form("job_task_select"):
            selected_job = st.selectbox(
                "Job",
                options=job_keys,
                format_func=lambda k: job_map.get(k, k),
            )
            apply_job = st.form_submit_button("Apply Job")
        if apply_job:
            st.session_state["drill_job"] = selected_job
        selected_job = st.session_state.get("drill_job", selected_job)

        task_month_job = scope_data["task_month"][
            scope_data["task_month"][COL_JOB_KEY] == selected_job
        ]
        dim_job_task_quote_job = scope_data["dim_job_task_quote"][
            scope_data["dim_job_task_quote"][COL_JOB_KEY] == selected_job
        ]
        task_summary = build_task_summary(
            task_month_job,
            dim_job_task_quote_job,
            filters.get("include_quote_only", True),
        )
        st.dataframe(task_summary, width="stretch", height=320)

with tab_insights:
    _render_callout("How to read: a short action list based on the current FY window.")
    insights = []
    if margin is not None and margin < MARGIN_TARGET:
        insights.append("Margin below target; reinforce pricing discipline on new quotes.")
    if quote_gap_pct is not None and quote_gap_pct > QUOTE_GAP_RISK:
        insights.append("Underquoting is material; tighten benchmark adherence.")
    if actual_cost > planned_cost_earned:
        insights.append("Delivery overruns exceed plan; review scope control and resourcing.")
    if not insights:
        insights.append("Performance is within target bands; monitor and scale best practices.")

    for item in insights[:3]:
        st.write(f"- {item}")

    st.subheader("Top Underquoted Jobs")
    if job_summary.empty:
        st.info("No jobs available for this scope.")
    else:
        underquoted_jobs = job_summary[job_summary["Quote_Gap"] > 0].head(10)
        st.dataframe(underquoted_jobs, width="stretch", height=240)

    st.subheader("Top Scope Creep Tasks")
    task_summary_all = build_task_summary(
        scope_data["task_month"],
        scope_data["dim_job_task_quote"],
        filters.get("include_quote_only", True),
    )
    scope_creep = task_summary_all[task_summary_all["Scope_Overrun"]].head(10)
    st.dataframe(scope_creep, width="stretch", height=240)

with tab_job:
    _render_callout("How to read: pick a job to generate a short diagnosis.")
    job_map = _job_label_map(filtered_scope)
    job_keys = sorted(filtered_scope[COL_JOB_KEY].dropna().astype(str).unique())
    if not job_keys:
        st.info("No jobs available.")
    else:
        selected_job = st.selectbox("Job", options=job_keys, format_func=lambda k: job_map.get(k, k))
        st.session_state["drill_job"] = selected_job
        job_task_month = scope_data["task_month"][
            scope_data["task_month"][COL_JOB_KEY] == selected_job
        ]
        job_dim_quote = scope_data["dim_job_task_quote"][
            scope_data["dim_job_task_quote"][COL_JOB_KEY] == selected_job
        ]
        job_task_summary = build_task_summary(
            job_task_month, job_dim_quote, filters.get("include_quote_only", True)
        )
        issues = []
        if (job_task_summary["Actual_Margin"] < 0).any():
            issues.append("Loss-making tasks detected.")
        if job_task_summary["Scope_Overrun"].any():
            issues.append("Scope creep present.")
        if job_task_summary["Unquoted_Work"].any():
            issues.append("Unquoted work present.")
        if not issues:
            issues.append("No major issues detected.")

        st.write("Summary line: Profitability drivers captured below.")
        st.write("Issues identified:")
        for issue in issues:
            st.write(f"- {issue}")
        st.dataframe(job_task_summary, width="stretch", height=320)

with tab_drivers:
    _render_callout("How to read: compare the dollar impact of the three driver buckets.")
    delivery_overrun = actual_cost - planned_cost_earned
    underquoting = quote_gap
    rate_erosion = benchmark_revenue - quoted_revenue
    drivers = pd.DataFrame(
        {
            "Driver": ["Delivery Overrun", "Underquoting", "Rate Erosion"],
            "Impact": [delivery_overrun, underquoting, rate_erosion],
        }
    )
    st.bar_chart(drivers.set_index("Driver"))

    st.subheader("Top Margin Erosion Jobs")
    if job_summary.empty:
        st.info("No jobs available for this scope.")
    else:
        erosion_jobs = job_summary.sort_values("Profit").head(15)
        st.dataframe(erosion_jobs, width="stretch", height=260)

with tab_recon:
    _render_callout("How to read: sanity checks to validate totals.")
    render_data_integrity(
        scope_data["task_month"],
        scope_data["unallocated"],
        scope_data["dim_job_month_window"],
        scope_data["dim_job_task_quote"],
        include_unallocated=filters.get("include_unallocated", True),
        quote_mode=filters.get("quote_mode", "Earned Quote Proxy"),
        is_lifetime=False,
        wrap_expander=False,
    )

with tab_quote:
    _render_callout("How to read: build a new quote from historical task patterns.")
    st.write("Select department and product to build a reference task library.")
    if "Department_Eff" in filtered_scope.columns:
        dept_options = sorted(filtered_scope["Department_Eff"].dropna().unique())
        selected_dept = st.selectbox("Department", dept_options)
    else:
        selected_dept = None

    product_options = sorted(filtered_scope["Product_Key"].dropna().unique())
    if product_options:
        selected_product = st.selectbox("Product", product_options)
    else:
        st.info("No products available for this scope.")
        selected_product = None

    reference = scope_data["task_month_lifetime"]
    if selected_dept:
        reference = reference[reference["Department_Eff"] == selected_dept]
    if selected_product:
        reference = reference[reference["Product_Key"] == selected_product]

    task_library = (
        reference.groupby(COL_TASK_KEY, as_index=False)
        .agg(
            Actual_Hours=(COL_HOURS, "sum"),
            Actual_Cost=(COL_COST, "sum"),
            Job_Count=(COL_JOB_KEY, "nunique"),
        )
        .sort_values("Job_Count", ascending=False)
    )
    if task_library.empty:
        st.info("No tasks available for this slice.")
    else:
        task_library = task_library.head(20)
        task_library["Proposed_Hours"] = task_library["Actual_Hours"].round(0)
        edited = st.data_editor(task_library, width="stretch", num_rows="dynamic")
        total_hours = edited["Proposed_Hours"].sum()
        actual_hours = edited["Actual_Hours"].sum()
        avg_cost_rate = safe_divide(edited["Actual_Cost"].sum(), edited["Actual_Hours"].sum())
        est_cost = total_hours * avg_cost_rate if avg_cost_rate else None
        st.write(f"Proposed hours: {_fmt_hours(total_hours)}")
        st.write(f"Estimated cost: {_fmt_currency(est_cost)}")
        if actual_hours and total_hours < actual_hours * 0.9:
            st.warning("Proposed hours are materially below historical actuals. Review scope.")

render_data_integrity(
    scope_data["task_month"],
    scope_data["unallocated"],
    scope_data["dim_job_month_window"],
    scope_data["dim_job_task_quote"],
    include_unallocated=filters.get("include_unallocated", True),
    quote_mode=filters.get("quote_mode", "Earned Quote Proxy"),
    is_lifetime=False,
)
