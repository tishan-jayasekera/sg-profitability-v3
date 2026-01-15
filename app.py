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
    build_task_summary,
    compute_monthly_metrics,
    compute_quote_by_task,
    safe_divide,
)
from lib.qa import render_data_integrity
from lib.semantic import build_dim_job_month, build_dim_job_task_quote
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


def _add_fiscal_year(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    fy_year = out[COL_MONTH_KEY].dt.year + (out[COL_MONTH_KEY].dt.month >= 7).astype(int)
    out["Fiscal_Year"] = "FY" + fy_year.astype(str)
    return out


def _group_summary(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    summary = (
        df.groupby(group_col, as_index=False)
        .agg(
            Actual_Revenue=(COL_ALLOCATED_REVENUE, "sum"),
            Actual_Cost=(COL_COST, "sum"),
            Actual_Hours=(COL_HOURS, "sum"),
        )
        .sort_values("Actual_Revenue", ascending=False)
    )
    summary["Actual_Profit"] = summary["Actual_Revenue"] - summary["Actual_Cost"]
    summary["Actual_Margin"] = safe_divide(summary["Actual_Profit"], summary["Actual_Revenue"])
    summary["Realized_Rate"] = safe_divide(summary["Actual_Revenue"], summary["Actual_Hours"])
    summary["Base_Cost_Rate"] = safe_divide(summary["Actual_Cost"], summary["Actual_Hours"])
    return summary


st.set_page_config(page_title="Executive Overview", layout="wide")
st.title("Executive Overview")
st.caption("Top-down view: company -> department -> product -> job -> task.")

df = load_data_enriched()
filters = render_sidebar(df)
scope_level = filters.get("scope_level", "Company")
scope_order = ["Company", "Department", "Product", "Job", "Task"]
max_scope_index = scope_order.index(scope_level) if scope_level in scope_order else 0
df_filtered = apply_filters(df, filters)
st.caption(
    "Scope: "
    f"{scope_level} | FYs: {', '.join(filters.get('fy_labels', []))} | "
    f"Quote: {filters.get('quote_mode', 'Earned Quote Proxy')}"
)

start = pd.to_datetime(filters["start_date"])
end = pd.to_datetime(filters["end_date"])
fy_labels = filters.get("fy_labels", [])
available_fys = (
    df_filtered["FY_Label"].dropna().astype(str).unique().tolist()
    if "FY_Label" in df_filtered.columns
    else []
)
is_lifetime = set(fy_labels) == set(available_fys)

actuals_window = df_filtered[
    df_filtered[COL_MONTH_KEY].notna()
    & df_filtered["FY_Label"].isin(fy_labels)
]
st.caption(
    f"Filtered rows: {len(df_filtered):,} | Actuals in window: {len(actuals_window):,}"
)
if actuals_window.empty:
    st.warning(
        "No actuals in the selected window. Showing quote-only data where available. "
        "Try the Lifetime preset for a full view."
    )

task_month = actuals_window[actuals_window[COL_TASK_KEY] != "__UNALLOCATED__"].copy()
unallocated = actuals_window[actuals_window[COL_TASK_KEY] == "__UNALLOCATED__"].copy()

task_month_lifetime = df_filtered[
    df_filtered[COL_MONTH_KEY].notna() & (df_filtered[COL_TASK_KEY] != "__UNALLOCATED__")
].copy()

dim_job_month = build_dim_job_month(df_filtered[df_filtered[COL_MONTH_KEY].notna()])
dim_job_month_window = dim_job_month[dim_job_month["FY_Label"].isin(fy_labels)]
dim_job_task_quote = build_dim_job_task_quote(df_filtered)

quote_mode = filters.get("quote_mode", "Earned Quote Proxy")
if not is_lifetime and quote_mode == "Lifetime Quote":
    st.warning(
        "You're viewing period actuals against lifetime quote. Consider Earned Quote Proxy."
    )

quote_by_task = compute_quote_by_task(
    quote_mode, task_month, task_month_lifetime, dim_job_task_quote
)

recognized_revenue = dim_job_month_window[COL_AMOUNT].sum(min_count=1)
allocated_revenue = task_month[COL_ALLOCATED_REVENUE].sum()
if filters["include_unallocated"]:
    allocated_revenue += unallocated[COL_ALLOCATED_REVENUE].sum()
actual_cost = task_month[COL_COST].sum()
actual_hours = task_month[COL_HOURS].sum()
actual_profit = allocated_revenue - actual_cost
actual_margin = safe_divide(actual_profit, allocated_revenue)

quoted_amount = quote_by_task["Quoted_Amount_Mode"].sum(min_count=1)
quoted_hours = quote_by_task["Quoted_Time_Mode"].sum(min_count=1)
quoted_rate = safe_divide(quoted_amount, quoted_hours)

st.subheader("Company Overview")
row1 = st.columns(4)
row1[0].metric("Recognized Revenue", _fmt_currency(recognized_revenue))
row1[1].metric("Allocated Revenue", _fmt_currency(allocated_revenue))
row1[2].metric("Actual Cost", _fmt_currency(actual_cost))
row1[3].metric("Actual Profit", _fmt_currency(actual_profit))

row2 = st.columns(4)
row2[0].metric("Actual Hours", _fmt_hours(actual_hours))
row2[1].metric("Realized Rate", _fmt_rate(safe_divide(allocated_revenue, actual_hours)))
row2[2].metric("Base Cost Rate", _fmt_rate(safe_divide(actual_cost, actual_hours)))
row2[3].metric("Quoted Rate", _fmt_rate(quoted_rate))

st.subheader("Trend")
monthly = compute_monthly_metrics(task_month, unallocated, filters["include_unallocated"])
if monthly.empty:
    st.info("No actuals to chart for this window.")
else:
    monthly = _add_fiscal_year(monthly)
    monthly["Revenue_Total"] = monthly["Revenue"] + monthly["Unallocated_Revenue"]
    if filters["trend_grain"] == "FY":
        trend = (
            monthly.groupby("Fiscal_Year", as_index=False)
            .agg(
                Revenue=("Revenue_Total", "sum"),
                Cost=("Cost", "sum"),
                Hours=("Hours", "sum"),
            )
            .sort_values("Fiscal_Year")
        )
        x_field = "Fiscal_Year:N"
    else:
        trend = (
            monthly.groupby(COL_MONTH_KEY, as_index=False)
            .agg(
                Revenue=("Revenue_Total", "sum"),
                Cost=("Cost", "sum"),
                Hours=("Hours", "sum"),
            )
            .sort_values(COL_MONTH_KEY)
        )
        x_field = f"{COL_MONTH_KEY}:T"

    trend["Profit"] = trend["Revenue"] - trend["Cost"]
    trend["Margin"] = safe_divide(trend["Profit"], trend["Revenue"])

    trend_long = trend.melt(
        id_vars=[trend.columns[0]],
        value_vars=["Revenue", "Cost", "Profit"],
        var_name="Metric",
        value_name="Value",
    )
    trend_chart = (
        alt.Chart(trend_long)
        .mark_bar()
        .encode(
            x=alt.X(x_field, title="Period"),
            y=alt.Y("Value:Q", title="Amount"),
            color="Metric:N",
            tooltip=[trend.columns[0], "Metric", "Value"],
        )
        .properties(height=280)
    )
    st.altair_chart(trend_chart, width="stretch")
    st.dataframe(trend, width="stretch", height=220)

if max_scope_index >= 1:
    st.subheader("Department View")
    if "Department_Eff" not in task_month.columns:
        st.info("No Department data available.")
    else:
        dept_summary = _group_summary(task_month, "Department_Eff")
        st.dataframe(dept_summary, width="stretch", height=300)
        if filters["include_unallocated"] and not unallocated.empty:
            st.caption("Unallocated revenue is excluded from department rollups.")

if max_scope_index >= 2:
    st.subheader("Product View")
    if "Product" not in task_month.columns:
        st.info("No Product data available.")
    else:
        product_summary = _group_summary(task_month, "Product")
        st.dataframe(product_summary, width="stretch", height=300)
        if filters["include_unallocated"] and not unallocated.empty:
            st.caption("Unallocated revenue is excluded from product rollups.")

if max_scope_index >= 3:
    st.subheader("Job View")
    job_summary = (
        task_month.groupby(COL_JOB_KEY, as_index=False)
        .agg(
            Allocated_Revenue=(COL_ALLOCATED_REVENUE, "sum"),
            Actual_Cost=(COL_COST, "sum"),
            Actual_Hours=(COL_HOURS, "sum"),
        )
        .merge(
            quote_by_task.groupby(COL_JOB_KEY, as_index=False)[
                ["Quoted_Time_Mode", "Quoted_Amount_Mode"]
            ].sum(min_count=1),
            on=COL_JOB_KEY,
            how="left",
        )
    )
    unalloc_by_job = (
        unallocated.groupby(COL_JOB_KEY, as_index=False)[COL_ALLOCATED_REVENUE]
        .sum()
        .rename(columns={COL_ALLOCATED_REVENUE: "Unallocated_Revenue"})
    )
    job_summary = job_summary.merge(unalloc_by_job, on=COL_JOB_KEY, how="left")
    job_summary["Unallocated_Revenue"] = job_summary["Unallocated_Revenue"].fillna(0)
    if filters["include_unallocated"]:
        job_summary["Actual_Revenue"] = (
            job_summary["Allocated_Revenue"] + job_summary["Unallocated_Revenue"]
        )
    else:
        job_summary["Actual_Revenue"] = job_summary["Allocated_Revenue"]

    job_summary["Actual_Profit"] = job_summary["Actual_Revenue"] - job_summary["Actual_Cost"]
    job_summary["Actual_Margin"] = safe_divide(
        job_summary["Actual_Profit"], job_summary["Actual_Revenue"]
    )
    job_summary["Realized_Rate"] = safe_divide(
        job_summary["Actual_Revenue"], job_summary["Actual_Hours"]
    )
    job_summary["Quoted_Rate"] = safe_divide(
        job_summary["Quoted_Amount_Mode"], job_summary["Quoted_Time_Mode"]
    )
    job_summary = job_summary.sort_values("Actual_Revenue", ascending=False)

    job_map = _job_label_map(df_filtered)
    if job_summary.empty:
        st.info("No job actuals in the selected window.")
    else:
        st.dataframe(job_summary, width="stretch", height=320)

        st.markdown("**Job Drilldown**")
        job_keys = job_summary[COL_JOB_KEY].astype(str).tolist()
        default_job = st.session_state.get("job_drill_selected", job_keys[0])
        with st.form("job_drill_form"):
            selected_job = st.selectbox(
                "Select a job for task-level detail",
                options=job_keys,
                index=job_keys.index(default_job) if default_job in job_keys else 0,
                format_func=lambda k: job_map.get(k, k),
                key="job_drill_input",
            )
            drill_apply = st.form_submit_button("Apply Job")
        if drill_apply:
            st.session_state["job_drill_selected"] = selected_job
        selected_job = st.session_state.get("job_drill_selected", selected_job)

        task_month_job = task_month[task_month[COL_JOB_KEY] == selected_job]
        dim_job_task_quote_job = dim_job_task_quote[
            dim_job_task_quote[COL_JOB_KEY] == selected_job
        ]
        quote_by_task_job = quote_by_task[quote_by_task[COL_JOB_KEY] == selected_job]

        task_summary = build_task_summary(
            task_month_job, dim_job_task_quote_job, filters["include_quote_only"]
        )
        task_summary = task_summary.merge(
            quote_by_task_job, on=[COL_JOB_KEY, COL_TASK_KEY], how="left"
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
        task_summary = task_summary.sort_values("Actual_Profit", ascending=False)
        st.subheader("Task View")
        columns = [
            COL_TASK_KEY,
            task_label_col,
            "Actual_Hours_Total",
            "Actual_Revenue",
            "Actual_Cost",
            "Actual_Profit",
            "Actual_Margin",
            "Realized_Rate",
            "Base_Cost_Rate",
            "Quoted_Time_Mode",
            "Quoted_Amount_Mode",
            "Quoted_Rate_Mode",
            "Unquoted_Work",
            "Scope_Overrun",
        ]
        columns = [c for c in columns if c in task_summary.columns]
        st.dataframe(task_summary[columns], width="stretch", height=320)

render_data_integrity(
    task_month,
    unallocated,
    dim_job_month_window,
    dim_job_task_quote,
    include_unallocated=filters["include_unallocated"],
    quote_mode=quote_mode,
    is_lifetime=is_lifetime,
)
