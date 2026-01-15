from __future__ import annotations

import pandas as pd

from lib.constants import (
    COL_ALLOCATED_REVENUE,
    COL_AMOUNT,
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


def safe_divide(n, d):
    if isinstance(n, pd.Series) or isinstance(d, pd.Series):
        result = n / d
        return result.where(d.notna() & (d != 0))
    if d is None or pd.isna(d) or d == 0:
        return None
    return n / d


def add_task_month_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Billable_Rate_Eff"] = out[COL_BILLABLE_RATE]
    out["Billable_Rate_Eff"] = out["Billable_Rate_Eff"].where(
        out["Billable_Rate_Eff"].notna(), out[COL_BILLABLE_RATE_QUOTE]
    )

    out["Base_Rate_Eff"] = out[COL_BASE_RATE]
    out["Base_Rate_Eff"] = out["Base_Rate_Eff"].where(
        out["Base_Rate_Eff"].notna(), out[COL_BASE_RATE_QUOTE]
    )

    base_cost_rate = safe_divide(out[COL_COST], out[COL_HOURS])
    out["Base_Rate_Eff"] = out["Base_Rate_Eff"].where(out["Base_Rate_Eff"].notna(), base_cost_rate)

    out["Base_Cost_Rate"] = base_cost_rate
    out["Realized_Rate"] = safe_divide(out[COL_ALLOCATED_REVENUE], out[COL_HOURS])

    out["Actual_Revenue"] = out[COL_ALLOCATED_REVENUE]
    out["Actual_Cost"] = out[COL_COST]
    out["Actual_Profit"] = out["Actual_Revenue"] - out["Actual_Cost"]
    out["Actual_Margin"] = safe_divide(out["Actual_Profit"], out["Actual_Revenue"])

    out["Ghost_Work_Month"] = (out[COL_HOURS] > 0) & (out[COL_ALLOCATED_REVENUE] == 0)
    out["Below_Cost_Month"] = (out[COL_HOURS] > 0) & (
        out["Realized_Rate"] < out["Base_Cost_Rate"]
    )
    out["Missing_RateCard"] = out["Billable_Rate_Eff"].isna()
    return out


def weighted_average(value: pd.Series, weight: pd.Series):
    valid = weight.notna() & (weight > 0) & value.notna()
    if not valid.any():
        return None
    return (value[valid] * weight[valid]).sum() / weight[valid].sum()


def build_task_summary(
    task_month: pd.DataFrame,
    dim_job_task_quote: pd.DataFrame,
    include_quote_only: bool,
) -> pd.DataFrame:
    actuals = (
        task_month.groupby([COL_JOB_KEY, COL_TASK_KEY], as_index=False)
        .agg(
            Actual_Hours_Total=(COL_HOURS, "sum"),
            Actual_Revenue=(COL_ALLOCATED_REVENUE, "sum"),
            Actual_Cost=(COL_COST, "sum"),
            Ghost_Work_Hours=(COL_HOURS, lambda s: s[task_month.loc[s.index, "Ghost_Work_Month"]].sum()),
            Below_Cost_Hours=(COL_HOURS, lambda s: s[task_month.loc[s.index, "Below_Cost_Month"]].sum()),
        )
    )

    actuals["Actual_Profit"] = actuals["Actual_Revenue"] - actuals["Actual_Cost"]
    actuals["Actual_Margin"] = safe_divide(actuals["Actual_Profit"], actuals["Actual_Revenue"])
    actuals["Realized_Rate"] = safe_divide(actuals["Actual_Revenue"], actuals["Actual_Hours_Total"])
    actuals["Base_Cost_Rate"] = safe_divide(actuals["Actual_Cost"], actuals["Actual_Hours_Total"])
    actuals["Below_Cost_Hours_Pct"] = safe_divide(actuals["Below_Cost_Hours"], actuals["Actual_Hours_Total"])

    join_how = "outer" if include_quote_only else "left"
    summary = actuals.merge(dim_job_task_quote, on=[COL_JOB_KEY, COL_TASK_KEY], how=join_how)

    summary["Quoted_Rate"] = safe_divide(summary[COL_QUOTED_AMOUNT], summary[COL_QUOTED_TIME])
    summary["Unquoted_Work"] = (summary["Actual_Hours_Total"] > 0) & (
        summary[COL_QUOTED_TIME].isna() | (summary[COL_QUOTED_TIME] == 0)
    )
    summary["Quoted_Not_Delivered"] = (summary[COL_QUOTED_TIME] > 0) & (
        summary["Actual_Hours_Total"].fillna(0) == 0
    )
    summary["Scope_Overrun"] = (summary[COL_QUOTED_TIME] > 0) & (
        summary["Actual_Hours_Total"].fillna(0) > summary[COL_QUOTED_TIME]
    )
    return summary


def compute_monthly_metrics(
    task_month: pd.DataFrame,
    unallocated: pd.DataFrame,
    include_unallocated: bool,
) -> pd.DataFrame:
    monthly = (
        task_month.groupby(COL_MONTH_KEY, as_index=False)
        .agg(
            Revenue=(COL_ALLOCATED_REVENUE, "sum"),
            Cost=(COL_COST, "sum"),
            Hours=(COL_HOURS, "sum"),
        )
        .sort_values(COL_MONTH_KEY)
    )

    monthly["Realized_Rate"] = safe_divide(monthly["Revenue"], monthly["Hours"])
    monthly["Base_Cost_Rate"] = safe_divide(monthly["Cost"], monthly["Hours"])

    billable = (
        task_month.groupby(COL_MONTH_KEY)
        .apply(lambda g: weighted_average(g["Billable_Rate_Eff"], g[COL_HOURS]))
        .reset_index(name="Billable_Rate_Wtd")
    )
    monthly = monthly.merge(billable, on=COL_MONTH_KEY, how="left")

    if include_unallocated and not unallocated.empty:
        unalloc_month = (
            unallocated.groupby(COL_MONTH_KEY, as_index=False)[COL_ALLOCATED_REVENUE]
            .sum()
            .rename(columns={COL_ALLOCATED_REVENUE: "Unallocated_Revenue"})
        )
        monthly = monthly.merge(unalloc_month, on=COL_MONTH_KEY, how="left")
    else:
        monthly["Unallocated_Revenue"] = 0.0

    return monthly


def compute_quote_by_task(
    quote_mode: str,
    task_month_window: pd.DataFrame,
    task_month_lifetime: pd.DataFrame,
    dim_job_task_quote: pd.DataFrame,
) -> pd.DataFrame:
    quote = dim_job_task_quote[[COL_JOB_KEY, COL_TASK_KEY, COL_QUOTED_TIME, COL_QUOTED_AMOUNT]].copy()
    if quote_mode == "Lifetime Quote":
        quote = quote.rename(
            columns={COL_QUOTED_TIME: "Quoted_Time_Mode", COL_QUOTED_AMOUNT: "Quoted_Amount_Mode"}
        )
        return quote

    lifetime_hours = (
        task_month_lifetime.groupby([COL_JOB_KEY, COL_TASK_KEY])[COL_HOURS]
        .sum()
        .rename("Lifetime_Hours")
        .reset_index()
    )
    window_hours = (
        task_month_window.groupby([COL_JOB_KEY, COL_TASK_KEY])[COL_HOURS]
        .sum()
        .rename("Window_Hours")
        .reset_index()
    )
    hours = window_hours.merge(lifetime_hours, on=[COL_JOB_KEY, COL_TASK_KEY], how="left")
    hours["Hour_Share"] = safe_divide(hours["Window_Hours"], hours["Lifetime_Hours"])

    quote = quote.merge(hours[[COL_JOB_KEY, COL_TASK_KEY, "Hour_Share"]], on=[COL_JOB_KEY, COL_TASK_KEY], how="left")
    quote["Quoted_Time_Mode"] = quote[COL_QUOTED_TIME] * quote["Hour_Share"]
    quote["Quoted_Amount_Mode"] = quote[COL_QUOTED_AMOUNT] * quote["Hour_Share"]
    quote.loc[quote["Hour_Share"].isna(), ["Quoted_Time_Mode", "Quoted_Amount_Mode"]] = None
    return quote[[COL_JOB_KEY, COL_TASK_KEY, "Quoted_Time_Mode", "Quoted_Amount_Mode"]]
