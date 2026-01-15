from __future__ import annotations

import pandas as pd
import streamlit as st

from lib.constants import (
    COL_ALLOCATED_REVENUE,
    COL_AMOUNT,
    COL_JOB_KEY,
    COL_MONTH_KEY,
    COL_QUOTED_AMOUNT,
    COL_QUOTED_TIME,
    COL_TASK_KEY,
)


def render_data_integrity(
    task_month: pd.DataFrame,
    unallocated: pd.DataFrame,
    dim_job_month: pd.DataFrame,
    dim_job_task_quote: pd.DataFrame,
    include_unallocated: bool,
    quote_mode: str,
    is_lifetime: bool,
    tolerance: float = 0.01,
) -> None:
    with st.expander("Data Integrity"):
        st.subheader("Allocation Closure (Job-Month)")
        alloc = (
            task_month.groupby([COL_JOB_KEY, COL_MONTH_KEY])[COL_ALLOCATED_REVENUE]
            .sum(min_count=1)
            .rename("Allocated_Revenue_Sum")
            .reset_index()
        )
        if include_unallocated and not unallocated.empty:
            unalloc = (
                unallocated.groupby([COL_JOB_KEY, COL_MONTH_KEY])[COL_ALLOCATED_REVENUE]
                .sum(min_count=1)
                .rename("Unallocated_Revenue_Sum")
                .reset_index()
            )
            alloc = alloc.merge(unalloc, on=[COL_JOB_KEY, COL_MONTH_KEY], how="left")
            alloc["Allocated_Revenue_Sum"] = alloc["Allocated_Revenue_Sum"] + alloc[
                "Unallocated_Revenue_Sum"
            ].fillna(0)

        job_month = dim_job_month[[COL_JOB_KEY, COL_MONTH_KEY, COL_AMOUNT]].copy()
        closure = alloc.merge(job_month, on=[COL_JOB_KEY, COL_MONTH_KEY], how="left")
        closure["Diff"] = (closure["Allocated_Revenue_Sum"] - closure[COL_AMOUNT]).abs()
        closure["Pass"] = closure["Diff"] <= tolerance
        passed = int(closure["Pass"].sum())
        failed = int((~closure["Pass"]).sum())
        st.write(f"Passed: {passed:,} | Failed: {failed:,} (tolerance {tolerance})")
        if failed:
            st.dataframe(
                closure.sort_values("Diff", ascending=False).head(10),
                use_container_width=True,
            )

        st.subheader("Revenue Consistency (Job Total)")
        job_amount = dim_job_month.groupby(COL_JOB_KEY)[COL_AMOUNT].sum(min_count=1)
        task_revenue = task_month.groupby(COL_JOB_KEY)[COL_ALLOCATED_REVENUE].sum(min_count=1)
        if include_unallocated and not unallocated.empty:
            task_revenue = task_revenue.add(
                unallocated.groupby(COL_JOB_KEY)[COL_ALLOCATED_REVENUE].sum(min_count=1),
                fill_value=0,
            )
        revenue_delta = (job_amount - task_revenue).abs().sum()
        st.write(f"Total absolute delta: {revenue_delta:,.2f}")

        st.subheader("Quote Dedupe Sanity")
        st.write(
            f"Unique job-task quote rows: {len(dim_job_task_quote):,} "
            f"(computed on unique Job_Key x Task_Key)."
        )
        if not is_lifetime and quote_mode == "Lifetime Quote":
            st.warning(
                "Window is not lifetime; quote totals are still lifetime per settings."
            )

        st.subheader("Coverage Diagnostics")
        if task_month.empty:
            missing_billable = None
            missing_base = None
        else:
            missing_billable = task_month["Billable_Rate_Eff"].isna().mean()
            missing_base = task_month["Base_Rate_Eff"].isna().mean()
        missing_quote = (
            None if dim_job_task_quote.empty else dim_job_task_quote[COL_QUOTED_TIME].isna().mean()
        )
        billable_txt = f"{missing_billable:.1%}" if missing_billable is not None else "N/A"
        base_txt = f"{missing_base:.1%}" if missing_base is not None else "N/A"
        quote_txt = f"{missing_quote:.1%}" if missing_quote is not None else "N/A"
        st.write(
            f"Missing Billable_Rate_Eff: {billable_txt} | "
            f"Missing Base_Rate_Eff: {base_txt} | "
            f"Tasks missing quote: {quote_txt}"
        )
