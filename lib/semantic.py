from __future__ import annotations

from typing import Iterable

import pandas as pd

from lib.constants import (
    COL_AMOUNT,
    COL_JOB_KEY,
    COL_JOB_MONTH_TOTAL_HOURS,
    COL_MONTH_KEY,
    COL_TASK_KEY,
    ROW_QUOTE_ONLY,
    ROW_TASK_MONTH,
    ROW_UNALLOCATED,
)
from lib.metrics import add_fiscal_fields


TASK_DIM_COLUMNS = [
    "[Job Task] Name",
    "Task Name",
    "Deliverable",
    "Role",
    "Task",
    "[Category] Category",
    "Task Category",
    "Product",
    "Function",
    "Business Unit",
    "Department",
    "Department_QUOTE",
    "Department_Eff",
    "Billable?",
    "Source",
]


def _existing(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def classify_row_type(df: pd.DataFrame) -> pd.Series:
    row_type = pd.Series("UNKNOWN", index=df.index)
    row_type[(df[COL_MONTH_KEY].notna()) & (df[COL_TASK_KEY] != "__UNALLOCATED__")] = ROW_TASK_MONTH
    row_type[(df[COL_MONTH_KEY].notna()) & (df[COL_TASK_KEY] == "__UNALLOCATED__")] = ROW_UNALLOCATED
    row_type[df[COL_MONTH_KEY].isna()] = ROW_QUOTE_ONLY
    return row_type


def add_row_type(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(Row_Type=classify_row_type(df))


def build_fact_task_month(df: pd.DataFrame) -> pd.DataFrame:
    if "Row_Type" not in df.columns:
        df = add_row_type(df)
    return df[df["Row_Type"] == ROW_TASK_MONTH].copy()


def build_unallocated(df: pd.DataFrame) -> pd.DataFrame:
    if "Row_Type" not in df.columns:
        df = add_row_type(df)
    return df[df["Row_Type"] == ROW_UNALLOCATED].copy()


def build_dim_job_month(df: pd.DataFrame) -> pd.DataFrame:
    job_month = df[df[COL_MONTH_KEY].notna()].copy()
    cols = [COL_AMOUNT, COL_JOB_MONTH_TOTAL_HOURS]
    cols = _existing(job_month, cols + ["Source"])
    agg = {col: "first" for col in cols}
    dim = (
        job_month.groupby([COL_JOB_KEY, COL_MONTH_KEY], as_index=False)
        .agg(agg)
        .sort_values([COL_JOB_KEY, COL_MONTH_KEY])
    )
    dim = add_fiscal_fields(dim)
    return dim


def _first_non_null(series: pd.Series):
    non_null = series.dropna()
    if non_null.empty:
        return None
    return non_null.iloc[0]


def build_dim_job_task_quote(df: pd.DataFrame) -> pd.DataFrame:
    if "Row_Type" not in df.columns:
        df = add_row_type(df)

    quote_source = df[df["Row_Type"].isin([ROW_TASK_MONTH, ROW_QUOTE_ONLY])].copy()
    quote_cols = _existing(
        quote_source,
        [
            "[Job Task] Quoted Time",
            "[Job Task] Quoted Amount",
            "[Task] Base Rate",
            "[Task] Billable Rate",
            "[Task] Base Rate_QUOTE",
            "[Task] Billable Rate_QUOTE",
        ]
        + TASK_DIM_COLUMNS,
    )
    agg = {col: _first_non_null for col in quote_cols}

    dim = (
        quote_source.groupby([COL_JOB_KEY, COL_TASK_KEY], as_index=False)
        .agg(agg)
        .sort_values([COL_JOB_KEY, COL_TASK_KEY])
    )
    return dim
