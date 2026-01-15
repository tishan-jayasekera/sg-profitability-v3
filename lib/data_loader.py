from pathlib import Path

import pandas as pd
import streamlit as st

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
    COL_JOB_MONTH_TOTAL_HOURS,
    COL_MONTH_KEY,
    COL_QUOTED_AMOUNT,
    COL_QUOTED_TIME,
    COL_TASK_KEY,
    REQUIRED_COLUMNS,
)


DATA_PATH = Path("data/Unified_Job_Profitability_Full_Data.csv")

NUMERIC_COLUMNS = [
    COL_HOURS,
    COL_COST,
    COL_ALLOCATED_REVENUE,
    COL_AMOUNT,
    COL_JOB_MONTH_TOTAL_HOURS,
    COL_QUOTED_TIME,
    COL_QUOTED_AMOUNT,
    COL_BASE_RATE,
    COL_BILLABLE_RATE,
    COL_BASE_RATE_QUOTE,
    COL_BILLABLE_RATE_QUOTE,
]


def _normalize_billable(series: pd.Series) -> pd.Series:
    billable = series.astype("string").str.strip().str.upper()
    billable = billable.where(billable.notna() & (billable != "NAN"))
    billable = billable.replace({"": None})
    return billable


@st.cache_data(show_spinner=False)
def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df[COL_MONTH_KEY] = pd.to_datetime(df[COL_MONTH_KEY], errors="coerce")

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df[COL_JOB_KEY] = df[COL_JOB_KEY].astype(str)
    df[COL_TASK_KEY] = df[COL_TASK_KEY].astype(str)

    if "Billable?" in df.columns:
        df["Billable?"] = _normalize_billable(df["Billable?"])

    if "Department" in df.columns or "Department_QUOTE" in df.columns:
        dept = df["Department"] if "Department" in df.columns else None
        dept_quote = df["Department_QUOTE"] if "Department_QUOTE" in df.columns else None
        if dept is not None and dept_quote is not None:
            df["Department_Eff"] = dept.where(dept.notna(), dept_quote)
        elif dept is not None:
            df["Department_Eff"] = dept
        elif dept_quote is not None:
            df["Department_Eff"] = dept_quote

    return df
