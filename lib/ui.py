from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from lib.constants import COL_JOB_KEY, COL_MONTH_KEY


def _sorted_unique(series: pd.Series) -> list[str]:
    values = series.dropna().astype(str).unique().tolist()
    return sorted(values)


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


def _preset_range(preset: str, lifetime_start: date, lifetime_end: date) -> tuple[date, date]:
    if preset == "Lifetime":
        return lifetime_start, lifetime_end
    if preset == "Last 12m":
        start = date(lifetime_end.year - 1, lifetime_end.month, 1)
        return start, lifetime_end
    if preset == "YTD":
        return date(lifetime_end.year, 1, 1), lifetime_end
    if preset == "FY":
        fy_start_year = lifetime_end.year if lifetime_end.month >= 7 else lifetime_end.year - 1
        return date(fy_start_year, 7, 1), lifetime_end
    return lifetime_start, lifetime_end


def render_sidebar(df: pd.DataFrame) -> dict:
    st.sidebar.header("Filters")

    job_map = _job_label_map(df)
    job_keys = sorted(job_map.keys())
    if not job_keys:
        st.error("No Job_Key values found.")
        st.stop()

    job_key = st.sidebar.selectbox(
        "Job",
        options=job_keys,
        format_func=lambda k: job_map.get(k, k),
        key="job_key",
    )

    job_df = df[df[COL_JOB_KEY] == job_key]
    lifetime = job_df[job_df[COL_MONTH_KEY].notna()][COL_MONTH_KEY]
    if lifetime.empty:
        lifetime_start = date.today()
        lifetime_end = date.today()
    else:
        lifetime_start = lifetime.min().date()
        lifetime_end = lifetime.max().date()

    preset = st.sidebar.selectbox(
        "Date Preset",
        options=["Lifetime", "FY", "YTD", "Last 12m", "Custom"],
        index=0,
        key="date_preset",
    )
    if preset != "Custom":
        date_range = _preset_range(preset, lifetime_start, lifetime_end)
        st.session_state["date_range"] = date_range

    date_range = st.sidebar.date_input(
        "Date Range (Month_Key)",
        value=st.session_state.get("date_range", (lifetime_start, lifetime_end)),
        key="date_range_input",
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        st.session_state["date_range"] = (start_date, end_date)
    else:
        start_date, end_date = lifetime_start, lifetime_end

    include_unallocated = st.sidebar.toggle(
        "Include UNALLOCATED revenue",
        value=st.session_state.get("include_unallocated", True),
        key="include_unallocated",
    )
    include_quote_only = st.sidebar.toggle(
        "Include QUOTE_ONLY tasks in tables",
        value=st.session_state.get("include_quote_only", True),
        key="include_quote_only",
    )

    filters = {"job_key": job_key, "start_date": start_date, "end_date": end_date}

    if "Department_Eff" in job_df.columns:
        filters["department"] = st.sidebar.multiselect(
            "Department",
            options=_sorted_unique(job_df["Department_Eff"]),
            default=st.session_state.get("department", []),
            key="department",
        )
    if "Function" in job_df.columns:
        filters["function"] = st.sidebar.multiselect(
            "Function",
            options=_sorted_unique(job_df["Function"]),
            default=st.session_state.get("function", []),
            key="function",
        )
    if "Business Unit" in job_df.columns:
        filters["business_unit"] = st.sidebar.multiselect(
            "Business Unit",
            options=_sorted_unique(job_df["Business Unit"]),
            default=st.session_state.get("business_unit", []),
            key="business_unit",
        )
    if "Billable?" in job_df.columns:
        filters["billable"] = st.sidebar.multiselect(
            "Billable?",
            options=_sorted_unique(job_df["Billable?"]),
            default=st.session_state.get("billable", []),
            key="billable",
        )
    if "Deliverable" in job_df.columns:
        filters["deliverable"] = st.sidebar.multiselect(
            "Deliverable",
            options=_sorted_unique(job_df["Deliverable"]),
            default=st.session_state.get("deliverable", []),
            key="deliverable",
        )
    if "Role" in job_df.columns:
        filters["role"] = st.sidebar.multiselect(
            "Role",
            options=_sorted_unique(job_df["Role"]),
            default=st.session_state.get("role", []),
            key="role",
        )
    if "Task" in job_df.columns:
        filters["task"] = st.sidebar.multiselect(
            "Task",
            options=_sorted_unique(job_df["Task"]),
            default=st.session_state.get("task", []),
            key="task",
        )
    if "Source" in job_df.columns:
        filters["source"] = st.sidebar.multiselect(
            "Revenue Source",
            options=_sorted_unique(job_df["Source"]),
            default=st.session_state.get("source", []),
            key="source",
        )

    filters["include_unallocated"] = include_unallocated
    filters["include_quote_only"] = include_quote_only
    filters["lifetime_start"] = lifetime_start
    filters["lifetime_end"] = lifetime_end
    return filters


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    out = df.copy()
    if filters.get("department"):
        out = out[out["Department_Eff"].isin(filters["department"])]
    if filters.get("function"):
        out = out[out["Function"].isin(filters["function"])]
    if filters.get("business_unit"):
        out = out[out["Business Unit"].isin(filters["business_unit"])]
    if filters.get("billable"):
        out = out[out["Billable?"].isin(filters["billable"])]
    if filters.get("deliverable"):
        out = out[out["Deliverable"].isin(filters["deliverable"])]
    if filters.get("role"):
        out = out[out["Role"].isin(filters["role"])]
    if filters.get("task"):
        out = out[out["Task"].isin(filters["task"])]
    if filters.get("source"):
        out = out[out["Source"].isin(filters["source"])]
    return out
