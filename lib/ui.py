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


def preset_range(preset: str, lifetime_start: date, lifetime_end: date) -> tuple[date, date]:
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


def _sort_fy_labels(labels: list[str]) -> list[str]:
    def _fy_key(label: str) -> int:
        if not label:
            return 0
        return int(label.replace("FY", ""))

    return sorted(labels, key=_fy_key)


def job_label_map(df: pd.DataFrame) -> dict[str, str]:
    return _job_label_map(df)

FILTER_KEYS = [
    "department",
    "function",
    "business_unit",
    "billable",
    "deliverable",
    "role",
    "task",
    "source",
]


def _reset_filters() -> None:
    for key in FILTER_KEYS:
        st.session_state[key] = []


def render_sidebar(df: pd.DataFrame) -> dict:
    st.sidebar.header("Filters")

    lifetime = df[df[COL_MONTH_KEY].notna()][COL_MONTH_KEY]
    if lifetime.empty:
        lifetime_start = date.today()
        lifetime_end = date.today()
    else:
        lifetime_start = lifetime.min().date()
        lifetime_end = lifetime.max().date()

    fy_labels = []
    if "FY_Label" in df.columns:
        fy_labels = _sort_fy_labels(
            df["FY_Label"].dropna().astype(str).unique().tolist()
        )
    if not fy_labels:
        fy_labels = ["FY0"]

    current_fy = fy_labels[-1]
    last_3_fys = fy_labels[-3:] if len(fy_labels) >= 3 else fy_labels

    if "applied_filters" not in st.session_state:
        st.session_state["applied_filters"] = {}
        st.session_state["applied_fy_preset"] = "Last 3 FYs"
        st.session_state["applied_fy_labels"] = last_3_fys
        st.session_state["applied_trend_grain"] = "FY"
        st.session_state["applied_scope_level"] = "Company"
        st.session_state["applied_include_unallocated"] = True
        st.session_state["applied_include_quote_only"] = True
        st.session_state["applied_quote_mode"] = "Earned Quote Proxy"

    with st.sidebar.form("filters_form"):
        scope_level = st.selectbox(
            "Scope Level",
            options=["Company", "Department", "Product", "Job"],
            index=["Company", "Department", "Product", "Job"].index(
                st.session_state.get("applied_scope_level", "Company")
            ),
            key="scope_level_input",
        )
        fy_preset = st.selectbox(
            "FY Preset",
            options=["Last 3 FYs", "FYTD", "All FYs", "Custom"],
            index=["Last 3 FYs", "FYTD", "All FYs", "Custom"].index(
                st.session_state.get("applied_fy_preset", "Last 3 FYs")
            ),
            key="fy_preset_input",
        )
        fy_selected = st.multiselect(
            "Fiscal Years",
            options=fy_labels,
            default=st.session_state.get("applied_fy_labels", last_3_fys),
            key="fy_labels_input",
            disabled=fy_preset != "Custom",
        )
        trend_grain = st.radio(
            "Trend Granularity",
            options=["FY", "Month"],
            index=["FY", "Month"].index(
                st.session_state.get("applied_trend_grain", "FY")
            ),
            horizontal=True,
            key="trend_grain_input",
        )

        include_unallocated = st.toggle(
            "Include UNALLOCATED revenue",
            value=st.session_state.get("applied_include_unallocated", True),
            key="include_unallocated_input",
        )
        include_quote_only = st.toggle(
            "Include QUOTE_ONLY tasks in tables",
            value=st.session_state.get("applied_include_quote_only", True),
            key="include_quote_only_input",
        )

        quote_default = st.session_state.get("applied_quote_mode", "Earned Quote Proxy")
        quote_mode = st.selectbox(
            "Quote Alignment",
            options=["Earned Quote Proxy", "Lifetime Quote"],
            index=["Earned Quote Proxy", "Lifetime Quote"].index(quote_default),
            key="quote_mode_input",
        )

        def _multiselect(label: str, column: str, key: str) -> list[str]:
            if column not in df.columns:
                return []
            return st.multiselect(
                label,
                options=_sorted_unique(df[column]),
                default=st.session_state.get("applied_filters", {}).get(key, []),
                key=f"{key}_input",
            )

        selected_department = _multiselect("Department", "Department_Eff", "department")
        selected_function = _multiselect("Function", "Function", "function")
        selected_business_unit = _multiselect("Business Unit", "Business Unit", "business_unit")
        selected_billable = _multiselect("Billable?", "Billable?", "billable")
        selected_deliverable = _multiselect("Deliverable", "Deliverable", "deliverable")
        selected_role = _multiselect("Role", "Role", "role")
        selected_task = _multiselect("Task", "Task", "task")
        selected_source = _multiselect("Revenue Source", "Source", "source")

        applied = st.form_submit_button("Apply Filters")

    if st.sidebar.button("Reset Filters"):
        _reset_filters()
        st.session_state["applied_filters"] = {}
        st.session_state["applied_fy_preset"] = "Last 3 FYs"
        st.session_state["applied_fy_labels"] = last_3_fys
        st.session_state["applied_trend_grain"] = "FY"
        st.session_state["applied_scope_level"] = "Company"
        st.session_state["applied_include_unallocated"] = True
        st.session_state["applied_include_quote_only"] = True
        st.session_state["applied_quote_mode"] = "Earned Quote Proxy"
        st.session_state["filters_version"] = st.session_state.get("filters_version", 0) + 1
    st.sidebar.caption("Changes apply when you click Apply Filters.")

    if applied:
        st.session_state["applied_scope_level"] = scope_level
        st.session_state["applied_fy_preset"] = fy_preset
        if fy_preset == "Custom":
            st.session_state["applied_fy_labels"] = fy_selected or last_3_fys
        elif fy_preset == "FYTD":
            st.session_state["applied_fy_labels"] = [current_fy]
        elif fy_preset == "All FYs":
            st.session_state["applied_fy_labels"] = fy_labels
        else:
            st.session_state["applied_fy_labels"] = last_3_fys
        st.session_state["applied_trend_grain"] = trend_grain
        st.session_state["applied_include_unallocated"] = include_unallocated
        st.session_state["applied_include_quote_only"] = include_quote_only
        st.session_state["applied_quote_mode"] = quote_mode
        st.session_state["applied_filters"] = {
            "department": selected_department,
            "function": selected_function,
            "business_unit": selected_business_unit,
            "billable": selected_billable,
            "deliverable": selected_deliverable,
            "role": selected_role,
            "task": selected_task,
            "source": selected_source,
        }
        st.session_state["filters_version"] = st.session_state.get("filters_version", 0) + 1

    applied_filters = st.session_state.get("applied_filters", {})
    fy_selection = st.session_state.get("applied_fy_labels", last_3_fys)
    if not fy_selection:
        fy_selection = last_3_fys or fy_labels
    start_date, end_date = lifetime_start, lifetime_end
    if "FY_Label" in df.columns and fy_selection:
        fy_df = df[df["FY_Label"].isin(fy_selection) & df[COL_MONTH_KEY].notna()]
        if not fy_df.empty:
            start_date = fy_df[COL_MONTH_KEY].min().date()
            end_date = fy_df[COL_MONTH_KEY].max().date()

    filters = {
        "start_date": start_date,
        "end_date": end_date,
        "fy_labels": fy_selection,
        "trend_grain": st.session_state.get("applied_trend_grain", "FY"),
        "scope_level": st.session_state.get("applied_scope_level", "Company"),
        "quote_mode": st.session_state.get("applied_quote_mode", "Earned Quote Proxy"),
        "include_unallocated": st.session_state.get("applied_include_unallocated", True),
        "include_quote_only": st.session_state.get("applied_include_quote_only", True),
        "lifetime_start": lifetime_start,
        "lifetime_end": lifetime_end,
    }
    filters.update(applied_filters)
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
