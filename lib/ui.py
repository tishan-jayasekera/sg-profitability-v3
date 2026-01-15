from __future__ import annotations

import streamlit as st


def render_sidebar(df) -> dict:
    st.sidebar.header("Filters")

    fy_labels = []
    if "FY_Label" in df.columns:
        fy_labels = sorted(df["FY_Label"].dropna().astype(str).unique().tolist())
    if not fy_labels:
        fy_labels = ["FY0"]

    current_fy = fy_labels[-1]
    last_3_fys = fy_labels[-3:] if len(fy_labels) >= 3 else fy_labels

    if "applied_fy_preset" not in st.session_state:
        st.session_state["applied_fy_preset"] = "Last 3 FYs"
        st.session_state["applied_fy_labels"] = last_3_fys
        st.session_state["applied_quote_mode"] = "Earned Quote Proxy"
        st.session_state["applied_include_unallocated"] = True
        st.session_state["applied_include_quote_only"] = True

    with st.sidebar.form("filters_form"):
        fy_preset = st.selectbox(
            "FY Preset",
            options=["Last 3 FYs", "FYTD", "All FYs", "Custom"],
            index=["Last 3 FYs", "FYTD", "All FYs", "Custom"].index(
                st.session_state.get("applied_fy_preset", "Last 3 FYs")
            ),
        )
        fy_selected = st.multiselect(
            "Fiscal Years",
            options=fy_labels,
            default=st.session_state.get("applied_fy_labels", last_3_fys),
            disabled=fy_preset != "Custom",
        )
        quote_mode = st.selectbox(
            "Quote Alignment",
            options=["Earned Quote Proxy", "Lifetime Quote"],
            index=["Earned Quote Proxy", "Lifetime Quote"].index(
                st.session_state.get("applied_quote_mode", "Earned Quote Proxy")
            ),
        )
        include_unallocated = st.toggle(
            "Include UNALLOCATED revenue",
            value=st.session_state.get("applied_include_unallocated", True),
        )
        include_quote_only = st.toggle(
            "Include quote-only tasks in tables",
            value=st.session_state.get("applied_include_quote_only", True),
        )

        applied = st.form_submit_button("Apply")

    if st.sidebar.button("Reset"):
        st.session_state["applied_fy_preset"] = "Last 3 FYs"
        st.session_state["applied_fy_labels"] = last_3_fys
        st.session_state["applied_quote_mode"] = "Earned Quote Proxy"
        st.session_state["applied_include_unallocated"] = True
        st.session_state["applied_include_quote_only"] = True

    if applied:
        st.session_state["applied_fy_preset"] = fy_preset
        if fy_preset == "Custom":
            st.session_state["applied_fy_labels"] = fy_selected or last_3_fys
        elif fy_preset == "FYTD":
            st.session_state["applied_fy_labels"] = [current_fy]
        elif fy_preset == "All FYs":
            st.session_state["applied_fy_labels"] = fy_labels
        else:
            st.session_state["applied_fy_labels"] = last_3_fys
        st.session_state["applied_quote_mode"] = quote_mode
        st.session_state["applied_include_unallocated"] = include_unallocated
        st.session_state["applied_include_quote_only"] = include_quote_only

    fy_selection = st.session_state.get("applied_fy_labels", last_3_fys)
    return {
        "fy_labels": fy_selection,
        "quote_mode": st.session_state.get("applied_quote_mode", "Earned Quote Proxy"),
        "include_unallocated": st.session_state.get("applied_include_unallocated", True),
        "include_quote_only": st.session_state.get("applied_include_quote_only", True),
    }


def apply_filters(df, filters: dict):
    return df
