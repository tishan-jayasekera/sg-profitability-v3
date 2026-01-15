#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "Job_Key",
    "Task_Key",
    "Month_Key",
    "Hours",
    "Cost",
    "Allocated_Revenue",
    "Amount",
    "Job_Month_Total_Hours",
    "Source",
    "[Job Task] Quoted Time",
    "[Job Task] Quoted Amount",
    "[Task] Base Rate",
    "[Task] Billable Rate",
    "[Task] Base Rate_QUOTE",
    "[Task] Billable Rate_QUOTE",
]


def load_minimal(path: Path) -> pd.DataFrame:
    cols = pd.read_csv(path, nrows=0).columns.tolist()
    missing = [c for c in REQUIRED_COLUMNS if c not in cols]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    usecols = list(dict.fromkeys(REQUIRED_COLUMNS + ["Revenue_Share"]))
    return pd.read_csv(path, usecols=usecols, low_memory=False)


def classify_row_type(df: pd.DataFrame) -> pd.Series:
    row_type = pd.Series("UNKNOWN", index=df.index)
    row_type[(df["Month_Key"].notna()) & (df["Task_Key"] != "__UNALLOCATED__")] = "TASK_MONTH"
    row_type[(df["Month_Key"].notna()) & (df["Task_Key"] == "__UNALLOCATED__")] = "UNALLOCATED_REV"
    row_type[df["Month_Key"].isna()] = "QUOTE_ONLY"
    return row_type


def main() -> int:
    parser = argparse.ArgumentParser(description="Run data readiness checks.")
    parser.add_argument(
        "--path",
        default="data/Unified_Job_Profitability_Full_Data.csv",
        help="CSV path",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.01,
        help="Tolerance for allocation closure checks",
    )
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise FileNotFoundError(path)

    df = load_minimal(path)
    row_type = classify_row_type(df)

    print("Data Readiness Summary")
    print("======================")
    print(f"Rows: {len(df):,}")
    print(f"Jobs: {df['Job_Key'].nunique():,}")
    print(f"Tasks: {df['Task_Key'].nunique():,}")
    print(f"Months: {df['Month_Key'].nunique(dropna=True):,}")
    print("")
    print("Row Types")
    print("---------")
    for k, v in row_type.value_counts().items():
        print(f"{k}: {v:,}")
    print("")

    task_month = df[row_type == "TASK_MONTH"]
    job_month = df[df["Month_Key"].notna()]

    # TASK_MONTH grain uniqueness
    dup = (
        task_month.groupby(["Job_Key", "Task_Key", "Month_Key"]).size().sort_values(ascending=False)
    )
    dup_over = dup[dup > 1]
    print("Grain Checks")
    print("------------")
    print(f"TASK_MONTH duplicates (>1 row per JobxTaskxMonth): {len(dup_over):,}")

    # Job-month Amount consistency
    amt_var = (
        job_month.groupby(["Job_Key", "Month_Key"])["Amount"].nunique(dropna=False)
    )
    amt_var_over = amt_var[amt_var > 1]
    print(f"Job-month Amount variation (>1 unique Amount): {len(amt_var_over):,}")

    # Job-month total hours consistency
    hours_var = (
        job_month.groupby(["Job_Key", "Month_Key"])["Job_Month_Total_Hours"].nunique(dropna=False)
    )
    hours_var_over = hours_var[hours_var > 1]
    print(f"Job-month total hours variation (>1 unique value): {len(hours_var_over):,}")

    # Allocation closure per job-month
    alloc_sum = job_month.groupby(["Job_Key", "Month_Key"])["Allocated_Revenue"].sum(min_count=1)
    amt = job_month.groupby(["Job_Key", "Month_Key"])["Amount"].first()
    alloc_diff = (alloc_sum - amt).abs()
    alloc_failed = alloc_diff[alloc_diff > args.tolerance]
    print(
        f"Allocation closure failures (> {args.tolerance}): {len(alloc_failed):,}"
    )
    print("")

    # Quote field consistency per job-task
    print("Quote Consistency")
    print("-----------------")
    for col in ["[Job Task] Quoted Time", "[Job Task] Quoted Amount"]:
        nunique = df.groupby(["Job_Key", "Task_Key"])[col].nunique(dropna=True)
        inconsistent = nunique[nunique > 1]
        print(f"{col}: {len(inconsistent):,} job-tasks with multiple values")
    print("")

    # Missing and edge cases
    print("Missing / Edge Cases")
    print("--------------------")
    print(
        "Hours>0 with Allocated_Revenue is null: "
        f"{((df['Hours'] > 0) & df['Allocated_Revenue'].isna()).sum():,}"
    )
    print(
        "Month present with Amount is null: "
        f"{((df['Month_Key'].notna()) & df['Amount'].isna()).sum():,}"
    )
    print(
        "Unallocated rows with Month missing: "
        f"{((df['Task_Key'] == '__UNALLOCATED__') & df['Month_Key'].isna()).sum():,}"
    )
    print(
        "Quote-only rows with quoted time/amount present: "
        f"{((row_type == 'QUOTE_ONLY') & (df['[Job Task] Quoted Amount'].notna() | df['[Job Task] Quoted Time'].notna())).sum():,}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
