"""
etl.py — ETL Pipeline Script
==============================
Task    : Task 1 — Data Preprocessing & ETL Pipeline
Module  : Principles of Data Science (7144)
Dataset : Online Retail II (Kaggle — mathchi)
Member  : Pasindu Ashan (COMScDS251P-001)
Branch  : feature-pasindu-ETL-EDA

Description
-----------
This script is the production-ready, importable version of the ETL pipeline
documented in notebooks/01_etl.ipynb. It can be run as a standalone script
or imported by other notebooks/scripts in the project.

Usage
-----
    python src/etl.py

    Or from another script:
        from src.etl import run_pipeline, load_cleaned_data

Output
------
    data/cleaned_retail_data.csv   — 333,234 clean records, 13 columns, UTF-8
    reports/pipeline_summary.png  — Pipeline impact bar chart
    reports/outliers_before.png   — Boxplots before IQR removal
    reports/outliers_after.png    — Boxplots after IQR removal
    reports/missing_values.png    — Missing value percentage chart
"""

# ── Standard Library ──────────────────────────────────────────
import os
import sys
import warnings
from pathlib import Path

# ── Third-Party Libraries ────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless backend — no display needed
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ── Path Configuration ────────────────────────────────────────
# Works whether run from project root or from src/ directory
ROOT_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT_DIR / "data"
REPORTS_DIR = ROOT_DIR / "reports"

KAGGLE_DATASET  = "mathchi/online-retail-ii-data-set-from-ml-repository"
RAW_FILENAME    = "Year 2010-2011.csv"
OUTPUT_FILENAME = "cleaned_retail_data.csv"

RAW_FILE    = DATA_DIR / RAW_FILENAME
OUTPUT_FILE = DATA_DIR / OUTPUT_FILENAME


# ══════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════

def _banner(title: str) -> None:
    """Print a section header to stdout."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def _save_figure(filename: str) -> None:
    """Save current matplotlib figure to the reports directory."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / filename
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved: {path}")


def _log_step(pipeline_log: list, step_name: str, before: int, after: int) -> None:
    """Record pipeline step impact and print progress."""
    removed = before - after
    pct = round(removed / before * 100, 2) if before > 0 else 0.0
    pipeline_log.append({
        "Step"     : step_name,
        "Before"   : before,
        "After"    : after,
        "Removed"  : removed,
        "% Removed": pct,
    })
    print(f"  [{step_name}]")
    print(f"    Before: {before:,}  |  After: {after:,}  |  Removed: {removed:,} rows  ({pct}%)")


# ══════════════════════════════════════════════════════════════
# STEP 1 — EXTRACT
# ══════════════════════════════════════════════════════════════

def extract(kaggle_download: bool = True) -> pd.DataFrame:
    """
    Extract the Online Retail II dataset.

    Parameters
    ----------
    kaggle_download : bool
        If True, download the dataset from Kaggle using kagglehub.
        If False, load from the local data/ directory.

    Returns
    -------
    pd.DataFrame
        Raw dataset with 541,910 rows and 8 columns.
    """
    _banner("STEP 1: EXTRACT")

    # ── 1A. Download from Kaggle (optional) ──────────────────
    if kaggle_download and not RAW_FILE.exists():
        try:
            import kagglehub  # pip install kagglehub
            print(f"  Downloading: {KAGGLE_DATASET}")
            download_path = kagglehub.dataset_download(KAGGLE_DATASET)
            print(f"  Downloaded to: {download_path}")

            # Locate the CSV inside the downloaded directory
            for root, _, files in os.walk(download_path):
                for fname in files:
                    if RAW_FILENAME in fname and fname.endswith(".csv"):
                        src = Path(root) / fname
                        DATA_DIR.mkdir(parents=True, exist_ok=True)
                        import shutil
                        shutil.copy(src, RAW_FILE)
                        print(f"  Copied to: {RAW_FILE}")
                        break
        except ImportError:
            print("  WARNING: kagglehub not installed. Trying local file.")
        except Exception as exc:
            print(f"  WARNING: Kaggle download failed ({exc}). Trying local file.")

    # ── 1B. Load CSV ──────────────────────────────────────────
    # ISO-8859-1 encoding is required to preserve special characters
    # in product descriptions (e.g., accented letters from UK/EU suppliers).
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Raw data file not found: {RAW_FILE}\n"
            "Run with kaggle_download=True or place the file manually."
        )

    print(f"  Loading: {RAW_FILE}")
    df = pd.read_csv(RAW_FILE, encoding="ISO-8859-1")

    # Strip accidental whitespace from column names
    df.columns = df.columns.str.strip()

    # ── 1C. Extraction Report ─────────────────────────────────
    print(f"\n  Total rows loaded : {df.shape[0]:,}")
    print(f"  Total columns     : {df.shape[1]}")
    print(f"  Column names      : {list(df.columns)}")
    print(f"  Memory usage      : {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")

    return df


# ══════════════════════════════════════════════════════════════
# STEP 2 — VALIDATE
# ══════════════════════════════════════════════════════════════

def validate(df: pd.DataFrame) -> dict:
    """
    Run a data quality audit across four dimensions:
    Completeness, Consistency, Uniqueness, Validity.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe from extract().

    Returns
    -------
    dict
        Summary of quality issues found.
    """
    _banner("STEP 2: VALIDATE — Data Quality Audit")

    # Detect column name variants (Kaggle vs UCI versions)
    invoice_col  = "Invoice"     if "Invoice"     in df.columns else "InvoiceNo"
    customer_col = "Customer ID" if "Customer ID" in df.columns else "CustomerID"

    # ── 2A. Missing Values ────────────────────────────────────
    null_report = pd.DataFrame({
        "Missing Count": df.isnull().sum(),
        "Missing %"    : (df.isnull().sum() / len(df) * 100).round(2),
    }).sort_values("Missing %", ascending=False)

    print("\n  --- Missing Value Report ---")
    print(null_report.to_string())

    missing_data = null_report[null_report["Missing Count"] > 0]
    if len(missing_data) > 0:
        plt.figure(figsize=(8, 4))
        colors = ["#C0392B" if p > 10 else "#E67E22" for p in missing_data["Missing %"]]
        bars = plt.bar(missing_data.index, missing_data["Missing %"], color=colors, edgecolor="white")
        plt.title("Missing Value % by Column", fontsize=13, fontweight="bold")
        plt.ylabel("Missing %")
        for bar, val in zip(bars, missing_data["Missing %"]):
            plt.text(bar.get_x() + bar.get_width() / 2, val + 0.3, f"{val}%",
                     ha="center", fontweight="bold", fontsize=10)
        plt.tight_layout()
        _save_figure("missing_values.png")

    # ── 2B. Consistency Issues ────────────────────────────────
    cancellations = df[invoice_col].astype(str).str.startswith("C").sum()
    negative_qty  = (df["Quantity"] < 0).sum()
    zero_qty      = (df["Quantity"] == 0).sum()
    bad_price     = (df["Price"] <= 0).sum()
    exact_dupes   = df.duplicated().sum()

    print("\n  --- Consistency Issues ---")
    print(f"  Cancellation invoices (prefix C) : {cancellations:,}")
    print(f"  Negative Quantity rows           : {negative_qty:,}")
    print(f"  Zero Quantity rows               : {zero_qty:,}")
    print(f"  Zero / Negative Price rows       : {bad_price:,}")
    print(f"  Exact duplicate rows             : {exact_dupes:,}")

    return {
        "invoice_col"   : invoice_col,
        "customer_col"  : customer_col,
        "cancellations" : cancellations,
        "negative_qty"  : negative_qty,
        "bad_price"     : bad_price,
        "exact_dupes"   : exact_dupes,
    }


# ══════════════════════════════════════════════════════════════
# STEP 3 — TRANSFORM
# ══════════════════════════════════════════════════════════════

def transform(df: pd.DataFrame, audit: dict) -> tuple[pd.DataFrame, list]:
    """
    Apply the 6-step cleaning and feature engineering pipeline.

    Steps
    -----
    3.1  Remove cancellation invoices (prefix 'C')
    3.2  Remove exact duplicate rows
    3.3  Drop rows with null Customer ID
    3.4  Filter Quantity <= 0 and Price <= 0
    3.5  IQR outlier removal on Quantity and Price (Tukey 1.5x fence)
    3.6  Feature engineering (datetime parsing + derived columns)

    Parameters
    ----------
    df    : pd.DataFrame  Raw or partially cleaned dataframe.
    audit : dict          Output from validate() containing column name variants.

    Returns
    -------
    tuple[pd.DataFrame, list]
        Cleaned dataframe and pipeline impact log.
    """
    _banner("STEP 3: TRANSFORM — 6-Step Cleaning Pipeline")

    invoice_col  = audit["invoice_col"]
    customer_col = audit["customer_col"]
    pipeline_log = []

    print(f"\n  Starting rows: {len(df):,}\n")

    # ── Step 3.1: Remove Cancellation Transactions ────────────
    # Invoices starting with 'C' are financial reversals (credits/returns).
    # They are NOT real sales — including them inflates negative revenue.
    # Assumption A3: See docs/assumptions.md
    before = len(df)
    df = df[~df[invoice_col].astype(str).str.startswith("C")]
    _log_step(pipeline_log, "3.1 Remove Cancellation Invoices", before, len(df))

    # ── Step 3.2: Remove Exact Duplicate Rows ────────────────
    # Exact duplicates arise from system-level logging errors.
    # Retaining them inflates transaction counts and biases frequency metrics.
    before = len(df)
    df = df.drop_duplicates()
    _log_step(pipeline_log, "3.2 Remove Exact Duplicates", before, len(df))

    # ── Step 3.3: Drop Rows with Missing Customer ID ──────────
    # Customer ID is the primary key for Task 3 Clustering (RFM).
    # Rows without it cannot be assigned to any customer segment.
    # Imputation is NOT appropriate — identity cannot be guessed.
    # Assumption A2: See docs/assumptions.md
    before = len(df)
    df = df.dropna(subset=[customer_col])
    _log_step(pipeline_log, "3.3 Drop Null Customer ID", before, len(df))

    # Cast Customer ID: float → int → string for clean representation
    df[customer_col] = df[customer_col].astype(int).astype(str)

    # ── Step 3.4: Filter Invalid Quantity and Price ───────────
    # Negative quantities (returns not caught by 3.1) and zero/negative
    # prices produce nonsensical Total Revenue values.
    # Both conditions are filtered simultaneously as a boolean mask.
    # Assumptions A4, A5: See docs/assumptions.md
    before = len(df)
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]
    _log_step(pipeline_log, "3.4 Filter Invalid Qty/Price", before, len(df))

    # ── Step 3.5: IQR Outlier Removal ────────────────────────
    # IQR method (Tukey, 1977) — distribution-agnostic, robust to skew.
    # Fence: Q1 - 1.5*IQR  to  Q3 + 1.5*IQR
    # Applied INDEPENDENTLY to Quantity and Price.
    # Hard removal chosen over capping — see Assumptions A6, A7, A8.

    # Visualise distributions BEFORE removal
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(y=df["Quantity"], ax=axes[0], color="#E74C3C")
    axes[0].set_title("Quantity — Before IQR Removal", fontweight="bold")
    sns.boxplot(y=df["Price"], ax=axes[1], color="#E74C3C")
    axes[1].set_title("Price — Before IQR Removal", fontweight="bold")
    plt.suptitle("Outlier Detection (Before)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save_figure("outliers_before.png")

    before = len(df)
    print("\n  IQR Bounds:")
    for col in ["Quantity", "Price"]:
        q1  = df[col].quantile(0.25)
        q3  = df[col].quantile(0.75)
        iqr = q3 - q1
        lo  = q1 - 1.5 * iqr
        hi  = q3 + 1.5 * iqr
        df  = df[(df[col] >= lo) & (df[col] <= hi)]
        print(f"    {col:<12}: Q1={q1:.2f}  Q3={q3:.2f}  IQR={iqr:.2f}  "
              f"Valid range=[{lo:.2f}, {hi:.2f}]")

    _log_step(pipeline_log, "3.5 IQR Outlier Removal", before, len(df))

    # Visualise distributions AFTER removal
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(y=df["Quantity"], ax=axes[0], color="#27AE60")
    axes[0].set_title("Quantity — After IQR Removal", fontweight="bold")
    sns.boxplot(y=df["Price"], ax=axes[1], color="#27AE60")
    axes[1].set_title("Price — After IQR Removal", fontweight="bold")
    plt.suptitle("Outlier Detection (After)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save_figure("outliers_after.png")

    # ── Step 3.6: Feature Engineering ────────────────────────
    # New columns support Task 2 (EDA), Task 3 (Clustering), Task 4 (Forecasting).
    # Computed AFTER cleaning to ensure all derived values are valid.
    # Assumption A10: See docs/assumptions.md

    df["InvoiceDate"]   = pd.to_datetime(df["InvoiceDate"])
    df["Year"]          = df["InvoiceDate"].dt.year        # Year-over-Year trends
    df["Month"]         = df["InvoiceDate"].dt.month       # Monthly seasonality
    df["DayOfWeek"]     = df["InvoiceDate"].dt.day_name()  # Weekly demand patterns
    df["Hour"]          = df["InvoiceDate"].dt.hour        # Intraday purchasing
    df["Total_Revenue"] = df["Quantity"] * df["Price"]     # Primary financial KPI
    df[customer_col]    = df[customer_col].astype(int)     # Final ID as integer

    before_fe = len(df)
    _log_step(pipeline_log, "3.6 Feature Engineering (no rows removed)", before_fe, len(df))

    print(f"\n  New columns added: Year, Month, DayOfWeek, Hour, Total_Revenue")
    print(f"  Final dataset shape : {df.shape}")
    print(f"\n--- Updated Data Types ---")
    print(df.dtypes.to_string())
    print(f"\n--- Cleaned Descriptive Statistics ---")
    print(df[["Quantity", "Price", "Total_Revenue"]].describe().round(2).to_string())

    return df, pipeline_log


# ══════════════════════════════════════════════════════════════
# STEP 4 — LOAD
# ══════════════════════════════════════════════════════════════

def load(df: pd.DataFrame) -> pd.DataFrame:
    """
    Export the cleaned dataset and run post-load validation.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned, feature-engineered dataframe from transform().

    Returns
    -------
    pd.DataFrame
        The reloaded dataframe (loaded fresh from disk for verification).
    """
    _banner("STEP 4: LOAD — Export to CSV")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    print(f"  Saved: {OUTPUT_FILE}")

    # ── Post-Load Validation ──────────────────────────────────
    # Reload from disk and verify — treats the file as an independent artefact.
    df_check = pd.read_csv(OUTPUT_FILE)

    customer_col = "Customer ID" if "Customer ID" in df_check.columns else "CustomerID"

    print("\n  --- Post-Load Validation Report ---")
    print(f"  Rows loaded from disk   : {df_check.shape[0]:,}")
    print(f"  Columns                 : {df_check.shape[1]}")
    print(f"  Null values remaining   : {df_check.isnull().sum().sum()}")
    print(f"  Negative Revenue rows   : {(df_check['Total_Revenue'] <= 0).sum()}")
    print(f"  Unique customers        : {df_check[customer_col].nunique():,}")
    print(f"  Unique products         : {df_check['StockCode'].nunique():,}")
    print(f"  Date range              : {df_check['InvoiceDate'].min()} → {df_check['InvoiceDate'].max()}")
    print(f"  Total Revenue (£)       : £{df_check['Total_Revenue'].sum():,.2f}")

    checks_passed = (
        df_check.isnull().sum().sum() == 0
        and (df_check["Total_Revenue"] <= 0).sum() == 0
    )

    if checks_passed:
        print("\n  ✅ VALIDATION PASSED — Dataset is clean and ready for analysis!")
    else:
        print("\n  ❌ VALIDATION FAILED — Review pipeline steps above.")

    return df_check


# ══════════════════════════════════════════════════════════════
# STEP 5 — REPORTING
# ══════════════════════════════════════════════════════════════

def report(pipeline_log: list, clean_rows: int) -> None:
    """
    Generate the pipeline impact summary table and bar chart.

    Parameters
    ----------
    pipeline_log : list  Output from transform().
    clean_rows   : int   Final clean record count.
    """
    _banner("STEP 5: REPORTING — Pipeline Summary")

    log_df = pd.DataFrame(pipeline_log)
    print("\n  --- ETL Pipeline Impact Summary ---")
    print(log_df.to_string(index=False))

    # Bar chart
    all_labels = ["0. Raw Extract"] + [row["Step"] for row in pipeline_log]
    all_counts = [pipeline_log[0]["Before"]] + [row["After"] for row in pipeline_log]
    colors = ["#1F3864", "#C0392B", "#C0392B", "#C0392B",
              "#E67E22", "#E67E22", "#27AE60"]
    colors = colors[:len(all_labels)]

    plt.figure(figsize=(14, 6))
    bars = plt.bar(range(len(all_labels)), all_counts,
                   color=colors, edgecolor="white", width=0.65)
    plt.xticks(range(len(all_labels)), all_labels, rotation=30, ha="right", fontsize=9)
    plt.ylabel("Number of Records", fontsize=11)
    plt.title("ETL Pipeline — Records at Each Stage", fontsize=14, fontweight="bold")
    plt.axhline(y=clean_rows, color="green", linestyle="--", linewidth=1.5,
                label=f"Final Output: {clean_rows:,} records")
    for bar, val in zip(bars, all_counts):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 1500,
                 f"{val:,}", ha="center", fontsize=8, fontweight="bold")
    plt.legend(fontsize=10)
    plt.tight_layout()
    _save_figure("pipeline_summary.png")

    raw_rows = pipeline_log[0]["Before"]
    removed  = raw_rows - clean_rows
    print(f"\n  Raw records    : {raw_rows:,}")
    print(f"  Clean records  : {clean_rows:,}")
    print(f"  Removed total  : {removed:,}  ({removed / raw_rows * 100:.1f}% of raw)")


# ══════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════

def run_pipeline(kaggle_download: bool = True) -> pd.DataFrame:
    """
    Run the complete ETL pipeline end-to-end.

    Parameters
    ----------
    kaggle_download : bool
        Whether to attempt downloading from Kaggle (default: True).

    Returns
    -------
    pd.DataFrame
        The final cleaned and validated dataframe loaded from disk.
    """
    print("=" * 60)
    print("  ETL PIPELINE — Online Retail II Dataset")
    print("  Principles of Data Science 7144 — Group Assignment")
    print("=" * 60)

    raw_df             = extract(kaggle_download=kaggle_download)
    audit              = validate(raw_df)
    clean_df, log      = transform(raw_df, audit)
    validated_df       = load(clean_df)
    report(log, len(validated_df))

    _banner("ETL PIPELINE COMPLETE")
    print(f"  Output : {OUTPUT_FILE}")
    print(f"  Charts : {REPORTS_DIR}/")
    print(f"  Rows   : {len(validated_df):,}")
    print(f"  Cols   : {validated_df.shape[1]}")
    print()

    return validated_df


def load_cleaned_data() -> pd.DataFrame:
    """
    Load the pre-cleaned dataset from disk (no pipeline re-run).

    Returns
    -------
    pd.DataFrame
        The cleaned dataset. Raises FileNotFoundError if ETL has not been run yet.
    """
    if not OUTPUT_FILE.exists():
        raise FileNotFoundError(
            f"Cleaned data not found: {OUTPUT_FILE}\n"
            "Run run_pipeline() first to generate the cleaned dataset."
        )
    df = pd.read_csv(OUTPUT_FILE)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Run the full pipeline when executed directly:
    #   python src/etl.py
    run_pipeline(kaggle_download=True)
