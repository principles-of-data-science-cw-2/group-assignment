# ETL Pipeline — Challenges & Solutions

**Task:** Task 1 — Data Preprocessing & ETL Pipeline  
**Module:** Principles of Data Science (7144)  
**Institution:** Coventry University — NIBM Campus  
**Member:** Pasindu Ashan (COMScDS251P-001)  
**Branch:** feature-pasindu-ETL-EDA  

---

## Overview

This document records every significant challenge encountered during the development of the ETL pipeline, the root cause of each, the solution implemented, and the lesson learned. These are genuine issues encountered during development, documented for transparency and future reproducibility.

---

## Challenge 1 — Dataset Download: Kaggle API Key Setup

| Field | Detail |
|-------|--------|
| **Challenge** | The initial plan was to use the official `kaggle` library, which requires a `kaggle.json` API token placed in `~/.kaggle/`. Setting this up manually on Windows caused `FileNotFoundError` and path resolution issues. |
| **Root Cause** | Windows path handling differs from the Unix paths assumed by the kaggle CLI. The `.kaggle` directory was not being found at `C:\Users\User\.kaggle\kaggle.json`. |
| **Solution** | Switched to `kagglehub` library which handles authentication transparently via browser-based OAuth on first use, caching credentials automatically. The download path is returned as a string and the CSV file is located by walking the directory tree. |
| **Code Fix** | `download_path = kagglehub.dataset_download('mathchi/online-retail-ii-data-set-from-ml-repository')` |
| **Lesson Learned** | `kagglehub` is significantly simpler for notebook-based workflows than the official `kaggle` CLI. It avoids manual credential setup and works cross-platform. |

---

## Challenge 2 — Encoding Error on CSV Load

| Field | Detail |
|-------|--------|
| **Challenge** | First attempt to load the CSV using default `pd.read_csv()` (UTF-8 encoding) raised `UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe9 in position 12` |
| **Root Cause** | The dataset contains product descriptions with special characters (e.g., accented letters: `Crème`, `café`) that are encoded in ISO-8859-1 (Latin-1), not UTF-8. The file was originally generated from a legacy UK retail system. |
| **Solution** | Explicitly specified `encoding='ISO-8859-1'` in the `pd.read_csv()` call. |
| **Code Fix** | `df = pd.read_csv(csv_file, encoding='ISO-8859-1')` |
| **Lesson Learned** | Always inspect the encoding of CSV files from legacy business systems. UK/European retail datasets frequently use ISO-8859-1 due to historical Windows code page usage. |

---

## Challenge 3 — Column Name Inconsistency Between Dataset Versions

| Field | Detail |
|-------|--------|
| **Challenge** | The Kaggle dataset uses `Invoice` as the column name, but earlier versions of the same dataset (UCI repository) use `InvoiceNo`. Similarly, `Customer ID` (with space) vs `CustomerID` (no space). Running the notebook on a different version caused `KeyError: 'InvoiceNo'`. |
| **Root Cause** | The Kaggle host (mathchi) reformatted the column names when uploading the UCI dataset to Kaggle. Different team members working from different download sources would encounter different column names. |
| **Solution** | Implemented auto-detection logic: `invoice_col = 'Invoice' if 'Invoice' in df.columns else 'InvoiceNo'` and similarly for `Customer ID`. All downstream references use these variables instead of hardcoded strings. |
| **Lesson Learned** | Never hardcode column names when working with public datasets that may have multiple version histories. Build auto-detection from the first cell. |

---

## Challenge 4 — Hardcoded File Paths Broke on Other Machines

| Field | Detail |
|-------|--------|
| **Challenge** | Early notebook versions used an absolute Windows path `C:\Users\User\Documents\...` for both the raw data file and the output CSV. When the notebook was cloned and run by another team member, all file I/O cells failed with `FileNotFoundError`. |
| **Root Cause** | Absolute paths are machine-specific and cannot be shared across different operating systems or user accounts. |
| **Solution** | Switched to relative paths (`../data/Year 2010-2011.csv` and `../data/cleaned_retail_data.csv`) anchored to the `notebooks/` directory. Added `os.makedirs('../data', exist_ok=True)` to create the output directory if it does not exist. |
| **Lesson Learned** | Always use relative paths in collaborative notebooks. Document the expected directory structure in `README.md` so all team members can reproduce the environment. |

---

## Challenge 5 — Cancellation Prefix Check Missed Some Returns

| Field | Detail |
|-------|--------|
| **Challenge** | After Step 3.1 removed invoices starting with `'C'`, Step 3.4 still found 10,624 rows with negative `Quantity`. These were return/credit entries that did not use the cancellation invoice prefix. |
| **Root Cause** | Not all return transactions follow the `'C'` prefix convention. Some entries appear to be manual adjustments or system corrections logged without the standard prefix. |
| **Solution** | Step 3.4 independently filters `Quantity > 0` as a separate boolean mask, acting as a safety net for returns that were not caught by the prefix check. The two steps are complementary, not redundant. |
| **Lesson Learned** | Data cleaning steps should be layered defensively. Never assume a single rule will capture all variations of the same data quality issue. Document each step's independent rationale. |

---

## Challenge 6 — IQR Removed Unexpectedly Large Fraction of Data

| Field | Detail |
|-------|--------|
| **Challenge** | Step 3.5 (IQR outlier removal) removed 59,459 rows — 15.1% of the post-Step-3.4 dataset. This was significantly larger than initially expected and raised concern about over-filtering. |
| **Root Cause** | The `Quantity` distribution is extremely right-skewed. A small number of wholesale B2B orders (e.g., Qty = 80,995 units; Qty = 12,000 units) pulled Q3 and IQR far right, setting a relatively low upper fence of 27 units. |
| **Solution** | After inspection, the removed records were verified to be genuine B2B wholesale orders (consistent product descriptions: "ASSORTED" in bulk, pallets, cases). The 1.5×IQR fence was confirmed as correct. A boxplot before/after comparison was added to the notebook for visual verification. |
| **Decision** | Accepted the 59,459 removals as correct. The analysis targets B2C consumer behaviour, and the fence appropriately separates the two transaction types. Documented in `assumptions.md` as Assumption A7. |
| **Lesson Learned** | Always visualise the distribution before and after outlier removal. A large removal count is not necessarily wrong — it may reflect a genuine structural split in the data (B2B vs B2C). |

---

## Challenge 7 — Feature Engineering: InvoiceDate Type Conversion

| Field | Detail |
|-------|--------|
| **Challenge** | After loading the CSV, `InvoiceDate` was stored as `object` (string). Attempting to extract `.dt.year` on a string column raised `AttributeError: Can only use .dt accessor with datetimelike values`. |
| **Root Cause** | `pd.read_csv()` does not automatically parse date columns unless `parse_dates` is specified. The date format in this file (`12/1/2010 8:26`) is not standard ISO 8601 and would require format specification. |
| **Solution** | Applied `pd.to_datetime(df['InvoiceDate'])` in Step 3.6 after all cleaning steps. `pandas` correctly infers the `MM/DD/YYYY HH:MM` format automatically. All temporal features (`Year`, `Month`, `DayOfWeek`, `Hour`) are then extracted using the `.dt` accessor. |
| **Lesson Learned** | Always cast date columns explicitly using `pd.to_datetime()` rather than relying on `parse_dates` in `read_csv`, which can fail silently or produce incorrect results for non-ISO formats. |

---

## Challenge 8 — Post-Load Validation Catches Silent Errors

| Field | Detail |
|-------|--------|
| **Challenge** | During development, an ordering bug caused feature engineering to run before the Customer ID cast, resulting in the column having incorrect dtype. The error was not visible until downstream tasks failed. |
| **Root Cause** | Notebook cells were run out of order during iterative development. The pipeline state was inconsistent. |
| **Solution** | Added a formal post-load validation block (Step 4) that reloads the exported CSV from disk and checks: row count, column count, null values, negative revenue, unique customer count, and date range. Any deviation surfaces immediately as a printed warning. |
| **Lesson Learned** | Always implement a post-load validation step that treats the output file as an independent artefact. Reloading from disk and re-checking is the only way to guarantee the saved file matches the in-memory state. |

---

## Challenge 9 — Git: Large CSV File Rejected by GitHub

| Field | Detail |
|-------|--------|
| **Challenge** | Attempting to commit `Year 2010-2011.csv` (44 MB) and `cleaned_retail_data.csv` (32 MB) to the repository caused `git push` to fail with `error: File exceeds GitHub's file size limit of 100 MB` warning and slow operations. |
| **Root Cause** | GitHub enforces a 100 MB hard limit per file and recommends keeping files under 50 MB. Large data files should never be tracked in Git. |
| **Solution** | Added `*.csv` and `*.xlsx` to `.gitignore`. The raw data file must be downloaded independently using the `kagglehub` code in `01_etl.ipynb`. The cleaned output file is regenerated by running the notebook. `README.md` was updated with setup instructions. |
| **Lesson Learned** | Data files must never be committed to Git repositories. The notebook code itself is the reproducible artefact — running it from scratch should regenerate all data outputs. |

---

## Challenge 10 — Branch Naming Conflict

| Field | Detail |
|-------|--------|
| **Challenge** | The initial branch was named `feature-pasindu-EDA` but the scope also included ETL. This caused confusion when merging with other team members' EDA branch (`feature-anuradha-EDA`). |
| **Root Cause** | Branch names were not agreed upon as a team before development began. |
| **Solution** | Renamed the branch to `feature-pasindu-ETL-EDA` to clearly reflect the scope. Updated all documentation and the group presentation to reflect the correct branch name. |
| **Lesson Learned** | Agree on a branch naming convention (`feature-{member}-{task}`) before any coding begins. This prevents merge conflicts and makes the Git history readable for all team members. |

---

## Summary Table

| # | Challenge | Category | Impact | Status |
|---|-----------|----------|--------|--------|
| 1 | Kaggle API key setup | Environment | High | ✅ Resolved — switched to kagglehub |
| 2 | CSV encoding error | Data | High | ✅ Resolved — ISO-8859-1 specified |
| 3 | Column name inconsistency | Data | Medium | ✅ Resolved — auto-detection logic |
| 4 | Hardcoded file paths | Collaboration | High | ✅ Resolved — relative paths |
| 5 | Returns not caught by prefix filter | Data | Medium | ✅ Resolved — layered filter in Step 3.4 |
| 6 | IQR removed large fraction | Methodology | Medium | ✅ Accepted — B2B/B2C structural split |
| 7 | Date column type conversion | Code | Low | ✅ Resolved — pd.to_datetime() |
| 8 | Silent errors in pipeline | Quality | High | ✅ Resolved — post-load validation |
| 9 | Large CSV file rejected by GitHub | Git | High | ✅ Resolved — .gitignore + README |
| 10 | Branch naming conflict | Collaboration | Low | ✅ Resolved — renamed branch |

---

## References

- Kaggle Hub documentation: https://github.com/Kaggle/kagglehub
- GitHub file size limits: https://docs.github.com/en/repositories/working-with-files/managing-large-files
- Pandas datetime documentation: https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html
