# ETL Pipeline — Assumptions & Design Decisions

**Task:** Task 1 — Data Preprocessing & ETL Pipeline  
**Module:** Principles of Data Science (7144)  
**Institution:** Coventry University — NIBM Campus  
**Member:** Pasindu Ashan (COMScDS251P-001)  
**Branch:** feature-pasindu-ETL-EDA  

---

## Overview

This document records every design assumption made during the ETL pipeline development for the Online Retail II dataset. Each assumption is documented with its justification, the alternative that was considered, and the reason that alternative was rejected. This ensures the pipeline is transparent, reproducible, and academically defensible.

---

## A1 — Primary Data Source

| Field | Detail |
|-------|--------|
| **Assumption** | `Year 2010-2011.csv` (Kaggle version 3) is the sole and authoritative data source |
| **Justification** | The dataset covers the full 12-month period (Dec 2010 – Dec 2011) required by the assignment. Version 3 is the latest published release with 541,910 records across 8 columns. |
| **Alternative Considered** | Using the `.xlsx` format also available on the repository |
| **Why Rejected** | CSV loads faster, requires no openpyxl dependency for large files, and produces identical data. ISO-8859-1 encoding is explicitly required to preserve special characters in product descriptions. |

---

## A2 — Guest Checkout Exclusion (Null Customer ID)

| Field | Detail |
|-------|--------|
| **Assumption** | Rows with a missing `Customer ID` are treated as anonymous guest checkouts and excluded from the pipeline |
| **Justification** | `Customer ID` is the primary key for Task 3 Customer Segmentation (RFM Clustering). Without it, a transaction cannot be assigned to any customer segment. 135,080 rows (24.93%) have no Customer ID. |
| **Alternative Considered** | KNN imputation or mode-fill using Country + StockCode grouping |
| **Why Rejected** | Customer identity cannot be reliably inferred from transaction metadata. Synthetic IDs would manufacture false customer profiles, introducing bias into clustering centroids and distorting RFM scores. |

---

## A3 — Cancellation Invoice Removal

| Field | Detail |
|-------|--------|
| **Assumption** | Invoices with a prefix of `'C'` (e.g., C536379) are financial reversal entries and must be removed before analysis |
| **Justification** | These entries represent reversed or cancelled transactions. They are not genuine consumer purchase events. Including them inflates negative revenue values and distorts the baseline for clustering and forecasting. 9,288 such rows were identified. |
| **Alternative Considered** | Revenue netting — matching each cancellation to its original invoice and subtracting the amount |
| **Why Rejected** | This requires reliable invoice-level matching which is not guaranteed in the dataset (some originals may be missing or in a different date range). Netting introduces cascading complexity without analytical benefit for the downstream segmentation tasks. |

---

## A4 — Zero-Price and Negative-Price Exclusion

| Field | Detail |
|-------|--------|
| **Assumption** | Records with `Price ≤ 0` are gifts, promotional items, or data entry errors and are excluded |
| **Justification** | Zero-price items carry no monetary value. They make `Total_Revenue = 0` for that line, which distorts RFM Monetary scoring, revenue aggregations, and the IQR fence used in Step 3.5. 2,517 such records were found. |
| **Alternative Considered** | Flag-and-retain — keeping the rows but marking them with a `is_gift` boolean column |
| **Why Rejected** | Retaining zero-price rows would break the homogeneity assumption required for clustering. The additional flag column adds complexity without analytical value for the defined tasks. |

---

## A5 — Negative and Zero Quantity Exclusion

| Field | Detail |
|-------|--------|
| **Assumption** | Records with `Quantity ≤ 0` not already caught by Step 3.1 (cancellation prefix check) are return or adjustment entries and are excluded |
| **Justification** | Negative quantities produce negative `Total_Revenue` values when multiplied by Price. These distort revenue distributions, clustering centroids, and forecasting baselines. 10,624 negative quantity rows were identified in the raw data. |
| **Alternative Considered** | Treating negative quantities as returns and offsetting them against the parent purchase |
| **Why Rejected** | Parent invoice matching is unreliable without a consistent return reference system in the data. Both filters (Qty > 0 AND Price > 0) are applied simultaneously as a single mask for efficiency. |

---

## A6 — IQR Threshold: 1.5× (Tukey Standard Fence)

| Field | Detail |
|-------|--------|
| **Assumption** | The standard Tukey (1977) mild outlier fence of `Q1 − 1.5×IQR` to `Q3 + 1.5×IQR` is appropriate for both `Quantity` and `Price` |
| **Justification** | IQR is robust and distribution-agnostic — important since both columns are heavily right-skewed. The 1.5× multiplier is the universally accepted academic standard for mild outliers. Computed bounds: Quantity valid range [−13, 27] → effective [1, 27]; Price valid range [−2.50, 7.50] → effective [£0.01, £7.50]. |
| **Alternative Considered** | 3×IQR fence (extreme outlier threshold) |
| **Why Rejected** | The 3×IQR fence is too permissive. It retains records with Quantity = 80,995 units and Price = £38,970 per unit — clearly wholesale B2B anomalies that are categorically different from typical consumer transactions and would severely distort RFM-based clustering. |

---

## A7 — Hard Removal vs Capping (Winsorisation)

| Field | Detail |
|-------|--------|
| **Assumption** | Outliers identified by IQR are removed entirely (hard removal) rather than capped at the fence values |
| **Justification** | The extreme values (e.g. Qty = 80,995) represent wholesale B2B orders — a fundamentally different transaction type from the B2C retail behaviour this analysis targets. Capping these at 27 units would still retain a data point that does not represent consumer behaviour, distorting RFM Monetary scores. |
| **Alternative Considered** | Winsorisation — capping outlier values at the IQR fence boundaries |
| **Why Rejected** | Capping changes the value but not the type of transaction. A wholesale order capped at 27 units is still not a consumer purchase. Hard removal produces a homogeneous, model-ready dataset. |

---

## A8 — IQR Applied Independently to Quantity and Price

| Field | Detail |
|-------|--------|
| **Assumption** | IQR outlier detection is applied to `Quantity` and `Price` as separate, independent operations |
| **Justification** | Applying IQR independently targets dimension-specific extremes. A high-Quantity/low-Price combination (e.g., 5,000 units at £0.01) could survive a joint `Total_Revenue` IQR check but both individual values may be statistically anomalous. |
| **Alternative Considered** | Apply IQR only to `Total_Revenue` (derived metric) |
| **Why Rejected** | Applying the fence to `Total_Revenue` only would miss cases where a single dimension is extreme but the product is within normal range. Independent application is more granular and defensible. |

---

## A9 — Description Nulls Retained

| Field | Detail |
|-------|--------|
| **Assumption** | Rows with a missing `Description` (1,454 rows, 0.27%) are retained — only the Description value is missing, not the row |
| **Justification** | `Description` is a qualitative text field that is not used in any quantitative modelling task (clustering uses RFM, forecasting uses temporal aggregations). Dropping 1,454 otherwise valid transaction rows would unnecessarily reduce the dataset. |
| **Alternative Considered** | Fill with StockCode-based lookup (mode of Description per StockCode) |
| **Why Rejected** | Text imputation for an unused feature adds computational complexity with no analytical benefit for the defined downstream tasks. |

---

## A10 — Feature Engineering After Cleaning

| Field | Detail |
|-------|--------|
| **Assumption** | `Total_Revenue` and all temporal features (`Year`, `Month`, `DayOfWeek`, `Hour`) are computed after all cleaning steps are complete |
| **Justification** | Computing `Total_Revenue` before cleaning would embed erroneous values: e.g., negative Quantity × negative Price = positive Revenue, which would survive further cleaning as a false positive. Post-cleaning computation guarantees all derived values are valid. |
| **Alternative Considered** | Compute features before cleaning and use them as additional filtering signals |
| **Why Rejected** | Creates circular logic — a derived metric used to filter the inputs it was derived from. Post-cleaning computation is simpler, deterministic, and produces provably correct values. |

---

## Pipeline Impact Summary

| Step | Before | After | Removed | % of Raw |
|------|--------|-------|---------|----------|
| 3.1 Remove Cancellations | 541,910 | 532,622 | 9,288 | 1.71% |
| 3.2 Remove Duplicates | 532,622 | 527,391 | 5,231 | 0.97% |
| 3.3 Drop Null Customer ID | 527,391 | 392,733 | 134,658 | 24.85% |
| 3.4 Filter Invalid Qty/Price | 392,733 | 392,693 | 40 | 0.01% |
| 3.5 IQR Outlier Removal | 392,693 | 333,234 | 59,459 | 10.97% |
| 3.6 Feature Engineering | 333,234 | 333,234 | 0 | 0.00% |
| **TOTAL** | **541,910** | **333,234** | **208,676** | **38.5%** |

---

## References

- Tukey, J. W. (1977). *Exploratory Data Analysis*. Addison-Wesley.
- Wang, R. Y., & Strong, D. M. (1996). Beyond accuracy: What data quality means to data consumers. *Journal of Management Information Systems*, 12(4), 5–33.
- McKinney, W. (2022). *Python for Data Analysis* (3rd ed.). O'Reilly Media.
