# EDA Assumptions

**Module:** Principles of Data Science (7144)
**Section:** Exploratory Data Analysis
**Author:** Anuradha Dhananjanee

---

## 1. Data Source Assumptions

- The cleaned dataset (`cleaned_retail_data.csv`) produced by the ETL pipeline (01_etl.ipynb) is used as the sole input for EDA.
- All ETL transformations (duplicate removal, cancellation filtering, outlier handling) are assumed to have been correctly applied before EDA begins.
- The dataset represents a UK-based online retail store selling gift and homeware items between December 2010 and December 2011.

---

## 2. Data Quality Assumptions

- After ETL cleaning, all remaining rows are assumed to be valid transactions with positive Quantity and Price values.
- Customer IDs are assumed to uniquely identify individual customers (no shared accounts).
- The `Total_Revenue` column is assumed to be correctly calculated as `Quantity × Price` during ETL.
- Missing Customer IDs were removed during ETL — no further imputation is performed in EDA.

---

## 3. Time-Based Column Assumptions

- `Month`, `Hour`, `DayOfWeek`, and `Week` columns are derived from `InvoiceDate` at the start of EDA.
- All timestamps are assumed to be in UTC / UK local time — no timezone conversion is applied.
- Business hours are assumed to be 6:00–20:00 based on observed transaction hour distribution.

---

## 4. Statistical Assumptions

- Descriptive statistics (mean, std, skewness, kurtosis) are computed on the cleaned dataset without further transformation.
- The IQR outlier method applied during ETL is assumed sufficient — no additional outlier removal is performed in EDA.
- Anomaly detection uses the Mean + 3σ threshold, which assumes approximately normal distribution of `Total_Revenue` after cleaning.
- Pearson correlation is used for correlation analysis, assuming linear relationships between numerical variables.

---

## 5. RFM Analysis Assumptions

- Reference date for Recency calculation is set to one day after the last recorded transaction date.
- Frequency is measured as the number of unique invoices per customer (not individual line items).
- Monetary value is the total revenue summed across all transactions per customer.
- RFM analysis in EDA is exploratory only — formal RFM scoring is handled in the Clustering notebook (03_clustering.ipynb).

---

## 6. Geographic Assumptions

- Country names are used as-is from the dataset — no standardisation or mapping is applied.
- "United Kingdom" is treated as the domestic market; all other countries are treated as international markets.
- Revenue comparisons across countries do not account for currency differences (all values assumed in GBP).

---

## 7. Visualisation Assumptions

- All charts use a 5000-row random sample (random_state=42) only for scatter plots, to improve rendering speed. All other charts use the full dataset.
- Charts are saved to the `reports/` folder which is auto-created if it does not exist.
- Colour choices are for visual clarity only and carry no statistical meaning.
