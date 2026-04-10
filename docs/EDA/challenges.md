# EDA Challenges & Solutions

**Module:** Principles of Data Science (7144)
**Section:** Exploratory Data Analysis
**Author:** Anuradha Dhananjanee

---

## 1. Hardcoded File Path

**Challenge:**
The original EDA notebook used an absolute file path (`C:\Users\User\Documents\...`) which only worked on the original developer's machine. Running it on any other team member's computer caused an immediate `FileNotFoundError`.

**Solution:**
Replaced the hardcoded path with a dynamic relative path using `os.path.dirname(os.getcwd())` to navigate from the `notebooks/` folder up to the project root, then into `data/`. This ensures the notebook runs correctly on any machine regardless of where the project is saved.

---

## 2. Missing Time-Based Columns

**Challenge:**
Several sections of the notebook referenced columns `DayOfWeek`, `Hour`, and `Month` which did not exist in the cleaned CSV file. This caused `KeyError` crashes in Sections 2D and 3B, preventing the notebook from running fully.

**Solution:**
Added column extraction from `InvoiceDate` at the top of Section 1 using pandas `.dt` accessor:
- `df['Month'] = df['InvoiceDate'].dt.month`
- `df['Hour'] = df['InvoiceDate'].dt.hour`
- `df['DayOfWeek'] = df['InvoiceDate'].dt.day_name()`

This ensures all derived time columns are available before any section references them.

---

## 3. Invoice Column Name Inconsistency

**Challenge:**
The raw dataset uses `InvoiceNo` as the column name, but after ETL cleaning the column was renamed to `Invoice`. The EDA notebook inconsistently checked for both names in different cells, making the code messy and error-prone.

**Solution:**
Detected the correct column name once in Section 1 and stored it in a variable (`invoice_col`). All subsequent sections reference `invoice_col` instead of hardcoding the column name, making the code cleaner and more robust.

---

## 4. Large Dataset Performance

**Challenge:**
The dataset contains over 333,000 rows. Rendering scatter plots with all rows caused slow rendering and dense, unreadable visualisations.

**Solution:**
Used `df.sample(5000, random_state=42)` for scatter plots only. All other charts (bar charts, histograms, line charts) use the full dataset to ensure statistical accuracy. `random_state=42` ensures reproducibility.

---

## 5. UK Revenue Dominance Skewing Visualisations

**Challenge:**
The United Kingdom accounts for the vast majority of revenue (~85%+). When plotting revenue by country, the UK bar completely dwarfed all other countries, making international market comparison impossible to read.

**Solution:**
Created two separate charts in Section 6 — one showing all countries including UK, and a second chart excluding the UK to make international market differences clearly visible and comparable.

---

## 6. Reports Folder Not in Repository

**Challenge:**
The notebook saves charts to a `reports/` folder, but this folder was not included in the repository structure. Running the notebook on a freshly cloned repo caused a `FileNotFoundError` when trying to save chart images.

**Solution:**
Added `os.makedirs(REPORTS_FOLDER, exist_ok=True)` in Section 1. This automatically creates the `reports/` folder if it does not exist, without raising an error if it already does.

---

## 7. Trend Visibility in Monthly Revenue Chart

**Challenge:**
The monthly revenue line chart showed high volatility month-to-month, making it difficult to clearly identify the overall upward trend leading into Q4.

**Solution:**
Added a 3-month rolling average line (`rolling(3).mean()`) overlaid on the monthly revenue chart. This smooths short-term fluctuations and makes the seasonal trend more visually apparent.
