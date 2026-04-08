# Forecasting Assumptions

## Data Assumptions
- `data/cleaned_retail_data.csv` exists and is readable by the script.
- The dataset contains the required columns: `Invoice`, `InvoiceDate`, `Total_Revenue`, `Quantity`, `Price`, `Customer ID`, and `StockCode`.
- The dataset also contains `Country`, because the supervised features use country-level revenue concentration.
- `InvoiceDate` values are parseable by `datetime.fromisoformat`.
- `Total_Revenue`, `Quantity`, and `Price` can be converted to `float`.
- `Invoice` values are reliable enough that counting unique invoice IDs per day is a reasonable proxy for daily order volume.
- Blank-like customer IDs such as empty strings or `nan` should be treated as unidentified customers rather than valid customer records.

## Aggregation Assumptions
- Missing calendar dates between the first and last observed transaction date represent genuine zero-activity days.
- Empty `Customer ID` and `StockCode` values should be ignored when counting unique customers and unique stock codes.
- `avg_unit_price` can be approximated as the average of row-level `Price` values for a day, computed as total `Price` divided by transaction line count.
- Revenue shares by time block, product mix, and geography are meaningful daily summary signals for short-term demand.
- The configured `LARGE_ORDER_THRESHOLD = 200.0` is a reasonable cutoff for separating unusually large invoices from normal orders.

## Modeling Assumptions
- A 7-day repeating pattern is a valid baseline for short-term demand.
- Recent lagged behavior and rolling averages contain enough signal to support the supervised forecast.
- Calendar effects such as weekday, month position, and month-end behavior are relevant to daily demand.
- Rolling volatility and cyclic calendar encodings add useful information beyond raw lag values.
- Forecasting supporting series recursively is an acceptable way to avoid future-feature leakage.
- Negative model outputs are not meaningful for these business targets and should be clipped to zero.
- Ratio-like support series such as revenue shares and customer mix should remain inside realistic bounds when recursively forecast.

## History And Evaluation Assumptions
- There is enough history to support the configured lag structure and holdout window.
- In practice, the workflow needs more than `58` daily observations because `MIN_HISTORY = 28` and `TEST_DAYS = 30`.
- The most recent 30 days are representative enough to serve as the holdout window.
- `RMSE` is the primary selection metric, with `MAE` used as a tie-breaker.
- `WAPE` is suitable because the aggregated series may contain zero-activity days.

## Operational Assumptions
- A 30-day horizon is the intended planning horizon for this workflow.
- The best model may differ by target, so revenue and order volume should be selected independently.
- Forecast outputs are meant for short-term operational guidance rather than long-range strategic planning.
