# Forecasting Methodology

## Purpose
`src/forecasting.py` builds a short-term daily forecasting workflow for two targets:

- `daily_revenue`
- `daily_order_volume`

The script compares a simple baseline against a supervised model, selects the lower-error option for each target, and produces a 30-day forward forecast.

## Input Data And Daily Feature Construction
The workflow reads `data/cleaned_retail_data.csv` and requires these columns:

- `Invoice`
- `InvoiceDate`
- `Total_Revenue`
- `Quantity`
- `Price`
- `Customer ID`
- `StockCode`
- `Country`

The script parses `InvoiceDate`, aggregates transaction rows to one record per calendar day, and then fills every date between the first and last observed day. Dates with no transactions are kept as zero-activity days.

Each daily record contains:

- `daily_revenue`: sum of `Total_Revenue`
- `daily_order_volume`: count of unique invoices
- `line_count`: number of transaction rows
- `daily_quantity`: sum of `Quantity`
- `unique_customers`: count of non-empty customer IDs
- `unique_stockcodes`: count of non-empty stock codes
- `avg_unit_price`: total `Price` divided by `line_count` with zero-safe handling
- `guest_checkout_ratio`: share of daily lines without a valid customer ID
- `active_hours_count`: number of active invoice hours in the day
- `revenue_per_line`: daily revenue divided by line count
- `quantity_per_order`: daily quantity divided by order volume
- `median_order_value`: median invoice revenue for the day
- `max_order_value`: largest invoice revenue for the day
- `large_order_count`: number of invoices above the configured large-order threshold
- `top5_product_revenue_share`: share of revenue captured by the top five stock codes
- `product_herfindahl`: concentration of product revenue within the day
- `uk_revenue_share`: share of revenue from `United Kingdom`
- `international_country_count`: number of non-UK countries active that day
- `top_country_concentration`: share of revenue contributed by the leading country
- `new_customer_count`: count of first-seen customers on that day
- `returning_customer_ratio`: share of known customers among identified customers
- `morning_revenue_share`: share of daily revenue booked between 06:00 and 11:59
- `afternoon_revenue_share`: share of daily revenue booked between 12:00 and 17:59

## Forecasting Models
The script evaluates two models:

- `Seasonal Naive`: repeats the latest 7-day pattern
- `Gradient Boosting`: uses `HistGradientBoostingRegressor`

The supervised path models not only the two targets but also these supporting series:

- `line_count`
- `daily_quantity`
- `unique_customers`
- `unique_stockcodes`
- `avg_unit_price`
- `guest_checkout_ratio`
- `active_hours_count`
- `revenue_per_line`
- `quantity_per_order`
- `median_order_value`
- `max_order_value`
- `large_order_count`
- `top5_product_revenue_share`
- `product_herfindahl`
- `uk_revenue_share`
- `international_country_count`
- `top_country_concentration`
- `new_customer_count`
- `returning_customer_ratio`
- `morning_revenue_share`
- `afternoon_revenue_share`

This allows the future feature rows to be built from recursively forecast supporting signals instead of from unknown future values.

## Feature Engineering
The gradient-boosting feature row contains calendar features:

- day of week
- day of month
- day of year
- ISO week of year
- month
- quarter
- weekend flag
- month-start flag
- month-end flag
- sine and cosine encodings for weekday
- sine and cosine encodings for month

It also contains lag and rolling-window features for every modeled series:

- lags at `1`, `7`, `14`, and `28` days
- rolling means over `7` and `28` days
- rolling standard deviations over `7` and `28` days

An additional derived series, `avg_order_value`, is computed as `daily_revenue / daily_order_volume` with zero-safe division and included through the same lag and rolling-window pattern.

## Training, Validation, And Selection
The workflow uses a chronological holdout:

- training period: all daily observations except the final 30 days
- test period: the final 30 days

The baseline model forecasts the test window by repeating the latest weekly pattern from the training period. The supervised model is trained on expanding historical windows and forecasts recursively for the same 30-day holdout horizon.

For each target, the script computes:

- `MAE`
- `RMSE`
- `WAPE`

Model selection is target-specific:

- choose the model with the lowest `RMSE`
- break ties with lower `MAE`

The selected model for each target is then used to create the next 30 days of dated forecast rows.

## Workflow Outputs
`run_forecasting_workflow()` returns a dictionary containing:

- source row count
- aggregated daily records
- target series
- per-model holdout metrics
- selected future forecast rows
- summary rows
- comparison payloads for plotting
- a 7-row preview per target

When `main()` runs, the script also:

- prints the forecasting window summary
- prints the model evaluation table
- prints the 30-day forecast summary table
- prints the first 7 forecasted days per target
- saves three PNG files under `outputs/forecasting`

The saved plots are:

- `model_metric_comparison.png`
- `holdout_comparison.png`
- `future_forecast_comparison.png`
