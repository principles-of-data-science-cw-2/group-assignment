# Forecasting Methodology

## Objective
The forecasting task predicts short-term operational trends from the cleaned retail transaction dataset. Two daily targets are modeled because they support both financial and operational decisions:

- `daily_revenue`: total sales revenue per day.
- `daily_order_volume`: number of unique invoices processed per day.

## Data Preparation
The model uses `data/cleaned_retail_data.csv`, which already contains cleaned transaction records and a derived `Total_Revenue` field. For forecasting:

1. `InvoiceDate` is parsed as a timestamp.
2. Transactions are aggregated to a daily grain.
3. Missing calendar days are retained and filled with zero activity.

Keeping zero-activity days matters because the dataset shows a recurring no-trade Saturday pattern. Removing those dates would hide a real operational rhythm and weaken weekly seasonality detection.

## Modeling Approach
The script in `src/forecasting.py` evaluates two time-series forecasting approaches:

- `Seasonal Naive`: repeats the most recent 7-day demand pattern
- `Holt-Winters`: additive trend plus additive weekly seasonality with a 7-day period

This approach was selected because:

- the dataset covers about one year, which is too short for reliable multi-year seasonal modeling;
- the series is continuous enough at the daily level to learn short-horizon weekly demand patterns;
- the business question is operational, so interpretable short-term forecasts are more useful than a highly complex model.

The script scores both models on a holdout window and uses the lower-RMSE model for the forward 30-day forecast. On this dataset, the simpler weekly seasonal-naive model performed better than Holt-Winters for both targets, so it is the recommended operational forecast.

## Validation Strategy
Time-series data cannot be shuffled without leaking future information into the past. The validation design therefore uses a chronological split:

- training set: all observations except the final 30 days
- test set: the most recent 30 days

The model is fit on the training period, forecasts the next 30 days, and is then scored against the held-out test window.

## Evaluation Metrics
The forecasting script reports:

- `MAE`: average absolute forecast error in the original unit
- `RMSE`: penalizes larger misses more heavily than MAE
- `WAPE`: expresses total absolute error as a percentage of total actual volume and remains stable when some days have zero activity

`Accuracy` is not the primary metric here because the targets are continuous regression outputs rather than class labels. `MAPE` was not retained because the project intentionally keeps zero-activity days in the series, which makes per-row percentage error unstable and potentially misleading.

## Observed Performance
Holdout evaluation on the last 30 days produced these results:

- `daily_revenue` Seasonal Naive: MAE `5822.61`, RMSE `8048.46`, WAPE `27.42%`
- `daily_revenue` Holt-Winters: MAE `13193.88`, RMSE `15816.71`, WAPE `62.14%`
- `daily_order_volume` Seasonal Naive: MAE `14.70`, RMSE `19.06`, WAPE `17.84%`
- `daily_order_volume` Holt-Winters: MAE `56.81`, RMSE `67.30`, WAPE `68.94%`

These results indicate that the recent weekly trading pattern is a stronger short-term predictor than a trend-plus-seasonality model trained on the available history.

## Operational Use
These forecasts can support:

- staffing plans by estimating daily order-processing pressure
- inventory and replenishment timing by anticipating short-term sales movement
- cash-flow and promotion planning by projecting near-term revenue patterns

The 30-day horizon is intentionally conservative because it is more actionable for operational teams and more defensible given the available history.
