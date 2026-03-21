# Forecasting Assumptions

## Data Assumptions
- `data/cleaned_retail_data.csv` is the approved post-ETL dataset and is accurate enough to support forecasting analysis.
- `InvoiceDate` values are valid timestamps and can be safely aggregated to the daily level without timezone adjustments.
- `Total_Revenue` correctly reflects cleaned transaction value and can be summed as the revenue target.
- The cleaned dataset has already handled major data-quality issues such as invalid prices, negative quantities, and duplicate problem records.

## Time-Series Assumptions
- Missing calendar dates in the aggregated series represent genuine zero-activity days rather than missing source data.
- The recurring no-trade Saturday pattern is treated as part of the business’s real operating cycle.
- Recent weekly behavior is informative enough for short-term forecasting, even though the dataset covers only a little more than one year.
- A 30-day forecast horizon is appropriate for tactical planning, but not for long-range strategic budgeting.

## Modeling Assumptions
- Daily revenue and daily order volume are the most useful targets for operational forecasting in this project.
- Order volume is represented by the number of unique invoices per day.
- A simple weekly seasonal pattern can explain more of the short-term signal than a highly complex model with limited history.
- No external regressors are available, so promotions, holidays, supply disruptions, or marketing campaigns are not modeled directly.

## Evaluation Assumptions
- The last 30 days provide a reasonable holdout window for judging short-term forecast quality.
- `MAE` and `RMSE` are appropriate because both forecasting targets are continuous numerical values.
- `WAPE` is preferred over `MAPE` because zero-activity days are intentionally retained in the series and would make per-row percentage error unstable.
- Lower holdout error is the main criterion for selecting the model used for the final forward forecast.

## Business Assumptions
- The forecast is intended to support staffing, replenishment, and short-term revenue planning rather than precise day-level financial commitments.
- Forecast outputs should be interpreted as directional decision support, not guaranteed future results.
- If the underlying business process changes materially, the current forecasting assumptions should be revisited before reusing the model.
