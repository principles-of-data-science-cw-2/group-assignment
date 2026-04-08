# Forecasting Challenges And Operational Impact

## Key Challenges

### 1. Strict input requirements
The script depends on a specific schema and parsing behavior. Missing columns such as `Country`, non-numeric values, or non-ISO-style timestamps will fail early and stop the workflow.

### 2. Zero-day interpretation
The aggregation step fills every calendar date between the first and last observed day. That is useful for modeling, but it assumes every missing date represents true zero activity rather than missing source data.

### 3. Limited history requirement
The supervised model needs enough history for both lag features and evaluation. With `LAG_STEPS = [1, 7, 14, 28]` and a 30-day holdout, short datasets cannot be forecast by this workflow.

### 4. Recursive forecast error
`Gradient Boosting` predicts future supporting series and then reuses those predictions to build later feature rows. That avoids leakage, but it also means forecast errors can compound across the 30-day horizon, especially now that the recursive path includes richer customer, geography, basket-value, and product-mix signals.

### 5. No external drivers
The model only uses transaction-derived fields and calendar features. It does not know about promotions, holidays, price changes outside the recorded data, supply issues, or other business events.

### 6. Daily aggregation tradeoff
Aggregating to the daily level reduces transaction noise, but it also removes intraday timing, item mix detail, and within-day operational patterns that could matter for some use cases.

### 7. Feature richness versus stability
The richer feature set gives the supervised model more context, but some engineered ratios can become noisy on low-activity days. The script clips bounded recursive outputs such as shares and ratios, but unstable low-volume behavior can still reduce forecast quality.

### 8. Per-target model differences
The workflow selects the best model separately for each target. That is useful operationally, but it means the final forecasting stack can be mixed rather than using one consistent model everywhere.

## Mitigations In The Script
- The workflow includes a transparent `Seasonal Naive` baseline instead of relying only on a complex model.
- The supervised feature set combines richer transaction-derived features with calendar features, lag features, rolling means, and rolling standard deviations across several daily support series.
- Recursive prediction of support series is used to avoid future-feature leakage.
- Negative predictions are clipped to zero before they enter the forecast outputs.
- Ratio-like features such as revenue shares and customer mix are clipped to realistic bounds during recursive forecasting.
- Holdout evaluation is chronological and target-specific.

## Operational Impact
- Better `daily_order_volume` forecasts can support staffing and workload planning.
- Better `daily_revenue` forecasts can support short-term purchasing and cash planning.
- The 7-row preview output makes it easy to inspect the immediate forecast horizon before using the full 30-day summary.
- The saved comparison charts provide a quick way to compare actuals, holdout predictions, and future projections.

## Remaining Risks
- Forecast quality can degrade quickly when recent behavior changes abruptly.
- Repeated weekly patterns may underreact to sudden shifts.
- Recursive supervised forecasts may drift as the horizon extends.
- The script does not explain why a given forecast changed beyond the engineered feature set and error metrics it reports.
