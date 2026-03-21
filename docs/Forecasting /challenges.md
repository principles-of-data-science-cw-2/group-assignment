# Forecasting Challenges And Operational Impact

## Key Challenges

### 1. Limited historical window
The cleaned dataset spans just over one year. That is enough for short-term forecasting, but not enough to model richer annual seasonality with confidence.

### 2. Calendar irregularities
The business does not trade every day. In this dataset, Saturday activity is consistently absent, so the model has to distinguish between a real zero-demand pattern and missing data.

### 3. No external explanatory variables
The dataset contains transactions only. It does not include promotions, holidays, marketing campaigns, supplier issues, or customer service events that could explain sudden demand changes.

### 4. Retail volatility
Daily retail demand can spike or dip sharply because of bulk orders, seasonal shopping periods, or country-level demand shifts. This increases forecast error risk, especially on peak days.

### 5. Model complexity does not guarantee better accuracy
Testing showed that Holt-Winters underperformed the simpler weekly seasonal-naive model on both daily revenue and daily order volume. With only about one year of history, a more complex model can overfit the noise instead of improving the forecast.

## Mitigations Used
- Daily aggregation smooths line-item noise while keeping the forecast operationally useful.
- Weekly seasonality is modeled explicitly because it is visible in the trading pattern.
- A seasonal-naive baseline is included so the final model can be judged against a simple and transparent benchmark.
- The lower-error model is used for forward forecasts instead of assuming the more complex method is automatically better.
- A 30-day holdout window is used to evaluate realistic near-term performance.

## Potential Operational Impact
- Better order-volume forecasts can improve staffing plans for picking, packing, and customer support.
- Better revenue forecasts can improve purchasing cadence and short-term cash-flow planning.
- Repeating low-demand days can be used to schedule maintenance, training, or lower-priority administrative work.
- Forecasted peaks can trigger earlier replenishment or temporary capacity increases.

## Remaining Risks
- Forecasts may underperform during promotions, holidays, or structural business changes because those drivers are not modeled directly.
- If the cleaned dataset later changes its treatment of cancellations or returns, forecast behavior may shift as well.
- Long-horizon forecasts should be treated cautiously because the available history is relatively short.
