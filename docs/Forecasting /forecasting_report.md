# Forecasting Approach Report

## 1. Overview

Our forecasting pipeline predicts short-term retail demand over a **30-day horizon** for two primary targets — **daily revenue** and **daily order volume**. We evaluate two competing models on a **30-day holdout window** and automatically select the better performer (by RMSE, with MAE as a tiebreaker) for generating the final forward-looking forecast.

---

## 2. Models Used

### 2.1 Seasonal Naive (Baseline)

The **Seasonal Naive** model is the simplest form of time-series forecasting. It assumes that the future will repeat the most recently observed seasonal cycle exactly. In our implementation:

- We take the **last 7 days** (one weekly cycle) of training data.
- Those 7 values are tiled forward to fill the entire forecast horizon.
- Predictions are clipped at zero to prevent negative forecasts.

This model captures no trends, no feature interactions, and no external drivers — it simply replays the last week on repeat. It serves as a **minimum-performance benchmark**: any model we deploy should comfortably beat this baseline, otherwise the added complexity is not justified.

### 2.2 Gradient Boosting (Supervised Model)

The **Gradient Boosting** model (`HistGradientBoostingRegressor` from scikit-learn) is a tree-ensemble method that learns non-linear relationships from engineered features. Our implementation uses:

| Feature Category | Details |
|---|---|
| **Calendar features** | Day of week, day of month, month, quarter, weekend flag, month-start/end flags |
| **Lag features** | Values at *t−1, t−7, t−14, t−28* for each of 8 modelled series |
| **Rolling statistics** | 7-day and 28-day rolling means for each series |

Key characteristics:
- **Recursive multi-step forecasting**: each predicted day is appended to the history and used to compute features for the next day, avoiding exogenous data leakage.
- **Multi-output**: separate regressors are fitted for all 7 modelled series (revenue, order volume, line count, quantity, unique customers, unique stock codes, average unit price), ensuring that supporting features remain internally consistent during the forecast loop.
- Hyperparameters are conservatively tuned (`learning_rate=0.05`, `max_depth=4`, `max_iter=400`, `min_samples_leaf=10`) to guard against overfitting.

---

## 3. Why Should a Model Outperform the Naive Baseline?

A Seasonal Naive forecast ignores **all information except the most recent weekly pattern**. In practice, real-world retail demand is influenced by:

- **Trends** — revenue may be growing or declining over weeks and months.
- **Day-specific effects** — e.g., month-end surges, weekend dips.
- **Cross-series signals** — a spike in unique customers often precedes a revenue uptick.
- **Momentum and volatility** — rolling averages smooth out noise that a naive repeat amplifies.

A well-engineered supervised model can exploit these signals. If it *cannot* beat the naive baseline, it usually means one of two things: the series is inherently dominated by a fixed weekly cycle (low signal-to-noise), or the feature engineering and model complexity are not well matched to the target.

---

## 4. Our Key Observation: Mixed Results Across Targets

> [!IMPORTANT]
> Gradient Boosting **outperforms** Seasonal Naive on **daily revenue** but **underperforms** on **daily order volume**.

### Why Gradient Boosting wins on daily revenue

- Revenue has higher variance and is influenced by order sizes, product mix, and pricing — all of which the lag and rolling features capture well.
- Calendar effects (month-end, weekday vs. weekend) create exploitable patterns beyond a simple weekly repeat.
- The model's ability to learn non-linear interactions (e.g., high average unit price + high customer count → revenue spike) gives it a clear advantage.

### Why Gradient Boosting struggles on daily order volume

- Order volume tends to follow a **stable, highly periodic weekly cycle** — customers place a similar number of orders each Monday, Tuesday, etc.
- With low variance and strong weekly periodicity, the Seasonal Naive baseline is already a near-optimal forecast, leaving very little room for improvement.
- The Gradient Boosting model introduces additional noise through recursive prediction errors that compound day-over-day. In a low-variance series, even small compounding errors degrade performance below the naive floor.

---

## 5. Other Findings

| Finding | Detail |
|---|---|
| **Model selection is target-specific** | The pipeline correctly selects the best model *per target*, so we may deploy Gradient Boosting for revenue and Seasonal Naive for order volume simultaneously. |
| **Recursive forecasting introduces error accumulation** | Because each predicted value feeds into the next step's features, early errors in supporting series (e.g., `unique_customers`) can cascade into later target predictions. |
| **Feature richness** | We engineer **8 modelled series × (4 lags + 2 rolling means) + 7 calendar features = 55 features**, giving the Gradient Boosting model a rich but potentially noisy input space. |
| **Conservative hyperparameters** | The chosen settings (`max_depth=4`, `min_samples_leaf=10`) limit model capacity, which helps prevent overfitting on our relatively short daily history. |
| **Zero-clipping** | Both models clip forecasts at zero, ensuring we never predict negative revenue or negative order counts. |
| **Gap-filling in history** | Missing dates in the raw transaction data are filled with zeros, which is reasonable for truly inactive days but could bias the model if gaps are due to data collection issues. |

---

## 6. Challenges

1. **Limited training history** — daily aggregation from transaction data may yield only a few hundred observations, which constrains how much the Gradient Boosting model can learn.
2. **Recursive error compounding** — multi-step ahead predictions accumulate errors; each forecast step relies on previously predicted (not actual) values.
3. **No external regressors** — promotions, holidays, and macroeconomic factors are not included; the models rely solely on endogenous series and calendar features.
4. **Single seasonal period (7 days)** — the Seasonal Naive model and lag design assume a weekly cycle only; monthly or yearly seasonality is not explicitly captured.
5. **Evaluation on a single holdout split** — the 30-day holdout is one fixed window; results could vary with different time periods (no cross-validation across time).

---

## 7. Assumptions

- **Stationarity of weekly patterns** — we assume the most recent week is representative of near-future demand.
- **No structural breaks** — the models assume no major business changes (store closures, new product launches) within the forecast horizon.
- **Zero-fill for missing days** — we assume that days with no transactions genuinely had zero activity, rather than representing data outages.
- **Independent targets** — while we model supporting series jointly, we evaluate and select models for each target independently.
- **Static hyperparameters** — Gradient Boosting parameters are fixed, not tuned per run or per series.
