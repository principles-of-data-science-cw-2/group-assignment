from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "cleaned_retail_data.csv"
TEST_DAYS = 30
FORECAST_DAYS = 30
SEASONAL_PERIODS = 7
SMOOTHING_GRID = [0.2, 0.4, 0.6, 0.8]


@dataclass
class HoltWintersState:
    alpha: float
    beta: float
    gamma: float
    level: float
    trend: float
    seasonals: list[float]
    sse: float


def load_daily_targets(data_path: Path) -> tuple[list[date], dict[str, list[float]]]:
    required_columns = {
        "Invoice",
        "InvoiceDate",
        "Total_Revenue",
    }
    revenue_by_day: dict[date, float] = defaultdict(float)
    orders_by_day: dict[date, set[str]] = defaultdict(set)
    min_day: date | None = None
    max_day: date | None = None

    with data_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Dataset is empty.")
        missing_columns = required_columns.difference(reader.fieldnames)
        if missing_columns:
            missing_list = ", ".join(sorted(missing_columns))
            raise ValueError(f"Dataset is missing required columns: {missing_list}")

        for row in reader:
            timestamp = datetime.fromisoformat(row["InvoiceDate"])
            current_day = timestamp.date()
            revenue_by_day[current_day] += float(row["Total_Revenue"])
            orders_by_day[current_day].add(row["Invoice"])

            if min_day is None or current_day < min_day:
                min_day = current_day
            if max_day is None or current_day > max_day:
                max_day = current_day

    if min_day is None or max_day is None:
        raise ValueError("Dataset does not contain any forecastable records.")

    dates: list[date] = []
    daily_revenue: list[float] = []
    daily_order_volume: list[float] = []

    current_day = min_day
    while current_day <= max_day:
        dates.append(current_day)
        daily_revenue.append(revenue_by_day.get(current_day, 0.0))
        daily_order_volume.append(float(len(orders_by_day.get(current_day, set()))))
        current_day += timedelta(days=1)

    return dates, {
        "daily_revenue": daily_revenue,
        "daily_order_volume": daily_order_volume,
    }


def initial_trend(series: list[float], seasonal_periods: int) -> float:
    return sum(
        (series[i + seasonal_periods] - series[i]) / seasonal_periods
        for i in range(seasonal_periods)
    ) / seasonal_periods


def initial_seasonals(series: list[float], seasonal_periods: int) -> list[float]:
    season_count = len(series) // seasonal_periods
    season_averages = []
    for season_index in range(season_count):
        start = season_index * seasonal_periods
        stop = start + seasonal_periods
        season_values = series[start:stop]
        season_averages.append(sum(season_values) / seasonal_periods)

    seasonals = []
    for position in range(seasonal_periods):
        seasonal_offset = 0.0
        for season_index in range(season_count):
            observation_index = season_index * seasonal_periods + position
            seasonal_offset += series[observation_index] - season_averages[season_index]
        seasonals.append(seasonal_offset / season_count)
    return seasonals


def fit_additive_holt_winters(
    series: list[float],
    seasonal_periods: int,
    alpha: float,
    beta: float,
    gamma: float,
) -> HoltWintersState:
    if len(series) < seasonal_periods * 2:
        raise ValueError("At least two full seasons are required for Holt-Winters.")

    level = sum(series[:seasonal_periods]) / seasonal_periods
    trend = initial_trend(series, seasonal_periods)
    seasonals = initial_seasonals(series, seasonal_periods)
    sse = 0.0

    for index in range(seasonal_periods, len(series)):
        observation = series[index]
        seasonal_component = seasonals[index % seasonal_periods]
        fitted_value = level + trend + seasonal_component
        sse += (observation - fitted_value) ** 2

        previous_level = level
        level = alpha * (observation - seasonal_component) + (1 - alpha) * (level + trend)
        trend = beta * (level - previous_level) + (1 - beta) * trend
        seasonals[index % seasonal_periods] = (
            gamma * (observation - level) + (1 - gamma) * seasonal_component
        )

    return HoltWintersState(alpha, beta, gamma, level, trend, seasonals, sse)


def optimize_holt_winters(series: list[float], seasonal_periods: int) -> HoltWintersState:
    best_state: HoltWintersState | None = None
    for alpha in SMOOTHING_GRID:
        for beta in SMOOTHING_GRID:
            for gamma in SMOOTHING_GRID:
                state = fit_additive_holt_winters(series, seasonal_periods, alpha, beta, gamma)
                if best_state is None or state.sse < best_state.sse:
                    best_state = state

    if best_state is None:
        raise RuntimeError("Failed to fit Holt-Winters model.")
    return best_state


def forecast_from_state(
    state: HoltWintersState,
    horizon: int,
    seasonal_periods: int,
    observed_length: int,
) -> list[float]:
    forecasts = []
    for step in range(horizon):
        seasonal_component = state.seasonals[(observed_length + step) % seasonal_periods]
        value = state.level + (step + 1) * state.trend + seasonal_component
        forecasts.append(max(0.0, value))
    return forecasts


def seasonal_naive_forecast(train: list[float], horizon: int, seasonal_periods: int) -> list[float]:
    pattern = train[-seasonal_periods:]
    return [max(0.0, pattern[index % seasonal_periods]) for index in range(horizon)]


def mae(actual: list[float], predicted: list[float]) -> float:
    return sum(abs(a - p) for a, p in zip(actual, predicted)) / len(actual)


def rmse(actual: list[float], predicted: list[float]) -> float:
    squared_error = sum((a - p) ** 2 for a, p in zip(actual, predicted)) / len(actual)
    return math.sqrt(squared_error)


def mape(actual: list[float], predicted: list[float]) -> float:
    non_zero_pairs = [(a, p) for a, p in zip(actual, predicted) if a != 0]
    if not non_zero_pairs:
        return 0.0
    return sum(abs(a - p) / a for a, p in non_zero_pairs) * 100 / len(non_zero_pairs)


def format_number(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def render_table(rows: list[dict[str, object]], headers: list[str]) -> str:
    widths = []
    for header in headers:
        max_cell_width = max(len(format_number(row[header])) for row in rows)
        widths.append(max(len(header), max_cell_width))

    header_row = "  ".join(header.ljust(width) for header, width in zip(headers, widths))
    divider = "  ".join("-" * width for width in widths)
    body = [
        "  ".join(format_number(row[header]).ljust(width) for header, width in zip(headers, widths))
        for row in rows
    ]
    return "\n".join([header_row, divider, *body])


def summarize_forecast(target: str, model: str, values: list[float]) -> dict[str, object]:
    return {
        "target": target,
        "selected_model": model,
        "30_day_total_forecast": round(sum(values), 2),
        "average_daily_forecast": round(sum(values) / len(values), 2),
        "peak_day_forecast": round(max(values), 2),
    }


def evaluate_target(target: str, series: list[float], dates: list[date]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if len(series) <= TEST_DAYS + SEASONAL_PERIODS * 2:
        raise ValueError(f"Not enough history to forecast {target}.")

    train = series[:-TEST_DAYS]
    test = series[-TEST_DAYS:]

    baseline_predictions = seasonal_naive_forecast(train, TEST_DAYS, SEASONAL_PERIODS)
    hw_state = optimize_holt_winters(train, SEASONAL_PERIODS)
    hw_predictions = forecast_from_state(hw_state, TEST_DAYS, SEASONAL_PERIODS, len(train))

    metric_rows = [
        {
            "target": target,
            "model": "Seasonal Naive",
            "mae": round(mae(test, baseline_predictions), 2),
            "rmse": round(rmse(test, baseline_predictions), 2),
            "mape": round(mape(test, baseline_predictions), 2),
        },
        {
            "target": target,
            "model": "Holt-Winters",
            "mae": round(mae(test, hw_predictions), 2),
            "rmse": round(rmse(test, hw_predictions), 2),
            "mape": round(mape(test, hw_predictions), 2),
        },
    ]

    selected_row = min(metric_rows, key=lambda row: (row["rmse"], row["mae"]))
    selected_model = selected_row["model"]
    if selected_model == "Seasonal Naive":
        future_predictions = seasonal_naive_forecast(series, FORECAST_DAYS, SEASONAL_PERIODS)
    else:
        full_state = optimize_holt_winters(series, SEASONAL_PERIODS)
        future_predictions = forecast_from_state(full_state, FORECAST_DAYS, SEASONAL_PERIODS, len(series))

    future_dates = [dates[-1] + timedelta(days=offset) for offset in range(1, FORECAST_DAYS + 1)]
    forecast_rows = [
        {
            "date": future_date.isoformat(),
            "target": target,
            "model": selected_model,
            "forecast": round(value, 2),
        }
        for future_date, value in zip(future_dates, future_predictions)
    ]
    return metric_rows, forecast_rows


def main() -> None:
    dates, targets = load_daily_targets(DATA_PATH)

    metrics: list[dict[str, object]] = []
    forecasts: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []

    for target_name, series in targets.items():
        target_metrics, target_forecasts = evaluate_target(target_name, series, dates)
        metrics.extend(target_metrics)
        forecasts.extend(target_forecasts)
        summaries.append(
            summarize_forecast(
                target_name,
                target_forecasts[0]["model"],
                [row["forecast"] for row in target_forecasts],
            )
        )

    print("Forecasting window:")
    print(f"  Observations: {len(dates)} daily records")
    print(f"  Date range: {dates[0].isoformat()} to {dates[-1].isoformat()}")
    print(f"  Holdout window: last {TEST_DAYS} days")
    print(f"  Future horizon: next {FORECAST_DAYS} days")

    print("\nModel evaluation:")
    print(render_table(metrics, ["target", "model", "mae", "rmse", "mape"]))

    print("\n30-day forecast summary:")
    print(render_table(summaries, ["target", "selected_model", "30_day_total_forecast", "average_daily_forecast", "peak_day_forecast"]))

    print("\nNext 7 forecasted days:")
    preview_rows = []
    seen_by_target: dict[str, int] = defaultdict(int)
    for row in forecasts:
        if seen_by_target[row["target"]] < 7:
            preview_rows.append(row)
            seen_by_target[row["target"]] += 1
    print(render_table(preview_rows, ["date", "target", "model", "forecast"]))


if __name__ == "__main__":
    main()
