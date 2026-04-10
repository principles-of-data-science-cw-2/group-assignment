"""Forecast short-term retail demand using baseline and supervised models."""

from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "cleaned_retail_data.csv"
VISUALS_DIR = BASE_DIR / "outputs" / "forecasting"
TEST_DAYS = 30
FORECAST_DAYS = 30
SEASONAL_PERIODS = 7
LAG_STEPS = [1, 7, 14, 28]
ROLLING_WINDOWS = [7, 28]
TARGET_SERIES = ["daily_revenue", "daily_order_volume"]
MODELLED_SERIES = [
    "daily_revenue",
    "daily_order_volume",
    "line_count",
    "daily_quantity",
    "unique_customers",
    "unique_stockcodes",
    "avg_unit_price",
]
MIN_HISTORY = max(max(LAG_STEPS), max(ROLLING_WINDOWS))
GRADIENT_BOOSTING_PARAMS = {
    "learning_rate": 0.05,
    "max_depth": 4,
    "max_iter": 400,
    "min_samples_leaf": 10,
    "random_state": 0,
}
BASELINE_MODEL = "Seasonal Naive"
SUPERVISED_MODEL = "Gradient Boosting"
COMPARISON_MODELS = [BASELINE_MODEL, SUPERVISED_MODEL]
MODEL_COLORS = {
    "Actual": "#111827",
    BASELINE_MODEL: "#2563eb",
    SUPERVISED_MODEL: "#dc2626",
}
MODEL_LINESTYLES = {
    BASELINE_MODEL: "--",
    SUPERVISED_MODEL: "-",
}
REQUIRED_COLUMNS = {
    "Invoice",
    "InvoiceDate",
    "Total_Revenue",
    "Quantity",
    "Price",
    "Customer ID",
    "StockCode",
}


def normalize_identifier_series(series: pd.Series) -> pd.Series:
    """Normalize string-like identifiers and treat blank-like values as missing."""

    normalized = series.fillna("").astype(str).str.strip()
    return normalized.mask(normalized.str.lower().isin({"", "nan", "none"}), pd.NA)


def build_history_frame(history_records: list[dict[str, object]]) -> pd.DataFrame:
    """Convert the list-based history payload into a typed DataFrame."""

    history_frame = pd.DataFrame.from_records(history_records).copy()
    history_frame["date"] = pd.to_datetime(history_frame["date"])

    numeric_columns = MODELLED_SERIES + TARGET_SERIES
    for column in numeric_columns:
        if column in history_frame:
            history_frame[column] = pd.to_numeric(history_frame[column], errors="raise")

    history_frame["avg_order_value"] = (
        history_frame["daily_revenue"]
        .div(history_frame["daily_order_volume"].replace(0, np.nan))
        .fillna(0.0)
    )
    return history_frame


def load_daily_feature_records(data_path: Path) -> tuple[int, list[dict[str, object]]]:
    """Aggregate the raw transaction rows into daily target and feature records."""

    try:
        source_frame = pd.read_csv(data_path)
    except pd.errors.EmptyDataError as error:
        raise ValueError("Dataset is empty.") from error
    if source_frame.empty:
        raise ValueError("Dataset does not contain any forecastable records.")

    missing_columns = REQUIRED_COLUMNS.difference(source_frame.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {missing_list}")

    source_frame = source_frame.copy()
    source_frame["InvoiceDate"] = pd.to_datetime(source_frame["InvoiceDate"], errors="raise")
    for column in ["Total_Revenue", "Quantity", "Price"]:
        source_frame[column] = pd.to_numeric(source_frame[column], errors="raise")

    source_frame["Customer ID"] = normalize_identifier_series(source_frame["Customer ID"])
    source_frame["StockCode"] = normalize_identifier_series(source_frame["StockCode"])
    source_frame["date"] = source_frame["InvoiceDate"].dt.normalize()

    daily_frame = (
        source_frame.groupby("date", sort=True)
        .agg(
            daily_revenue=("Total_Revenue", "sum"),
            daily_order_volume=("Invoice", "nunique"),
            line_count=("Invoice", "size"),
            daily_quantity=("Quantity", "sum"),
            unique_customers=("Customer ID", "nunique"),
            unique_stockcodes=("StockCode", "nunique"),
            avg_unit_price=("Price", "mean"),
        )
        .astype(float)
    )

    full_date_index = pd.date_range(
        start=daily_frame.index.min(),
        end=daily_frame.index.max(),
        freq="D",
        name="date",
    )
    daily_frame = daily_frame.reindex(full_date_index).fillna(0.0)
    daily_frame["date"] = daily_frame.index.date

    ordered_columns = [
        "date",
        "daily_revenue",
        "daily_order_volume",
        "line_count",
        "daily_quantity",
        "unique_customers",
        "unique_stockcodes",
        "avg_unit_price",
    ]
    records = daily_frame.reset_index(drop=True)[ordered_columns].to_dict("records")
    return len(source_frame), records


def extract_dates_and_targets(
    daily_records: list[dict[str, object]],
) -> tuple[list[date], dict[str, list[float]]]:
    """Split the daily records into the target series consumed by the reporting flow."""

    dates = [record["date"] for record in daily_records]
    targets = {
        target_name: [float(record[target_name]) for record in daily_records]
        for target_name in TARGET_SERIES
    }
    return dates, targets


def seasonal_naive_forecast(
    train: list[float],
    horizon: int,
    seasonal_periods: int,
) -> list[float]:
    """Repeat the most recent seasonal pattern as a baseline forecast."""

    pattern = np.asarray(train[-seasonal_periods:], dtype=float)
    if pattern.size == 0:
        return []
    return np.clip(np.resize(pattern, horizon), a_min=0.0, a_max=None).tolist()


def build_calendar_feature_row(next_day: date) -> dict[str, float]:
    """Return the calendar features used by the supervised model."""

    timestamp = pd.Timestamp(next_day)
    return {
        "day_of_week": float(timestamp.dayofweek),
        "day_of_month": float(timestamp.day),
        "month": float(timestamp.month),
        "quarter": float(timestamp.quarter),
        "is_weekend": float(timestamp.dayofweek >= 5),
        "is_month_start": float(timestamp.is_month_start),
        "is_month_end": float(timestamp.is_month_end),
    }


def add_history_window_features(
    feature_row: dict[str, float],
    series_name: str,
    series_values: list[float],
) -> None:
    """Add lagged and rolling history features for one modeled series."""

    series = pd.Series(series_values, dtype=float)
    for lag in LAG_STEPS:
        feature_row[f"{series_name}_lag_{lag}"] = float(series.iloc[-lag])

    for window in ROLLING_WINDOWS:
        rolling_mean = series.rolling(window=window, min_periods=window).mean().iloc[-1]
        feature_row[f"{series_name}_rolling_mean_{window}"] = float(rolling_mean)


def build_gradient_boosting_feature_row(
    history_records: list[dict[str, object]],
    next_day: date,
) -> dict[str, float]:
    """Build one supervised feature row from the expanding historical record."""

    if len(history_records) < MIN_HISTORY:
        raise ValueError("Not enough history to build supervised features.")

    history_frame = build_history_frame(history_records)
    feature_row = build_calendar_feature_row(next_day)

    for series_name in [*MODELLED_SERIES, "avg_order_value"]:
        add_history_window_features(
            feature_row,
            series_name,
            history_frame[series_name].tolist(),
        )
    return feature_row


def build_supervised_training_frame(
    history_records: list[dict[str, object]],
) -> tuple[pd.DataFrame, list[str]]:
    """Build the feature matrix for all training rows with pandas shifts and rolling windows."""

    history_frame = build_history_frame(history_records)
    feature_frame = pd.DataFrame(index=history_frame.index)
    date_series = history_frame["date"]

    feature_frame["day_of_week"] = date_series.dt.dayofweek.astype(float)
    feature_frame["day_of_month"] = date_series.dt.day.astype(float)
    feature_frame["month"] = date_series.dt.month.astype(float)
    feature_frame["quarter"] = date_series.dt.quarter.astype(float)
    feature_frame["is_weekend"] = (date_series.dt.dayofweek >= 5).astype(float)
    feature_frame["is_month_start"] = date_series.dt.is_month_start.astype(float)
    feature_frame["is_month_end"] = date_series.dt.is_month_end.astype(float)

    for series_name in [*MODELLED_SERIES, "avg_order_value"]:
        series = history_frame[series_name]
        shifted_series = series.shift(1)
        for lag in LAG_STEPS:
            feature_frame[f"{series_name}_lag_{lag}"] = series.shift(lag)
        for window in ROLLING_WINDOWS:
            feature_frame[f"{series_name}_rolling_mean_{window}"] = shifted_series.rolling(
                window=window,
                min_periods=window,
            ).mean()

    feature_names = list(feature_frame.columns)
    training_frame = pd.concat([feature_frame, history_frame[MODELLED_SERIES]], axis=1).iloc[MIN_HISTORY:]
    return training_frame.dropna().reset_index(drop=True), feature_names


def fit_gradient_boosting_models(
    history_records: list[dict[str, object]],
) -> tuple[list[str], dict[str, HistGradientBoostingRegressor]]:
    """Fit one gradient boosting regressor per modeled daily series."""

    if len(history_records) <= MIN_HISTORY:
        raise ValueError("Not enough history to fit the gradient boosting models.")

    training_frame, feature_names = build_supervised_training_frame(history_records)
    feature_rows = training_frame[feature_names]

    models: dict[str, HistGradientBoostingRegressor] = {}
    for series_name in MODELLED_SERIES:
        model = HistGradientBoostingRegressor(**GRADIENT_BOOSTING_PARAMS)
        model.fit(feature_rows, training_frame[series_name])
        models[series_name] = model

    return feature_names, models


def forecast_with_gradient_boosting(
    history_records: list[dict[str, object]],
    horizon: int,
) -> dict[str, list[float]]:
    """Recursively forecast all modeled daily series from transaction-derived features."""

    feature_names, models = fit_gradient_boosting_models(history_records)
    simulated_history = [record.copy() for record in history_records]
    predictions = {series_name: [] for series_name in MODELLED_SERIES}

    # Forecast supporting feature series alongside the targets to avoid exogenous leakage.
    for _ in range(horizon):
        next_day = simulated_history[-1]["date"] + timedelta(days=1)
        feature_row = build_gradient_boosting_feature_row(simulated_history, next_day)
        feature_frame = pd.DataFrame([feature_row], columns=feature_names)

        next_record: dict[str, object] = {"date": next_day}
        for series_name, model in models.items():
            predicted_value = float(np.clip(model.predict(feature_frame)[0], a_min=0.0, a_max=None))
            next_record[series_name] = predicted_value
            predictions[series_name].append(predicted_value)

        simulated_history.append(next_record)

    return predictions


def score_predictions(actual: list[float], predicted: list[float]) -> dict[str, float]:
    """Calculate MAE, RMSE, and WAPE for a forecast."""

    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    total_absolute_error = float(np.abs(actual_array - predicted_array).sum())
    total_actual = float(np.abs(actual_array).sum())

    if total_actual == 0:
        wape = 0.0 if total_absolute_error == 0 else float("inf")
    else:
        wape = (total_absolute_error / total_actual) * 100

    return {
        "mae": float(mean_absolute_error(actual_array, predicted_array)),
        "rmse": float(np.sqrt(mean_squared_error(actual_array, predicted_array))),
        "wape": wape,
    }


def format_cell(value: object) -> str:
    """Format a table cell for plain-text output."""

    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def render_table(rows: list[dict[str, object]], headers: list[str]) -> str:
    """Render a list of dictionaries as a simple aligned text table."""

    widths: list[int] = []
    for header in headers:
        column_width = len(header)
        for row in rows:
            column_width = max(column_width, len(format_cell(row[header])))
        widths.append(column_width)

    header_row = "  ".join(header.ljust(width) for header, width in zip(headers, widths))
    divider = "  ".join("-" * width for width in widths)

    body_rows: list[str] = []
    for row in rows:
        values = []
        for header, width in zip(headers, widths):
            values.append(format_cell(row[header]).ljust(width))
        body_rows.append("  ".join(values))

    return "\n".join([header_row, divider, *body_rows])


def format_target_label(target: str) -> str:
    """Return a presentation-friendly label for a target series."""

    return target.replace("_", " ").title()


def build_metric_lookup(metric_rows: list[dict[str, object]]) -> dict[str, dict[str, dict[str, float]]]:
    """Index metric rows by target and model for plotting."""

    metric_lookup: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for row in metric_rows:
        target = str(row["target"])
        model = str(row["model"])
        metric_lookup[target][model] = {
            "mae": float(row["mae"]),
            "rmse": float(row["rmse"]),
            "wape": float(row["wape"]),
        }
    return metric_lookup


def plot_metric_comparison(
    metric_rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    """Save grouped metric charts comparing the remaining candidate models."""

    metric_lookup = build_metric_lookup(metric_rows)
    targets = list(metric_lookup)
    metrics = ["mae", "rmse", "wape"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 4.5))

    for axis, metric_name in zip(axes, metrics):
        positions = list(range(len(targets)))
        width = 0.22

        for offset, model_name in enumerate(COMPARISON_MODELS):
            model_values = [
                metric_lookup[target][model_name][metric_name]
                for target in targets
            ]
            centered_positions = [
                position + (offset - (len(COMPARISON_MODELS) - 1) / 2) * width
                for position in positions
            ]
            axis.bar(
                centered_positions,
                model_values,
                width=width,
                color=MODEL_COLORS[model_name],
                label=model_name,
            )
        axis.set_xticks(positions)
        axis.set_xticklabels([format_target_label(target) for target in targets], rotation=12)
        axis.set_title(metric_name.upper())
        axis.grid(axis="y", alpha=0.25)

    axes[0].legend(loc="upper right")
    fig.suptitle("Holdout Metrics Across Both Models", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_holdout_comparison(
    comparison_rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    """Save holdout charts comparing actuals against all candidate predictions."""

    fig, axes = plt.subplots(len(comparison_rows), 1, figsize=(12, 4.5 * len(comparison_rows)))
    if len(comparison_rows) == 1:
        axes = [axes]

    for axis, comparison in zip(axes, comparison_rows):
        holdout_dates = comparison["holdout_dates"]
        axis.plot(
            holdout_dates,
            comparison["actual_holdout"],
            color=MODEL_COLORS["Actual"],
            linewidth=2.4,
            label="Actual",
        )

        for model_name in COMPARISON_MODELS:
            axis.plot(
                holdout_dates,
                comparison["holdout_predictions"][model_name],
                color=MODEL_COLORS[model_name],
                linewidth=2.0,
                linestyle=MODEL_LINESTYLES[model_name],
                label=model_name,
            )

        axis.set_title(f"{format_target_label(comparison['target'])}: Holdout Comparison")
        axis.grid(alpha=0.25)
        axis.legend(loc="upper left")

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_future_comparison(
    comparison_rows: list[dict[str, object]],
    output_path: Path,
) -> None:
    """Save future forecast charts comparing all candidate projections."""

    fig, axes = plt.subplots(len(comparison_rows), 1, figsize=(12, 4.5 * len(comparison_rows)))
    if len(comparison_rows) == 1:
        axes = [axes]

    for axis, comparison in zip(axes, comparison_rows):
        future_dates = comparison["future_dates"]
        for model_name in COMPARISON_MODELS:
            axis.plot(
                future_dates,
                comparison["future_predictions"][model_name],
                color=MODEL_COLORS[model_name],
                linewidth=2.2,
                linestyle=MODEL_LINESTYLES[model_name],
                label=model_name,
            )

        axis.set_title(f"{format_target_label(comparison['target'])}: Next 30 Days")
        axis.grid(alpha=0.25)
        axis.legend(loc="upper left")

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_visualizations(results: dict[str, object]) -> list[Path]:
    """Create the comparison charts used in the forecast write-up and slides."""

    plt.switch_backend("Agg")
    VISUALS_DIR.mkdir(parents=True, exist_ok=True)
    metric_path = VISUALS_DIR / "model_metric_comparison.png"
    holdout_path = VISUALS_DIR / "holdout_comparison.png"
    future_path = VISUALS_DIR / "future_forecast_comparison.png"

    plot_metric_comparison(results["metrics"], metric_path)
    plot_holdout_comparison(results["comparisons"], holdout_path)
    plot_future_comparison(results["comparisons"], future_path)
    return [metric_path, holdout_path, future_path]


def build_metric_rows(
    target: str,
    test: list[float],
    predictions_by_model: dict[str, list[float]],
) -> list[dict[str, object]]:
    """Build one metric row per candidate model."""

    metric_rows: list[dict[str, object]] = []
    for model_name, predicted_values in predictions_by_model.items():
        scores = score_predictions(test, predicted_values)
        metric_rows.append(
            {
                "target": target,
                "model": model_name,
                "mae": round(scores["mae"], 2),
                "rmse": round(scores["rmse"], 2),
                "wape": round(scores["wape"], 2),
            }
        )
    return metric_rows


def select_best_model(metric_rows: list[dict[str, object]]) -> str:
    """Choose the model with the lowest RMSE, breaking ties with MAE."""

    return min(metric_rows, key=lambda row: (row["rmse"], row["mae"]))["model"]


def build_forecast_rows(
    target: str,
    model_name: str,
    values: list[float],
    last_observed_date: date,
) -> list[dict[str, object]]:
    """Build the dated forecast rows for one target."""

    forecast_rows: list[dict[str, object]] = []
    for offset, value in enumerate(values, start=1):
        forecast_rows.append(
            {
                "date": (last_observed_date + timedelta(days=offset)).isoformat(),
                "target": target,
                "model": model_name,
                "forecast": round(value, 2),
            }
        )
    return forecast_rows


def summarize_forecast_rows(
    target: str,
    forecast_rows: list[dict[str, object]],
) -> dict[str, object]:
    """Summarize the selected forecast for one target."""

    forecast_values = [row["forecast"] for row in forecast_rows]
    total_forecast = sum(forecast_values)
    return {
        "target": target,
        "selected_model": forecast_rows[0]["model"],
        "30_day_total_forecast": round(total_forecast, 2),
        "average_daily_forecast": round(total_forecast / len(forecast_values), 2),
        "peak_day_forecast": round(max(forecast_values), 2),
    }


def build_preview_rows(
    forecasts: list[dict[str, object]],
    limit_per_target: int = 7,
) -> list[dict[str, object]]:
    """Return the first few forecast rows for each target."""

    preview_rows: list[dict[str, object]] = []
    seen_by_target: dict[str, int] = defaultdict(int)

    for row in forecasts:
        target_name = str(row["target"])
        if seen_by_target[target_name] >= limit_per_target:
            continue
        preview_rows.append(row)
        seen_by_target[target_name] += 1

    return preview_rows


def evaluate_target(
    target: str,
    series: list[float],
    dates: list[date],
    gradient_boosting_test_predictions: list[float],
    gradient_boosting_future_predictions: list[float],
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    """Evaluate the remaining candidate models on a holdout window."""

    if len(series) <= TEST_DAYS + MIN_HISTORY:
        raise ValueError(f"Not enough history to forecast {target}.")

    train = series[:-TEST_DAYS]
    test = series[-TEST_DAYS:]
    holdout_dates = dates[-TEST_DAYS:]

    predictions_by_model = {
        BASELINE_MODEL: seasonal_naive_forecast(train, TEST_DAYS, SEASONAL_PERIODS),
        SUPERVISED_MODEL: gradient_boosting_test_predictions,
    }
    metric_rows = build_metric_rows(target, test, predictions_by_model)
    selected_model = select_best_model(metric_rows)
    future_predictions_by_model = {
        BASELINE_MODEL: seasonal_naive_forecast(
            series,
            FORECAST_DAYS,
            SEASONAL_PERIODS,
        ),
        SUPERVISED_MODEL: gradient_boosting_future_predictions,
    }

    future_predictions = future_predictions_by_model[selected_model]

    forecast_rows = build_forecast_rows(
        target,
        selected_model,
        future_predictions,
        dates[-1],
    )
    comparison_row = {
        "target": target,
        "selected_model": selected_model,
        "holdout_dates": holdout_dates,
        "actual_holdout": test,
        "holdout_predictions": predictions_by_model,
        "future_dates": [dates[-1] + timedelta(days=offset) for offset in range(1, FORECAST_DAYS + 1)],
        "future_predictions": future_predictions_by_model,
    }
    return metric_rows, forecast_rows, comparison_row


def run_forecasting_workflow(data_path: Path = DATA_PATH) -> dict[str, object]:
    """Run the forecasting workflow once and return all derived outputs."""

    source_row_count, daily_records = load_daily_feature_records(data_path)
    dates, targets = extract_dates_and_targets(daily_records)

    holdout_gradient_boosting_predictions = forecast_with_gradient_boosting(
        daily_records[:-TEST_DAYS],
        TEST_DAYS,
    )
    future_gradient_boosting_predictions = forecast_with_gradient_boosting(
        daily_records,
        FORECAST_DAYS,
    )

    all_metrics: list[dict[str, object]] = []
    all_forecasts: list[dict[str, object]] = []
    all_summaries: list[dict[str, object]] = []
    all_comparisons: list[dict[str, object]] = []

    for target_name, series in targets.items():
        metric_rows, forecast_rows, comparison_row = evaluate_target(
            target_name,
            series,
            dates,
            holdout_gradient_boosting_predictions[target_name],
            future_gradient_boosting_predictions[target_name],
        )
        all_metrics.extend(metric_rows)
        all_forecasts.extend(forecast_rows)
        all_summaries.append(summarize_forecast_rows(target_name, forecast_rows))
        all_comparisons.append(comparison_row)

    return {
        "source_row_count": source_row_count,
        "daily_records": daily_records,
        "dates": dates,
        "targets": targets,
        "target_names": list(targets),
        "metrics": all_metrics,
        "forecasts": all_forecasts,
        "summaries": all_summaries,
        "comparisons": all_comparisons,
        "preview_rows": build_preview_rows(all_forecasts),
    }


def main() -> None:
    """Run the forecasting workflow and print evaluation and forecast summaries."""

    results = run_forecasting_workflow(DATA_PATH)
    dates = results["dates"]
    metrics = results["metrics"]
    summaries = results["summaries"]
    preview_rows = results["preview_rows"]
    visual_paths = build_visualizations(results)

    print("Forecasting window:")
    print(f"  Source rows: {results['source_row_count']} transaction lines")
    print(f"  Aggregated observations: {len(dates)} daily records")
    print(f"  Date range: {dates[0].isoformat()} to {dates[-1].isoformat()}")
    print(f"  Holdout window: last {TEST_DAYS} days")
    print(f"  Future horizon: next {FORECAST_DAYS} days")

    print("\nModel evaluation:")
    print(render_table(metrics, ["target", "model", "mae", "rmse", "wape"]))

    print("\n30-day forecast summary:")
    print(
        render_table(
            summaries,
            [
                "target",
                "selected_model",
                "30_day_total_forecast",
                "average_daily_forecast",
                "peak_day_forecast",
            ],
        )
    )

    print("\nNext 7 forecasted days:")
    print(render_table(preview_rows, ["date", "target", "model", "forecast"]))

    print("\nSaved visuals:")
    for visual_path in visual_paths:
        print(f"  {visual_path}")


if __name__ == "__main__":
    main()
