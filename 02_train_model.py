import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from aqi_calculations import haversine_km


TARGET = "pm25_ug_m3"
GROUP = "station_id"
TIMESTAMP = "observed_at"
FEATURES = (
    "temperature_2m",
    "relative_humidity_2m",
    "wind_u_10m",
    "wind_v_10m",
    "road_length_50m_m",
    "major_road_length_50m_m",
    "road_length_100m_m",
    "major_road_length_100m_m",
    "road_length_500m_m",
    "major_road_length_500m_m",
    "road_length_1000m_m",
    "major_road_length_1000m_m",
    "nearest_major_road_m",
    "major_road_within_1100m",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
)


def _model() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        max_features=0.8,
        random_state=42,
        n_jobs=-1,
    )


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(actual, predicted)),
        "rmse_ug_m3": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mae_ug_m3": float(mean_absolute_error(actual, predicted)),
    }


def _idw_prediction(target: pd.Series, reference: pd.DataFrame) -> float | None:
    if reference.empty:
        return None
    distances = np.array(
        [
            haversine_km(
                float(target["latitude"]),
                float(target["longitude"]),
                float(row["latitude"]),
                float(row["longitude"]),
            )
            for _, row in reference.iterrows()
        ],
        dtype=float,
    )
    if np.any(distances < 1e-9):
        return float(reference.iloc[int(np.argmin(distances))][TARGET])
    weights = 1 / distances**2
    return float(np.average(reference[TARGET].to_numpy(dtype=float), weights=weights))


def evaluate_dataset(data: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    required = {TARGET, GROUP, TIMESTAMP, "latitude", "longitude", *FEATURES}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")

    frame = data.copy()
    frame[TIMESTAMP] = pd.to_datetime(frame[TIMESTAMP], utc=True, errors="coerce")
    for column in (TARGET, "latitude", "longitude", *FEATURES):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=[TARGET, GROUP, TIMESTAMP, *FEATURES])
    frame = frame[np.isfinite(frame[[TARGET, *FEATURES]]).all(axis=1)]
    if frame[GROUP].nunique() < 5:
        raise ValueError("At least five distinct stations are required for spatial validation.")
    if frame[TIMESTAMP].nunique() < 5:
        raise ValueError("At least five distinct timestamps are required for temporal validation.")

    x = frame.loc[:, FEATURES]
    y = frame[TARGET].to_numpy(dtype=float)
    groups = frame[GROUP].astype(str).to_numpy()
    splitter = GroupKFold(n_splits=min(5, frame[GROUP].nunique()))
    spatial_predictions = np.full(len(frame), np.nan)
    spatial_idw_predictions = np.full(len(frame), np.nan)
    for train_indices, test_indices in splitter.split(x, y, groups):
        spatial_predictions[test_indices] = _model().fit(
            x.iloc[train_indices], y[train_indices]
        ).predict(x.iloc[test_indices])
        fold_train = frame.iloc[train_indices]
        for row_index in test_indices:
            target = frame.iloc[row_index]
            same_time = fold_train[fold_train[TIMESTAMP] == target[TIMESTAMP]]
            prediction = _idw_prediction(target, same_time)
            if prediction is not None:
                spatial_idw_predictions[row_index] = prediction
    spatial_valid = np.isfinite(spatial_predictions)
    spatial_idw_valid = np.isfinite(spatial_idw_predictions)

    unique_stations = frame[GROUP].astype(str).drop_duplicates().to_numpy()
    heldout_station_split = GroupShuffleSplit(
        n_splits=1, test_size=0.2, random_state=42
    )
    station_train_indices, station_test_indices = next(
        heldout_station_split.split(unique_stations, groups=unique_stations)
    )
    train_stations = set(unique_stations[station_train_indices])
    test_stations = set(unique_stations[station_test_indices])
    cutoff = frame[TIMESTAMP].quantile(0.8)
    temporal_train = frame[TIMESTAMP] < cutoff
    temporal_test = frame[TIMESTAMP] >= cutoff
    train_mask = frame[GROUP].astype(str).isin(train_stations) & temporal_train
    test_mask = frame[GROUP].astype(str).isin(test_stations) & temporal_test
    if train_mask.sum() < 2 or test_mask.sum() < 2:
        raise ValueError(
            "Spatial-temporal holdout is too small; collect more stations or "
            "a longer observation period."
        )

    temporal_model = _model().fit(x.loc[train_mask], y[train_mask.to_numpy()])
    temporal_predictions = temporal_model.predict(x.loc[test_mask])
    spatial_temporal = _metrics(y[test_mask.to_numpy()], temporal_predictions)
    latest_training_observations = (
        frame.loc[train_mask]
        .sort_values(TIMESTAMP)
        .groupby(GROUP, as_index=False)
        .tail(1)
    )
    temporal_idw_predictions = np.array(
        [
            _idw_prediction(row, latest_training_observations)
            for _, row in frame.loc[test_mask].iterrows()
        ],
        dtype=float,
    )
    temporal_idw_valid = np.isfinite(temporal_idw_predictions)
    metrics: dict[str, Any] = {
        "spatial_group_kfold": _metrics(
            y[spatial_valid], spatial_predictions[spatial_valid]
        ),
        "spatial_group_idw_baseline": (
            _metrics(y[spatial_idw_valid], spatial_idw_predictions[spatial_idw_valid])
            if spatial_idw_valid.sum() >= 2
            else None
        ),
        "spatial_temporal_holdout": spatial_temporal,
        "spatial_temporal_last_observation_idw_baseline": (
            _metrics(
                y[test_mask.to_numpy()][temporal_idw_valid],
                temporal_idw_predictions[temporal_idw_valid],
            )
            if temporal_idw_valid.sum() >= 2
            else None
        ),
        "spatial_temporal_split": {
            "time_cutoff_utc": cutoff.isoformat(),
            "train_station_count": len(train_stations),
            "test_station_count": len(test_stations),
            "train_rows": int(train_mask.sum()),
            "test_rows": int(test_mask.sum()),
        },
        "rows_used": int(len(frame)),
        "stations_used": int(frame[GROUP].nunique()),
        "feature_importance": dict(
            sorted(
                zip(FEATURES, temporal_model.feature_importances_),
                key=lambda item: item[1],
                reverse=True,
            )
        ),
        "data_scope": "Real OpenAQ hourly station observations paired with OSM and Open-Meteo features.",
    }
    return metrics, frame


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate and train the PM2.5 land-use regression model."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).with_name("data") / "india_pm25_features.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("data") / "lur",
    )
    args = parser.parse_args()
    if not args.dataset.is_file():
        parser.error(
            f"Dataset not found: {args.dataset}. Run 01_get_data.py first."
        )

    dataset = pd.read_csv(args.dataset)
    metrics, frame = evaluate_dataset(dataset)
    final_model = _model().fit(frame.loc[:, FEATURES], frame[TARGET].to_numpy())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "india_pm25_lur.joblib"
    metrics_path = args.output_dir / "validation_metrics.json"
    from joblib import dump

    dump(
        {
            "model": final_model,
            "features": list(FEATURES),
            "target": TARGET,
            "units": "µg/m³",
            "validation": metrics,
        },
        model_path,
    )
    metrics_path.write_text(
        json.dumps(metrics, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, allow_nan=False))
    print(f"Saved model bundle to {model_path}")
    print(f"Saved validation metrics to {metrics_path}")


if __name__ == "__main__":
    main()
