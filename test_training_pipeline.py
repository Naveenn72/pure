from datetime import datetime, timedelta, timezone
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString


def load_script(name: str) -> object:
    spec = spec_from_file_location(name, Path(__file__).with_name(f"{name}.py"))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {name} for tests.")
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


data_pipeline = load_script("01_get_data")
training_pipeline = load_script("02_train_model")


class TrainingPipelineTests(unittest.TestCase):
    def test_openaq_fetch_keeps_only_pm25_concentration_records(self) -> None:
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        end = start + timedelta(hours=2)
        calls = []

        def fake_get_json(url: str, **kwargs: object) -> dict[str, object]:
            calls.append((url, kwargs))
            if url.endswith("/locations"):
                return {
                    "results": [
                        {
                            "id": 17,
                            "name": "Sensor site",
                            "coordinates": {"latitude": "12.9", "longitude": "77.6"},
                            "sensors": [
                                {
                                    "id": 51,
                                    "parameter": {"name": "pm25", "units": "µg/m³"},
                                },
                                {
                                    "id": 52,
                                    "parameter": {"name": "pm10", "units": "µg/m³"},
                                },
                                {
                                    "id": 53,
                                    "parameter": {"name": "pm25", "units": "ppm"},
                                },
                            ],
                        }
                    ]
                }
            return {
                "results": [
                    {
                        "value": 21.5,
                        "period": {
                            "datetimeTo": {
                                "utc": (start + timedelta(hours=1)).isoformat()
                            }
                        },
                    },
                    {
                        "value": 999.0,
                        "period": {
                            "datetimeTo": {
                                "utc": (start + timedelta(hours=3)).isoformat()
                            }
                        },
                    },
                ]
            }

        with patch.object(data_pipeline, "_get_json", side_effect=fake_get_json):
            observations = data_pipeline.fetch_openaq_observations(
                12.9, 77.6, 25, start, end, "sample-key"
            )

        self.assertEqual(len(observations), 1)
        self.assertEqual(observations.iloc[0]["pm25_ug_m3"], 21.5)
        self.assertEqual(observations.iloc[0]["station_id"], "17")
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][1]["headers"]["X-API-Key"], "sample-key")

    def test_osm_road_features_measure_buffered_length_in_meters(self) -> None:
        edges = gpd.GeoDataFrame(
            {
                "highway": ["primary", "residential"],
                "geometry": [
                    LineString([(77.599, 12.9), (77.601, 12.9)]),
                    LineString([(77.6, 12.899), (77.6, 12.901)]),
                ],
            },
            crs="EPSG:4326",
        )
        with (
            patch.object(data_pipeline.ox, "graph_from_point", return_value="graph"),
            patch.object(
                data_pipeline.ox,
                "graph_to_gdfs",
                return_value=(None, edges),
            ),
        ):
            features = data_pipeline.extract_road_features(12.9, 77.6)

        self.assertGreater(features["road_length_100m_m"], 0)
        self.assertGreater(features["major_road_length_100m_m"], 0)
        self.assertEqual(features["major_road_within_1100m"], 1.0)

    def test_dataset_pairs_observations_to_exact_hourly_weather(self) -> None:
        first_time = datetime(2026, 1, 1, 0, tzinfo=timezone.utc)
        observations = pd.DataFrame(
            [
                {
                    "station_id": "station-1",
                    "station_name": "Test station",
                    "latitude": 12.9,
                    "longitude": 77.6,
                    "observed_at": first_time,
                    "pm25_ug_m3": 14.0,
                    "source": "OpenAQ hourly sensor measurement",
                },
                {
                    "station_id": "station-1",
                    "station_name": "Test station",
                    "latitude": 12.9,
                    "longitude": 77.6,
                    "observed_at": first_time + timedelta(hours=1),
                    "pm25_ug_m3": 17.0,
                    "source": "OpenAQ hourly sensor measurement",
                },
            ]
        )
        weather = pd.DataFrame(
            [
                {
                    "observed_at": first_time,
                    "temperature_2m": 28.0,
                    "relative_humidity_2m": 65,
                    "wind_speed_10m": 5,
                    "wind_direction_10m": 180,
                }
            ]
        )
        with (
            patch.object(
                data_pipeline,
                "extract_road_features",
                return_value={"road_length_100m_m": 150.0},
            ),
            patch.object(data_pipeline, "fetch_hourly_weather", return_value=weather),
        ):
            dataset = data_pipeline.build_training_dataset(
                observations,
                first_time,
                first_time + timedelta(hours=2),
            )

        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset.iloc[0]["pm25_ug_m3"], 14.0)
        self.assertEqual(dataset.iloc[0]["temperature_2m"], 28.0)
        self.assertAlmostEqual(dataset.iloc[0]["wind_u_10m"], 0.0)
        self.assertAlmostEqual(dataset.iloc[0]["wind_v_10m"], 5.0)
        self.assertEqual(dataset.iloc[0]["road_length_100m_m"], 150.0)
        self.assertIn("hour_sin", dataset.columns)

    def test_validation_holds_out_station_groups_and_latest_time_block(self) -> None:
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        rows = []
        for station_index in range(6):
            for hour in range(10):
                row = {
                    "station_id": f"station-{station_index}",
                    "latitude": 12.0 + station_index * 0.02,
                    "longitude": 77.0 + station_index * 0.01,
                    "observed_at": start + timedelta(hours=hour),
                    "pm25_ug_m3": 12 + station_index * 2 + hour * 0.5,
                }
                row.update(
                    {
                        feature: (
                            float(station_index + hour + feature_index + 1)
                            if feature not in {
                                "wind_direction_10m",
                                "major_road_within_1100m",
                                "hour_sin",
                                "hour_cos",
                                "month_sin",
                                "month_cos",
                            }
                            else float((station_index * 7 + hour + feature_index) % 360)
                        )
                        for feature_index, feature in enumerate(training_pipeline.FEATURES)
                    }
                )
                rows.append(row)
        metrics, validated = training_pipeline.evaluate_dataset(pd.DataFrame(rows))

        self.assertEqual(metrics["stations_used"], 6)
        self.assertEqual(metrics["rows_used"], len(rows))
        self.assertGreater(metrics["spatial_temporal_split"]["test_rows"], 0)
        self.assertGreaterEqual(
            metrics["spatial_temporal_split"]["time_cutoff_utc"],
            "2026-01-01T07:00:00",
        )
        self.assertEqual(len(validated), len(rows))
        self.assertIn("r2", metrics["spatial_group_kfold"])
        self.assertIn("rmse_ug_m3", metrics["spatial_group_idw_baseline"])
        self.assertIn("rmse_ug_m3", metrics["spatial_temporal_holdout"])
        self.assertIn("road_length_1000m_m", metrics["feature_importance"])

    def test_training_rejects_insufficient_real_station_groups(self) -> None:
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        rows = []
        for station_index in range(4):
            for hour in range(10):
                row = {
                    "station_id": f"station-{station_index}",
                    "latitude": 12.0 + station_index * 0.02,
                    "longitude": 77.0 + station_index * 0.01,
                    "observed_at": start + timedelta(hours=hour),
                    "pm25_ug_m3": float(10 + hour),
                }
                row.update({feature: float(hour + 1) for feature in training_pipeline.FEATURES})
                rows.append(row)

        with self.assertRaisesRegex(ValueError, "five distinct stations"):
            training_pipeline.evaluate_dataset(pd.DataFrame(rows))


if __name__ == "__main__":
    unittest.main()
