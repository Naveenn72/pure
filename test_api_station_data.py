from datetime import datetime, timedelta, timezone
from importlib.util import module_from_spec, spec_from_file_location
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from aqi_calculations import StationReading, haversine_km


spec = spec_from_file_location(
    "pureair_api", Path(__file__).with_name("03_api.py")
)
if spec is None or spec.loader is None:
    raise RuntimeError("Could not load the PureAir API module for testing.")
api_module = module_from_spec(spec)
sys.modules[spec.name] = api_module
spec.loader.exec_module(api_module)


class StationDataEndpointTests(unittest.TestCase):
    @staticmethod
    def model_fallback_payload() -> dict[str, object]:
        return {
            "available": True,
            "data_type": "model_estimate",
            "provider": "open_meteo_cams",
            "aqi": 101.0,
            "aqi_standard": "U.S. AQI scale (Open-Meteo model)",
            "pm2_5": 25.2,
            "pollutants": {"pm2_5": 25.2, "pm10": 28.7},
            "time": "2026-10-06T12:30",
            "timezone": "Asia/Kolkata",
            "method": "Open-Meteo CAMS model grid estimate",
            "window_hours": None,
            "aqi_interval": None,
            "interval_method": "No calibrated uncertainty interval.",
            "source": "Open-Meteo Air Quality API (CAMS model)",
            "stations": [],
        }

    def test_road_midpoint_uses_distance_along_geometry(self) -> None:
        coordinates = [(12.0, 77.0), (12.0, 77.01), (12.02, 77.01)]
        midpoint = api_module._road_midpoint(coordinates)
        length = sum(
            haversine_km(start[0], start[1], end[0], end[1])
            for start, end in zip(coordinates, coordinates[1:])
        )
        distance_to_midpoint = haversine_km(*coordinates[0], *coordinates[1]) + (
            haversine_km(*coordinates[1], *midpoint)
        )

        self.assertAlmostEqual(distance_to_midpoint, length / 2, places=6)

    def test_location_timezone_returns_valid_timezone_for_coordinates(self) -> None:
        with patch.object(
            api_module.requests,
            "get",
            return_value=FakeResponse({"timezone": "Asia/Kolkata"}),
        ) as request:
            result = api_module.location_timezone(lat=12.9, lon=77.6)

        self.assertEqual(result, {"timezone": "Asia/Kolkata"})
        self.assertEqual(
            request.call_args.kwargs["params"]["timezone"],
            "auto",
        )

    def test_location_timezone_rejects_unknown_timezone(self) -> None:
        with patch.object(
            api_module.requests,
            "get",
            return_value=FakeResponse({"timezone": "Not/A_Real_Zone"}),
        ):
            with self.assertRaises(api_module.HTTPException) as error:
                api_module.location_timezone(lat=12.9, lon=77.6)

        self.assertEqual(error.exception.status_code, 502)

    def test_sensor_placement_ranks_roads_and_discloses_proxy_inputs(self) -> None:
        now = datetime.now().astimezone()
        stations = [
            StationReading(
                name=f"station-{index}",
                latitude=12.0 + index * 0.01,
                longitude=77.0,
                country_code="IN",
                timezone="Asia/Kolkata",
                distance_km=1.0,
                pm25_24h=10 + index * 5,
                observed_at=now,
                observations=24,
            )
            for index in range(5)
        ]
        roads = [
            {
                "tags": {"highway": "residential"},
                "geometry": [
                    {"lat": 12.01, "lon": 77.01},
                    {"lat": 12.011, "lon": 77.011},
                ],
            },
            {
                "tags": {"highway": "primary"},
                "geometry": [
                    {"lat": 12.015, "lon": 77.015},
                    {"lat": 12.0152, "lon": 77.0152},
                ],
            },
        ]

        with (
            patch.dict(os.environ, {"OPENAQ_API_KEY": "test-key"}),
            patch.object(api_module, "_get_station_readings", return_value=stations),
            patch.object(api_module, "_overpass_roads", return_value=roads),
        ):
            result = api_module.sensor_placement(lat=12.0, lon=77.0, limit=10)

        self.assertEqual(result["candidate_count"], 2)
        self.assertFalse(result["traffic_density_measured"])
        self.assertEqual(result["candidates"][0]["road_type"], "primary")
        self.assertIn("field surveys", result["limitation"])

    def test_air_quality_includes_empirical_interval_with_five_stations(self) -> None:
        now = datetime.now().astimezone()
        stations = [
            StationReading(
                name=f"station-{index}",
                latitude=40.70 + index * 0.01,
                longitude=-74.01 + index * 0.002,
                country_code="US",
                timezone="America/New_York",
                distance_km=index + 1,
                pm25_24h=12 + index * 2,
                observed_at=now,
                observations=24,
            )
            for index in range(5)
        ]
        with (
            patch.object(api_module, "_get_openaq_api_key", return_value="test-key"),
            patch.object(api_module, "_get_station_readings", return_value=stations),
        ):
            result = api_module.air_quality(lat=40.715, lon=-74.005)

        self.assertIsNotNone(result["aqi_interval"])
        self.assertEqual(result["aqi_interval"]["level"], 0.9)
        self.assertIn("not a calibrated guarantee", result["interval_method"])

    def test_api_key_can_be_loaded_from_project_env_file(self) -> None:
        with TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text(
                "# local credentials\n"
                "OPENAQ_API_KEY='sample-key' # comment\n"
                "GOOGLE_MAPS_API_KEY=google-sample-key\n"
                "CPCB_DATA_GOV_IN_API_KEY=cpcb-sample-key\n",
                encoding="utf-8",
            )
            with (
                patch.dict(os.environ, {"OPENAQ_API_KEY": ""}),
                patch.object(api_module, "ENV_FILE", env_file),
            ):
                self.assertEqual(api_module._get_openaq_api_key(), "sample-key")
                self.assertEqual(
                    api_module._get_google_air_quality_api_key(),
                    "google-sample-key",
                )
                self.assertEqual(
                    api_module._get_cpcb_api_key(),
                    "cpcb-sample-key",
                )

    def test_openaq_connection_errors_do_not_expose_transport_details(self) -> None:
        with patch.object(
            api_module.requests,
            "get",
            side_effect=api_module.requests.ConnectionError("private network detail"),
        ):
            with self.assertRaises(api_module.HTTPException) as error:
                api_module._openaq_get("locations", "test-key")

        self.assertEqual(
            error.exception.detail,
            "Could not retrieve monitoring-station data from OpenAQ.",
        )
        self.assertNotIn("private network detail", error.exception.detail)

    def test_cpcb_station_returns_fresh_nearby_sub_indices_without_fake_pm25(self) -> None:
        now = datetime(2026, 10, 8, 4, 0, tzinfo=timezone.utc)
        records = [
            {
                "station": "CPCB station",
                "latitude": "12.9000",
                "longitude": "77.6000",
                "last_update": "08-10-2026 09:30:00",
                "pollutant_id": "PM2.5",
                "pollutant_avg": "68",
            },
            {
                "station": "CPCB station",
                "latitude": "12.9000",
                "longitude": "77.6000",
                "last_update": "08-10-2026 09:30:00",
                "pollutant_id": "PM10",
                "pollutant_avg": "82",
            },
            {
                "station": "Distant station",
                "latitude": "13.1000",
                "longitude": "77.6000",
                "last_update": "08-10-2026 09:30:00",
                "pollutant_id": "PM2.5",
                "pollutant_avg": "140",
            },
            {
                "station": "Stale station",
                "latitude": "12.9000",
                "longitude": "77.6000",
                "last_update": "08-10-2026 06:00:00",
                "pollutant_id": "PM2.5",
                "pollutant_avg": "200",
            },
        ]
        with patch.object(
            api_module.requests,
            "get",
            return_value=FakeResponse({"records": records, "total": "4"}),
        ) as request:
            result = api_module._cpcb_air_quality(
                12.9, 77.6, "test-cpcb-key", now
            )

        self.assertEqual(result["provider"], "cpcb")
        self.assertEqual(result["aqi"], 82.0)
        self.assertIsNone(result["pm2_5"])
        self.assertEqual(result["pm2_5_sub_index"], 68.0)
        self.assertEqual(result["pollutants"]["PM10"], 82.0)
        self.assertEqual(result["stations"][0]["name"], "CPCB station")
        self.assertEqual(
            request.call_args.kwargs["params"]["api-key"],
            "test-cpcb-key",
        )

    def test_cpcb_station_is_used_before_google_for_india(self) -> None:
        cpcb_payload = {
            "available": True,
            "data_type": "station_observation",
            "provider": "cpcb",
            "aqi": 82.0,
            "aqi_standard": "India CPCB pollutant sub-index (maximum available)",
            "pm2_5": None,
            "stations": [{"name": "CPCB station", "distance_km": 1.0}],
        }
        with (
            patch.object(api_module, "_get_cpcb_api_key", return_value="test-key"),
            patch.object(api_module, "_cpcb_air_quality", return_value=cpcb_payload),
            patch.object(api_module, "_get_google_air_quality_api_key") as google_key,
        ):
            result = api_module._modeled_fallback(
                12.9, 77.6, "No qualifying OpenAQ station readings."
            )

        self.assertEqual(result["provider"], "cpcb")
        self.assertIn("No qualifying OpenAQ", result["fallback_reason"])
        google_key.assert_not_called()

    def test_openaq_is_checked_before_cpcb_for_india(self) -> None:
        cpcb_payload = {
            "available": True,
            "data_type": "station_observation",
            "provider": "cpcb",
            "aqi": 82.0,
            "aqi_standard": "India CPCB pollutant sub-index (maximum available)",
            "pm2_5": None,
            "stations": [{"name": "CPCB station", "distance_km": 1.0}],
        }
        with (
            patch.object(api_module, "_get_cpcb_api_key", return_value="test-key"),
            patch.object(api_module, "_cpcb_air_quality", return_value=cpcb_payload),
            patch.object(
                api_module, "_get_openaq_api_key", return_value="openaq-key"
            ),
            patch.object(api_module, "_get_station_readings", return_value=[]) as openaq,
        ):
            result = api_module.air_quality(lat=12.9, lon=77.6)

        self.assertEqual(result["provider"], "cpcb")
        openaq.assert_called_once()

    def test_missing_openaq_key_falls_back_to_model_estimate(self) -> None:
        with (
            patch.dict(os.environ, {"OPENAQ_API_KEY": ""}),
            patch.object(
                api_module,
                "ENV_FILE",
                Path(__file__).with_name("missing-openaq-test.env"),
            ),
            patch.object(api_module, "_get_google_air_quality_api_key", return_value=None),
            patch.object(
                api_module,
                "_open_meteo_air_quality",
                return_value=self.model_fallback_payload(),
            ),
        ):
            result = api_module.air_quality(lat=40.7, lon=-74.0)

        self.assertTrue(result["available"])
        self.assertEqual(result["data_type"], "model_estimate")
        self.assertEqual(result["aqi"], 101.0)
        self.assertIn("not configured", result["fallback_reason"])
        self.assertIn("not a ground", result["transparency_note"])

    def test_open_meteo_fallback_parses_current_model_response(self) -> None:
        now = datetime(2026, 10, 6, 7, 0, tzinfo=timezone.utc)
        hourly_times = [
            (now - timedelta(hours=24 - index)).isoformat().replace("+00:00", "Z")
            for index in range(24)
        ]
        hourly_values: list[float | None] = [float(index) for index in range(24)]
        hourly_times.extend(
            [
                (now + timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
                "invalid timestamp",
            ]
        )
        hourly_values.extend([999.0, None])
        payload = {
            "timezone": "GMT",
            "current": {
                "time": now.isoformat().replace("+00:00", "Z"),
                "pm2_5": 25.2,
                "pm10": 28.7,
                "nitrogen_dioxide": 4.9,
                "sulphur_dioxide": 8.4,
                "ozone": 160.0,
                "us_aqi": 101,
            },
            "hourly": {"time": hourly_times, "pm2_5": hourly_values},
        }
        with patch.object(
            api_module.requests, "get", return_value=FakeResponse(payload)
        ) as request:
            result = api_module._open_meteo_air_quality(12.88, 77.60, now)

        self.assertEqual(result["data_type"], "model_estimate")
        self.assertEqual(
            result["model_validation"], "not_validated_for_street_level"
        )
        self.assertEqual(result["aqi"], 101)
        self.assertEqual(result["pm2_5"], 25.2)
        self.assertEqual(result["pm2_5_24h"], 12.0)
        self.assertEqual(result["pm2_5_24h_observations"], 23)
        self.assertEqual(result["pollutants"]["ozone"], 160.0)
        self.assertIsNone(result["aqi_interval"])
        self.assertEqual(result["window_hours"], 1)
        self.assertEqual(
            request.call_args.kwargs["params"]["timezone"],
            "UTC",
        )

    def test_google_air_quality_parses_universal_aqi_and_pm25(self) -> None:
        now = datetime(2026, 10, 6, 7, 0, tzinfo=timezone.utc)
        payload = {
            "dateTime": now.isoformat().replace("+00:00", "Z"),
            "indexes": [
                {"code": "uaqi", "aqi": 72},
                {"code": "usa_epa", "aqi": 48},
            ],
            "pollutants": [
                {
                    "code": "pm25",
                    "concentration": {
                        "value": 18.4,
                        "units": "MICROGRAMS_PER_CUBIC_METER",
                    },
                },
                {
                    "code": "no2",
                    "concentration": {"value": 12.0, "units": "PARTS_PER_BILLION"},
                },
            ],
        }
        with patch.object(
            api_module.requests, "post", return_value=FakeResponse(payload)
        ) as request:
            result = api_module._google_air_quality(12.9, 77.6, "test-key", now)

        self.assertEqual(result["provider"], "google_air_quality")
        self.assertEqual(result["aqi"], 72.0)
        self.assertEqual(result["pm2_5"], 18.4)
        self.assertIsNone(result["pm2_5_24h"])
        self.assertEqual(
            result["pollutant_units"]["nitrogen_dioxide"],
            "PARTS_PER_BILLION",
        )
        self.assertEqual(
            request.call_args.kwargs["json"]["location"],
            {"latitude": 12.9, "longitude": 77.6},
        )
        self.assertTrue(request.call_args.kwargs["json"]["universalAqi"])

    def test_google_air_quality_is_used_before_cams_fallback(self) -> None:
        google_payload = {
            "available": True,
            "data_type": "model_estimate",
            "provider": "google_air_quality",
            "aqi": 72.0,
            "aqi_standard": "Google Universal AQI",
            "pm2_5": 18.4,
            "pm2_5_24h": None,
            "aqi_interval": None,
            "stations": [],
        }
        with (
            patch.object(api_module, "_get_openaq_api_key", return_value=None),
            patch.object(
                api_module,
                "_get_google_air_quality_api_key",
                return_value="test-google-key",
            ),
            patch.object(api_module, "_google_air_quality", return_value=google_payload),
            patch.object(api_module, "_open_meteo_air_quality") as cams_request,
        ):
            result = api_module.air_quality(lat=12.9, lon=77.6)

        self.assertEqual(result["provider"], "google_air_quality")
        self.assertEqual(result["aqi"], 72.0)
        self.assertIn("not a direct measurement", result["transparency_note"])
        cams_request.assert_not_called()

    def test_google_failure_is_disclosed_before_cams_fallback(self) -> None:
        with (
            patch.object(api_module, "_get_openaq_api_key", return_value=None),
            patch.object(
                api_module,
                "_get_google_air_quality_api_key",
                return_value="test-google-key",
            ),
            patch.object(
                api_module,
                "_google_air_quality",
                side_effect=api_module.HTTPException(
                    status_code=502, detail="Google returned no current AQI."
                ),
            ),
            patch.object(
                api_module,
                "_open_meteo_air_quality",
                return_value=self.model_fallback_payload(),
            ),
        ):
            result = api_module.air_quality(lat=12.9, lon=77.6)

        self.assertEqual(result["provider"], "open_meteo_cams")
        self.assertIn("Google Air Quality API was unavailable", result["fallback_reason"])
        self.assertIn("Google returned no current AQI", result["fallback_reason"])

    def test_open_meteo_hourly_mean_is_unavailable_when_stale_or_sparse(self) -> None:
        now = datetime(2026, 10, 6, 7, 0, tzinfo=timezone.utc)
        payload = {
            "current": {
                "time": now.isoformat(),
                "pm2_5": 25.2,
                "us_aqi": 101,
            },
            "hourly": {
                "time": [
                    (now - timedelta(hours=10 + index)).isoformat()
                    for index in range(20)
                ],
                "pm2_5": [10.0] * 20,
            },
        }
        with patch.object(
            api_module.requests, "get", return_value=FakeResponse(payload)
        ):
            result = api_module._open_meteo_air_quality(12.88, 77.60, now)

        self.assertIsNone(result["pm2_5_24h"])
        self.assertEqual(result["pm2_5_24h_observations"], 14)

    def test_insufficient_recent_stations_falls_back_to_model_estimate(self) -> None:
        with (
            patch.object(api_module, "_get_openaq_api_key", return_value="test-key"),
            patch.object(api_module, "_get_station_readings", return_value=[]),
            patch.object(
                api_module,
                "_open_meteo_air_quality",
                return_value=self.model_fallback_payload(),
            ),
        ):
            result = api_module.air_quality(lat=12.9, lon=77.6)

        self.assertTrue(result["available"])
        self.assertEqual(result["data_type"], "model_estimate")
        self.assertIn("No OpenAQ station readings", result["fallback_reason"])

    def test_air_quality_uses_recent_station_data_and_returns_idw_result(self) -> None:
        station_specs = (
            {
                "id": 101,
                "name": "Nearby station",
                "latitude": 40.70,
                "longitude": -74.00,
                "pm25": 10.0,
            },
            {
                "id": 202,
                "name": "Further station",
                "latitude": 40.70,
                "longitude": -74.05,
                "pm25": 20.0,
            },
        )

        def fake_get(url: str, **kwargs: object) -> object:
            if url.endswith("/locations"):
                return FakeResponse(
                    {
                        "results": [
                            {
                                "id": station["id"] + 1000,
                                "name": station["name"],
                                "timezone": "America/New_York",
                                "country": {"code": "US"},
                                "coordinates": {
                                    "latitude": station["latitude"],
                                    "longitude": station["longitude"],
                                },
                                "sensors": [
                                    {
                                        "id": station["id"],
                                        "parameter": {
                                            "name": "pm25",
                                            "units": "µg/m³",
                                        },
                                    }
                                ],
                            }
                            for station in station_specs
                        ]
                    }
                )

            sensor_id = int(url.rstrip("/").rsplit("/", maxsplit=2)[-2])
            station = next(item for item in station_specs if item["id"] == sensor_id)
            params = kwargs.get("params")
            if params is None:
                raise AssertionError("Hourly measurements must include a time range.")
            if not isinstance(params, dict):
                raise AssertionError("Hourly query parameters must be a mapping.")
            start = datetime.fromisoformat(
                str(params["datetime_from"]).replace("Z", "+00:00")
            )
            hourly_results = [
                {
                    "value": station["pm25"],
                    "period": {
                        "datetimeTo": {
                            "utc": (start + timedelta(hours=hour)).isoformat()
                        }
                    },
                }
                for hour in range(1, 25)
            ]
            return FakeResponse({"results": hourly_results})

        with (
            patch.dict(os.environ, {"OPENAQ_API_KEY": "test-key"}),
            patch.object(api_module.requests, "get", side_effect=fake_get),
        ):
            result = api_module.air_quality(lat=40.7128, lon=-74.006)

        self.assertEqual(result["aqi_standard"], "U.S. EPA")
        self.assertEqual(result["method"], "IDW (power 2)")
        self.assertEqual(result["window_hours"], 24)
        self.assertEqual(len(result["stations"]), 2)
        self.assertIsNotNone(result["station_spatial_std_pm25"])
        self.assertEqual(
            result["stations"][0]["name"],
            "Nearby station",
        )
        self.assertIn("distance_km", result["stations"][0])
        self.assertGreater(result["pm2_5"], 10)
        self.assertLess(result["pm2_5"], 20)


class FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.status_code = 200
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self.payload


if __name__ == "__main__":
    unittest.main()
