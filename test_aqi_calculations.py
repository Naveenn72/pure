from datetime import datetime, timezone
import unittest

from aqi_calculations import (
    StationReading,
    aqi_from_pm25,
    idw_pm25,
    idw_spatial_loo_interval,
    idw_spatial_std,
)


def make_station(name: str, latitude: float, pm25: float) -> StationReading:
    return StationReading(
        name=name,
        latitude=latitude,
        longitude=0,
        country_code="US",
        timezone="UTC",
        distance_km=0,
        pm25_24h=pm25,
        observed_at=datetime.now(timezone.utc),
        observations=24,
    )


class AQICalculationTests(unittest.TestCase):
    def test_country_breakpoints_and_caps(self) -> None:
        cases = (
            (9.0, "US", 50, "U.S. EPA"),
            (12.0, "US", 56, "U.S. EPA"),
            (30.0, "IN", 50, "India CPCB"),
            (45.0, "IN", 75, "India CPCB"),
            (600.0, "US", 500, "U.S. EPA"),
            (600.0, "IN", 500, "India CPCB"),
        )
        for concentration, country_code, expected_aqi, expected_standard in cases:
            with self.subTest(country_code=country_code, concentration=concentration):
                self.assertEqual(
                    aqi_from_pm25(concentration, country_code),
                    (expected_aqi, expected_standard),
                )

    def test_idw_weights_nearer_station_more_heavily(self) -> None:
        nearby = make_station("near", 0.01, 100)
        distant = make_station("far", 1, 0)

        result = idw_pm25([nearby, distant], 0, 0)

        self.assertGreater(result, 99)

    def test_idw_returns_exact_station_value_at_station_coordinates(self) -> None:
        station = make_station("at target", 0, 42)

        self.assertEqual(idw_pm25([station], 0, 0), 42)

    def test_aqi_rejects_unconfigured_country_standard(self) -> None:
        with self.assertRaisesRegex(ValueError, "configured for country"):
            aqi_from_pm25(10, "GB")

    def test_spatial_leave_one_out_interval_requires_and_uses_multiple_stations(self) -> None:
        stations = [
            make_station(f"station-{index}", index * 0.01, 10 + index * 2)
            for index in range(5)
        ]

        interval = idw_spatial_loo_interval(stations, 0.025, 0.005)
        spatial_std = idw_spatial_std(stations, 0.025, 0.005)

        self.assertIsNotNone(interval)
        assert interval is not None
        estimate = idw_pm25(stations, 0.025, 0.005)
        self.assertLessEqual(interval[0], estimate)
        self.assertGreaterEqual(interval[1], estimate)
        self.assertGreater(spatial_std, 0)

    def test_spatial_leave_one_out_interval_requires_five_stations(self) -> None:
        stations = [make_station(f"station-{index}", index * 0.01, 10) for index in range(4)]

        self.assertIsNone(idw_spatial_loo_interval(stations, 0.02, 0))


if __name__ == "__main__":
    unittest.main()
