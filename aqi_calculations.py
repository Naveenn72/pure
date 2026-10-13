from dataclasses import dataclass
from datetime import datetime
import math
from typing import Sequence


@dataclass(frozen=True)
class StationReading:
    name: str
    latitude: float
    longitude: float
    country_code: str
    timezone: str | None
    distance_km: float
    pm25_24h: float
    observed_at: datetime
    observations: int


_BREAKPOINTS: dict[str, tuple[tuple[float, float, int, int], ...]] = {
    "US": (
        (0.0, 9.0, 0, 50),
        (9.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 125.4, 151, 200),
        (125.5, 225.4, 201, 300),
        (225.5, 325.4, 301, 400),
        (325.5, 500.4, 401, 500),
    ),
    "IN": (
        (0, 30, 0, 50),
        (31, 60, 51, 100),
        (61, 90, 101, 200),
        (91, 120, 201, 300),
        (121, 250, 301, 400),
        (251, 500, 401, 500),
    ),
}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    earth_radius_km = 6371.0088
    lat1_radians = math.radians(lat1)
    lat2_radians = math.radians(lat2)
    delta_latitude = math.radians(lat2 - lat1)
    delta_longitude = math.radians(lon2 - lon1)
    haversine = (
        math.sin(delta_latitude / 2) ** 2
        + math.cos(lat1_radians)
        * math.cos(lat2_radians)
        * math.sin(delta_longitude / 2) ** 2
    )
    return 2 * earth_radius_km * math.asin(min(1.0, math.sqrt(haversine)))


def idw_pm25(
    stations: Sequence[StationReading],
    latitude: float,
    longitude: float,
    power: float = 2.0,
) -> float:
    if not stations:
        raise ValueError("At least one station is required for IDW interpolation.")
    if not math.isfinite(power) or power <= 0:
        raise ValueError("The IDW power must be a positive finite number.")

    weighted_total = 0.0
    weight_sum = 0.0
    for station in stations:
        distance = haversine_km(
            latitude, longitude, station.latitude, station.longitude
        )
        if distance < 1e-9:
            return station.pm25_24h
        weight = 1 / distance**power
        weighted_total += station.pm25_24h * weight
        weight_sum += weight
    return weighted_total / weight_sum


def idw_spatial_std(
    stations: Sequence[StationReading],
    latitude: float,
    longitude: float,
    power: float = 2.0,
) -> float:
    if not stations:
        raise ValueError("At least one station is required for spatial uncertainty.")
    estimate = idw_pm25(stations, latitude, longitude, power)
    weights: list[float] = []
    for station in stations:
        distance = haversine_km(
            latitude, longitude, station.latitude, station.longitude
        )
        if distance < 1e-9:
            return 0.0
        weights.append(1 / distance**power)
    weight_sum = sum(weights)
    variance = sum(
        weight * (station.pm25_24h - estimate) ** 2
        for station, weight in zip(stations, weights)
    ) / weight_sum
    return math.sqrt(variance)


def idw_spatial_loo_interval(
    stations: Sequence[StationReading],
    latitude: float,
    longitude: float,
    coverage: float = 0.9,
    power: float = 2.0,
) -> tuple[float, float] | None:
    if not 0 < coverage < 1:
        raise ValueError("Interval coverage must be between zero and one.")
    if len(stations) < 5:
        return None

    absolute_errors = [
        abs(
            station.pm25_24h
            - idw_pm25(
                [other for index, other in enumerate(stations) if index != held_out],
                station.latitude,
                station.longitude,
                power,
            )
        )
        for held_out, station in enumerate(stations)
    ]
    absolute_errors.sort()
    rank = min(
        len(absolute_errors),
        math.ceil((len(absolute_errors) + 1) * coverage),
    )
    error_quantile = absolute_errors[rank - 1]
    estimate = idw_pm25(stations, latitude, longitude, power)
    return max(0.0, estimate - error_quantile), estimate + error_quantile


def aqi_from_pm25(pm25: float, country_code: str) -> tuple[int, str]:
    if not math.isfinite(pm25) or pm25 < 0:
        raise ValueError("PM2.5 concentration must be a finite, non-negative value.")

    code = country_code.upper()
    breakpoints = _BREAKPOINTS.get(code)
    if breakpoints is None:
        raise ValueError(
            f"No PM2.5 AQI breakpoint standard is configured for country {code or 'unknown'}."
        )

    if code == "US":
        concentration = math.floor(min(pm25, 500.4) * 10) / 10
        standard = "U.S. EPA"
    else:
        concentration = float(math.floor(min(pm25, 500)))
        standard = "India CPCB"

    for concentration_low, concentration_high, index_low, index_high in breakpoints:
        if concentration_low <= concentration <= concentration_high:
            index = (
                (index_high - index_low)
                / (concentration_high - concentration_low)
                * (concentration - concentration_low)
                + index_low
            )
            return math.floor(index + 0.5), standard
    return 500, standard
