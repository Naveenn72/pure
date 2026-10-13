import argparse
from datetime import datetime, timedelta, timezone
import math
import os
from pathlib import Path
import sys
from typing import Any

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
import requests
from shapely.geometry import Point


OPENAQ_BASE_URL = "https://api.openaq.org/v3"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
ENV_FILE = Path(__file__).with_name(".env")
ROAD_RADII_METERS = (50, 100, 500, 1000)
MAJOR_ROADS = {"motorway", "trunk", "primary", "secondary"}


def _configured_key(name: str) -> str | None:
    value = os.getenv(name)
    if value:
        return value.strip().strip("'\"") or None
    try:
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            key, separator, candidate = line.partition("=")
            if separator and key.strip() == name:
                return candidate.split("#", maxsplit=1)[0].strip().strip("'\"") or None
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise RuntimeError("Could not read the local API configuration file.") from exc
    return None


def _get_json(
    url: str, *, params: dict[str, Any], headers: dict[str, str] | None = None
) -> dict[str, Any]:
    try:
        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, OSError) as exc:
        raise RuntimeError(
            "Data provider request failed; check network access and API credentials."
        ) from exc
    except ValueError as exc:
        raise RuntimeError("Data provider returned invalid JSON.") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Data provider returned an unexpected response.")
    return payload


def _parse_utc(value: Any) -> datetime | None:
    if isinstance(value, dict):
        value = value.get("utc")
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def fetch_openaq_observations(
    latitude: float,
    longitude: float,
    radius_km: float,
    start: datetime,
    end: datetime,
    api_key: str,
) -> pd.DataFrame:
    headers = {"X-API-Key": api_key}
    locations = _get_json(
        f"{OPENAQ_BASE_URL}/locations",
        headers=headers,
        params={
            "coordinates": f"{latitude},{longitude}",
            "radius": round(radius_km * 1000),
            "parameters_id": 2,
            "mobile": False,
            "limit": 100,
        },
    ).get("results")
    if not isinstance(locations, list):
        raise RuntimeError("OpenAQ returned an unexpected locations response.")

    observations: list[dict[str, Any]] = []
    for location in locations:
        if not isinstance(location, dict):
            continue
        coordinates = location.get("coordinates")
        sensors = location.get("sensors")
        if not isinstance(coordinates, dict) or not isinstance(sensors, list):
            continue
        try:
            station_lat = float(coordinates["latitude"])
            station_lon = float(coordinates["longitude"])
            station_id = str(location["id"])
        except (KeyError, TypeError, ValueError):
            continue
        for sensor in sensors:
            if not isinstance(sensor, dict):
                continue
            parameter = sensor.get("parameter")
            if not isinstance(parameter, dict):
                continue
            parameter_name = str(parameter.get("name", "")).casefold().replace(".", "")
            units = (
                str(parameter.get("units", ""))
                .casefold()
                .replace("μ", "u")
                .replace("µ", "u")
                .replace(" ", "")
                .replace("³", "3")
                .replace("^", "")
            )
            if parameter_name != "pm25" or units not in {
                "ug/m3",
                "micrograms/m3",
                "microgram/m3",
            }:
                continue
            try:
                sensor_id = int(sensor["id"])
            except (KeyError, TypeError, ValueError):
                continue
            current = start
            while current < end:
                chunk_end = min(current + timedelta(days=7), end)
                result = _get_json(
                    f"{OPENAQ_BASE_URL}/sensors/{sensor_id}/hours",
                    headers=headers,
                    params={
                        "datetime_from": current.isoformat().replace("+00:00", "Z"),
                        "datetime_to": chunk_end.isoformat().replace("+00:00", "Z"),
                        "limit": 1000,
                    },
                ).get("results")
                if not isinstance(result, list):
                    raise RuntimeError("OpenAQ returned unexpected hourly data.")
                for item in result:
                    if not isinstance(item, dict):
                        continue
                    value = item.get("value")
                    period = item.get("period")
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, (float, int))
                        or not math.isfinite(value)
                        or value < 0
                        or not isinstance(period, dict)
                    ):
                        continue
                    observed_at = _parse_utc(
                        period.get("datetimeTo") or period.get("datetime_to")
                    )
                    if observed_at is None or not start < observed_at <= end:
                        continue
                    observations.append(
                        {
                            "station_id": station_id,
                            "station_name": str(location.get("name") or ""),
                            "latitude": station_lat,
                            "longitude": station_lon,
                            "observed_at": observed_at,
                            "pm25_ug_m3": float(value),
                            "source": "OpenAQ hourly sensor measurement",
                        }
                    )
                current = chunk_end

    columns = [
        "station_id",
        "station_name",
        "latitude",
        "longitude",
        "observed_at",
        "pm25_ug_m3",
        "source",
    ]
    if not observations:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(observations, columns=columns)
        .drop_duplicates(["station_id", "observed_at"])
        .sort_values(["station_id", "observed_at"])
        .reset_index(drop=True)
    )


def extract_road_features(
    latitude: float, longitude: float
) -> dict[str, float]:
    graph = ox.graph_from_point(
        (latitude, longitude),
        dist=1100,
        network_type="drive",
        simplify=True,
    )
    _, edges = ox.graph_to_gdfs(graph, nodes=True, edges=True)
    if edges.empty:
        raise RuntimeError("OpenStreetMap returned no roads near a station.")
    projected_crs = edges.estimate_utm_crs()
    projected_edges = edges.to_crs(projected_crs)
    center = (
        gpd.GeoSeries([Point(longitude, latitude)], crs="EPSG:4326")
        .to_crs(projected_crs)
        .iloc[0]
    )
    road_types = projected_edges.get(
        "highway", pd.Series(index=projected_edges.index, dtype=object)
    )
    if road_types.map(lambda value: isinstance(value, list)).any():
        road_types = road_types.map(
            lambda value: value[0] if isinstance(value, list) and value else value
        )

    features: dict[str, float] = {}
    for radius in ROAD_RADII_METERS:
        buffer = center.buffer(radius)
        clipped_length = projected_edges.geometry.intersection(buffer).length
        features[f"road_length_{radius}m_m"] = float(clipped_length.sum())
        features[f"major_road_length_{radius}m_m"] = float(
            clipped_length[road_types.isin(MAJOR_ROADS)].sum()
        )
    primary_roads = projected_edges[road_types.isin(MAJOR_ROADS)]
    if primary_roads.empty:
        features["nearest_major_road_m"] = 1100.0
        features["major_road_within_1100m"] = 0.0
    else:
        features["nearest_major_road_m"] = float(
            min(primary_roads.geometry.distance(center).min(), 1100.0)
        )
        features["major_road_within_1100m"] = 1.0
    return features


def fetch_hourly_weather(
    latitude: float, longitude: float, start: datetime, end: datetime
) -> pd.DataFrame:
    payload = _get_json(
        OPEN_METEO_ARCHIVE_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start.date().isoformat(),
            "end_date": end.date().isoformat(),
            "hourly": (
                "temperature_2m,relative_humidity_2m,wind_speed_10m,"
                "wind_direction_10m"
            ),
            "timezone": "UTC",
        },
    )
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict) or not isinstance(hourly.get("time"), list):
        raise RuntimeError("Open-Meteo returned unexpected hourly weather data.")
    rows = {
        "observed_at": [_parse_utc(value) for value in hourly["time"]],
    }
    for column in (
        "temperature_2m",
        "relative_humidity_2m",
        "wind_speed_10m",
        "wind_direction_10m",
    ):
        values = hourly.get(column)
        if not isinstance(values, list) or len(values) != len(rows["observed_at"]):
            raise RuntimeError(f"Open-Meteo omitted the {column} hourly series.")
        rows[column] = values
    return pd.DataFrame(rows).dropna(subset=["observed_at"])


def build_training_dataset(
    observations: pd.DataFrame,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    enriched: list[pd.DataFrame] = []
    for _, station_data in observations.groupby("station_id", sort=False):
        first = station_data.iloc[0]
        road_features = extract_road_features(first["latitude"], first["longitude"])
        weather = fetch_hourly_weather(first["latitude"], first["longitude"], start, end)
        joined = station_data.merge(weather, on="observed_at", how="inner")
        wind_direction = np.deg2rad(joined["wind_direction_10m"])
        joined["wind_u_10m"] = (
            -joined["wind_speed_10m"] * np.sin(wind_direction)
        )
        joined["wind_v_10m"] = (
            -joined["wind_speed_10m"] * np.cos(wind_direction)
        )
        for name, value in road_features.items():
            joined[name] = value
        joined["hour_sin"] = joined["observed_at"].map(
            lambda value: math.sin(2 * math.pi * value.hour / 24)
        )
        joined["hour_cos"] = joined["observed_at"].map(
            lambda value: math.cos(2 * math.pi * value.hour / 24)
        )
        joined["month_sin"] = joined["observed_at"].map(
            lambda value: math.sin(2 * math.pi * value.month / 12)
        )
        joined["month_cos"] = joined["observed_at"].map(
            lambda value: math.cos(2 * math.pi * value.month / 12)
        )
        enriched.append(joined)
    if not enriched:
        return pd.DataFrame()
    return pd.concat(enriched, ignore_index=True).sort_values(
        ["station_id", "observed_at"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a real India-region PM2.5 training dataset."
    )
    parser.add_argument("--latitude", type=float, required=True)
    parser.add_argument("--longitude", type=float, required=True)
    parser.add_argument("--radius-km", type=float, default=25)
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("data") / "india_pm25_features.csv",
    )
    args = parser.parse_args()
    if (
        not -90 <= args.latitude <= 90
        or not -180 <= args.longitude <= 180
        or not math.isfinite(args.radius_km)
        or args.radius_km <= 0
        or args.days < 1
    ):
        parser.error("Coordinates, radius, and days must be valid positive ranges.")

    api_key = _configured_key("OPENAQ_API_KEY")
    if not api_key:
        parser.error("Set OPENAQ_API_KEY in the environment or project .env file.")
    end = (datetime.now(timezone.utc) - timedelta(days=7)).replace(
        minute=0, second=0, microsecond=0
    )
    start = end - timedelta(days=args.days)
    observations = fetch_openaq_observations(
        args.latitude,
        args.longitude,
        args.radius_km,
        start,
        end,
        api_key,
    )
    if observations.empty:
        raise RuntimeError("OpenAQ returned no PM2.5 observations in the requested period.")
    dataset = build_training_dataset(observations, start, end)
    if dataset.empty:
        raise RuntimeError("No observation rows matched the hourly weather data.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.output, index=False)
    print(
        f"Saved {len(dataset)} paired rows from "
        f"{dataset['station_id'].nunique()} stations to {args.output}."
    )


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, requests.RequestException, OSError) as exc:
        print(f"Data acquisition failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
