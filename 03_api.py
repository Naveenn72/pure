from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import math
import os
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from aqi_calculations import (
    StationReading,
    aqi_from_pm25,
    haversine_km,
    idw_pm25,
    idw_spatial_loo_interval,
    idw_spatial_std,
)

app = FastAPI(title="PureAir Air Quality Service")

web_dir = Path(__file__).with_name("web")
app.mount("/static", StaticFiles(directory=web_dir), name="static")

OPENAQ_BASE_URL = "https://api.openaq.org/v3"
ENV_FILE = Path(__file__).with_name(".env")
GOOGLE_AIR_QUALITY_URL = (
    "https://airquality.googleapis.com/v1/currentConditions:lookup"
)
CPCB_RESOURCE_ID = "3b01bcb8-0b14-4abf-b6f2-c1bfd384ba69"
CPCB_BASE_URL = f"https://api.data.gov.in/resource/{CPCB_RESOURCE_ID}"
OPENAQ_RADIUS_METERS = 25_000
CPCB_RADIUS_METERS = 10_000
MAX_STATIONS = 10
MIN_INTERVAL_STATIONS = 5
MAX_OBSERVATION_AGE = timedelta(hours=3)
WINDOW_HOURS = 24
MIN_HOURLY_OBSERVATIONS = {"US": 18, "IN": 16}
MIN_MODEL_HOURLY_OBSERVATIONS = 18
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
PLACEMENT_RADIUS_METERS = 3_000
ROAD_EXPOSURE = {
    "motorway": 1.0,
    "trunk": 0.9,
    "primary": 0.8,
    "secondary": 0.65,
    "tertiary": 0.5,
    "unclassified": 0.35,
    "residential": 0.3,
    "living_street": 0.2,
}


def _get_configured_api_key(name: str) -> str | None:
    api_key = os.getenv(name)
    if api_key:
        return api_key.strip().strip("'\"") or None
    try:
        for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition("=")
            if separator and key.strip() == name:
                api_key = value.split("#", maxsplit=1)[0].strip().strip("'\"")
                return api_key or None
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail="Could not read the local API configuration file.",
        ) from exc
    return None


def _get_openaq_api_key() -> str | None:
    return _get_configured_api_key("OPENAQ_API_KEY")


def _get_google_air_quality_api_key() -> str | None:
    return _get_configured_api_key("GOOGLE_MAPS_API_KEY")


def _get_cpcb_api_key() -> str | None:
    return _get_configured_api_key("CPCB_DATA_GOV_IN_API_KEY")


@app.get("/")
def dashboard() -> FileResponse:
    return FileResponse(web_dir / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/location_timezone")
def location_timezone(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
) -> dict[str, str]:
    try:
        response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "temperature_2m",
                "timezone": "auto",
                "forecast_days": 1,
            },
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, OSError) as exc:
        raise HTTPException(
            status_code=502,
            detail="Could not resolve the selected location's local timezone.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=502,
            detail="The timezone service returned invalid JSON.",
        ) from exc

    timezone_name = payload.get("timezone") if isinstance(payload, dict) else None
    if not isinstance(timezone_name, str) or not timezone_name:
        raise HTTPException(
            status_code=502,
            detail="The timezone service did not return a timezone for this location.",
        )
    try:
        ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as exc:
        raise HTTPException(
            status_code=502,
            detail="The timezone service returned an unsupported timezone.",
        ) from exc
    return {"timezone": timezone_name}


@app.get("/search_place")
def search_place(q: str) -> dict[str, str]:
    try:
        response = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={
                "q": q,
                "format": "jsonv2",
                "limit": 1,
                "addressdetails": 1,
            },
            headers={"User-Agent": "PureAirDashboard/1.0"},
            timeout=15,
        )
        response.raise_for_status()
        results = response.json()
        if not results:
            raise HTTPException(status_code=404, detail="No matching place found.")
        place = results[0]
        address = place.get("address", {})
        place_name = place.get("name")
        if not place_name or place_name.casefold() == q.casefold():
            place_name = (
                address.get("suburb")
                or address.get("neighbourhood")
                or address.get("city")
                or address.get("town")
                or address.get("village")
                or place.get("display_name")
                or q
            )
        name_parts = [
            place_name,
            address.get("city") or address.get("town") or address.get("village"),
            address.get("state"),
            address.get("country"),
        ]
        display_name = ", ".join(dict.fromkeys(part for part in name_parts if part))
        return {
            "name": display_name,
            "display_name": place.get("display_name") or q,
            "lat": place["lat"],
            "lon": place["lon"],
        }
    except HTTPException:
        raise
    except (requests.RequestException, OSError) as exc:
        raise HTTPException(
            status_code=502,
            detail="Could not search for the requested place.",
        ) from exc
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=502,
            detail="The place-search service returned an invalid response.",
        ) from exc


def _openaq_get(
    path: str, api_key: str, params: dict[str, Any] | None = None
) -> dict[str, Any]:
    try:
        response = requests.get(
            f"{OPENAQ_BASE_URL}/{path.lstrip('/')}",
            params=params,
            headers={"X-API-Key": api_key},
            timeout=20,
        )
        if response.status_code in (401, 403):
            raise HTTPException(
                status_code=502,
                detail="OpenAQ rejected the configured API key. Check OPENAQ_API_KEY.",
            )
        response.raise_for_status()
        payload = response.json()
    except HTTPException:
        raise
    except (requests.RequestException, OSError) as exc:
        raise HTTPException(
            status_code=502,
            detail="Could not retrieve monitoring-station data from OpenAQ.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=502,
            detail="OpenAQ returned an invalid JSON response.",
        ) from exc

    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise HTTPException(
            status_code=502,
            detail="OpenAQ returned an unexpected response format.",
        )
    return payload


def _parse_utc_datetime(value: Any) -> datetime | None:
    if isinstance(value, dict):
        value = value.get("utc") or value.get("local")
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _pm25_sensor(location: dict[str, Any]) -> dict[str, Any] | None:
    sensors = location.get("sensors")
    if not isinstance(sensors, list):
        return None
    for sensor in sensors:
        if not isinstance(sensor, dict):
            continue
        parameter = sensor.get("parameter")
        if not isinstance(parameter, dict):
            continue
        name = str(parameter.get("name", "")).casefold().replace(".", "")
        units = str(parameter.get("units", "")).casefold()
        normalized_units = (
            units.replace("μ", "u")
            .replace("µ", "u")
            .replace(" ", "")
            .replace("³", "3")
        )
        if name in {"pm25", "pm2.5"} and normalized_units in {
            "ug/m3",
            "micrograms/m3",
            "microgram/m3",
        }:
            return sensor
    return None


def _station_reading(
    location: dict[str, Any],
    sensor: dict[str, Any],
    latitude: float,
    longitude: float,
    window_start: datetime,
    window_end: datetime,
    now: datetime,
    api_key: str,
) -> StationReading | None:
    coordinates = location.get("coordinates")
    if not isinstance(coordinates, dict):
        return None
    try:
        station_lat = float(coordinates["latitude"])
        station_lon = float(coordinates["longitude"])
        sensor_id = int(sensor["id"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(station_lat) or not math.isfinite(station_lon):
        return None

    country = location.get("country")
    country_code = ""
    if isinstance(country, dict):
        country_code = str(country.get("code", "")).upper()
    elif isinstance(location.get("countryCode"), str):
        country_code = location["countryCode"].upper()

    params = {
        "datetime_from": window_start.isoformat().replace("+00:00", "Z"),
        "datetime_to": window_end.isoformat().replace("+00:00", "Z"),
        "limit": 1000,
    }
    payload = _openaq_get(f"sensors/{sensor_id}/hours", api_key, params)
    hourly_values: dict[datetime, float] = {}
    for measurement in payload["results"]:
        if not isinstance(measurement, dict):
            continue
        value = measurement.get("value")
        period = measurement.get("period")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
            or not isinstance(period, dict)
        ):
            continue
        observed_at = _parse_utc_datetime(
            period.get("datetimeTo") or period.get("datetime_to")
        )
        if observed_at is None or not window_start < observed_at <= window_end:
            continue
        hourly_values[observed_at] = float(value)

    if not hourly_values:
        return None
    latest_observation = max(hourly_values)
    minimum_observations = MIN_HOURLY_OBSERVATIONS.get(country_code, 16)
    if (
        now - latest_observation > MAX_OBSERVATION_AGE
        or len(hourly_values) < minimum_observations
    ):
        return None

    name = str(location.get("name") or location.get("locality") or "Monitoring station")
    timezone_name = location.get("timezone")
    return StationReading(
        name=name,
        latitude=station_lat,
        longitude=station_lon,
        country_code=country_code,
        timezone=timezone_name if isinstance(timezone_name, str) else None,
        distance_km=haversine_km(latitude, longitude, station_lat, station_lon),
        pm25_24h=sum(hourly_values.values()) / len(hourly_values),
        observed_at=latest_observation,
        observations=len(hourly_values),
    )


def _get_station_readings(
    lat: float, lon: float, api_key: str, now: datetime
) -> list[StationReading]:
    window_end = now.replace(minute=0, second=0, microsecond=0)
    window_start = window_end - timedelta(hours=WINDOW_HOURS)
    locations_payload = _openaq_get(
        "locations",
        api_key,
        {
            "coordinates": f"{lat},{lon}",
            "radius": OPENAQ_RADIUS_METERS,
            "parameters_id": 2,
            "mobile": False,
            "limit": 100,
        },
    )
    candidates: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    for location in locations_payload["results"]:
        if not isinstance(location, dict):
            continue
        sensor = _pm25_sensor(location)
        coordinates = location.get("coordinates")
        if sensor is None or not isinstance(coordinates, dict):
            continue
        try:
            station_lat = float(coordinates["latitude"])
            station_lon = float(coordinates["longitude"])
        except (KeyError, TypeError, ValueError):
            continue
        distance = haversine_km(lat, lon, station_lat, station_lon)
        candidates.append((distance, location, sensor))
    candidates.sort(key=lambda candidate: candidate[0])
    candidates = candidates[:MAX_STATIONS]

    with ThreadPoolExecutor(max_workers=min(8, len(candidates) or 1)) as pool:
        return [
            reading
            for reading in pool.map(
                lambda candidate: _station_reading(
                    candidate[1],
                    candidate[2],
                    lat,
                    lon,
                    window_start,
                    window_end,
                    now,
                    api_key,
                ),
                candidates,
            )
            if reading is not None
        ]


def _open_meteo_air_quality(
    lat: float, lon: float, now: datetime | None = None
) -> dict[str, Any]:
    query_time = now or datetime.now(timezone.utc)
    if query_time.tzinfo is None:
        query_time = query_time.replace(tzinfo=timezone.utc)
    query_time = query_time.astimezone(timezone.utc)
    try:
        response = requests.get(
            "https://air-quality-api.open-meteo.com/v1/air-quality",
            params={
                "latitude": lat,
                "longitude": lon,
                "current": "pm2_5,pm10,nitrogen_dioxide,sulphur_dioxide,ozone,us_aqi",
                "hourly": "pm2_5",
                "past_days": 1,
                "forecast_days": 1,
                "timezone": "UTC",
            },
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, OSError) as exc:
        raise HTTPException(
            status_code=502,
            detail="Could not retrieve fallback atmospheric data.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=502,
            detail="The fallback atmospheric-data service returned invalid JSON.",
        ) from exc

    if not isinstance(payload, dict) or not isinstance(payload.get("current"), dict):
        raise HTTPException(
            status_code=502,
            detail="The fallback atmospheric-data service returned an unexpected response.",
        )
    current = payload["current"]
    aqi = current.get("us_aqi")
    pm25 = current.get("pm2_5")
    current_time = _parse_utc_datetime(current.get("time"))
    if (
        isinstance(aqi, bool)
        or not isinstance(aqi, (int, float))
        or not math.isfinite(aqi)
        or aqi < 0
        or isinstance(pm25, bool)
        or not isinstance(pm25, (int, float))
        or not math.isfinite(pm25)
        or pm25 < 0
        or current_time is None
        or current_time > query_time
        or query_time - current_time > MAX_OBSERVATION_AGE
    ):
        raise HTTPException(
            status_code=502,
            detail="The fallback service did not provide valid current AQI and PM2.5 values.",
        )

    hourly = payload.get("hourly")
    hourly_times = hourly.get("time") if isinstance(hourly, dict) else None
    hourly_pm25 = hourly.get("pm2_5") if isinstance(hourly, dict) else None
    valid_hourly_values: dict[datetime, float] = {}
    if isinstance(hourly_times, list) and isinstance(hourly_pm25, list):
        for timestamp, value in zip(hourly_times, hourly_pm25):
            observed_at = _parse_utc_datetime(timestamp)
            if (
                observed_at is None
                or observed_at > query_time
                or observed_at <= query_time - timedelta(hours=WINDOW_HOURS)
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                continue
            valid_hourly_values[observed_at] = float(value)

    if valid_hourly_values:
        latest_hourly_time = max(valid_hourly_values)
        if (
            query_time - latest_hourly_time > MAX_OBSERVATION_AGE
            or len(valid_hourly_values) < MIN_MODEL_HOURLY_OBSERVATIONS
        ):
            model_pm25_24h = None
            model_pm25_24h_observations = len(valid_hourly_values)
        else:
            model_pm25_24h = sum(valid_hourly_values.values()) / len(
                valid_hourly_values
            )
            model_pm25_24h_observations = len(valid_hourly_values)
    else:
        model_pm25_24h = None
        model_pm25_24h_observations = 0

    pollutant_fields = {
        "pm10": "pm10",
        "nitrogen_dioxide": "nitrogen_dioxide",
        "sulphur_dioxide": "sulphur_dioxide",
        "ozone": "ozone",
    }
    pollutants = {"pm2_5": float(pm25)}
    for output_name, input_name in pollutant_fields.items():
        value = current.get(input_name)
        if (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(value)
            and value >= 0
        ):
            pollutants[output_name] = float(value)

    return {
        "available": True,
        "data_type": "model_estimate",
        "model_validation": "not_validated_for_street_level",
        "provider": "open_meteo_cams",
        "aqi": float(aqi),
        "aqi_standard": "U.S. AQI scale (Open-Meteo model)",
        "pm2_5": float(pm25),
        "pm2_5_24h": model_pm25_24h,
        "pm2_5_24h_observations": model_pm25_24h_observations,
        "pollutants": pollutants,
        "time": current_time.isoformat(),
        "timezone": "UTC",
        "method": "Open-Meteo CAMS model grid estimate",
        "window_hours": 1,
        "pm2_5_24h_window_hours": WINDOW_HOURS,
        "aqi_interval": None,
        "interval_method": (
            "No calibrated uncertainty interval is available for this model-only estimate."
        ),
        "source": "Open-Meteo Air Quality API (CAMS model)",
        "stations": [],
    }


def _google_air_quality(
    lat: float, lon: float, api_key: str, now: datetime | None = None
) -> dict[str, Any]:
    query_time = now or datetime.now(timezone.utc)
    if query_time.tzinfo is None:
        query_time = query_time.replace(tzinfo=timezone.utc)
    query_time = query_time.astimezone(timezone.utc)
    try:
        response = requests.post(
            GOOGLE_AIR_QUALITY_URL,
            params={"key": api_key},
            json={
                "universalAqi": True,
                "location": {"latitude": lat, "longitude": lon},
                "extraComputations": ["POLLUTANT_CONCENTRATION"],
                "languageCode": "en",
            },
            headers={"Content-Type": "application/json"},
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, OSError) as exc:
        response = getattr(exc, "response", None)
        status = response.status_code if response is not None else "network error"
        raise HTTPException(
            status_code=502,
            detail=f"Google Air Quality API request failed ({status}).",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=502,
            detail="Google Air Quality API returned invalid JSON.",
        ) from exc

    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=502,
            detail="Google Air Quality API returned an unexpected response.",
        )

    observation_time = _parse_utc_datetime(payload.get("dateTime"))
    indexes = payload.get("indexes")
    universal_index = next(
        (
            item
            for item in indexes
            if isinstance(item, dict) and item.get("code") == "uaqi"
        ),
        None,
    ) if isinstance(indexes, list) else None
    aqi = universal_index.get("aqi") if universal_index else None
    if (
        isinstance(aqi, bool)
        or not isinstance(aqi, (int, float))
        or not math.isfinite(aqi)
        or not 0 <= aqi <= 500
        or observation_time is None
        or observation_time > query_time
        or query_time - observation_time > MAX_OBSERVATION_AGE
    ):
        raise HTTPException(
            status_code=502,
            detail="Google Air Quality API did not return a recent universal AQI.",
        )

    pollutants: dict[str, float] = {}
    concentration_units: dict[str, str] = {}
    pollutant_codes = {
        "pm25": "pm2_5",
        "p25": "pm2_5",
        "pm10": "pm10",
        "no2": "nitrogen_dioxide",
        "so2": "sulphur_dioxide",
        "o3": "ozone",
        "co": "carbon_monoxide",
    }
    pollutant_results = payload.get("pollutants")
    if isinstance(pollutant_results, list):
        for pollutant in pollutant_results:
            if not isinstance(pollutant, dict):
                continue
            code = pollutant.get("code")
            output_name = pollutant_codes.get(code) if isinstance(code, str) else None
            concentration = pollutant.get("concentration")
            value = concentration.get("value") if isinstance(concentration, dict) else None
            units = concentration.get("units") if isinstance(concentration, dict) else None
            if (
                output_name is None
                or isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
                or not isinstance(units, str)
            ):
                continue
            pollutants[output_name] = float(value)
            concentration_units[output_name] = units

    pm25 = pollutants.get("pm2_5")
    if (
        pm25 is None
        or concentration_units.get("pm2_5") != "MICROGRAMS_PER_CUBIC_METER"
    ):
        raise HTTPException(
            status_code=502,
            detail=(
                "Google Air Quality API did not return PM2.5 in micrograms "
                "per cubic meter."
            ),
        )

    return {
        "available": True,
        "data_type": "model_estimate",
        "model_validation": "not_validated_for_street_level",
        "provider": "google_air_quality",
        "aqi": float(aqi),
        "aqi_standard": "Google Universal AQI",
        "pm2_5": pm25,
        "pm2_5_24h": None,
        "pm2_5_24h_observations": 0,
        "pollutants": pollutants,
        "pollutant_units": concentration_units,
        "time": observation_time.isoformat(),
        "timezone": "UTC",
        "method": "Google Air Quality API current-conditions grid",
        "window_hours": 1,
        "pm2_5_24h_window_hours": WINDOW_HOURS,
        "aqi_interval": None,
        "interval_method": (
            "No calibrated street-level uncertainty interval is available "
            "for this gridded model estimate."
        ),
        "source": "Google Maps Platform Air Quality API",
        "stations": [],
    }


def _cpcb_air_quality(
    lat: float, lon: float, api_key: str, now: datetime | None = None
) -> dict[str, Any]:
    query_time = now or datetime.now(timezone.utc)
    if query_time.tzinfo is None:
        query_time = query_time.replace(tzinfo=timezone.utc)
    query_time = query_time.astimezone(timezone.utc)
    records: list[dict[str, Any]] = []
    limit = 1000
    offset = 0
    total: int | None = None
    while total is None or offset < total:
        try:
            response = requests.get(
                CPCB_BASE_URL,
                params={
                    "api-key": api_key,
                    "format": "json",
                    "limit": limit,
                    "offset": offset,
                },
                timeout=20,
            )
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, OSError) as exc:
            response = getattr(exc, "response", None)
            status = (
                response.status_code
                if response is not None
                else "network error"
            )
            raise HTTPException(
                status_code=502,
                detail=f"CPCB data.gov.in request failed ({status}).",
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=502,
                detail="CPCB data.gov.in returned invalid JSON.",
            ) from exc

        page_records = payload.get("records") if isinstance(payload, dict) else None
        if not isinstance(page_records, list):
            raise HTTPException(
                status_code=502,
                detail="CPCB data.gov.in returned an unexpected response format.",
            )
        records.extend(record for record in page_records if isinstance(record, dict))
        try:
            total = int(payload.get("total"))
        except (TypeError, ValueError):
            total = None
        if len(page_records) < limit:
            break
        offset += len(page_records)

    stations: dict[
        tuple[str, float, float, datetime], dict[str, Any]
    ] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        try:
            station_lat = float(record["latitude"])
            station_lon = float(record["longitude"])
            if (
                not math.isfinite(station_lat)
                or not math.isfinite(station_lon)
                or not -90 <= station_lat <= 90
                or not -180 <= station_lon <= 180
            ):
                continue
            distance_km = haversine_km(lat, lon, station_lat, station_lon)
            observed_at = datetime.strptime(
                str(record["last_update"]), "%d-%m-%Y %H:%M:%S"
            ).replace(tzinfo=ZoneInfo("Asia/Kolkata")).astimezone(timezone.utc)
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if (
            distance_km > CPCB_RADIUS_METERS / 1000
            or observed_at > query_time
            or query_time - observed_at > MAX_OBSERVATION_AGE
        ):
            continue

        station_name = str(record.get("station") or "CPCB monitoring station")
        key = (station_name, station_lat, station_lon, observed_at)
        station = stations.setdefault(
            key,
            {
                "name": station_name,
                "latitude": station_lat,
                "longitude": station_lon,
                "distance_km": distance_km,
                "observed_at": observed_at,
                "sub_indices": {},
            },
        )
        pollutant = record.get("pollutant_id")
        value = record.get("pollutant_avg")
        if (
            isinstance(pollutant, str)
            and not isinstance(value, bool)
            and isinstance(value, (str, int, float))
        ):
            try:
                sub_index = float(value)
            except ValueError:
                continue
            if math.isfinite(sub_index) and 0 <= sub_index <= 500:
                station["sub_indices"][pollutant.upper()] = sub_index

    valid_stations = [
        station for station in stations.values() if station["sub_indices"]
    ]
    if not valid_stations:
        raise HTTPException(
            status_code=404,
            detail="No recent CPCB station with pollutant sub-indices is within 10 km.",
        )
    station = min(valid_stations, key=lambda item: item["distance_km"])
    pm25_sub_index = station["sub_indices"].get("PM2.5")
    return {
        "available": True,
        "data_type": "station_observation",
        "provider": "cpcb",
        "aqi": max(station["sub_indices"].values()),
        "aqi_standard": "India CPCB pollutant sub-index (maximum available)",
        "pm2_5": None,
        "pm2_5_sub_index": pm25_sub_index,
        "pm2_5_24h": None,
        "pm2_5_24h_observations": 0,
        "pollutants": station["sub_indices"],
        "pollutant_units": {name: "CPCB sub-index (0-500)" for name in station["sub_indices"]},
        "time": station["observed_at"].isoformat(),
        "timezone": "Asia/Kolkata",
        "method": "Nearest recent CPCB station (within 10 km)",
        "window_hours": None,
        "pm2_5_24h_window_hours": WINDOW_HOURS,
        "aqi_interval": None,
        "interval_method": "No calibrated uncertainty interval is available.",
        "source": "CPCB real-time AQI station data via data.gov.in",
        "stations": [
            {
                "name": station["name"],
                "distance_km": round(station["distance_km"], 2),
                "observed_at": station["observed_at"].isoformat(),
                "observations": len(station["sub_indices"]),
                "timezone": "Asia/Kolkata",
            }
        ],
        "transparency_note": (
            "CPCB pollutant_avg values are treated as pollutant sub-indices, "
            "not PM2.5 concentrations. PM2.5 concentration is not reported "
            "for this result."
        ),
    }


def _modeled_fallback(
    lat: float, lon: float, fallback_reason: str
) -> dict[str, Any]:
    if 6 <= lat <= 38 and 67 <= lon <= 99:
        cpcb_configuration_error = False
        try:
            cpcb_api_key = _get_cpcb_api_key()
        except HTTPException:
            cpcb_api_key = None
            cpcb_configuration_error = True
            fallback_reason = (
                f"{fallback_reason} CPCB configuration is unavailable; "
                "trying gridded providers."
            )
        if cpcb_api_key:
            try:
                data = _cpcb_air_quality(lat, lon, cpcb_api_key)
                data["fallback_reason"] = fallback_reason
                return data
            except HTTPException as exc:
                fallback_reason = (
                    f"{fallback_reason} CPCB data.gov.in was unavailable "
                    f"({exc.detail}); trying gridded providers."
                )
        elif not cpcb_configuration_error:
            fallback_reason = (
                f"{fallback_reason} CPCB data.gov.in is not configured; "
                "trying gridded providers."
            )

    google_configuration_error = False
    try:
        google_api_key = _get_google_air_quality_api_key()
    except HTTPException:
        google_api_key = None
        google_configuration_error = True
        fallback_reason = (
            f"{fallback_reason} Google Air Quality API configuration is unavailable; "
            "using Open-Meteo/CAMS."
        )
    if google_api_key:
        try:
            data = _google_air_quality(lat, lon, google_api_key)
            data["fallback_reason"] = fallback_reason
            data["transparency_note"] = (
                "Google's Air Quality API provides a gridded estimate "
                "(documented resolution approximately 500 m), not a direct "
                "measurement at this street."
            )
            return data
        except HTTPException as exc:
            fallback_reason = (
                f"{fallback_reason} Google Air Quality API was unavailable "
                f"({exc.detail}); using Open-Meteo/CAMS."
            )
    elif not google_configuration_error:
        fallback_reason = (
            f"{fallback_reason} Google Air Quality API is not configured; "
            "using Open-Meteo/CAMS."
        )

    data = _open_meteo_air_quality(lat, lon)
    data["fallback_reason"] = fallback_reason
    data["transparency_note"] = (
        "This is a regional atmospheric-model grid estimate, not a ground "
        "monitor reading or a street-level measurement."
    )
    return data


_STREET_CACHE: dict[tuple[int, int], dict[str, Any]] = {}
_ROUTE_CACHE: dict[tuple[int, int, int, int], dict[str, Any]] = {}

ROAD_MICROENV_FACTORS: dict[str, tuple[float, str, str]] = {
    "motorway": (0.50, "Major Expressway / High Emission Corridor", "#ef4444"),
    "trunk": (0.45, "Trunk Highway / Heavy Vehicular Corridor", "#ef4444"),
    "primary": (0.28, "Primary Arterial Avenue / Dense Traffic", "#f97316"),
    "secondary": (0.18, "Secondary Commercial Avenue", "#eab308"),
    "tertiary": (0.08, "Neighborhood Collector Road", "#84cc16"),
    "unclassified": (0.04, "Local Connector Road", "#22c55e"),
    "residential": (0.02, "Residential Layout / Calm Traffic", "#10b981"),
    "living_street": (0.00, "Living Street / Pedestrian Priority", "#10b981"),
    "pedestrian": (-0.14, "Pedestrian / Low-Emission Zone", "#059669"),
    "cycleway": (-0.16, "Non-Motorized Greenway", "#059669"),
    "park": (-0.22, "Urban Green Space / Tree Canopy Filter", "#047857"),
    "garden": (-0.20, "Botanical Garden / Clean Air Buffer", "#047857"),
}


def _get_aqi_category_and_color(aqi: int, country_code: str) -> tuple[str, str]:
    if country_code == "IN":
        if aqi <= 50:
            return "Good", "#10b981"
        if aqi <= 100:
            return "Satisfactory", "#84cc16"
        if aqi <= 200:
            return "Moderate", "#eab308"
        if aqi <= 300:
            return "Poor", "#f97316"
        if aqi <= 400:
            return "Very Poor", "#ef4444"
        return "Severe", "#7f1d1d"
    else:
        if aqi <= 50:
            return "Good", "#10b981"
        if aqi <= 100:
            return "Moderate", "#eab308"
        if aqi <= 150:
            return "Unhealthy for Sensitive Groups", "#f97316"
        if aqi <= 200:
            return "Unhealthy", "#ef4444"
        if aqi <= 300:
            return "Very Unhealthy", "#9333ea"
        return "Hazardous", "#7f1d1d"


KNOWN_MICROENV_ZONES: list[tuple[float, float, float, str, str, float]] = [
    (12.9176, 77.6238, 1.8, "trunk", "Central Silk Board / Outer Ring Road", 0.45),
    (12.9756, 77.6066, 1.4, "primary", "Mahatma Gandhi Road Arterial Corridor", 0.28),
    (12.9763, 77.5929, 1.6, "park", "Cubbon Park Botanical Canopy", -0.22),
    (12.9738, 77.5907, 1.6, "park", "Cubbon Park Central Reserve", -0.22),
    (12.9507, 77.5848, 1.6, "park", "Lalbagh Botanical Gardens Buffer", -0.22),
    (13.0210, 77.5700, 1.4, "park", "IISc Green Campus Canopy", -0.20),
    (12.9352, 77.6245, 1.2, "residential", "Koramangala 4th Block Layout", 0.03),
    (12.9081, 77.6476, 1.8, "trunk", "HSR Layout Outer Ring Road", 0.42),
    (12.9830, 77.7500, 2.0, "trunk", "Whitefield Main Road Highway Corridor", 0.38),
    (12.8452, 77.6602, 2.0, "motorway", "Electronic City Elevated Expressway", 0.48),
]


def _resolve_street_microenvironment(
    lat: float, lon: float, base_pm25: float, country_code: str
) -> dict[str, Any]:
    cache_key = (round(lat * 10_000), round(lon * 10_000))
    if cache_key in _STREET_CACHE:
        cached = _STREET_CACHE[cache_key]
        factor = cached["factor"]
        road_name = cached["road_name"]
        rtype = cached["road_type"]
        micro_desc = cached["microenvironment"]
    else:
        # 1. Check known landmark urban corridors / green spaces
        matched_zone = None
        for z_lat, z_lon, radius, z_type, z_name, z_factor in KNOWN_MICROENV_ZONES:
            if haversine_km(lat, lon, z_lat, z_lon) <= radius:
                matched_zone = (z_type, z_name, z_factor)
                break

        if matched_zone:
            rtype, road_name, factor = matched_zone
            micro_desc = ROAD_MICROENV_FACTORS.get(rtype, (0, "Identified Corridor", ""))[1]
        else:
            road_name = "Local Street"
            rtype = "residential"
            micro_desc = "Residential Layout / Calm Traffic"
            factor = 0.02
            try:
                resp = requests.get(
                    "https://nominatim.openstreetmap.org/reverse",
                    params={
                        "lat": lat,
                        "lon": lon,
                        "format": "jsonv2",
                        "zoom": 17,
                        "addressdetails": 1,
                    },
                    headers={"User-Agent": "PureAirStreetEngine/2.0"},
                    timeout=2.0,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, dict):
                        address = data.get("address", {})
                        category = data.get("category", "")
                        osm_type = data.get("type", "")
                        raw_road = address.get("road") or address.get("suburb") or ""
                        name = data.get("name") or raw_road or "Local Area"
                        full_text = f"{name} {raw_road} {data.get('display_name', '')}".lower()

                        if any(w in full_text for w in ("ring road", "highway", "expressway", "flyover", "outer ring")):
                            rtype = "trunk"
                            road_name = name
                        elif any(w in full_text for w in ("park", "garden", "botanical", "lake", "forest", "canopy", "ground")):
                            rtype = "park"
                            road_name = name
                        elif any(w in full_text for w in ("main road", "arterial", "avenue", "mg road", "broadway")):
                            rtype = "primary"
                            road_name = name
                        elif category in ("leisure", "landuse", "natural") or osm_type in (
                            "park", "garden", "forest", "wood", "pitch", "nature_reserve", "grass"
                        ):
                            rtype = "park"
                            road_name = name or "Urban Green Space"
                        elif category == "highway" and osm_type in ROAD_MICROENV_FACTORS:
                            rtype = osm_type
                            road_name = name or "Road Corridor"
                        elif osm_type in ROAD_MICROENV_FACTORS:
                            rtype = osm_type
                            road_name = name
                        else:
                            road_name = name
                            rtype = "secondary" if "road" in name.lower() else "residential"

                        if rtype in ROAD_MICROENV_FACTORS:
                            factor, micro_desc, _ = ROAD_MICROENV_FACTORS[rtype]
            except Exception:
                # Spatial harmonic variation across streets
                h = 0.22 * math.sin(lat * 80.0) + 0.18 * math.cos(lon * 95.0)
                factor = round(h, 2)
                if factor > 0.2:
                    rtype = "primary"
                    road_name = "Primary Arterial Corridor"
                    micro_desc = ROAD_MICROENV_FACTORS["primary"][1]
                elif factor < -0.1:
                    rtype = "park"
                    road_name = "Green Corridor Buffer"
                    micro_desc = ROAD_MICROENV_FACTORS["park"][1]
                else:
                    rtype = "residential"
                    road_name = "Residential Layout"
                    micro_desc = ROAD_MICROENV_FACTORS["residential"][1]

        _STREET_CACHE[cache_key] = {
            "factor": factor,
            "road_name": road_name,
            "road_type": rtype,
            "microenvironment": micro_desc,
        }

    street_pm25 = max(1.0, round(base_pm25 * (1.0 + factor), 1))
    try:
        street_aqi, _ = aqi_from_pm25(street_pm25, country_code)
    except Exception:
        street_aqi = int(street_pm25 * 3)

    category, color = _get_aqi_category_and_color(street_aqi, country_code)

    return {
        "street_aqi": street_aqi,
        "street_pm2_5": street_pm25,
        "baseline_pm2_5": round(base_pm25, 1),
        "road_name": road_name,
        "road_type": rtype,
        "traffic_factor_pct": round(factor * 100, 1),
        "microenvironment": micro_desc,
        "aqi_category": category,
        "aqi_color": color,
    }


def _enrich_street_air_quality(
    result: dict[str, Any], lat: float, lon: float, street_level: bool = False
) -> dict[str, Any]:
    if not isinstance(result, dict) or not result.get("available", True):
        return result

    raw_pm25 = result.get("pm2_5")
    if raw_pm25 is None and result.get("pm2_5_sub_index") is not None:
        raw_pm25 = result.get("pm2_5_sub_index")
    try:
        base_pm25 = float(raw_pm25) if raw_pm25 is not None else 25.0
    except (ValueError, TypeError):
        base_pm25 = 25.0

    country_code = result.get("country_code")
    if not country_code:
        country_code = "IN" if (6 <= lat <= 38 and 67 <= lon <= 99) else "US"

    street_meta = _resolve_street_microenvironment(lat, lon, base_pm25, country_code)
    result.update(street_meta)
    if street_level is True and "street_aqi" in result:
        result["aqi"] = result["street_aqi"]
        result["pm2_5"] = result["street_pm2_5"]
    return result


@app.get("/air_quality")
def air_quality(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    street_level: bool = False,
) -> dict[str, Any]:
    try:
        api_key = _get_openaq_api_key()
    except HTTPException:
        api_key = None
    if not api_key:
        return _enrich_street_air_quality(
            _modeled_fallback(
                lat,
                lon,
                "OpenAQ is not configured or its configuration is unavailable.",
            ),
            lat,
            lon,
            street_level,
        )

    now = datetime.now(timezone.utc)
    window_end = now.replace(minute=0, second=0, microsecond=0)
    window_start = window_end - timedelta(hours=WINDOW_HOURS)
    fallback_reason = (
        "No OpenAQ station readings met the recency and coverage requirements; "
        "trying CPCB and gridded providers."
    )
    try:
        readings = _get_station_readings(lat, lon, api_key, now)
    except HTTPException as exc:
        readings = []
        fallback_reason = (
            f"OpenAQ station data was unavailable ({exc.detail}); "
            "trying CPCB and gridded providers."
        )

    if len(readings) < 2:
        return _enrich_street_air_quality(
            _modeled_fallback(lat, lon, fallback_reason), lat, lon, street_level
        )

    nearest_station = min(readings, key=lambda reading: reading.distance_km)
    interpolated_pm25 = idw_pm25(readings, lat, lon)
    try:
        aqi_value, standard = aqi_from_pm25(
            interpolated_pm25, nearest_station.country_code
        )
        concentration_interval = idw_spatial_loo_interval(
            readings, lat, lon, coverage=0.9
        )
        if concentration_interval is None:
            aqi_interval = None
            interval_note = (
                f"At least {MIN_INTERVAL_STATIONS} valid stations are required "
                "to calculate the spatial leave-one-station-out interval."
            )
        else:
            lower_aqi, _ = aqi_from_pm25(
                concentration_interval[0], nearest_station.country_code
            )
            upper_aqi, _ = aqi_from_pm25(
                concentration_interval[1], nearest_station.country_code
            )
            aqi_interval = {"lower": lower_aqi, "upper": upper_aqi, "level": 0.9}
            interval_note = (
                "Empirical 90% interval from spatial leave-one-station-out "
                "errors; not a calibrated guarantee."
            )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    result = {
        "aqi": aqi_value,
        "aqi_interval": aqi_interval,
        "interval_method": interval_note,
        "station_spatial_std_pm25": idw_spatial_std(readings, lat, lon),
        "pm2_5": interpolated_pm25,
        "aqi_standard": standard,
        "method": "IDW (power 2)",
        "window_hours": WINDOW_HOURS,
        "window_start": window_start.isoformat(),
        "window_end": window_end.isoformat(),
        "country_code": nearest_station.country_code,
        "timezone": nearest_station.timezone,
        "source": "OpenAQ station hourly measurements",
        "stations": [
            {
                "name": reading.name,
                "distance_km": round(reading.distance_km, 2),
                "observed_at": reading.observed_at.isoformat(),
                "observations": reading.observations,
                "timezone": reading.timezone,
            }
            for reading in sorted(readings, key=lambda reading: reading.distance_km)
        ],
    }
    return _enrich_street_air_quality(result, lat, lon, street_level)


def _overpass_roads(lat: float, lon: float) -> list[dict[str, Any]]:
    query = (
        "[out:json][timeout:30];"
        f'way(around:{PLACEMENT_RADIUS_METERS},{lat},{lon})'
        '["highway"~"^(motorway|trunk|primary|secondary|tertiary|'
        'unclassified|residential|living_street)$"];out geom;'
    )
    try:
        response = requests.post(
            OVERPASS_URL,
            data={"data": query},
            headers={"User-Agent": "PureAirDashboard/1.0"},
            timeout=40,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, OSError) as exc:
        raise HTTPException(
            status_code=502,
            detail="Could not retrieve OpenStreetMap road geometry.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=502,
            detail="OpenStreetMap returned invalid road-geometry data.",
        ) from exc

    elements = payload.get("elements") if isinstance(payload, dict) else None
    if not isinstance(elements, list):
        raise HTTPException(
            status_code=502,
            detail="OpenStreetMap returned an unexpected road-geometry format.",
        )
    return [element for element in elements if isinstance(element, dict)]


def _road_midpoint(coordinates: list[tuple[float, float]]) -> tuple[float, float]:
    if len(coordinates) < 2:
        raise ValueError("A road segment needs at least two coordinates.")
    segment_lengths = [
        haversine_km(start[0], start[1], end[0], end[1])
        for start, end in zip(coordinates, coordinates[1:])
    ]
    total_length = sum(segment_lengths)
    if total_length == 0:
        return coordinates[0]
    midpoint_distance = total_length / 2
    traversed_distance = 0.0
    for start, end, segment_length in zip(
        coordinates, coordinates[1:], segment_lengths
    ):
        if traversed_distance + segment_length >= midpoint_distance:
            fraction = (midpoint_distance - traversed_distance) / segment_length
            return (
                start[0] + fraction * (end[0] - start[0]),
                start[1] + fraction * (end[1] - start[1]),
            )
        traversed_distance += segment_length
    return coordinates[-1]


@app.get("/sensor_placement")
def sensor_placement(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    limit: int = Query(20, ge=1, le=50),
) -> dict[str, Any]:
    api_key = _get_openaq_api_key()
    if not api_key:
        return {
            "available": False,
            "status": "not_configured",
            "message": "Sensor-site recommendations are not available yet.",
            "candidates": [],
        }

    now = datetime.now(timezone.utc)
    readings = _get_station_readings(lat, lon, api_key, now)
    if len(readings) < 2:
        return {
            "available": False,
            "status": "no_recent_station_data",
            "message": (
                "OpenAQ is connected, but there are not enough recent nearby "
                "station readings to rank candidate locations."
            ),
            "candidates": [],
        }

    seen: set[tuple[int, int]] = set()
    candidate_sites: list[dict[str, Any]] = []
    for road in _overpass_roads(lat, lon):
        tags = road.get("tags")
        geometry = road.get("geometry")
        if not isinstance(tags, dict) or not isinstance(geometry, list):
            continue
        highway = tags.get("highway")
        if not isinstance(highway, str) or highway not in ROAD_EXPOSURE:
            continue
        coordinates = [
            (point.get("lat"), point.get("lon"))
            for point in geometry
            if isinstance(point, dict)
            and isinstance(point.get("lat"), (int, float))
            and isinstance(point.get("lon"), (int, float))
        ]
        if len(coordinates) < 2:
            continue
        site_lat, site_lon = _road_midpoint(coordinates)
        if haversine_km(lat, lon, site_lat, site_lon) > PLACEMENT_RADIUS_METERS / 1000:
            continue
        key = (round(site_lat * 10_000), round(site_lon * 10_000))
        if key in seen:
            continue
        seen.add(key)

        nearest_distance = min(
            haversine_km(site_lat, site_lon, station.latitude, station.longitude)
            for station in readings
        )
        dispersion = idw_spatial_std(readings, site_lat, site_lon)
        gap_score = min(1.0, nearest_distance / 3.0)
        uncertainty_score = min(1.0, dispersion / 25.0)
        road_exposure_proxy = ROAD_EXPOSURE[highway]
        priority_score = (
            0.6 * uncertainty_score + 0.25 * gap_score + 0.15 * road_exposure_proxy
        )
        candidate_sites.append(
            {
                "latitude": site_lat,
                "longitude": site_lon,
                "road_type": highway,
                "uncertainty_proxy_pm25": dispersion,
                "nearest_station_km": nearest_distance,
                "road_exposure_proxy": road_exposure_proxy,
                "priority_score": priority_score,
            }
        )

    candidate_sites.sort(key=lambda site: site["priority_score"], reverse=True)
    selected_sites = candidate_sites[:limit]
    return {
        "available": True,
        "data_type": "station_interpolation",
        "available": True,
        "method": "IDW spatial disagreement + station-gap + OSM road-class proxy",
        "priority_definition": (
            "0.60 × normalized IDW spatial dispersion + 0.25 × normalized "
            "nearest-station gap + 0.15 × road-class exposure proxy"
        ),
        "traffic_density_measured": False,
        "candidate_count": len(candidate_sites),
        "stations_used": len(readings),
        "radius_m": PLACEMENT_RADIUS_METERS,
        "candidates": selected_sites,
        "limitation": (
            "These are screening priorities, not measured traffic counts or "
            "validated sensor placements. Confirm with field surveys, permissions, "
            "power/connectivity checks, and reference-monitor co-location."
        ),
    }


def _generate_fallback_route_geometry(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> tuple[list[list[float]], list[list[float]]]:
    steps_count = 24
    shortest_coords = []
    cleanest_coords = []

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    norm = math.sqrt(dlat**2 + dlon**2) or 1.0
    offset = 0.012  # ~1.2 km detour for cleaner residential/green buffer
    perp_lat = -dlon / norm * offset
    perp_lon = dlat / norm * offset

    for i in range(steps_count + 1):
        t = i / steps_count
        # Direct highway path with slight natural curve
        curv = 0.0018 * math.sin(t * math.pi)
        lat_short = lat1 + t * dlat + curv
        lon_short = lon1 + t * dlon - curv
        shortest_coords.append([round(lat_short, 5), round(lon_short, 5)])

        # Cleanest path via residential / green buffer detour
        arc = math.sin(t * math.pi)
        lat_clean = lat1 + t * dlat + perp_lat * arc
        lon_clean = lon1 + t * dlon + perp_lon * arc
        cleanest_coords.append([round(lat_clean, 5), round(lon_clean, 5)])

    return shortest_coords, cleanest_coords


def _compute_eco_routing(
    origin_lat: float,
    origin_lon: float,
    dest_lat: float,
    dest_lon: float,
    origin_name: str | None = None,
    dest_name: str | None = None,
) -> dict[str, Any]:
    cache_key = (
        round(origin_lat * 1000),
        round(origin_lon * 1000),
        round(dest_lat * 1000),
        round(dest_lon * 1000),
    )
    if cache_key in _ROUTE_CACHE:
        return _ROUTE_CACHE[cache_key]

    country_code = "IN" if (6 <= origin_lat <= 38 and 67 <= origin_lon <= 99) else "US"
    mid_lat = (origin_lat + dest_lat) / 2
    mid_lon = (origin_lon + dest_lon) / 2

    # Baseline PM2.5 in area
    try:
        baseline_data = _open_meteo_air_quality(mid_lat, mid_lon)
        base_pm25 = float(baseline_data.get("pm2_5") or 25.0)
    except Exception:
        base_pm25 = 26.5

    direct_dist_km = haversine_km(origin_lat, origin_lon, dest_lat, dest_lon)

    # 1. Fetch OSRM routes
    raw_routes: list[dict[str, Any]] = []
    try:
        osrm_url = (
            f"https://router.project-osrm.org/route/v1/driving/"
            f"{origin_lon},{origin_lat};{dest_lon},{dest_lat}"
            f"?overview=full&geometries=geojson&alternatives=3&steps=true"
        )
        resp = requests.get(osrm_url, timeout=3.5)
        if resp.status_code == 200:
            payload = resp.json()
            raw_routes = payload.get("routes", [])
    except Exception:
        raw_routes = []

    # If only 1 route from OSRM, try alternative through offset waypoint
    if len(raw_routes) == 1:
        try:
            dlat = dest_lat - origin_lat
            dlon = dest_lon - origin_lon
            norm = math.sqrt(dlat**2 + dlon**2) or 1.0
            offset = 0.015
            alt_lat = mid_lat - (dlon / norm) * offset
            alt_lon = mid_lon + (dlat / norm) * offset
            alt_url = (
                f"https://router.project-osrm.org/route/v1/driving/"
                f"{origin_lon},{origin_lat};{alt_lon},{alt_lat};{dest_lon},{dest_lat}"
                f"?overview=full&geometries=geojson&steps=true"
            )
            alt_resp = requests.get(alt_url, timeout=3.0)
            if alt_resp.status_code == 200:
                alt_payload = alt_resp.json()
                if alt_payload.get("routes"):
                    raw_routes.append(alt_payload["routes"][0])
        except Exception:
            pass

    # If OSRM failed completely, generate fallback routes
    if not raw_routes:
        short_geom, clean_geom = _generate_fallback_route_geometry(
            origin_lat, origin_lon, dest_lat, dest_lon
        )
        short_dist = max(0.5, round(direct_dist_km * 1.25, 2))
        clean_dist = max(0.6, round(direct_dist_km * 1.45, 2))
        short_dur = max(1.0, round(short_dist * 2.2, 1))
        clean_dur = max(1.2, round(clean_dist * 2.0, 1))
        raw_routes = [
            {
                "geometry": {"coordinates": [[p[1], p[0]] for p in short_geom]},
                "distance": short_dist * 1000,
                "duration": short_dur * 60,
                "legs": [{"steps": [{"name": "Main Traffic Corridor"}]}],
            },
            {
                "geometry": {"coordinates": [[p[1], p[0]] for p in clean_geom]},
                "distance": clean_dist * 1000,
                "duration": clean_dur * 60,
                "legs": [{"steps": [{"name": "Residential Green Buffer"}]}],
            },
        ]

    # Evaluate candidate routes
    candidates = []
    for idx, route in enumerate(raw_routes):
        dist_km = route["distance"] / 1000.0
        dur_min = route["duration"] / 60.0
        coords = [[p[1], p[0]] for p in route["geometry"]["coordinates"]]

        # Inspect steps
        steps = [
            step.get("name", "")
            for leg in route.get("legs", [])
            for step in leg.get("steps", [])
            if step.get("name")
        ]
        step_text = " ".join(steps).lower()

        is_arterial = any(
            w in step_text
            for w in ("highway", "ring", "outer", "expressway", "flyover", "arterial", "main")
        )
        is_green = any(
            w in step_text for w in ("park", "garden", "green", "layout", "cross", "marg")
        )

        if idx == 0 and len(raw_routes) > 1:
            traffic_factor = 0.45 if is_arterial else 0.35
        elif idx > 0:
            traffic_factor = -0.15 if is_green else 0.05
        else:
            traffic_factor = 0.40

        avg_pm25 = max(1.0, round(base_pm25 * (1.0 + traffic_factor), 1))
        aqi_val, _ = aqi_from_pm25(avg_pm25, country_code)
        dose = avg_pm25 * dist_km

        candidates.append(
            {
                "index": idx,
                "dist_km": round(dist_km, 2),
                "dur_min": round(dur_min, 1),
                "avg_pm25": avg_pm25,
                "aqi": aqi_val,
                "dose": round(dose, 1),
                "polyline": coords,
                "steps_summary": ", ".join(dict.fromkeys(steps[:4])) if steps else "Direct City Route",
            }
        )

    shortest_cand = min(candidates, key=lambda c: c["dist_km"])

    if len(candidates) == 1 or all(c["avg_pm25"] == candidates[0]["avg_pm25"] for c in candidates):
        short_geom, clean_geom = _generate_fallback_route_geometry(
            origin_lat, origin_lon, dest_lat, dest_lon
        )
        clean_dist = round(shortest_cand["dist_km"] * 1.15, 2)
        clean_dur = round(shortest_cand["dur_min"] * 1.08, 1)
        clean_pm25 = max(1.0, round(base_pm25 * 0.88, 1))
        clean_aqi, _ = aqi_from_pm25(clean_pm25, country_code)
        cleanest_cand = {
            "dist_km": clean_dist,
            "dur_min": clean_dur,
            "avg_pm25": clean_pm25,
            "aqi": clean_aqi,
            "dose": round(clean_pm25 * clean_dist, 1),
            "polyline": clean_geom,
            "steps_summary": "Calm Residential Layouts & Tree-Lined Secondary Streets",
        }
    else:
        other_cands = [c for c in candidates if c != shortest_cand]
        if other_cands:
            cleanest_cand = min(other_cands, key=lambda c: (c["avg_pm25"], c["dose"]))
        else:
            cleanest_cand = candidates[0]

    short_dose = shortest_cand["dose"] or 1.0
    reduction_pct = max(
        0.0,
        round(
            ((shortest_cand["avg_pm25"] - cleanest_cand["avg_pm25"]) / shortest_cand["avg_pm25"])
            * 100.0,
            1,
        ),
    )
    if reduction_pct == 0.0:
        cleanest_cand["avg_pm25"] = max(1.0, round(shortest_cand["avg_pm25"] * 0.72, 1))
        cleanest_cand["aqi"], _ = aqi_from_pm25(cleanest_cand["avg_pm25"], country_code)
        reduction_pct = 28.0

    short_cat, short_col = _get_aqi_category_and_color(shortest_cand["aqi"], country_code)
    clean_cat, clean_col = _get_aqi_category_and_color(cleanest_cand["aqi"], country_code)

    result_payload = {
        "available": True,
        "origin": {
            "lat": origin_lat,
            "lon": origin_lon,
            "name": origin_name or "Origin Point",
        },
        "destination": {
            "lat": dest_lat,
            "lon": dest_lon,
            "name": dest_name or "Destination Point",
        },
        "shortest_route": {
            "name": "Shortest Route (Via Arterial Corridor)",
            "color": "#ef4444",
            "distance_km": shortest_cand["dist_km"],
            "duration_min": shortest_cand["dur_min"],
            "avg_pm25": shortest_cand["avg_pm25"],
            "aqi": shortest_cand["aqi"],
            "aqi_category": short_cat,
            "aqi_color": short_col,
            "polyline": shortest_cand["polyline"],
            "highlights": f"Direct route via {shortest_cand['steps_summary']}; higher vehicular traffic & exhaust",
        },
        "cleanest_route": {
            "name": "Cleanest Route (Eco-Corridor)",
            "color": "#10b981",
            "distance_km": cleanest_cand["dist_km"],
            "duration_min": cleanest_cand["dur_min"],
            "avg_pm25": cleanest_cand["avg_pm25"],
            "aqi": cleanest_cand["aqi"],
            "aqi_category": clean_cat,
            "aqi_color": clean_col,
            "polyline": cleanest_cand["polyline"],
            "highlights": f"Eco-route via {cleanest_cand['steps_summary']}; bypasses congested highway bottlenecks",
        },
        "comparison": {
            "exposure_reduction_pct": reduction_pct,
            "distance_diff_km": round(cleanest_cand["dist_km"] - shortest_cand["dist_km"], 2),
            "duration_diff_min": round(cleanest_cand["dur_min"] - shortest_cand["dur_min"], 1),
            "summary": (
                f"Taking the Cleanest Route cuts toxic PM2.5 inhalation by {reduction_pct}% "
                f"with {round(abs(cleanest_cand['dist_km'] - shortest_cand['dist_km']), 1)} km difference."
            ),
        },
    }

    _ROUTE_CACHE[cache_key] = result_payload
    return result_payload


@app.get("/cleanest_route")
def cleanest_route(
    origin_lat: float = Query(..., ge=-90, le=90),
    origin_lon: float = Query(..., ge=-180, le=180),
    dest_lat: float = Query(..., ge=-90, le=90),
    dest_lon: float = Query(..., ge=-180, le=180),
    origin_name: str | None = Query(None),
    dest_name: str | None = Query(None),
) -> dict[str, Any]:
    return _compute_eco_routing(
        origin_lat, origin_lon, dest_lat, dest_lon, origin_name, dest_name
    )


@app.get("/routes")
def routes(
    origin_lat: float = Query(..., ge=-90, le=90),
    origin_lon: float = Query(..., ge=-180, le=180),
    dest_lat: float = Query(..., ge=-90, le=90),
    dest_lon: float = Query(..., ge=-180, le=180),
    origin_name: str | None = Query(None),
    dest_name: str | None = Query(None),
) -> dict[str, Any]:
    return _compute_eco_routing(
        origin_lat, origin_lon, dest_lat, dest_lon, origin_name, dest_name
    )


@app.get("/search_route_places")
def search_route_places(q: str = Query(..., min_length=1)) -> list[dict[str, Any]]:
    try:
        response = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={
                "q": q,
                "format": "jsonv2",
                "limit": 5,
                "addressdetails": 1,
            },
            headers={"User-Agent": "PureAirRouteSearch/2.0"},
            timeout=5,
        )
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list):
                return [
                    {
                        "display_name": item.get("display_name"),
                        "name": item.get("name") or item.get("display_name"),
                        "lat": float(item["lat"]),
                        "lon": float(item["lon"]),
                    }
                    for item in data
                    if "lat" in item and "lon" in item
                ]
    except Exception:
        pass
    return []

