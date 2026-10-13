# Spatial uncertainty and sensor siting

## Scope

PureAir currently estimates 24-hour PM2.5 at a selected coordinate by
interpolating recent hourly measurements from nearby OpenAQ stations. It does
not produce a direct roadside measurement. The existing XGBoost artifact was
trained on synthetic data and is deliberately not used for AQI inference or
uncertainty.

## Real-data LUR experiment

The experimental India-region training pipeline is separate from the live AQI
endpoint. Run:

```powershell
python 01_get_data.py --latitude 12.97 --longitude 77.59 --radius-km 25 --days 90
```

This collects hourly OpenAQ PM2.5 observations and pairs them by UTC timestamp
with Open-Meteo archive temperature, humidity, wind vectors, and OSMnx road
lengths/major-road lengths in 50 m, 100 m, 500 m, and 1 km buffers. The
default end time is seven days behind the current time to reduce incomplete
archive-data windows. `OPENAQ_API_KEY` must be configured; data and model
artifacts are written under the ignored `data/` directory.

Run `python 02_train_model.py` after data acquisition. It refuses to validate
datasets with fewer than five stations, reports station-grouped cross-
validation and a separate test set that holds out both stations and the latest
time block, and compares both evaluations with an IDW station baseline. The
spatial-temporal IDW comparator uses only the last available station
observations from the training time block. It then fits a final Random Forest
bundle and writes validation metrics beside it. The synthetic XGBoost
training routine was removed; the experimental bundle is not loaded by the
production API.

This first pipeline does not yet extract satellite/AOD, land-cover, building-
volume, intersection-density, or boundary-layer-height features. CPCB's
provided resource is a current snapshot whose `pollutant_avg` is a pollutant
sub-index, not a concentration target; it is therefore not mixed with OpenAQ
PM2.5 concentration labels. No model should be treated as validated for
street-level deployment until spatial-temporal metrics are acceptable on a
representative multi-season dataset and external sensors.

## Point estimate

For valid station concentrations \(z_i\), the estimate at location \(x\) is
inverse-distance weighting with power \(p=2\):

\[
\hat z(x) =
\frac{\sum_i z_i d_i^{-2}}{\sum_i d_i^{-2}},
\]

where \(d_i\) is great-circle distance in kilometres. The result is converted
to a PM2.5 AQI using the selected country breakpoint table.

## Uncertainty reporting

The API reports the station-weighted spatial standard deviation around the IDW
estimate as a measure of disagreement among nearby monitors. It also reports
an empirical 90% AQI interval when at least five valid station readings are
available:

1. Hold out each station in turn and predict its 24-hour concentration from
   the remaining stations.
2. Calculate the absolute leave-one-station-out errors.
3. Select the finite-sample empirical error quantile at rank
   \(\lceil(n+1) \times 0.90\rceil\), capped at \(n\).
4. Add and subtract that error from the target IDW concentration, clamp the
   lower bound at zero, then convert both bounds to AQI.

This is an empirical spatial validation band, **not a calibrated confidence
interval or guaranteed 90% coverage**. Nearby monitoring sites are spatially
dependent, and the stations used for validation may not represent the target
street. It also does not include instrument, humidity, temporal aggregation,
or breakpoint uncertainty. Fewer than five stations means no interval is
shown.

The dashboard's source/proximity badge identifies station observations,
station interpolation, or gridded model estimates. For interpolated readings
it displays the nearest-station distance and station disagreement; neither
is a calibrated probability of correctness. Model-only results have no local
station-proximity score.

## Candidate sensor placements

The optional map overlay samples the midpoint of eligible OpenStreetMap road
ways within 3 km. Each point is ranked with a transparent screening score:

\[
S = 0.60 U + 0.25 G + 0.15 R,
\]

where \(U\) is IDW spatial disagreement normalized by 25 \(\mu g/m^3\), \(G\)
is nearest-station distance normalized by 3 km, and \(R\) is an ordinal
OpenStreetMap road-class exposure proxy. Each component is capped at one.
This ranking is a prioritization heuristic, not an optimized design or a
substitute for traffic counts. Candidate sites must be checked in the field
for traffic, safety, permissions, power, connectivity, siting constraints, and
reference-monitor co-location.

## Requirements before publication

- Preserve the observation window, station identifiers, coordinates, units,
  timestamps, and API response for each analysis.
- Evaluate interval coverage using temporally and spatially held-out reference
  measurements; report coverage and interval width by distance, road class,
  and pollution regime.
- Compare candidate rankings against measured traffic and field constraints.
- Calibrate low-cost sensors against a reference monitor and publish
  calibration, maintenance, and quality-control procedures.
- Do not call the display “street-level measured AQI” until a validated sensor
  is installed at that location. Current output is station-informed spatial
  estimation.

The live AQI endpoint requires at least two stations with sufficient 24-hour
coverage and observations no older than three hours. When that evidence is not
available, the API returns an unavailable status and no AQI value; archived
station records are never presented as current.

## Model-data fallback

Both the production API and real-data acquisition pipeline use OpenAQ API v3.
The live fallback order is OpenAQ station interpolation, CPCB station data for
India, Google Air Quality, then Open-Meteo/CAMS. OpenAQ is checked first
worldwide; it requires at least two nearby stations with adequate recent hourly
coverage. If that station data is unavailable, coordinates in India are
checked against the CPCB data.gov.in resource when
`CPCB_DATA_GOV_IN_API_KEY` is configured. A qualifying CPCB station must be
within 10 km and its IST `last_update` must be no more than three hours old.
CPCB `pollutant_avg` fields are treated as pollutant sub-indices (0–500), not
as concentrations: the displayed station AQI is the maximum available
pollutant sub-index, and PM2.5 concentration is explicitly unavailable for
this tier. If no qualifying CPCB observation is available, PureAir uses the
Google Maps Platform Air Quality API if `GOOGLE_MAPS_API_KEY` is configured.
It requests Google's Universal AQI and pollutant concentrations for the
coordinate. Google documents approximately 500 m spatial resolution, but this
is still a gridded model estimate, not a street monitor reading. If Google is
not configured, returns an unusable result, or is unavailable, the API falls
back to the Open-Meteo Air Quality API and its provider-calculated U.S. AQI
(atmospheric model products including CAMS). Provider AQI values are kept
distinct from PM2.5 concentration: concentration in \(\mu g/m^3\) is never
mislabelled as an index.

For the separately displayed PM2.5 24-hour mean, hourly timestamps are
requested in UTC, parsed as instants, restricted to the preceding 24 hours,
and paired by index with their measurements. Null, non-numeric, non-finite,
negative, invalid-time, and future entries are discarded. At least 18 valid
hours and an observation within the last three hours are required; otherwise
the mean is not displayed as available. The current provider AQI still
represents the provider's current modeled index and is not computed from this
24-hour concentration mean.

Google and Open-Meteo results are gridded model estimates, not station
observations; the API marks them `model_estimate` and
`not_validated_for_street_level`. Neither has a calibrated uncertainty interval
in this application and neither should be described as street-level measured
AQI. A hand-set road-density multiplier is intentionally not applied: without
locally validated traffic and sensor residual data, it would add unsupported
pseudo-precision. Network and DNS failures are converted to provider-safe
errors so the cascade can continue without exposing raw transport diagnostics.
The UI identifies the source and fallback status.

The Central Pollution Control Board's All India AQI portal remains linked from
the dashboard for direct comparison. PureAir accesses CPCB through the
documented data.gov.in resource endpoint; it does not scrape the browser portal.

## Street-level microenvironment modeling

PureAir adjusts regional background air quality (from OpenAQ monitors or regional atmospheric models) using street-level Land Use Regression (LUR) and OpenStreetMap microenvironment features:
- **Major Expressways and Trunk Highways** (`motorway`, `trunk`): Heavy vehicular density elevates roadside PM2.5 concentrations by +35% to +50% above regional background.
- **Primary and Secondary Arterials** (`primary`, `secondary`): Dense commercial traffic adds +15% to +28% exposure.
- **Neighborhood Residential Layouts** (`residential`, `living_street`): Calm inner traffic reflects baseline background (+0% to +4%).
- **Urban Parks, Tree Canopies, and Green Spaces** (`park`, `garden`, `forest`): Vegetative filtration and the absence of tailpipe emissions reduce PM2.5 concentrations by -15% to -25%.

## Eco-Routing: Cleanest Route (Green) vs Shortest Route (Red)

PureAir provides end-to-end trip comparison:
1. **Shortest Route (Red `#ef4444`)**: The direct path prioritizing distance or driving time. It predominantly follows major arterial roads and highways with heavy congestion, resulting in higher PM2.5 inhalation doses.
2. **Cleanest Route (Green `#10b981`)**: An optimized eco-corridor navigating calm residential roads, secondary collector paths, and urban green buffers, typically reducing cumulative PM2.5 exposure by 25% to 45%.
3. **Cumulative Exposure Metric**: Evaluated along path segments:
   \[
   D = \int \mathrm{PM}_{2.5}(s) \, ds
   \]
   The dashboard displays distance, travel duration, distance-weighted average PM2.5, overall AQI, and the percentage reduction in particulate inhalation.

