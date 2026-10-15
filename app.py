import json
import math
import requests
import pandas as pd
import streamlit as st
import folium
import plotly.express as px

from shapely.geometry import (
    Polygon, MultiPolygon, LineString, MultiLineString,
    Point, box, mapping
)
from shapely.ops import unary_union, transform as shapely_transform
from pyproj import Transformer
from streamlit_folium import st_folium


# ============================================================
# LADYWOOD ENVIRONMENTAL RISK & REHABILITATION DASHBOARD
# Data: Birmingham City Council ArcGIS services
# Air-quality data and calculations are intentionally excluded.
# ============================================================

st.set_page_config(
    page_title="LADYWOOD | Environmental Dashboard",
    page_icon="🌿",
    layout="wide"
)

st.title("LADYWOOD")
st.subheader("Environmental Risk & Rehabilitation Dashboard")
st.caption(
    "Ladywood, Birmingham, England | LSOA-level environmental assessment"
)

# ----------------------- CONFIGURATION -----------------------

CRVA_BASE = (
    "https://maps.birmingham.gov.uk/server/rest/services/"
    "CRVA/CRVA_2025/MapServer"
)

BROWNFIELD_BASE = (
    "https://maps.birmingham.gov.uk/server/rest/services/"
    "planning/HELAA/MapServer"
)

ROADS_BASE = (
    "https://maps.birmingham.gov.uk/server/rest/services/"
    "mybrummap/mybrummap_Transportation/MapServer"
)

# Known layer IDs from the project research.
LAYERS = {
    "ward": (CRVA_BASE, 14),
    "lsoa": (CRVA_BASE, 17),
    "flood_zone_3": (CRVA_BASE, 109),
    "surface_flood": (CRVA_BASE, 1546),
    "greenspace": (CRVA_BASE, 1549),
    "woodland": (CRVA_BASE, 1552),
    "buildings": (CRVA_BASE, 1550),
    "temperature": (CRVA_BASE, 18),
    "climate_zones": (CRVA_BASE, 4),
    "brownfield": (BROWNFIELD_BASE, 38),
    "roads": (ROADS_BASE, 25),
}

ARCGIS_TIMEOUT = 45
PAGE_SIZE = 1000
AREA_EPSG = "EPSG:27700"
WEB_EPSG = "EPSG:4326"

TO_WEB = Transformer.from_crs(
    AREA_EPSG, WEB_EPSG, always_xy=True
).transform

TO_AREA = Transformer.from_crs(
    WEB_EPSG, AREA_EPSG, always_xy=True
).transform


# ----------------------- HELPERS -----------------------------

def arcgis_url(service, layer_id):
    return f"{service}/{layer_id}/query"


def signed_area(coords):
    """Signed area of a ring; used to help identify polygon rings."""
    total = 0.0
    for i in range(len(coords)):
        x1, y1 = coords[i][0], coords[i][1]
        x2, y2 = coords[(i + 1) % len(coords)][0], coords[(i + 1) % len(coords)][1]
        total += x1 * y2 - x2 * y1
    return total / 2.0


def esri_geometry_to_shape(geometry):
    """
    Convert common Esri JSON geometries to Shapely.
    Polygon rings are grouped by containment to preserve holes
    where possible. Coordinates are requested in EPSG:27700.
    """
    if not geometry:
        return None

    if "rings" in geometry:
        rings = []
        for ring in geometry["rings"]:
            if len(ring) >= 4:
                try:
                    p = Polygon(ring)
                    if not p.is_empty and p.area > 0:
                        rings.append(p)
                except Exception:
                    pass

        if not rings:
            return None

        # A ring inside another ring is a hole or nested ring.
        # Assign each ring to the smallest containing outer ring.
        rings = sorted(rings, key=lambda p: p.area, reverse=True)
        outers = []
        holes_by_outer = {}

        for ring in rings:
            container = None
            container_area = None
            point = ring.representative_point()

            for outer in outers:
                if outer.contains(point):
                    if container_area is None or outer.area < container_area:
                        container = outer
                        container_area = outer.area

            if container is None:
                outers.append(ring)
                holes_by_outer[id(ring)] = []
            else:
                holes_by_outer.setdefault(id(container), []).append(
                    list(ring.exterior.coords)
                )

        polygons = []
        for outer in outers:
            holes = holes_by_outer.get(id(outer), [])
            try:
                polygon = Polygon(
                    list(outer.exterior.coords),
                    holes=holes
                )
                if not polygon.is_empty:
                    polygons.append(polygon)
            except Exception:
                polygons.append(outer)

        if not polygons:
            return None

        result = unary_union(polygons)
        return result if not result.is_empty else None

    if "paths" in geometry:
        lines = []
        for path in geometry["paths"]:
            if len(path) >= 2:
                try:
                    lines.append(LineString(path))
                except Exception:
                    pass

        if not lines:
            return None
        return lines[0] if len(lines) == 1 else MultiLineString(lines)

    if "x" in geometry and "y" in geometry:
        return Point(geometry["x"], geometry["y"])

    return None


def request_json(url, params):
    response = requests.get(
        url,
        params=params,
        timeout=ARCGIS_TIMEOUT
    )
    response.raise_for_status()
    payload = response.json()

    if "error" in payload:
        error = payload["error"]
        raise RuntimeError(
            error.get("message", "ArcGIS service returned an error")
        )

    return payload


@st.cache_data(ttl=3600, show_spinner=False)
def get_layer_metadata(service, layer_id):
    url = f"{service}/{layer_id}"
    return request_json(url, {"f": "json"})


@st.cache_data(ttl=3600, show_spinner=False)
def query_features(service, layer_id, where="1=1", bbox_web=None):
    """
    Retrieve ArcGIS features in object-ID batches.
    This avoids resultOffset pagination, which some services
    do not support.
    """
    url = arcgis_url(service, layer_id)

    base_params = {
        "f": "json",
        "where": where,
    }

    if bbox_web is not None:
        xmin, ymin, xmax, ymax = bbox_web
        base_params.update({
            "geometry": json.dumps({
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "spatialReference": {"wkid": 4326},
            }),
            "geometryType": "esriGeometryEnvelope",
            "inSR": 4326,
            "spatialRel": "esriSpatialRelIntersects",
        })

    # First ask ArcGIS for the matching feature IDs.
    id_params = dict(base_params)
    id_params.update({
        "returnIdsOnly": "true",
        "returnGeometry": "false",
    })

    id_result = request_json(url, id_params)
    object_ids = id_result.get("objectIds")

    if object_ids is None:
        raise RuntimeError(
            f"Layer {layer_id} did not return object IDs. "
            "Check the layer query endpoint and permissions."
        )

    if not object_ids:
        return []

    # Query manageable batches without using resultOffset.
    all_features = []
    batch_size = 200

    for start in range(0, len(object_ids), batch_size):
        batch_ids = object_ids[start:start + batch_size]

        feature_params = {
            "f": "json",
            "objectIds": ",".join(map(str, batch_ids)),
            "outFields": "*",
            "returnGeometry": "true",
            "outSR": 27700,
        }

        payload = request_json(url, feature_params)
        all_features.extend(payload.get("features", []))

    return all_features

def features_to_records(features):
    records = []

    for feature in features:
        attrs = feature.get("attributes", {}) or {}
        geom = esri_geometry_to_shape(feature.get("geometry"))

        if geom is None or geom.is_empty:
            continue

        record = dict(attrs)
        record["_geometry"] = geom
        records.append(record)

    return records


def first_value(record, candidates, default=None):
    """Find a field without assuming the source's exact capitalisation."""
    lower_map = {str(k).lower(): k for k in record.keys()}

    for candidate in candidates:
        key = lower_map.get(candidate.lower())
        if key is not None:
            value = record.get(key)
            if value not in (None, ""):
                return value

    return default


def numeric(value):
    try:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def safe_union(geometries):
    valid = [
        g for g in geometries
        if g is not None and not g.is_empty
    ]
    if not valid:
        return None
    try:
        return unary_union(valid)
    except Exception:
        cleaned = []
        for geom in valid:
            try:
                cleaned.append(geom.buffer(0))
            except Exception:
                pass
        return unary_union(cleaned) if cleaned else None


def intersection_area_pct(feature_geometry, study_geometry, denominator):
    if (
        feature_geometry is None
        or feature_geometry.is_empty
        or denominator <= 0
    ):
        return 0.0

    try:
        overlap = feature_geometry.intersection(study_geometry)
        return max(0.0, min(100.0, overlap.area / denominator * 100.0))
    except Exception:
        return None


def union_intersection_pct(geometries, study_geometry, denominator):
    if denominator <= 0:
        return 0.0

    combined = safe_union(geometries)
    if combined is None:
        return 0.0

    try:
        overlap = combined.intersection(study_geometry)
        return max(0.0, min(100.0, overlap.area / denominator * 100.0))
    except Exception:
        return None


def clip_geometry(geometry, clipper):
    if geometry is None or clipper is None:
        return None
    try:
        result = geometry.intersection(clipper)
        if result.is_empty:
            return None
        return result
    except Exception:
        return None


def project_to_web(geometry):
    if geometry is None or geometry.is_empty:
        return None
    try:
        return shapely_transform(TO_WEB, geometry)
    except Exception:
        return None


def normalise_series(series):
    """Min-max scale valid values to 0–1 for relative comparisons."""
    values = pd.to_numeric(series, errors="coerce")
    valid = values.dropna()

    result = pd.Series(float("nan"), index=series.index, dtype=float)

    if valid.empty:
        return result

    minimum = valid.min()
    maximum = valid.max()

    if maximum == minimum:
        result.loc[valid.index] = 0.5
    else:
        result.loc[valid.index] = (
            (valid - minimum) / (maximum - minimum)
        )

    return result


def risk_colour(risk_class):
    value = str(risk_class or "").strip().lower()

    if "high" in value and "medium" not in value:
        return "#d73027"
    if "medium" in value or "moderate" in value:
        return "#fdae61"
    if "low" in value:
        return "#1a9850"
    return "#808080"


def display_table(df, columns=None):
    view = df.copy()
    if columns:
        view = view[[c for c in columns if c in view.columns]]
    st.dataframe(view, use_container_width=True, hide_index=True)


# ------------------- LOAD LADYWOOD BOUNDARY ------------------

@st.cache_data(ttl=3600, show_spinner=False)
def load_study_area():
    service, layer_id = LAYERS["ward"]
    metadata = get_layer_metadata(service, layer_id)
    field_names = [
        f.get("name", "") for f in metadata.get("fields", [])
    ]

    ward_field = next(
        (f for f in field_names if f.lower() in
         ["wardnme", "wardname", "ward_name", "name"]),
        None
    )

    where = "1=1"
    if ward_field:
        where = f"{ward_field} LIKE '%Ladywood%'"

    features = query_features(service, layer_id, where=where)
    records = features_to_records(features)

    matches = []
    for record in records:
        values = " ".join(
            str(v) for k, v in record.items()
            if not k.startswith("_")
        ).lower()

        if "ladywood" in values:
            matches.append(record)

    if not matches:
        raise RuntimeError(
            "Ladywood ward boundary was not found in the council layer."
        )

    ward_geometry = safe_union(
        [r["_geometry"] for r in matches]
    )

    if ward_geometry is None or ward_geometry.is_empty:
        raise RuntimeError("Ladywood boundary geometry is empty.")

    return ward_geometry


def load_spatial_records(key, ward_geometry, bbox_web):
    service, layer_id = LAYERS[key]
    features = query_features(
        service,
        layer_id,
        bbox_web=bbox_web
    )
    records = features_to_records(features)

    clipped = []
    for record in records:
        geom = clip_geometry(record["_geometry"], ward_geometry)
        if geom is None:
            continue
        new_record = dict(record)
        new_record["_geometry"] = geom
        clipped.append(new_record)

    return clipped


# ---------------------- LOAD ALL DATA ------------------------

with st.spinner("Loading official Birmingham map layers..."):
    try:
        ward_geometry = load_study_area()
        ward_web = project_to_web(ward_geometry)
        ward_bounds_web = ward_web.bounds

        lsoa_records = load_spatial_records(
            "lsoa", ward_geometry, ward_bounds_web
        )

        if not lsoa_records:
            raise RuntimeError(
                "No LSOA features were returned for Ladywood."
            )

        flood3_records = load_spatial_records(
            "flood_zone_3", ward_geometry, ward_bounds_web
        )
        surface_records = load_spatial_records(
            "surface_flood", ward_geometry, ward_bounds_web
        )
        brownfield_records = load_spatial_records(
            "brownfield", ward_geometry, ward_bounds_web
        )
        green_records = load_spatial_records(
            "greenspace", ward_geometry, ward_bounds_web
        )
        woodland_records = load_spatial_records(
            "woodland", ward_geometry, ward_bounds_web
        )
        building_records = load_spatial_records(
            "buildings", ward_geometry, ward_bounds_web
        )
        temperature_records = load_spatial_records(
            "temperature", ward_geometry, ward_bounds_web
        )
        climate_zone_records = load_spatial_records(
            "climate_zones", ward_geometry, ward_bounds_web
        )
        road_records = load_spatial_records(
            "roads", ward_geometry, ward_bounds_web
        )

    except Exception as error:
        st.error(f"Unable to load the required map data: {error}")
        st.info(
            "Check the ArcGIS service URLs, layer permissions and "
            "internet connection. No-data results must not be read "
            "as zero exposure."
        )
        st.stop()


# ----------------------- LSOA METRICS ------------------------

flood3_geoms = [r["_geometry"] for r in flood3_records]
surface_geoms = [r["_geometry"] for r in surface_records]
brownfield_geoms = [r["_geometry"] for r in brownfield_records]
green_geoms = [r["_geometry"] for r in green_records]
woodland_geoms = [r["_geometry"] for r in woodland_records]
building_geoms = [r["_geometry"] for r in building_records]
temperature_geoms = [r["_geometry"] for r in temperature_records]

# Union geometry for combined flood footprint to avoid double counting.
combined_flood_geometries = flood3_geoms + surface_geoms
combined_flood_union = safe_union(combined_flood_geometries)

rows = []
lsoa_geometries = {}

for index, record in enumerate(lsoa_records):
    geom = record["_geometry"]
    area_m2 = geom.area

    if area_m2 <= 0:
        continue

    lsoa_code = first_value(
        record, ["LSOA21CD", "LSOA11CD", "LSOA_CODE", "LSOA_CODE21"],
        f"Unknown-{index + 1}"
    )
    lsoa_name = first_value(
        record, ["LSOA21NM", "LSOA11NM", "LSOA_NAME", "LSOA_NAME21"],
        str(lsoa_code)
    )

    crva_mean = numeric(first_value(record, ["MEAN"]))
    official_class = first_value(
        record,
        ["AVERAGE_RISK", "AVERAGE_RISK_CLASS", "RISK_CLASS"],
        "Not available"
    )

    flood3_pct = union_intersection_pct(
        flood3_geoms, geom, area_m2
    )
    surface_pct = union_intersection_pct(
        surface_geoms, geom, area_m2
    )

    combined_flood_pct = (
        combined_flood_union.intersection(geom).area / area_m2 * 100.0
        if combined_flood_union is not None else None
    )
    if combined_flood_pct is not None:
        combined_flood_pct = max(0.0, min(100.0, combined_flood_pct))

    brownfield_pct = union_intersection_pct(
        brownfield_geoms, geom, area_m2
    )
    greenspace_pct = union_intersection_pct(
        green_geoms, geom, area_m2
    )
    woodland_pct = union_intersection_pct(
        woodland_geoms, geom, area_m2
    )
    building_pct = union_intersection_pct(
        building_geoms, geom, area_m2
    )

    # Summarise numeric temperature attributes where available.
    temp_values = []
    for temp_record in temperature_records:
        temp_geom = temp_record["_geometry"]
        try:
            if temp_geom.intersects(geom):
                for field, value in temp_record.items():
                    if field.startswith("_"):
                        continue
                    field_lower = field.lower()
                    if (
                        "temp" in field_lower
                        or "maximum" in field_lower
                        or "mean" in field_lower
                    ):
                        val = numeric(value)
                        if val is not None:
                            temp_values.append(val)
        except Exception:
            pass

    temperature_value = (
        sum(temp_values) / len(temp_values) if temp_values else None
    )

    rows.append({
        "LSOA_Code": str(lsoa_code),
        "LSOA_Name": str(lsoa_name),
        "LSOA_Area_ha": area_m2 / 10000.0,
        "CRVA_Mean": crva_mean,
        "Official_CRVA_Risk_Class": str(official_class),
        "Flood_Zone_3_%": flood3_pct,
        "Surface_Flood_%": surface_pct,
        "Combined_Flood_Footprint_%": combined_flood_pct,
        "Brownfield_Exposure_%": brownfield_pct,
        "Greenspace_Coverage_%": greenspace_pct,
        "Woodland_Coverage_%": woodland_pct,
        "Building_Footprint_%": building_pct,
        "Temperature_Indicator": temperature_value,
    })

    lsoa_geometries[str(lsoa_code)] = geom


data = pd.DataFrame(rows)

if data.empty:
    st.error("No usable LSOA records were returned.")
    st.stop()

# Keep percentages within valid bounds.
percentage_columns = [
    "Flood_Zone_3_%",
    "Surface_Flood_%",
    "Combined_Flood_Footprint_%",
    "Brownfield_Exposure_%",
    "Greenspace_Coverage_%",
    "Woodland_Coverage_%",
    "Building_Footprint_%",
]
for column in percentage_columns:
    data[column] = pd.to_numeric(data[column], errors="coerce").clip(0, 100)

# A zero means the queried mapped layer had no overlap in that LSOA.
# A failed/unavailable layer is reported separately rather than silently
# replacing every missing result with zero.
layer_status = {
    "Flood Zone 3": len(flood3_records),
    "Surface-water flooding": len(surface_records),
    "Brownfield": len(brownfield_records),
    "Green space": len(green_records),
    "Woodland": len(woodland_records),
    "Buildings": len(building_records),
    "Temperature": len(temperature_records),
    "Roads": len(road_records),
}


# ------------------- INVESTIGATION PRIORITY ------------------

# Relative indicators are used only for ranking, not as official ratings.
# The priority score is not displayed as a council risk class.
data["CRVA_Relative"] = normalise_series(data["CRVA_Mean"])
data["Flood_Relative"] = normalise_series(
    data["Combined_Flood_Footprint_%"]
)
data["Brownfield_Relative"] = normalise_series(
    data["Brownfield_Exposure_%"]
)
data["Heat_Relative"] = normalise_series(
    data["Temperature_Indicator"]
)

# Low green-space coverage is treated as a potential pressure.
data["Greenspace_Deficit_Relative"] = (
    1 - normalise_series(data["Greenspace_Coverage_%"])
)

driver_columns = {
    "CRVA": "CRVA_Relative",
    "Flood exposure": "Flood_Relative",
    "Brownfield exposure": "Brownfield_Relative",
    "Temperature indicator": "Heat_Relative",
    "Green-space deficit": "Greenspace_Deficit_Relative",
}

# A driver is only assigned when at least one indicator is available.
def dominant_driver(row):
    available = {
        label: row[column]
        for label, column in driver_columns.items()
        if pd.notna(row.get(column))
    }
    if not available:
        return "Insufficient data"
    return max(available, key=available.get)


data["Dominant_Environmental_Driver"] = data.apply(
    dominant_driver, axis=1
)

priority_components = list(driver_columns.values())
data["__Priority_Internal"] = data[priority_components].mean(
    axis=1, skipna=True
)

data["Priority_Data_Indicators"] = data[priority_components].notna().sum(axis=1)

priority_table = data.sort_values(
    ["__Priority_Internal", "Priority_Data_Indicators"],
    ascending=[False, False],
    na_position="last"
).copy()

priority_table.insert(
    0, "Investigation_Rank",
    range(1, len(priority_table) + 1)
)

# Remove internal comparison columns from visible tables.
visible_data = data.drop(
    columns=[
        "CRVA_Relative", "Flood_Relative",
        "Brownfield_Relative", "Heat_Relative",
        "Greenspace_Deficit_Relative",
        "__Priority_Internal"
    ],
    errors="ignore"
)

priority_visible = priority_table[
    [
        "Investigation_Rank",
        "LSOA_Code",
        "LSOA_Name",
        "Official_CRVA_Risk_Class",
        "Dominant_Environmental_Driver",
        "Priority_Data_Indicators",
    ]
].copy()

assessment_table = data[
    [
        "LSOA_Code",
        "LSOA_Name",
        "CRVA_Mean",
        "Official_CRVA_Risk_Class",
        "Combined_Flood_Footprint_%",
        "Brownfield_Exposure_%",
        "Dominant_Environmental_Driver",
    ]
].copy()


# ---------------------- ROADS TABLE ---------------------------

def get_road_name(attrs):
    return first_value(
        attrs,
        [
            "ROAD_NAME", "ROADNAME", "STREET_NAME", "STREETNAME",
            "BRUM_ROAD_", "NAME", "ROAD", "FULLNAME"
        ],
        "Name not supplied"
    )


def get_road_class(attrs):
    return first_value(
        attrs,
        [
            "ROAD_CLASS", "ROADCLASS", "CLASSIFICATION",
            "BRUM_CLASS", "FUNCTIONAL_CLASS", "ROAD_TYPE",
            "ROADTYPE", "CLASS"
        ],
        "Class not supplied"
    )


road_rows = []

for record in road_records:
    road_geom = record["_geometry"]
    if road_geom is None or road_geom.is_empty:
        continue

    attrs = {
        k: v for k, v in record.items()
        if not k.startswith("_")
    }

    # Associate the road with the nearest/intersected Ladywood LSOA.
    best_code = None
    best_name = None
    best_distance = float("inf")
    road_length_by_lsoa = 0.0

    for lsoa_code, lsoa_geom in lsoa_geometries.items():
        try:
            distance = road_geom.distance(lsoa_geom)

            if distance < best_distance:
                best_distance = distance
                best_code = lsoa_code
                best_name = str(
                    data.loc[
                        data["LSOA_Code"] == lsoa_code,
                        "LSOA_Name"
                    ].iloc[0]
                )

            if road_geom.intersects(lsoa_geom):
                intersection = road_geom.intersection(lsoa_geom)
                road_length_by_lsoa += intersection.length
        except Exception:
            continue

    road_rows.append({
        "Road_Name": get_road_name(attrs),
        "Road_Class": get_road_class(attrs),
        "Associated_LSOA_Code": best_code or "Not determined",
        "Associated_LSOA_Name": best_name or "Not determined",
        "Nearest_LSOA_Distance_m": (
            round(best_distance, 1)
            if math.isfinite(best_distance) else None
        ),
        "Road_Length_Within_LSOA_m": (
            round(road_length_by_lsoa, 1)
            if road_length_by_lsoa > 0 else None
        ),
    })

roads_table = pd.DataFrame(road_rows)


# -------------------------- OVERVIEW --------------------------

st.markdown("---")
st.header("Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("LSOAs assessed", f"{len(data):,}")
c2.metric(
    "Ladywood area",
    f"{ward_geometry.area / 10000:,.1f} ha"
)
c3.metric(
    "Mapped brownfield",
    f"{safe_union(brownfield_geoms).area / 10000:,.1f} ha"
    if safe_union(brownfield_geoms) is not None else "No data"
)
c4.metric(
    "Mapped Flood Zone 3",
    f"{safe_union(flood3_geoms).area / 10000:,.1f} ha"
    if safe_union(flood3_geoms) is not None else "No data"
)

st.caption(
    "Area figures are calculated from retrieved mapped geometries clipped "
    "to the Ladywood boundary. They are not annual probabilities or "
    "predictions of future events."
)

with st.expander("Data availability and checks"):
    st.write(
        "Number of clipped features returned by each source layer. "
        "A count of zero may indicate no mapped features or a service/coverage "
        "issue; it should be checked before drawing conclusions."
    )
    st.dataframe(
        pd.DataFrame(
            [{"Layer": k, "Features returned": v}
             for k, v in layer_status.items()]
        ),
        use_container_width=True,
        hide_index=True
    )


# ---------------------- INTERACTIVE MAP -----------------------

st.markdown("---")
st.header("Interactive Ladywood environmental map")

map_options = st.multiselect(
    "Choose map layers",
    [
        "LSOA boundaries",
        "Flood Zone 3",
        "Surface-water flooding",
        "Brownfield sites",
        "Green space",
        "Woodland",
        "Buildings",
        "Roads",
        "LSOA priority markers",
    ],
    default=[
        "LSOA boundaries",
        "Flood Zone 3",
        "Surface-water flooding",
        "Brownfield sites",
        "Green space",
        "Roads",
        "LSOA priority markers",
    ]
)

ward_web = project_to_web(ward_geometry)
centre = ward_web.centroid

m = folium.Map(
    location=[centre.y, centre.x],
    zoom_start=13,
    tiles="OpenStreetMap",
    control_scale=True
)

folium.GeoJson(
    mapping(ward_web),
    name="Ladywood ward boundary",
    style_function=lambda feature: {
        "color": "#111111",
        "weight": 3,
        "fill": False,
    },
    tooltip="Ladywood ward boundary"
).add_to(m)


def add_polygon_layer(records, label, colour, fill_opacity=0.25):
    feature_collection = {"type": "FeatureCollection", "features": []}

    for record in records:
        web_geom = project_to_web(record["_geometry"])
        if web_geom is None:
            continue
        feature_collection["features"].append({
            "type": "Feature",
            "geometry": mapping(web_geom),
            "properties": {},
        })

    if feature_collection["features"]:
        folium.GeoJson(
            feature_collection,
            name=label,
            style_function=lambda feature, c=colour, o=fill_opacity: {
                "color": c,
                "weight": 1,
                "fillColor": c,
                "fillOpacity": o,
            },
            tooltip=label,
        ).add_to(m)


if "Flood Zone 3" in map_options:
    add_polygon_layer(
        flood3_records, "Flood Zone 3", "#2878b5", 0.30
    )

if "Surface-water flooding" in map_options:
    add_polygon_layer(
        surface_records, "Surface-water flooding", "#41b6c4", 0.30
    )

if "Brownfield sites" in map_options:
    add_polygon_layer(
        brownfield_records, "Brownfield sites", "#8c510a", 0.35
    )

if "Green space" in map_options:
    add_polygon_layer(
        green_records, "Green space", "#4daf4a", 0.25
    )

if "Woodland" in map_options:
    add_polygon_layer(
        woodland_records, "Woodland", "#006837", 0.35
    )

if "Buildings" in map_options:
    add_polygon_layer(
        building_records, "Buildings", "#777777", 0.20
    )

if "LSOA boundaries" in map_options:
    lsoa_fc = {"type": "FeatureCollection", "features": []}

    for record in lsoa_records:
        web_geom = project_to_web(record["_geometry"])
        if web_geom is None:
            continue

        code = str(first_value(
            record, ["LSOA21CD", "LSOA11CD"], "Unknown"
        ))
        name = str(first_value(
            record, ["LSOA21NM", "LSOA11NM"], code
        ))

        lsoa_fc["features"].append({
            "type": "Feature",
            "geometry": mapping(web_geom),
            "properties": {"LSOA": name, "Code": code},
        })

    folium.GeoJson(
        lsoa_fc,
        name="LSOA boundaries",
        style_function=lambda feature: {
            "color": "#555555",
            "weight": 1,
            "fillOpacity": 0.02,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=["LSOA", "Code"],
            aliases=["LSOA:", "Code:"]
        )
    ).add_to(m)

if "Roads" in map_options:
    road_fc = {"type": "FeatureCollection", "features": []}

    for record in road_records:
        web_geom = project_to_web(record["_geometry"])
        if web_geom is None:
            continue
        attrs = {
            k: v for k, v in record.items()
            if not k.startswith("_")
        }
        road_fc["features"].append({
            "type": "Feature",
            "geometry": mapping(web_geom),
            "properties": {
                "Road": str(get_road_name(attrs)),
                "Class": str(get_road_class(attrs)),
            },
        })

    if road_fc["features"]:
        folium.GeoJson(
            road_fc,
            name="Roads",
            style_function=lambda feature: {
                "color": "#333333",
                "weight": 2,
                "opacity": 0.75,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["Road", "Class"],
                aliases=["Road:", "Class:"]
            )
        ).add_to(m)

if "LSOA priority markers" in map_options:
    for _, row in priority_table.iterrows():
        code = row["LSOA_Code"]
        geom = lsoa_geometries.get(code)
        if geom is None:
            continue

        centroid = project_to_web(geom.representative_point())
        colour = risk_colour(row["Official_CRVA_Risk_Class"])

        folium.CircleMarker(
            location=[centroid.y, centroid.x],
            radius=7,
            color=colour,
            fill=True,
            fill_color=colour,
            fill_opacity=0.9,
            tooltip=f"{row['LSOA_Name']} | CRVA: "
                    f"{row['Official_CRVA_Risk_Class']}",
            popup=folium.Popup(
                f"<b>{row['LSOA_Name']}</b><br>"
                f"LSOA code: {code}<br>"
                f"Official CRVA class: "
                f"{row['Official_CRVA_Risk_Class']}<br>"
                f"CRVA mean: {row['CRVA_Mean']}<br>"
                f"Combined mapped flood footprint: "
                f"{row['Combined_Flood_Footprint_%']:.2f}%<br>"
                f"Brownfield exposure: "
                f"{row['Brownfield_Exposure_%']:.2f}%<br>"
                f"Dominant driver: "
                f"{row['Dominant_Environmental_Driver']}",
                max_width=360
            )
        ).add_to(m)

folium.LayerControl(collapsed=False).add_to(m)

st_folium(
    m,
    use_container_width=True,
    height=600,
    returned_objects=[]
)

st.caption(
    "Map marker colours represent the official CRVA category when "
    "available. Grey means the category was not recognised. They do not "
    "represent the separate project investigation ranking."
)


# ------------------- INDIVIDUAL GRAPHS ------------------------

st.markdown("---")
st.header("Environmental analysis by LSOA")

def bar_chart(df, x, y, title, y_label):
    chart_df = df[[x, y]].copy()
    chart_df[y] = pd.to_numeric(chart_df[y], errors="coerce")
    chart_df = chart_df.dropna(subset=[y]).sort_values(y, ascending=False)

    if chart_df.empty:
        st.warning(f"No valid values are available for {title}.")
        return

    fig = px.bar(
        chart_df,
        x=x,
        y=y,
        title=title,
        labels={x: "LSOA", y: y_label},
        hover_data=[x, y],
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        margin=dict(l=10, r=10, t=60, b=120),
    )
    st.plotly_chart(fig, use_container_width=True)


st.subheader("Climate and environmental vulnerability")
bar_chart(
    data, "LSOA_Name", "CRVA_Mean",
    "Official CRVA mean by LSOA",
    "CRVA mean"
)

if data["Temperature_Indicator"].notna().any():
    bar_chart(
        data, "LSOA_Name", "Temperature_Indicator",
        "Available temperature indicator by LSOA",
        "Source temperature value"
    )
else:
    st.info(
        "No usable numeric temperature attribute was extracted from "
        "the configured temperature layer. The layer metadata must be "
        "checked before presenting a temperature chart."
    )

st.subheader("Flood exposure")
bar_chart(
    data, "LSOA_Name", "Flood_Zone_3_%",
    "Flood Zone 3 mapped coverage by LSOA",
    "LSOA area covered (%)"
)
bar_chart(
    data, "LSOA_Name", "Surface_Flood_%",
    "Surface-water flood mapped coverage by LSOA",
    "LSOA area covered (%)"
)
bar_chart(
    data, "LSOA_Name", "Combined_Flood_Footprint_%",
    "Combined mapped flood footprint by LSOA",
    "LSOA area covered (%)"
)

st.subheader("Brownfield land")
bar_chart(
    data, "LSOA_Name", "Brownfield_Exposure_%",
    "Brownfield mapped coverage by LSOA",
    "LSOA area covered (%)"
)

st.subheader("Green infrastructure")
bar_chart(
    data, "LSOA_Name", "Greenspace_Coverage_%",
    "Green-space coverage by LSOA",
    "LSOA area covered (%)"
)
bar_chart(
    data, "LSOA_Name", "Woodland_Coverage_%",
    "Woodland coverage by LSOA",
    "LSOA area covered (%)"
)

st.subheader("Buildings")
bar_chart(
    data, "LSOA_Name", "Building_Footprint_%",
    "Building footprint coverage by LSOA",
    "LSOA area covered (%)"
)


# ---------------- COMBINED COMPARISON GRAPH -------------------

st.markdown("---")
st.header("Combined environmental comparison")

comparison_options = {
    "Flood Zone 3 (%)": "Flood_Zone_3_%",
    "Surface flood (%)": "Surface_Flood_%",
    "Combined flood footprint (%)": "Combined_Flood_Footprint_%",
    "Brownfield (%)": "Brownfield_Exposure_%",
    "Green space (%)": "Greenspace_Coverage_%",
    "Woodland (%)": "Woodland_Coverage_%",
    "Buildings (%)": "Building_Footprint_%",
    "CRVA mean": "CRVA_Mean",
}

selected_comparisons = st.multiselect(
    "Indicators to compare",
    list(comparison_options.keys()),
    default=[
        "Combined flood footprint (%)",
        "Brownfield (%)",
        "Green space (%)",
        "Buildings (%)",
    ]
)

if selected_comparisons:
    compare_df = data[["LSOA_Name"]].copy()

    for label in selected_comparisons:
        column = comparison_options[label]
        compare_df[label] = normalise_series(data[column])

    compare_long = compare_df.melt(
        id_vars="LSOA_Name",
        var_name="Indicator",
        value_name="Relative value (0–1)"
    ).dropna()

    if not compare_long.empty:
        fig = px.bar(
            compare_long,
            x="LSOA_Name",
            y="Relative value (0–1)",
            color="Indicator",
            barmode="group",
            title="Relative comparison of selected environmental indicators",
        )
        fig.update_layout(
            xaxis_tickangle=-45,
            margin=dict(l=10, r=10, t=60, b=120),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "This graph uses relative normalised values to compare indicators "
        "with different units. It is not a graph of raw percentages or "
        "an official Birmingham classification."
    )
else:
    st.info("Select at least one indicator to display the comparison.")


# -------------------- EXPOSURE TABLES -------------------------

st.markdown("---")
st.header("Exposure tables")

st.subheader("Flood Zone 3 exposure — highest first")
display_table(
    data.sort_values("Flood_Zone_3_%", ascending=False),
    [
        "LSOA_Code", "LSOA_Name", "LSOA_Area_ha",
        "Flood_Zone_3_%", "Official_CRVA_Risk_Class"
    ]
)

st.subheader("Surface-water flood exposure — highest first")
display_table(
    data.sort_values("Surface_Flood_%", ascending=False),
    [
        "LSOA_Code", "LSOA_Name", "LSOA_Area_ha",
        "Surface_Flood_%", "Official_CRVA_Risk_Class"
    ]
)

st.subheader("Brownfield exposure — highest first")
display_table(
    data.sort_values("Brownfield_Exposure_%", ascending=False),
    [
        "LSOA_Code", "LSOA_Name", "LSOA_Area_ha",
        "Brownfield_Exposure_%", "Official_CRVA_Risk_Class"
    ]
)

st.subheader("Green-space coverage — lowest first")
display_table(
    data.sort_values("Greenspace_Coverage_%", ascending=True),
    [
        "LSOA_Code", "LSOA_Name",
        "Greenspace_Coverage_%", "Woodland_Coverage_%"
    ]
)


# ---------------- INVESTIGATION PRIORITY TABLE ---------------

st.markdown("---")
st.header("Investigation-priority ranking")

st.write(
    "This table ranks all assessed LSOAs for further investigation. "
    "It is a project-specific relative ranking, not an official council "
    "risk category. Areas near the top should be investigated using "
    "site evidence before any intervention is selected."
)

display_table(priority_visible)

st.download_button(
    "Download investigation-priority table",
    data=priority_visible.to_csv(index=False).encode("utf-8"),
    file_name="ladywood_investigation_priority.csv",
    mime="text/csv"
)


# ---------------- ENVIRONMENTAL CLASSIFICATION ----------------

st.markdown("---")
st.header("Environmental classification assessment")

st.write(
    "The CRVA class below is the category supplied by the council. "
    "The dominant driver is a project comparison across the available "
    "indicators; it is not a diagnosis of the cause of environmental harm."
)

display_table(
    assessment_table.sort_values(
        "CRVA_Mean", ascending=False, na_position="last"
    )
)

st.download_button(
    "Download environmental assessment table",
    data=assessment_table.to_csv(index=False).encode("utf-8"),
    file_name="ladywood_environmental_assessment.csv",
    mime="text/csv"
)


# --------------------- ROAD ANALYSIS --------------------------

st.markdown("---")
st.header("Roads and associated LSOAs")

if roads_table.empty:
    st.warning(
        "No road features were returned from the configured road layer. "
        "Check the service URL, layer ID, and whether the layer supports "
        "the expected geometry query."
    )
else:
    st.write(
        "Road-to-LSOA associations are based on the retrieved geometries. "
        "Where a road crosses multiple LSOAs, this table currently reports "
        "its nearest LSOA and the total length intersecting Ladywood LSOAs; "
        "the length is not attributed to a single LSOA."
    )
    display_table(
        roads_table.sort_values("Road_Name"),
        [
            "Road_Name", "Road_Class", "Associated_LSOA_Code",
            "Associated_LSOA_Name", "Nearest_LSOA_Distance_m",
            "Road_Length_Within_LSOA_m"
        ]
    )

    st.download_button(
        "Download road table",
        data=roads_table.to_csv(index=False).encode("utf-8"),
        file_name="ladywood_roads_and_lsoas.csv",
        mime="text/csv"
    )


# ---------------- REHABILITATION RECOMMENDATIONS --------------

st.markdown("---")
st.header("Rehabilitation and monitoring recommendations")

st.write(
    "Recommendations below are preliminary investigation prompts based "
    "on mapped indicators. They do not replace site inspections, "
    "engineering design, flood modelling, contamination testing, or "
    "consultation with Birmingham City Council."
)

for _, row in priority_table.head(10).iterrows():
    name = row["LSOA_Name"]
    driver = row["Dominant_Environmental_Driver"]

    if driver == "Flood exposure":
        recommendation = (
            "Review drainage, surface-water flow paths and sustainable "
            "drainage opportunities; validate mapped exposure on site."
        )
    elif driver == "Brownfield exposure":
        recommendation = (
            "Check the mapped site record, planning status and land history. "
            "Undertake appropriate ground investigation before proposing "
            "remediation or redevelopment."
        )
    elif driver == "Green-space deficit":
        recommendation = (
            "Review opportunities for tree planting, green corridors, "
            "permeable surfaces and accessible green space."
        )
    elif driver == "Temperature indicator":
        recommendation = (
            "Check the temperature layer's date and units; consider "
            "shade and green infrastructure if local conditions support it."
        )
    elif driver == "CRVA":
        recommendation = (
            "Review the council's CRVA supporting indicators to determine "
            "which underlying pressures require a closer assessment."
        )
    else:
        recommendation = (
            "Collect more local evidence before selecting an intervention."
        )

    with st.expander(f"{name} — {driver}"):
        st.write(recommendation)
        st.caption(
            "This is a suggested next investigation, not a confirmed "
            "engineering prescription."
        )


# ---------------- SOURCES & METHODS --------------------------

st.markdown("---")
st.header("Data sources, methods and limitations")

st.markdown(
    """
**Birmingham City Council map services**

- [CRVA 2025 MapServer](https://maps.birmingham.gov.uk/server/rest/services/CRVA/CRVA_2025/MapServer)
- [Brownfield / HELAA MapServer](https://maps.birmingham.gov.uk/server/rest/services/planning/HELAA/MapServer)
- [Transportation MapServer](https://maps.birmingham.gov.uk/server/rest/services/mybrummap/mybrummap_Transportation/MapServer)

**Interpretation and limitations**

- `CRVA_Mean` and `Official_CRVA_Risk_Class` are taken from the council's
  LSOA layer when those fields are available.
- Flood and brownfield percentages are calculated as the share of the
  clipped LSOA geometry intersecting the retrieved mapped features.
- The combined flood footprint uses a union of Flood Zone 3 and surface
  flood geometries to avoid counting overlapping mapped areas twice.
- Brownfield mapping does not prove contamination.
- Mapped flood overlap is not an annual probability of flooding.
- Road proximity is not proof of pollution or environmental damage.
- The temperature indicator is only meaningful after the layer's field
  names, units and date have been checked.
- Missing or unavailable source data must be investigated and must not
  be interpreted as zero environmental risk.
- The investigation ranking is a project-specific relative ranking and
  is not an official Birmingham City Council classification.
- The CRVA is already a composite measure. Additional layers should be
  interpreted alongside it carefully to avoid double-counting pressures.
"""
)

st.caption(
    "Dashboard built for environmental data exploration and engineering "
    "project support. Verify source layer metadata and outputs before using "
    "results in formal reporting or design decisions."
)
