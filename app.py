import json
import requests
import pandas as pd
import streamlit as st
import folium
import plotly.express as px

from shapely.geometry import shape, mapping
from shapely.ops import unary_union
from streamlit_folium import st_folium


# ============================================================
# LADYWOOD ENVIRONMENTAL RISK DASHBOARD
# Uses published Birmingham City Council GIS services.
# No air-quality dataset or assumed official risk thresholds.
# ============================================================

st.set_page_config(
    page_title="Ladywood Environmental Dashboard",
    page_icon="🌍",
    layout="wide"
)

st.title("LADYWOOD")
st.subheader("Environmental Risk & Rehabilitation Dashboard")
st.caption(
    "Study area: Ladywood, Birmingham, United Kingdom. "
    "Environmental exposure indicators are screening evidence, "
    "not a substitute for a site investigation."
)

CRVA = (
    "https://maps.birmingham.gov.uk/server/rest/services/"
    "CRVA/CRVA_2025/MapServer"
)
BOUNDARIES = (
    "https://maps.birmingham.gov.uk/server/rest/services/"
    "mybrummap/mybrummap_Boundaries/MapServer"
)
LANDUSE = (
    "https://maps.birmingham.gov.uk/server/rest/services/"
    "mybrummap/mybrummap_LandUse/MapServer"
)
TRANSPORT = (
    "https://maps.birmingham.gov.uk/server/rest/services/"
    "mybrummap/mybrummap_Transportation/MapServer"
)

CRVA_LSOA = 17
CRVA_WARD = 12
FLOOD_ZONE_3 = 109
SURFACE_FLOOD = 1546
BROWNFIELD = 38
ROAD_CLASSIFICATION = 25

SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "LadywoodEnvironmentalStudentDashboard/1.0"
})


# ------------------------------------------------------------
# ArcGIS request helpers
# ------------------------------------------------------------

def arcgis_json(url, params=None, timeout=90):
    """Request JSON from an ArcGIS REST endpoint."""
    response = SESSION.get(
        url,
        params=params or {"f": "json"},
        timeout=timeout
    )
    response.raise_for_status()

    try:
        result = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"The GIS service did not return JSON: {url}"
        ) from exc

    if isinstance(result, dict) and "error" in result:
        error = result["error"]
        raise RuntimeError(
            f"ArcGIS error at {url}: "
            f"{error.get('message', error)}"
        )

    return result


@st.cache_data(ttl=3600, show_spinner=False)
def get_layer_features(service, layer_id, where="1=1",
                       out_fields="*", geometry_filter=None):
    """
    Direct feature query. Does not request object IDs and does not
    use resultOffset pagination.

    geometry_filter, if supplied, must be an ArcGIS geometry
    dictionary in the British National Grid coordinate system.
    """
    url = f"{service.rstrip('/')}/{layer_id}/query"

    params = {
        "f": "json",
        "where": where,
        "outFields": out_fields,
        "returnGeometry": "true",
        "outSR": "27700",
        "returnZ": "false",
        "returnM": "false"
    }

    if geometry_filter is not None:
        params.update({
            "geometry": json.dumps(geometry_filter),
            "geometryType": "esriGeometryPolygon",
            "inSR": "27700",
            "spatialRel": "esriSpatialRelIntersects"
        })

    result = arcgis_json(url, params)

    if "features" not in result:
        raise RuntimeError(
            f"Layer {layer_id} returned no feature list. "
            f"Response keys: {list(result.keys())}"
        )

    features = result["features"]

    # Explicitly warn rather than silently claiming completeness.
    if result.get("exceededTransferLimit"):
        st.warning(
            f"Layer {layer_id} reached the GIS service record limit. "
            "Its returned features may be incomplete."
        )

    return features


def esri_geometry_to_shape(geometry):
    """Convert ArcGIS polygon or multipolygon geometry to Shapely."""
    if not geometry:
        return None

    if "rings" in geometry:
        # Shapely's shape() handles polygon ring nesting and holes.
        return shape({
            "type": "Polygon",
            "coordinates": geometry["rings"]
        })

    if "x" in geometry and "y" in geometry:
        return shape({
            "type": "Point",
            "coordinates": (geometry["x"], geometry["y"])
        })

    if "paths" in geometry:
        return shape({
            "type": "MultiLineString",
            "coordinates": geometry["paths"]
        })

    return None


def features_to_records(features):
    """Convert ArcGIS features into rows with geometry."""
    rows = []

    for feature in features:
        attrs = feature.get("attributes", {}) or {}
        geom = esri_geometry_to_shape(feature.get("geometry"))

        if geom is not None and not geom.is_empty:
            rows.append({**attrs, "_geometry": geom})

    return rows


def safe_union(geometries):
    geometries = [
        g for g in geometries
        if g is not None and not g.is_empty and g.is_valid
    ]
    return unary_union(geometries) if geometries else None


def intersect_area_pct(feature_geom, exposure_geom):
    """Percentage of an LSOA intersecting an exposure polygon."""
    if feature_geom is None or exposure_geom is None:
        return None

    if feature_geom.is_empty or feature_geom.area <= 0:
        return None

    try:
        overlap = feature_geom.intersection(exposure_geom).area
        return round(100 * overlap / feature_geom.area, 2)
    except Exception:
        return None


def fmt(value, suffix=""):
    if pd.isna(value):
        return "Not available"
    return f"{value:,.2f}{suffix}"


# ------------------------------------------------------------
# Retrieve official Ladywood ward boundary
# ------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def load_ward_boundary():
    # Use the separate published ward-boundary service instead of
    # the problematic CRVA MapServer layer 14 endpoint.
    features = get_layer_features(
        BOUNDARIES,
        0,
        where="WARDNME = 'Ladywood'",
        out_fields="WARDNME,AREASQM,AREAHECTARES"
    )

    records = features_to_records(features)

    if not records:
        # Some services use slightly different ward-name casing.
        features = get_layer_features(
            BOUNDARIES,
            0,
            where="1=1",
            out_fields="WARDNME,AREASQM,AREAHECTARES"
        )
        records = [
            r for r in features_to_records(features)
            if str(r.get("WARDNME", "")).strip().lower() == "ladywood"
        ]

    if not records:
        raise RuntimeError(
            "Ladywood was not returned by Birmingham's ward-boundary "
            "service. The service may be unavailable or its ward names "
            "may have changed."
        )

    boundary = safe_union([r["_geometry"] for r in records])

    if boundary is None or boundary.is_empty:
        raise RuntimeError("The Ladywood boundary geometry is empty.")

    return mapping(boundary), boundary.area


# ------------------------------------------------------------
# Load and process data
# ------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def build_dataset():
    ward_geojson, _ = load_ward_boundary()
    ward_geom = shape(ward_geojson)

    # Query LSOAs by Ladywood's projected bounding box. This avoids
    # requesting every Birmingham LSOA polygon unnecessarily.
    minx, miny, maxx, maxy = ward_geom.bounds
    bbox = {
        "xmin": minx,
        "ymin": miny,
        "xmax": maxx,
        "ymax": maxy,
        "spatialReference": {"wkid": 27700}
    }

    lsoa_features = get_layer_features(
        CRVA, CRVA_LSOA,
        out_fields=(
            "LSOA21CD,LSOA21NM,MEAN,AVERAGE_RISK,"
            "MINIMUM_RISK,MAXIMUM_RISK"
        ),
        geometry_filter=bbox
    )

    lsoa_records = features_to_records(lsoa_features)
    lsoas = []

    for row in lsoa_records:
        geom = row["_geometry"]

        try:
            clipped = geom.intersection(ward_geom)
        except Exception:
            continue

        if clipped.is_empty or clipped.area <= 0:
            continue

        row["_geometry"] = clipped
        lsoas.append(row)

    if not lsoas:
        raise RuntimeError(
            "No LSOAs intersected Ladywood. Check the GIS service "
            "response and coordinate reference system."
        )

    # Query exposure layers spatially within the Ladywood bounding box.
    flood_features = get_layer_features(
        CRVA, FLOOD_ZONE_3, geometry_filter=bbox
    )
    surface_features = get_layer_features(
        CRVA, SURFACE_FLOOD, geometry_filter=bbox
    )
    brownfield_features = get_layer_features(
        LANDUSE, BROWNFIELD, geometry_filter=bbox
    )

    flood_union = safe_union([
        r["_geometry"] for r in features_to_records(flood_features)
    ])
    surface_union = safe_union([
        r["_geometry"] for r in features_to_records(surface_features)
    ])
    brownfield_union = safe_union([
        r["_geometry"] for r in features_to_records(brownfield_features)
    ])

    # Union of flood exposures prevents double-counting overlap.
    flood_union_total = safe_union([
        g for g in [flood_union, surface_union] if g is not None
    ])

    output = []

    for row in lsoas:
        geom = row["_geometry"]

        fz3_pct = intersect_area_pct(geom, flood_union)
        surface_pct = intersect_area_pct(geom, surface_union)
        total_flood_pct = intersect_area_pct(geom, flood_union_total)
        brownfield_pct = intersect_area_pct(geom, brownfield_union)

        crva_mean = pd.to_numeric(
            row.get("MEAN"), errors="coerce"
        )

        output.append({
            "LSOA code": row.get("LSOA21CD", "Unknown"),
            "LSOA name": row.get("LSOA21NM", "Unknown"),
            "CRVA mean": crva_mean,
            "Official CRVA risk class": row.get("AVERAGE_RISK"),
            "Flood Zone 3 exposure (%)": fz3_pct,
            "Surface flooding exposure (%)": surface_pct,
            "Combined flood exposure (%)": total_flood_pct,
            "Brownfield exposure (%)": brownfield_pct,
            "_geometry": geom
        })

    df = pd.DataFrame(output)

    # Do not create an invented score or assume official cutoffs.
    # This transparent ordering is for investigation prioritisation:
    # largest measured exposure first, with missing values kept missing.
    df["Investigation priority rank"] = (
        df["Combined flood exposure (%)"]
        .rank(method="min", ascending=False, na_option="bottom")
    )

    return df, ward_geojson, ward_geom, {
        "flood_zone_3": flood_union,
        "surface_flood": surface_union,
        "brownfield": brownfield_union
    }


@st.cache_data(ttl=3600, show_spinner=False)
def load_roads(bbox):
    try:
        features = get_layer_features(
            TRANSPORT,
            ROAD_CLASSIFICATION,
            out_fields="*",
            geometry_filter=bbox
        )
        return features_to_records(features)
    except Exception as exc:
        # Road data is supplementary. Keep the environmental dashboard
        # usable if this separate service is temporarily unavailable.
        return {"error": str(exc)}


# ------------------------------------------------------------
# Main dashboard
# ------------------------------------------------------------

try:
    with st.spinner("Loading Birmingham Council environmental GIS data..."):
        df, ward_geojson, ward_geom, exposure_geoms = build_dataset()

except Exception as exc:
    st.error("The official map data could not be loaded.")
    st.code(str(exc))
    st.info(
        "This message shows the underlying GIS service error. "
        "No dummy values are substituted for unavailable data."
    )
    st.stop()


# Summary metrics
ward_area_ha = ward_geom.area / 10000

flood3_area_ha = (
    exposure_geoms["flood_zone_3"].intersection(ward_geom).area / 10000
    if exposure_geoms["flood_zone_3"] is not None else None
)
surface_area_ha = (
    exposure_geoms["surface_flood"].intersection(ward_geom).area / 10000
    if exposure_geoms["surface_flood"] is not None else None
)
brownfield_area_ha = (
    exposure_geoms["brownfield"].intersection(ward_geom).area / 10000
    if exposure_geoms["brownfield"] is not None else None
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Ladywood area", fmt(ward_area_ha, " ha"))
m2.metric("Flood Zone 3", fmt(flood3_area_ha, " ha"))
m3.metric("Surface flooding", fmt(surface_area_ha, " ha"))
m4.metric("Brownfield sites", fmt(brownfield_area_ha, " ha"))

st.divider()

# Map controls
st.subheader("Interactive Ladywood map")

show_flood3 = st.checkbox("Show Flood Zone 3", value=True)
show_surface = st.checkbox("Show surface-flood extent", value=True)
show_brownfield = st.checkbox("Show brownfield register", value=True)
show_lsoas = st.checkbox("Show LSOA investigation markers", value=True)

# Map centre calculated from official ward geometry.
centroid = ward_geom.centroid
map_obj = folium.Map(
    location=[centroid.y / 111000, centroid.x / 70000 - 5.7],
    zoom_start=14,
    tiles="CartoDB positron"
)

# Transform British National Grid geometry to longitude/latitude.
from pyproj import Transformer
from shapely.ops import transform

to_wgs84 = Transformer.from_crs(
    "EPSG:27700", "EPSG:4326", always_xy=True
).transform


def add_polygon_layer(map_obj, geom, name, colour, fill_opacity=0.25):
    if geom is None or geom.is_empty:
        return
    geom_wgs = transform(to_wgs84, geom)
    folium.GeoJson(
        data=mapping(geom_wgs),
        name=name,
        style_function=lambda feature: {
            "color": colour,
            "weight": 1.5,
            "fillColor": colour,
            "fillOpacity": fill_opacity
        }
    ).add_to(map_obj)


add_polygon_layer(map_obj, ward_geom, "Ladywood boundary", "#202020", 0.02)

if show_flood3:
    add_polygon_layer(
        map_obj, exposure_geoms["flood_zone_3"],
        "Flood Zone 3", "#1769aa", 0.28
    )

if show_surface:
    add_polygon_layer(
        map_obj, exposure_geoms["surface_flood"],
        "Surface flooding", "#00a6d6", 0.25
    )

if show_brownfield:
    add_polygon_layer(
        map_obj, exposure_geoms["brownfield"],
        "Brownfield register", "#8c564b", 0.25
    )

if show_lsoas:
    for _, row in df.iterrows():
        geom = row["_geometry"]
        centre = geom.representative_point()
        lon, lat = to_wgs84(centre.x, centre.y)

        popup_html = f"""
        <b>{row['LSOA name']}</b><br>
        LSOA: {row['LSOA code']}<br>
        CRVA mean: {fmt(row['CRVA mean'])}<br>
        Official CRVA class: {row['Official CRVA risk class']}<br>
        Flood Zone 3: {fmt(row['Flood Zone 3 exposure (%)'], '%')}<br>
        Surface flooding: {fmt(row['Surface flooding exposure (%)'], '%')}<br>
        Combined flood exposure: {fmt(row['Combined flood exposure (%)'], '%')}<br>
        Brownfield exposure: {fmt(row['Brownfield exposure (%)'], '%')}
        """

        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            color="#7b3294",
            fill=True,
            fill_opacity=0.85,
            tooltip=str(row["LSOA name"]),
            popup=folium.Popup(popup_html, max_width=320)
        ).add_to(map_obj)

folium.LayerControl(collapsed=False).add_to(map_obj)
map_obj.fit_bounds([
    [transform(to_wgs84, ward_geom).bounds[1],
     transform(to_wgs84, ward_geom).bounds[0]],
    [transform(to_wgs84, ward_geom).bounds[3],
     transform(to_wgs84, ward_geom).bounds[2]]
])

st_folium(map_obj, use_container_width=True, height=620)

st.caption(
    "Map layers show spatial overlap with published datasets. "
    "Brownfield status does not by itself establish contamination. "
    "A polygon overlap does not prove that a particular property "
    "or resident is affected."
)

# ------------------------------------------------------------
# Exposure graphs
# ------------------------------------------------------------

st.divider()
st.subheader("Environmental exposure by LSOA")

chart_df = df.copy()

chart_options = {
    "CRVA mean": "CRVA mean",
    "Flood Zone 3 exposure (%)": "Flood Zone 3 exposure (%)",
    "Surface flooding exposure (%)": "Surface flooding exposure (%)",
    "Combined flood exposure (%)": "Combined flood exposure (%)",
    "Brownfield exposure (%)": "Brownfield exposure (%)"
}

for label, column in chart_options.items():
    plot_df = chart_df[["LSOA name", column]].copy()
    plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")
    plot_df = plot_df.dropna(subset=[column]).sort_values(
        column, ascending=False
    )

    st.markdown(f"**{label}**")
    if plot_df.empty:
        st.info(f"No numeric values are available for {label}.")
    else:
        fig = px.bar(
            plot_df,
            x=column,
            y="LSOA name",
            orientation="h",
            title=f"{label} by LSOA",
            labels={column: label, "LSOA name": "LSOA"}
        )
        fig.update_layout(
            yaxis={"categoryorder": "total ascending"},
            height=max(350, min(850, 28 * len(plot_df))),
            margin={"l": 15, "r": 15, "t": 55, "b": 30}
        )
        st.plotly_chart(fig, use_container_width=True)

# Combined comparison graph: measured indicators only.
st.markdown("**Combined exposure comparison**")
comparison_columns = [
    "Flood Zone 3 exposure (%)",
    "Surface flooding exposure (%)",
    "Brownfield exposure (%)"
]
comparison = df[["LSOA name"] + comparison_columns].melt(
    id_vars="LSOA name",
    var_name="Indicator",
    value_name="Exposure (%)"
)
comparison["Exposure (%)"] = pd.to_numeric(
    comparison["Exposure (%)"], errors="coerce"
)
comparison = comparison.dropna(subset=["Exposure (%)"])

if not comparison.empty:
    comparison_fig = px.bar(
        comparison,
        x="LSOA name",
        y="Exposure (%)",
        color="Indicator",
        barmode="group",
        title="Measured spatial exposure indicators by LSOA"
    )
    comparison_fig.update_layout(
        xaxis_tickangle=-45,
        height=500
    )
    st.plotly_chart(comparison_fig, use_container_width=True)
else:
    st.info("No numeric exposure values are available for comparison.")

# ------------------------------------------------------------
# Tables
# ------------------------------------------------------------

st.divider()
st.subheader("Environmental classification and evidence")

classification_cols = [
    "LSOA code",
    "LSOA name",
    "CRVA mean",
    "Official CRVA risk class",
    "Flood Zone 3 exposure (%)",
    "Surface flooding exposure (%)",
    "Brownfield exposure (%)"
]
st.dataframe(
    df[classification_cols].sort_values(
        "Combined flood exposure (%)",
        ascending=False,
        na_position="last"
    ),
    use_container_width=True,
    hide_index=True
)

st.subheader("Investigation priority list")

priority_df = df.sort_values(
    ["Combined flood exposure (%)", "Brownfield exposure (%)"],
    ascending=[False, False],
    na_position="last"
).copy()

st.dataframe(
    priority_df[[
        "LSOA code",
        "LSOA name",
        "Official CRVA risk class",
        "Flood Zone 3 exposure (%)",
        "Surface flooding exposure (%)",
        "Combined flood exposure (%)",
        "Brownfield exposure (%)"
    ]],
    use_container_width=True,
    hide_index=True
)

st.caption(
    "The investigation list is ordered by measured combined flood "
    "exposure, with brownfield exposure used as a secondary ordering "
    "factor. It is not an official Birmingham risk classification."
)

# ------------------------------------------------------------
# Road classification and nearest LSOA
# ------------------------------------------------------------

st.divider()
st.subheader("Road classification and nearby LSOA")

minx, miny, maxx, maxy = ward_geom.bounds
road_bbox = {
    "xmin": minx,
    "ymin": miny,
    "xmax": maxx,
    "ymax": maxy,
    "spatialReference": {"wkid": 27700}
}
roads = load_roads(road_bbox)

if isinstance(roads, dict) and "error" in roads:
    st.warning(
        "The road classification service could not be loaded. "
        f"Service response: {roads['error']}"
    )
elif not roads:
    st.info("No road features were returned for the Ladywood area.")
else:
    road_rows = []
    for road in roads:
        geom = road["_geometry"]
        if geom is None or geom.is_empty:
            continue

        try:
            if not geom.intersects(ward_geom):
                continue
            clipped_road = geom.intersection(ward_geom)
            if clipped_road.is_empty:
                continue
            midpoint = clipped_road.interpolate(0.5, normalized=True)
        except Exception:
            midpoint = geom.representative_point()

        nearest_idx = min(
            range(len(df)),
            key=lambda i: df.iloc[i]["_geometry"].distance(midpoint)
        )
        nearest_lsoa = df.iloc[nearest_idx]["LSOA name"]

        attrs = {k: v for k, v in road.items() if k != "_geometry"}
        road_name = next(
            (attrs[k] for k in attrs
             if "road" in k.lower() and "class" not in k.lower()
             and attrs[k] not in (None, "")),
            next(
                (attrs[k] for k in attrs
                 if "name" in k.lower() and attrs[k] not in (None, "")),
                "Not supplied"
            )
        )
        road_class = next(
            (attrs[k] for k in attrs
             if "class" in k.lower() and attrs[k] not in (None, "")),
            "Not supplied"
        )

        road_rows.append({
            "Road name": road_name,
            "Road classification": road_class,
            "Nearest LSOA": nearest_lsoa
        })

    if road_rows:
        roads_df = pd.DataFrame(road_rows).drop_duplicates()
        st.dataframe(roads_df, use_container_width=True, hide_index=True)
    else:
        st.info("No classified roads intersected the Ladywood boundary.")

# ------------------------------------------------------------
# Data downloads and methodology
# ------------------------------------------------------------

st.divider()
st.subheader("Download the processed LSOA data")

download_df = df.drop(columns=["_geometry"]).copy()
csv_data = download_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download LSOA environmental data (CSV)",
    data=csv_data,
    file_name="Ladywood_LSOA_Environmental_Exposure.csv",
    mime="text/csv"
)

st.subheader("Method and limitations")

st.markdown("""
- **Ward boundary:** Birmingham City Council's published ward boundary service.
- **LSOA and CRVA:** Birmingham City Council's CRVA 2025 map service.
- **Flood exposure:** spatial overlap with Flood Zone 3 and surface-flood extent polygons.
- **Combined flood exposure:** area of the union of the two flood layers intersecting each LSOA, divided by the clipped LSOA area. Overlapping flood polygons are not double-counted.
- **Brownfield exposure:** area overlapping the published Brownfield Register, divided by the clipped LSOA area. This is not proof of contamination.
- **Investigation priority:** transparent ordering of measured exposure, not an official council score.
- **Missing values:** remain missing; they are not silently converted to zero.
- **Risk thresholds:** no new red/amber/green thresholds or weighted environmental index are invented here. The published CRVA class is displayed as supplied by the council.
- **Road proximity:** the nearest LSOA is contextual information, not a measurement of traffic pollution or health impact.
- **Spatial limitations:** LSOA averages and overlap percentages are screening indicators. Site-level inspection and additional evidence are needed before selecting rehabilitation interventions.
""")

st.markdown(
    "**Data sources:** "
    "[Birmingham CRVA GIS service]("
    "https://maps.birmingham.gov.uk/server/rest/services/"
    "CRVA/CRVA_2025/MapServer), "
    "[Birmingham ward boundaries]("
    "https://maps.birmingham.gov.uk/server/rest/services/"
    "mybrummap/mybrummap_Boundaries/MapServer), "
    "[Birmingham Brownfield Register]("
    "https://www.birmingham.gov.uk/brownfieldregister), "
    "[Road classification service]("
    "https://maps.birmingham.gov.uk/server/rest/services/"
    "mybrummap/mybrummap_Transportation/MapServer)."
)
