#!/usr/bin/env python3
"""
SVG City Map to STL converter.
Extrudes roads and buildings from an SVG map into a 3D STL file.

Usage:
    pip install svgpathtools numpy shapely trimesh
    python svg_to_stl.py input.svg output.stl

Layer detection uses SVG element id/class/fill:
  - Buildings: id/class containing 'building', 'house', 'block', 'здание'
  - Roads:     id/class containing 'road', 'street', 'way', 'дорог', 'улиц'
  - Base:      everything else (parks, water, etc.)
"""

import sys
import re
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

try:
    from svgpathtools import svg2paths2, Path as SvgPath, Line, CubicBezier, QuadraticBezier, Arc
except ImportError:
    sys.exit("Install svgpathtools:  pip install svgpathtools")

try:
    from shapely.geometry import Polygon, MultiPolygon, LineString
    from shapely.ops import unary_union
    import shapely.errors
except ImportError:
    sys.exit("Install shapely:  pip install shapely")

try:
    import trimesh
    from trimesh.creation import extrude_polygon
except ImportError:
    sys.exit("Install trimesh:  pip install trimesh")


# ── Extrusion heights (mm) ─────────────────────────────────────────────────
BASE_HEIGHT      = 1.0   # ground plate
ROAD_HEIGHT      = 0.5   # roads sit slightly above base
BUILDING_HEIGHT  = 5.0   # default building height
OTHER_HEIGHT     = 0.2   # parks, water, etc. — barely raised

# ── Road stroke width → polygon expansion (mm in SVG units) ───────────────
ROAD_BUFFER      = 3.0   # half-width added around road centre-lines

SAMPLES_PER_CURVE = 32   # bezier/arc sampling resolution


# ──────────────────────────────────────────────────────────────────────────
def classify(attrs: dict) -> str:
    """Return 'building', 'road', or 'other' based on element attributes."""
    text = " ".join([
        attrs.get("id", ""),
        attrs.get("class", ""),
        attrs.get("inkscape:label", ""),
        attrs.get("fill", ""),
        attrs.get("stroke", ""),
    ]).lower()

    building_kw = ("building", "house", "block", "здани", "дом", "квартал")
    road_kw     = ("road", "street", "way", "highway", "дорог", "улиц", "шоссе", "проспект")

    if any(k in text for k in building_kw):
        return "building"
    if any(k in text for k in road_kw):
        return "road"
    return "other"


def svg_path_to_coords(path, n=SAMPLES_PER_CURVE):
    """Sample an svgpathtools Path into a list of (x, y) tuples."""
    coords = []
    for seg in path:
        if isinstance(seg, Line):
            coords.append((seg.start.real, seg.start.imag))
        else:
            for i in range(n):
                t = i / n
                pt = seg.point(t)
                coords.append((pt.real, pt.imag))
    if path:
        end = path[-1].end
        coords.append((end.real, end.imag))
    return coords


def coords_to_shapely(coords):
    """Convert coordinate list to a Shapely geometry (Polygon or LineString)."""
    if len(coords) < 2:
        return None
    # closed path → polygon
    if len(coords) >= 3 and np.allclose(coords[0], coords[-1], atol=1e-3):
        try:
            poly = Polygon(coords)
            if poly.is_valid and poly.area > 1e-6:
                return poly
            poly = poly.buffer(0)
            if not poly.is_empty:
                return poly
        except Exception:
            pass
    return LineString(coords)


def flip_y(geom, height):
    """SVG Y-axis points down; flip it for standard 3D coordinates."""
    if geom is None:
        return None
    from shapely.affinity import scale
    return scale(geom, yfact=-1, origin=(0, height / 2, 0))


def make_base_plate(all_geoms, thickness=BASE_HEIGHT):
    """Create a rectangular base under the whole model."""
    from shapely.ops import unary_union
    combined = unary_union([g for g in all_geoms if g is not None and not g.is_empty])
    if combined.is_empty:
        return None
    minx, miny, maxx, maxy = combined.bounds
    margin = 5
    plate = Polygon([
        (minx - margin, miny - margin),
        (maxx + margin, miny - margin),
        (maxx + margin, maxy + margin),
        (minx - margin, maxy + margin),
    ])
    return extrude_polygon(plate, thickness, engine="earcut")


def extrude(geom, height, z_offset=0.0):
    """Extrude a Shapely geometry to a trimesh mesh at z_offset."""
    meshes = []
    if geom is None or geom.is_empty:
        return None

    polys = []
    if isinstance(geom, Polygon):
        polys = [geom]
    elif isinstance(geom, MultiPolygon):
        polys = list(geom.geoms)
    elif isinstance(geom, LineString):
        polys = [geom.buffer(ROAD_BUFFER)]
    else:
        # GeometryCollection or other
        for g in geom.geoms:
            m = extrude(g, height, z_offset)
            if m is not None:
                meshes.append(m)
        return trimesh.util.concatenate(meshes) if meshes else None

    for poly in polys:
        if poly.is_empty or poly.area < 1e-6:
            continue
        try:
            m = extrude_polygon(poly, height, engine="earcut")
            m.apply_translation([0, 0, z_offset])
            meshes.append(m)
        except Exception as e:
            print(f"  [warn] extrude failed: {e}")

    return trimesh.util.concatenate(meshes) if meshes else None


# ──────────────────────────────────────────────────────────────────────────
def convert(svg_path: str, stl_path: str):
    print(f"Reading {svg_path} …")
    paths, attributes, svg_attrs = svg2paths2(svg_path)

    # Get SVG viewport height for Y-flip
    vb = svg_attrs.get("viewBox", "")
    try:
        parts = [float(x) for x in re.split(r"[\s,]+", vb.strip())]
        svg_height = parts[3]
    except Exception:
        svg_height = float(svg_attrs.get("height", "500").replace("px", ""))

    buckets = {"building": [], "road": [], "other": []}

    for path, attrs in zip(paths, attributes):
        kind = classify(attrs)
        coords = svg_path_to_coords(path)
        geom = coords_to_shapely(coords)
        if geom is None:
            continue
        geom = flip_y(geom, svg_height)
        buckets[kind].append(geom)

    print(f"  buildings: {len(buckets['building'])}, "
          f"roads: {len(buckets['road'])}, "
          f"other: {len(buckets['other'])}")

    all_geoms = buckets["building"] + buckets["road"] + buckets["other"]
    meshes = []

    # Base plate
    base = make_base_plate(all_geoms, BASE_HEIGHT)
    if base:
        meshes.append(base)

    # Roads
    for geom in buckets["road"]:
        m = extrude(geom, ROAD_HEIGHT, z_offset=BASE_HEIGHT)
        if m:
            meshes.append(m)

    # Buildings
    for geom in buckets["building"]:
        m = extrude(geom, BUILDING_HEIGHT, z_offset=BASE_HEIGHT)
        if m:
            meshes.append(m)

    # Other (parks, water)
    for geom in buckets["other"]:
        m = extrude(geom, OTHER_HEIGHT, z_offset=BASE_HEIGHT)
        if m:
            meshes.append(m)

    if not meshes:
        sys.exit("No geometry found — check that SVG contains filled paths/polygons.")

    print("Merging meshes …")
    result = trimesh.util.concatenate(meshes)

    print(f"Exporting to {stl_path} …")
    result.export(stl_path)
    print(f"Done! Vertices: {len(result.vertices)}, Faces: {len(result.faces)}")


def main():
    global ROAD_BUFFER, BUILDING_HEIGHT, ROAD_HEIGHT, BASE_HEIGHT

    parser = argparse.ArgumentParser(description="Convert SVG city map to STL")
    parser.add_argument("input",  help="Input SVG file")
    parser.add_argument("output", nargs="?", default=None, help="Output STL file (default: <input>.stl)")
    parser.add_argument("--road-buffer",     type=float, default=ROAD_BUFFER,     help="Road half-width in SVG units")
    parser.add_argument("--building-height", type=float, default=BUILDING_HEIGHT, help="Building extrusion height")
    parser.add_argument("--road-height",     type=float, default=ROAD_HEIGHT,     help="Road extrusion height")
    parser.add_argument("--base-height",     type=float, default=BASE_HEIGHT,     help="Base plate thickness")
    args = parser.parse_args()

    ROAD_BUFFER     = args.road_buffer
    BUILDING_HEIGHT = args.building_height
    ROAD_HEIGHT     = args.road_height
    BASE_HEIGHT     = args.base_height

    out = args.output or str(Path(args.input).with_suffix(".stl"))
    convert(args.input, out)


if __name__ == "__main__":
    main()
