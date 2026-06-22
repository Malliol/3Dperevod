#!/usr/bin/env python3
"""
SVG City Map to STL converter.
No trimesh triangulation engine required — mesh is built manually.

Usage:
    pip install svgpathtools numpy shapely
    python svg_to_stl.py input.svg output.stl
"""

import sys
import re
import struct
import argparse
from pathlib import Path

import numpy as np

try:
    from svgpathtools import svg2paths2, Line, CubicBezier, QuadraticBezier, Arc
except ImportError:
    sys.exit("Run:  pip install svgpathtools")

try:
    from shapely.geometry import Polygon, MultiPolygon, LineString
    from shapely.ops import unary_union
    from shapely.validation import make_valid
except ImportError:
    sys.exit("Run:  pip install shapely")

try:
    import numpy as np
except ImportError:
    sys.exit("Run:  pip install numpy")


# ── Heights (mm / SVG units) ───────────────────────────────────────────────
BASE_HEIGHT     = 1.0
ROAD_HEIGHT     = 0.8
BUILDING_HEIGHT = 5.0
OTHER_HEIGHT    = 0.3
ROAD_BUFFER     = 3.0
SAMPLES         = 32


# ── Classify SVG element ───────────────────────────────────────────────────
def classify(attrs):
    text = " ".join([
        attrs.get("id", ""), attrs.get("class", ""),
        attrs.get("inkscape:label", ""), attrs.get("fill", ""),
    ]).lower()
    if any(k in text for k in ("building","house","block","здани","дом","квартал")):
        return "building"
    if any(k in text for k in ("road","street","way","highway","дорог","улиц","шоссе")):
        return "road"
    return "other"


# ── SVG path → coordinate list ─────────────────────────────────────────────
def path_to_coords(path):
    coords = []
    for seg in path:
        if isinstance(seg, Line):
            coords.append((seg.start.real, seg.start.imag))
        else:
            for i in range(SAMPLES):
                pt = seg.point(i / SAMPLES)
                coords.append((pt.real, pt.imag))
    if path:
        e = path[-1].end
        coords.append((e.real, e.imag))
    return coords


def to_shapely(coords):
    if len(coords) < 2:
        return None
    if len(coords) >= 3 and np.allclose(coords[0], coords[-1], atol=1e-3):
        try:
            p = make_valid(Polygon(coords))
            return p if not p.is_empty else None
        except Exception:
            pass
    return LineString(coords)


def flip_y(geom, h):
    from shapely.affinity import scale
    return scale(geom, yfact=-1, origin=(0, h / 2, 0))


# ── Manual STL mesh builder ────────────────────────────────────────────────
class MeshBuilder:
    def __init__(self):
        self.triangles = []  # list of (3,3) arrays

    def add_quad(self, a, b, c, d):
        """Add two triangles forming a quad (a,b,c,d in order)."""
        self.triangles.append(np.array([a, b, c]))
        self.triangles.append(np.array([a, c, d]))

    def add_polygon_cap(self, ring, z, flip=False):
        """Fan-triangulate a flat polygon ring at height z."""
        pts = [(x, y) for x, y in ring]
        if len(pts) < 3:
            return
        # remove duplicate closing point
        if np.allclose(pts[0], pts[-1]):
            pts = pts[:-1]
        if len(pts) < 3:
            return
        o = np.array([pts[0][0], pts[0][1], z])
        for i in range(1, len(pts) - 1):
            a = np.array([pts[i][0],   pts[i][1],   z])
            b = np.array([pts[i+1][0], pts[i+1][1], z])
            tri = [o, a, b] if not flip else [o, b, a]
            self.triangles.append(np.array(tri))

    def add_walls(self, ring, z_bot, z_top):
        """Extrude the edges of a ring into vertical walls."""
        pts = list(ring)
        if np.allclose(pts[0], pts[-1]):
            pts = pts[:-1]
        n = len(pts)
        for i in range(n):
            x0, y0 = pts[i]
            x1, y1 = pts[(i+1) % n]
            bl = np.array([x0, y0, z_bot])
            br = np.array([x1, y1, z_bot])
            tr = np.array([x1, y1, z_top])
            tl = np.array([x0, y0, z_top])
            self.add_quad(bl, br, tr, tl)

    def extrude_poly(self, poly, z_bot, z_top):
        ring = list(poly.exterior.coords)
        self.add_polygon_cap(ring, z_bot, flip=True)
        self.add_polygon_cap(ring, z_top, flip=False)
        self.add_walls(ring, z_bot, z_top)
        for interior in poly.interiors:
            ir = list(interior.coords)
            self.add_polygon_cap(ir, z_bot, flip=False)
            self.add_polygon_cap(ir, z_top, flip=True)
            self.add_walls(ir, z_bot, z_top)

    def extrude_geom(self, geom, z_bot, z_top):
        if geom is None or geom.is_empty:
            return
        if isinstance(geom, Polygon):
            self.extrude_poly(geom, z_bot, z_top)
        elif isinstance(geom, MultiPolygon):
            for p in geom.geoms:
                self.extrude_poly(p, z_bot, z_top)
        elif isinstance(geom, LineString):
            self.extrude_geom(geom.buffer(ROAD_BUFFER), z_bot, z_top)
        else:
            for g in geom.geoms:
                self.extrude_geom(g, z_bot, z_top)

    def write_stl(self, path):
        tris = self.triangles
        print(f"Writing {len(tris)} triangles to {path} …")
        with open(path, "wb") as f:
            f.write(b"\x00" * 80)
            f.write(struct.pack("<I", len(tris)))
            for tri in tris:
                v0, v1, v2 = tri
                n = np.cross(v1 - v0, v2 - v0)
                nn = np.linalg.norm(n)
                n = n / nn if nn > 1e-10 else n
                f.write(struct.pack("<fff", *n))
                f.write(struct.pack("<fff", *v0))
                f.write(struct.pack("<fff", *v1))
                f.write(struct.pack("<fff", *v2))
                f.write(b"\x00\x00")
        print("Done!")


# ── Main conversion ────────────────────────────────────────────────────────
def convert(svg_path, stl_path):
    print(f"Reading {svg_path} …")
    paths, attributes, svg_attrs = svg2paths2(svg_path)

    vb = svg_attrs.get("viewBox", "")
    try:
        parts = [float(x) for x in re.split(r"[\s,]+", vb.strip())]
        svg_h = parts[3]
    except Exception:
        svg_h = float(re.sub(r"[^\d.]", "", svg_attrs.get("height", "500")) or 500)

    buckets = {"building": [], "road": [], "other": []}
    for path, attrs in zip(paths, attributes):
        coords = path_to_coords(path)
        geom = to_shapely(coords)
        if geom is None:
            continue
        geom = flip_y(geom, svg_h)
        buckets[classify(attrs)].append(geom)

    print(f"  buildings: {len(buckets['building'])}, "
          f"roads: {len(buckets['road'])}, "
          f"other: {len(buckets['other'])}")

    all_geoms = buckets["building"] + buckets["road"] + buckets["other"]
    if not all_geoms:
        sys.exit("No geometry found in SVG.")

    combined = unary_union([g for g in all_geoms if not g.is_empty])
    minx, miny, maxx, maxy = combined.bounds
    m = 5
    base_poly = Polygon([
        (minx-m, miny-m), (maxx+m, miny-m),
        (maxx+m, maxy+m), (minx-m, maxy+m),
    ])

    builder = MeshBuilder()
    builder.extrude_geom(base_poly, 0, BASE_HEIGHT)

    for geom in buckets["road"]:
        builder.extrude_geom(geom, BASE_HEIGHT, BASE_HEIGHT + ROAD_HEIGHT)

    for geom in buckets["building"]:
        builder.extrude_geom(geom, BASE_HEIGHT, BASE_HEIGHT + BUILDING_HEIGHT)

    for geom in buckets["other"]:
        builder.extrude_geom(geom, BASE_HEIGHT, BASE_HEIGHT + OTHER_HEIGHT)

    builder.write_stl(stl_path)


def main():
    global ROAD_BUFFER, BUILDING_HEIGHT, ROAD_HEIGHT, BASE_HEIGHT, OTHER_HEIGHT

    parser = argparse.ArgumentParser(description="SVG city map → STL")
    parser.add_argument("input")
    parser.add_argument("output", nargs="?", default=None)
    parser.add_argument("--road-buffer",     type=float, default=ROAD_BUFFER)
    parser.add_argument("--building-height", type=float, default=BUILDING_HEIGHT)
    parser.add_argument("--road-height",     type=float, default=ROAD_HEIGHT)
    parser.add_argument("--base-height",     type=float, default=BASE_HEIGHT)
    args = parser.parse_args()

    ROAD_BUFFER     = args.road_buffer
    BUILDING_HEIGHT = args.building_height
    ROAD_HEIGHT     = args.road_height
    BASE_HEIGHT     = args.base_height

    out = args.output or str(Path(args.input).with_suffix(".stl"))
    convert(args.input, out)


if __name__ == "__main__":
    main()
