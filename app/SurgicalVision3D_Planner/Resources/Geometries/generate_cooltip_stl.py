from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


OUT_DIR = Path(__file__).resolve().parent
LATITUDE_BANDS = 64
LONGITUDE_BANDS = 128
# Optional simplification for single-tip lesions around the applicator axis (local Z).
FORCE_AXISYMMETRIC_SINGLE_TIP = False


@dataclass(frozen=True)
class CooltipSpec:
    file_name: str
    diameter_x_mm: float
    diameter_y_mm: float
    diameter_z_mm: float
    is_single_tip: bool


COOLTIP_SPECS: list[CooltipSpec] = [
    CooltipSpec("09_cooltip-single-2cm-ic_exvivo.stl", 26.2, 26.2, 30.1, True),
    CooltipSpec("10_cooltip-single-2cm-ic_invivo.stl", 20.0, 20.0, 26.2, True),
    CooltipSpec("11_cooltip-single-3cm-ic_exvivo.stl", 32.5, 34.1, 40.9, True),
    CooltipSpec("12_cooltip-single-3cm-ic_invivo.stl", 18.5, 21.8, 36.7, True),
    CooltipSpec("13_cooltip-cluster-2_5cm_nominal.stl", 45.0, 45.0, 45.0, False),
    CooltipSpec("14_cooltip-multi3-3cm-switched_invivo.stl", 43.1, 43.1, 43.1, False),
]


def _cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _sub(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _normalize(v: tuple[float, float, float]) -> tuple[float, float, float]:
    magnitude = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if magnitude <= 1e-12:
        return (0.0, 0.0, 0.0)
    return (v[0] / magnitude, v[1] / magnitude, v[2] / magnitude)


def _triangle_with_outward_normal(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    c: tuple[float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
    ab = _sub(b, a)
    ac = _sub(c, a)
    normal = _cross(ab, ac)
    centroid = (
        (a[0] + b[0] + c[0]) / 3.0,
        (a[1] + b[1] + c[1]) / 3.0,
        (a[2] + b[2] + c[2]) / 3.0,
    )
    if normal[0] * centroid[0] + normal[1] * centroid[1] + normal[2] * centroid[2] < 0.0:
        b, c = c, b
        ab = _sub(b, a)
        ac = _sub(c, a)
        normal = _cross(ab, ac)
    return _normalize(normal), a, b, c


def _build_ellipsoid_triangles(
    rx: float,
    ry: float,
    rz: float,
    latitude_bands: int = LATITUDE_BANDS,
    longitude_bands: int = LONGITUDE_BANDS,
) -> list[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]]:
    if latitude_bands < 3 or longitude_bands < 3:
        raise ValueError("Need at least 3 latitude bands and 3 longitude bands.")

    def ring_point(lat_index: int, lon_index: int) -> tuple[float, float, float]:
        phi = -0.5 * math.pi + (lat_index * math.pi / latitude_bands)
        theta = 2.0 * math.pi * (lon_index / longitude_bands)
        cos_phi = math.cos(phi)
        return (
            rx * cos_phi * math.cos(theta),
            ry * cos_phi * math.sin(theta),
            rz * math.sin(phi),
        )

    south_pole = (0.0, 0.0, -rz)
    north_pole = (0.0, 0.0, rz)
    interior_rings: list[list[tuple[float, float, float]]] = []
    for lat in range(1, latitude_bands):
        ring = [ring_point(lat, lon) for lon in range(longitude_bands)]
        interior_rings.append(ring)

    triangles: list[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]] = []

    first_ring = interior_rings[0]
    for lon in range(longitude_bands):
        nxt = (lon + 1) % longitude_bands
        triangles.append(_triangle_with_outward_normal(south_pole, first_ring[nxt], first_ring[lon]))

    for ring_index in range(len(interior_rings) - 1):
        lower = interior_rings[ring_index]
        upper = interior_rings[ring_index + 1]
        for lon in range(longitude_bands):
            nxt = (lon + 1) % longitude_bands
            triangles.append(_triangle_with_outward_normal(lower[lon], lower[nxt], upper[nxt]))
            triangles.append(_triangle_with_outward_normal(lower[lon], upper[nxt], upper[lon]))

    last_ring = interior_rings[-1]
    for lon in range(longitude_bands):
        nxt = (lon + 1) % longitude_bands
        triangles.append(_triangle_with_outward_normal(north_pole, last_ring[lon], last_ring[nxt]))

    return triangles


def _write_binary_stl(
    output_path: Path,
    triangles: Iterable[
        tuple[
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[float, float, float],
        ]
    ],
) -> None:
    triangle_list = list(triangles)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = b"SurgicalVision3D Cool-tip generated mesh"
    header = header + (b" " * (80 - len(header)))

    with output_path.open("wb") as stream:
        stream.write(header[:80])
        stream.write(struct.pack("<I", len(triangle_list)))
        for normal, v1, v2, v3 in triangle_list:
            stream.write(
                struct.pack(
                    "<12fH",
                    normal[0], normal[1], normal[2],
                    v1[0], v1[1], v1[2],
                    v2[0], v2[1], v2[2],
                    v3[0], v3[1], v3[2],
                    0,
                )
            )


def _effective_diameters(spec: CooltipSpec) -> tuple[float, float, float]:
    dx = float(spec.diameter_x_mm)
    dy = float(spec.diameter_y_mm)
    dz = float(spec.diameter_z_mm)
    if FORCE_AXISYMMETRIC_SINGLE_TIP and spec.is_single_tip:
        transverse = 0.5 * (dx + dy)
        dx = transverse
        dy = transverse
    return dx, dy, dz


def generate() -> None:
    for spec in COOLTIP_SPECS:
        diameter_x_mm, diameter_y_mm, diameter_z_mm = _effective_diameters(spec)
        radius_x = 0.5 * diameter_x_mm
        radius_y = 0.5 * diameter_y_mm
        radius_z = 0.5 * diameter_z_mm
        triangles = _build_ellipsoid_triangles(radius_x, radius_y, radius_z)
        output_path = OUT_DIR / spec.file_name
        _write_binary_stl(output_path, triangles)
        print(
            f"Wrote {output_path.name}: diameters(mm)=({diameter_x_mm:.2f}, {diameter_y_mm:.2f}, {diameter_z_mm:.2f}) "
            f"triangles={len(triangles)}"
        )


if __name__ == "__main__":
    generate()
