"""
Single-sweep LiDAR free-space grid and queries for meshes and boxes (analysis of SAM3D Objects output).

A voxel is
  FREE      if at least one beam passed through it (one traversal is enough: single sweep, no aggregation)
  OCCUPIED  if a beam ended in it
  UNKNOWN   otherwise (never touched, or occluded behind a return)
Same log-odds bookkeeping and thresholds as contribution_ideas/phase0 (LO_FREE -0.4, LO_OCC +0.85, free < -0.2,
occupied > 0.5). Differences: numpy instead of numba (numba is not in the container), and a fixed-step ray march
(step = half a voxel) instead of Amanatides-Woo; a beam can clip the corner of a voxel without counting it.

Everything is in the ego frame of the sweep's keyframe; the ray origin is the LiDAR position in that frame (t_l2e).
"""
from typing import Dict, Iterable, Optional

import numpy as np

LO_FREE, LO_OCC, LO_MIN, LO_MAX = -0.4, 0.85, -5.0, 10.0
THR_FREE, THR_OCC = -0.2, 0.5
FREE, UNKNOWN, OCC = -1, 0, 1


class FreeSpaceGrid:
    """Log-odds voxel grid built from one LiDAR sweep."""

    def __init__(self, lo: np.ndarray, origin: np.ndarray, vs: float, sensor: np.ndarray):
        self.lo = lo                                  # (NX, NY, NZ) float32 log-odds
        self.origin = np.asarray(origin, np.float64)  # lower corner of voxel (0, 0, 0)
        self.vs = float(vs)
        self.sensor = np.asarray(sensor, np.float64)  # ray origin (LiDAR position, ego frame)

    @property
    def shape(self):
        return self.lo.shape

    def index(self, P: np.ndarray):
        """Voxel indices (M, 3) int64 and an in-grid flag (M,) for points P (M, 3)."""
        ijk = np.floor((np.asarray(P, np.float64) - self.origin) / self.vs).astype(np.int64)
        ok = np.all((ijk >= 0) & (ijk < np.array(self.shape)), axis=1)
        return ijk, ok

    def state(self, P: np.ndarray) -> np.ndarray:
        """FREE / UNKNOWN / OCC (int8) at points P. Outside the grid counts as UNKNOWN."""
        ijk, ok = self.index(P)
        out = np.zeros(len(ijk), np.int8)
        v = self.lo[ijk[ok, 0], ijk[ok, 1], ijk[ok, 2]]
        s = np.zeros(len(v), np.int8)
        s[v < THR_FREE] = FREE
        s[v > THR_OCC] = OCC
        out[ok] = s
        return out

    def state_grid(self) -> np.ndarray:
        s = np.zeros(self.lo.shape, np.int8)
        s[self.lo < THR_FREE] = FREE
        s[self.lo > THR_OCC] = OCC
        return s

    def dist_to_occupied(self, P: np.ndarray, max_dist: float = 20.0) -> np.ndarray:
        """
        For each point, how far BEHIND it (along the ray from the sensor through it) the first OCCUPIED voxel lies [m].
        Small = the point sits right in front of a return (surface). Large = far in front of the nearest evidence.
        inf = no occupied voxel within `max_dist` (or the ray leaves the grid first).
        """
        P = np.asarray(P, np.float64)
        d = P - self.sensor
        r = np.linalg.norm(d, axis=1)
        u = d / np.maximum(r, 1e-9)[:, None]
        out = np.full(len(P), np.inf)
        active = np.arange(len(P))
        step = self.vs / 2.0
        t = 0.0
        occ = self.lo > THR_OCC
        while t <= max_dist and len(active):
            t += step
            Q = P[active] + u[active] * t
            ijk, ok = self.index(Q)
            hit = np.zeros(len(active), bool)
            hit[ok] = occ[ijk[ok, 0], ijk[ok, 1], ijk[ok, 2]]
            out[active[hit]] = t
            active = active[~hit & ok]
        return out


def build_grid(pts_ego: np.ndarray, sensor: np.ndarray, vs: float = 0.1, xy_half: float = 50.0,
               z_min: float = -2.0, z_max: float = 5.0, min_range: float = 0.5, max_range: float = 80.0,
               max_samples_per_chunk: int = 15_000_000) -> FreeSpaceGrid:
    """
    Ray-cast one sweep. pts_ego (N, 3): returns in the ego frame; sensor (3,): LiDAR origin in the same frame.
    Grid: x, y in [-xy_half, xy_half], z in [z_min, z_max], voxel size `vs`.
    """
    sensor = np.asarray(sensor, np.float64)
    origin = np.array([-xy_half, -xy_half, z_min], np.float64)
    shape = (int(round(2 * xy_half / vs)), int(round(2 * xy_half / vs)), int(round((z_max - z_min) / vs)))
    n_vox = int(np.prod(shape))
    step = vs / 2.0

    P = np.asarray(pts_ego, np.float64)
    d = P - sensor
    r = np.linalg.norm(d, axis=1)
    keep = (r >= min_range) & (r <= max_range)
    P, d, r = P[keep], d[keep], r[keep]
    u = d / r[:, None]

    def flat(ijk):
        return (ijk[:, 0] * shape[1] + ijk[:, 1]) * shape[2] + ijk[:, 2]

    def inside(ijk):
        return np.all((ijk >= 0) & (ijk < np.array(shape)), axis=1)

    # endpoints -> hits
    end_ijk = np.floor((P - origin) / vs).astype(np.int64)
    end_ok = inside(end_ijk)
    hits = np.bincount(flat(end_ijk[end_ok]), minlength=n_vox).astype(np.int32)

    # traversed voxels -> free counts (each ray counts a voxel once, its own end voxel not at all)
    free = np.zeros(n_vox, np.int32)
    counts = np.floor(r / step).astype(np.int64)
    order = np.arange(len(P))
    i0 = 0
    while i0 < len(order):
        # chunk of rays whose total number of samples stays below the limit
        csum = np.cumsum(counts[i0:])
        n = int(np.searchsorted(csum, max_samples_per_chunk)) + 1
        sl = slice(i0, min(i0 + n, len(order)))
        i0 = sl.stop
        c = counts[sl]
        tot = int(c.sum())
        if tot == 0:
            continue
        ray = np.repeat(np.arange(sl.start, sl.stop), c)
        first = np.repeat(np.cumsum(c) - c, c)
        k = np.arange(tot) - first
        Q = sensor + u[ray] * (k[:, None] * step)
        ijk = np.floor((Q - origin) / vs).astype(np.int64)
        ok = inside(ijk) & np.any(ijk != end_ijk[ray], axis=1)
        f = flat(ijk[ok])
        rr = ray[ok]
        newv = np.ones(len(f), bool)                       # drop consecutive repeats of a voxel within one ray
        newv[1:] = (f[1:] != f[:-1]) | (rr[1:] != rr[:-1])
        free += np.bincount(f[newv], minlength=n_vox).astype(np.int32)

    lo = np.clip(free * LO_FREE + hits * LO_OCC, LO_MIN, LO_MAX).astype(np.float32).reshape(shape)
    return FreeSpaceGrid(lo, origin, vs, sensor)


# ── Geometry helpers ─────────────────────────────────────────────────────────────

def sample_mesh_surface(verts: np.ndarray, faces: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """n points uniformly on the mesh surface (area-weighted)."""
    v = np.asarray(verts, np.float64)
    f = np.asarray(faces, np.int64)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(f), size=n, p=area / area.sum())
    r1, r2 = rng.random(n), rng.random(n)
    s = np.sqrt(r1)
    return a[idx] * (1 - s)[:, None] + b[idx] * (s * (1 - r2))[:, None] + c[idx] * (s * r2)[:, None]


def mesh_area(verts: np.ndarray, faces: np.ndarray) -> float:
    v = np.asarray(verts, np.float64)
    f = np.asarray(faces, np.int64)
    return float(0.5 * np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1).sum())


def mesh_volume_points(verts: np.ndarray, faces: np.ndarray, vs: float = 0.1, seed: int = 0) -> np.ndarray:
    """
    Centres of the voxels inside the mesh: surface voxelised densely, then enclosed holes filled. Needs a closed shell;
    where the mesh is open or thin (spokes, thin fins) only the surface voxels remain, so the volume is an underestimate.
    """
    from scipy import ndimage
    n = int(max(5000, min(2_000_000, 30.0 * mesh_area(verts, faces) / vs ** 2)))   # ~30 samples per voxel face: no gaps in the shell
    S = sample_mesh_surface(verts, faces, n, seed)
    lo = S.min(axis=0) - 2 * vs
    ijk = np.floor((S - lo) / vs).astype(np.int64)
    shape = ijk.max(axis=0) + 3
    occ = np.zeros(shape, bool)
    occ[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True
    occ = ndimage.binary_fill_holes(occ)
    return lo + (np.argwhere(occ) + 0.5) * vs


def box_lattice(center, length: float, width: float, height: float, yaw: float, vs: float = 0.1) -> np.ndarray:
    """Voxel-centre lattice inside a gravity-aligned oriented box (length along yaw)."""
    nx, ny, nz = (max(1, int(round(s / vs))) for s in (length, width, height))
    gx = (np.arange(nx) + 0.5) / nx * length - length / 2
    gy = (np.arange(ny) + 0.5) / ny * width - width / 2
    gz = (np.arange(nz) + 0.5) / nz * height - height / 2
    X, Y, Z = np.meshgrid(gx, gy, gz, indexing='ij')
    c, s = np.cos(yaw), np.sin(yaw)
    x = c * X - s * Y
    y = s * X + c * Y
    return np.stack([x.ravel(), y.ravel(), Z.ravel()], axis=1) + np.asarray(center, np.float64)


# ── The statistic ─────────────────────────────────────────────────────────────────

def freespace_stats(grid: FreeSpaceGrid, P: np.ndarray, margins: Iterable[float] = (0.0, 0.1, 0.2, 0.3),
                    volume: bool = False, claim: float = 0.2) -> Dict[str, float]:
    """
    Share of the points P (surface samples or volume lattice) in certified-free space.

    frac_free / frac_occ / frac_unknown : state of the voxel holding each point.
    frac_free_m<d>                      : free AND at least d metres in front of the nearest occupied voxel along the
                                          beam. d filters out points that merely sit at a surface (beam grazing,
                                          range noise ~ few cm, small pose error). Use these for the claim.
    depth_p50 / depth_p90 / depth_max   : among free points, how far in front of the nearest occupied voxel [m].
    viol_depth_p50 / viol_depth_p90     : the same, among the violating points only (free and >= `claim` in front) = penetration depth.
    volume=True adds vol_m3 (total) and free_vol_m3(d) = count * vs^3, comparable across mesh / OBB / GT box.
    """
    P = np.asarray(P, np.float64)
    st = grid.state(P)
    free = st == FREE
    out: Dict[str, float] = dict(n=int(len(P)), frac_free=float(free.mean()), frac_occ=float((st == OCC).mean()),
                                 frac_unknown=float((st == UNKNOWN).mean()))
    depth = np.full(len(P), np.nan)
    if free.any():
        depth[free] = grid.dist_to_occupied(P[free])
    for m in margins:
        out[f'frac_free_m{m:g}'] = float((free & (np.nan_to_num(depth, nan=-1.0) >= m)).mean())
    dv = depth[free]
    fin = dv[np.isfinite(dv)]
    out['depth_p50'] = float(np.median(fin)) if len(fin) else float('nan')
    out['depth_p90'] = float(np.percentile(fin, 90)) if len(fin) else float('nan')
    out['depth_max'] = float(fin.max()) if len(fin) else float('nan')
    viol = fin[fin >= claim]
    out['viol_depth_p50'] = float(np.median(viol)) if len(viol) else float('nan')
    out['viol_depth_p90'] = float(np.percentile(viol, 90)) if len(viol) else float('nan')
    if volume:
        v3 = grid.vs ** 3
        out['vol_m3'] = len(P) * v3
        for m in margins:
            out[f'free_vol_m3_m{m:g}'] = out[f'frac_free_m{m:g}'] * len(P) * v3
    return out


def local_ground_z(pts: np.ndarray, center, length: float, width: float, yaw: float, ring=(0.5, 2.5), min_pts: int = 30, max_spread: float = 0.3) -> float:
    """
    Ground height next to an object: returns in a ring around its footprint (0.5-2.5 m outside a box of length x width at
    yaw), lowest quarter of their heights, median of those. NaN if the ring holds fewer than `min_pts` returns or the lowest quarter spreads over more than `max_spread` m (no clear ground level). An estimate
    from the single sweep, not a ground model: kerbs, walls and neighbouring cars can bias it by a few centimetres.
    """
    c, s = np.cos(-yaw), np.sin(-yaw)
    d = np.asarray(pts, np.float64)[:, :2] - np.asarray(center, np.float64)[:2]
    lx, ly = c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]
    q = np.maximum(np.abs(lx) - length / 2, np.abs(ly) - width / 2)      # distance outside the footprint (Chebyshev-like)
    sel = (q > ring[0]) & (q < ring[1])
    z = np.asarray(pts, np.float64)[sel, 2]
    if len(z) < min_pts:
        return float('nan')
    low = z[z <= np.percentile(z, 25)]
    if np.percentile(low, 90) - np.percentile(low, 10) > max_spread:     # no consistent ground level in the ring (walls, kerbs, slope)
        return float('nan')
    return float(np.median(low))
