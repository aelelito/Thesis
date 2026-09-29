import numpy as np

# Classes whose full vertex set is kept: their OBB is refitted together with the rider's mesh (_merge_rider_obbs),
# and that PCA fit depends on how many vertices each mesh contributes, so a subsampled bike would change it.
FULL_MESH_CLASSES = ('bicycle', 'motorcycle')


def subsample_vertices(vertices: np.ndarray, n_points) -> np.ndarray:
    """
    Deterministic uniform random subset of `n_points` vertices (index order kept), or the input
    unchanged if `n_points` is falsy or not smaller than the mesh. The same input always gives the
    same subset, so a fresh run and one resumed from a checkpoint see identical vertices.
    """
    if not n_points or len(vertices) <= n_points:
        return vertices
    idx = np.sort(np.random.default_rng(0).choice(len(vertices), int(n_points), replace=False))
    return vertices[idx]


def attach_full_obbs(objs: list, frame) -> list:
    """
    Fit each object's gravity-aligned OBB on its FULL mesh and store it as r['obb_raw'] (corners, center, dims, yaw
    in the ego frame, yaw still ambiguous by 180 deg). Call this BEFORE slim_object_results: the checkpoint then keeps
    the exact full-mesh box next to the reduced vertices, and _postprocess_frame uses it instead of refitting.
    """
    from ..fitting.obb import compute_obb_gravity_aligned
    for r in objs:
        corners, center, dims, yaw = compute_obb_gravity_aligned(r['vertices'], frame.R_c2e, frame.t_c2e, ground_z=None)
        r['obb_raw'] = dict(corners=corners, center=center, dims=dims, yaw=float(yaw))
    return objs


def slim_object_results(objs: list, n_points, keep_full=FULL_MESH_CLASSES) -> list:
    """
    Reduce each SAM3D Objects mesh to `n_points` of its vertices and drop the faces (a 300k-vertex mesh is ~5 MB per
    object; nothing downstream uses the faces). Classes in `keep_full` keep all vertices (faces are still dropped).
    n_points falsy -> unchanged. Idempotent, and also applied to checkpoints written before slimming existed.
    """
    if not n_points:
        return objs
    for r in objs:
        if r.get('prompt') not in keep_full:
            r['vertices'] = subsample_vertices(r['vertices'], n_points)
        r['faces'] = None
    return objs
