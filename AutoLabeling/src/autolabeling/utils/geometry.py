import numpy as np

# Ordered edge pairs for an 8-corner OBB wireframe.
# Corners are indexed as: 0-3 bottom face, 4-7 top face (same XY order).
_BBOX_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
    (4, 5), (5, 6), (6, 7), (7, 4),  # top face
    (0, 4), (1, 5), (2, 6), (3, 7),  # verticals
]


def cam_to_ego(pts: np.ndarray, R_c2e: np.ndarray, t_c2e: np.ndarray) -> np.ndarray:
    """Transform (N, 3) points from camera (R3/OpenCV) to ego frame."""
    return (R_c2e @ pts.T).T + t_c2e


def ego_to_global(pts: np.ndarray, R_e2g: np.ndarray, t_e2g: np.ndarray) -> np.ndarray:
    """Transform (N, 3) points from ego frame to global frame."""
    return (R_e2g @ pts.T).T + t_e2g


def project(pts: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Project (N, 3) camera-space points to pixel coordinates.

    Returns (N, 2) float32 array; points behind the camera get [-9999, -9999].
    """
    z = pts[:, 2]
    valid = z > 0
    uv = np.full((len(pts), 2), -9999.0, dtype=np.float32)
    uv[valid, 0] = pts[valid, 0] / z[valid] * K[0, 0] + K[0, 2]
    uv[valid, 1] = pts[valid, 1] / z[valid] * K[1, 1] + K[1, 2]
    return uv
