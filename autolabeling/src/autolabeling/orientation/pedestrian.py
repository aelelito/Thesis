"""
Pedestrian heading estimation from MHR70 skeleton keypoints.

This module will grow as orientation estimation is improved for other classes.
At stage 1, only pedestrian heading is fully resolved (no 180° ambiguity).
All other classes return None — ambiguity is accepted until LiDAR is added.
"""
import numpy as np

# MHR70 keypoint indices used for heading estimation
_NOSE       = 0
_L_SHOULDER = 5
_R_SHOULDER = 6


def facing_direction(joints: np.ndarray) -> np.ndarray:
    """
    Estimate horizontal facing direction of a pedestrian from MHR70 skeleton.

    Uses the shoulder line to define a perpendicular facing vector in the XZ
    plane, then disambiguates using the nose position (nose should be in front
    of the shoulder midpoint).

    Parameters
    ----------
    joints : (70, 3) float32  MHR70 keypoints in body-relative camera space

    Returns
    -------
    fwd : (3,) float32  unit vector in XZ plane (Y=0). Fully resolved — no 180° ambiguity.
    """
    shoulder_vec = joints[_R_SHOULDER] - joints[_L_SHOULDER]
    sv_xz = np.array([shoulder_vec[0], 0.0, shoulder_vec[2]], dtype=np.float32)
    sv_xz /= np.linalg.norm(sv_xz) + 1e-8

    # 90° rotation in XZ gives a facing candidate
    fwd = np.array([-sv_xz[2], 0.0, sv_xz[0]], dtype=np.float32)

    # Nose must be in front of the shoulder midpoint — disambiguate
    shoulder_mid = (joints[_L_SHOULDER] + joints[_R_SHOULDER]) / 2.0
    if np.dot(fwd, joints[_NOSE] - shoulder_mid) < 0:
        fwd = -fwd

    return fwd
