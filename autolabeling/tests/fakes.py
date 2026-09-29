"""Shared synthetic data for the CPU tests."""
from pathlib import Path

import numpy as np


class FakeNusc:
    """5 sweeps; ego drives +1 m/sweep along global x. Static world points + one point on
    the ego body (0,0,1.5 in every sweep's OWN frame) + two ground points."""
    WORLD = np.array([[10, 2, 1.0], [10.2, 2.1, 1.2],      # static object (non-ground)
                      [6, -3, 0.0], [7, -3, 0.02]])         # ground

    def __init__(self, d, n=5):
        self.sd, self.ep = {}, {}
        self.cal = {'cal0': {'rotation': [1, 0, 0, 0], 'translation': [0, 0, 0]}}
        for i in range(n):
            path = Path(d) / f'{i}.bin'
            self.sd[f'sd{i}'] = dict(
                prev=f'sd{i-1}' if i > 0 else '', next=f'sd{i+1}' if i < n - 1 else '',
                calibrated_sensor_token='cal0', ego_pose_token=f'ep{i}', filename=str(path))
            self.ep[f'ep{i}'] = dict(rotation=[1, 0, 0, 0], translation=[float(i), 0, 0])
            own = np.vstack([self.WORLD - [i, 0, 0], [0, 0, 1.5]])
            np.hstack([own, np.zeros((len(own), 2))]).astype(np.float32).tofile(path)

    def get(self, table, tok):
        return {'sample_data': self.sd, 'calibrated_sensor': self.cal, 'ego_pose': self.ep}[table][tok]

    def get_sample_data_path(self, tok):
        return self.sd[tok]['filename']


class FakeGroundFilter:
    def __init__(self):
        self.calls = []

    def segment(self, pts):
        self.calls.append(pts.copy())
        return pts[:, 2] > 0.1          # True = non-ground
