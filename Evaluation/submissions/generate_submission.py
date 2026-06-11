"""
Convert pseudo-label detections into a nuScenes-format submission JSON.

Autolabeling pipeline produces bounding boxes per sample. Collect them as PseudoBox
objects and call generate_submission() to write the submission file.

Coordinate conventions (nuScenes world frame):
  translation : [x, y, z]          — box centre in metres
  size        : [width, length, height]
  yaw         : rotation around the z-axis in radians (counter-clockwise positive)
  velocity    : [vx, vy]            — m/s in the world frame (use [0.0, 0.0] if unknown)
  score       : confidence in [0, 1]

Example usage:
    from generate_submission import PseudoBox, generate_submission
    from pathlib import Path

    boxes_per_sample = {
        "<sample_token>": [
            PseudoBox(
                translation=[x, y, z],
                size=[w, l, h],
                yaw=theta,
                detection_name="car",   # must match the chosen mapping_name
                score=0.8,
            ),
        ],
    }

    generate_submission(
        boxes_per_sample=boxes_per_sample,
        split="train",
        mapping_name="8class",
        out_path=Path("results/my_labels.json"),
    )
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from pyquaternion import Quaternion

from class_remapping import VALID_MAPPINGS, get_classes


@dataclass
class PseudoBox:
    """
    A single predicted 3D bounding box produced by your pipeline.

    Fields:
        translation:    [x, y, z] — box centre in the nuScenes world frame (metres)
        size:           [width, length, height] — box dimensions (metres)
        yaw:            rotation around the z-axis in radians (counter-clockwise positive)
        detection_name: class name — must match the classes defined for mapping_name
        score:          detection confidence in [0.0, 1.0]
        velocity:       [vx, vy] in m/s (world frame); defaults to [0.0, 0.0]
    """
    translation:    List[float]
    size:           List[float]
    yaw:            float
    detection_name: str
    score:          float
    velocity:       List[float] = field(default_factory=lambda: [0.0, 0.0])

    def to_submission_dict(self, sample_token: str) -> dict:
        """Serialise to the nuScenes submission box format."""
        rotation = Quaternion(axis=[0, 0, 1], angle=self.yaw).elements.tolist()  # [w, x, y, z]
        return {
            "sample_token":    sample_token,
            "translation":     list(self.translation),
            "size":            list(self.size),
            "rotation":        rotation,
            "velocity":        list(self.velocity),
            "detection_name":  self.detection_name,
            "detection_score": float(self.score),
            "attribute_name":  "",
        }


def generate_submission(
    boxes_per_sample: Dict[str, List[PseudoBox]],
    split: str,
    mapping_name: str,
    out_path: Path,
) -> None:
    """
    Write a nuScenes-format submission JSON from your pipeline's predictions.

    Args:
        boxes_per_sample: dict mapping sample_token → list of PseudoBox.
                          Every sample token in the evaluation split must be
                          present as a key (use an empty list for samples with
                          no detections).
        split:            "train" or "val" — must match the split you will
                          evaluate on.
        mapping_name:     "1class", "3class", or "8class" — determines which
                          class names are valid in detection_name.
        out_path:         Where to write the JSON file.
    """
    if mapping_name not in VALID_MAPPINGS:
        raise ValueError(
            f"Unknown mapping_name '{mapping_name}'. Expected one of: {VALID_MAPPINGS}"
        )

    valid_classes = set(get_classes(mapping_name))
    for sample_token, boxes in boxes_per_sample.items():
        for box in boxes:
            if box.detection_name not in valid_classes:
                raise ValueError(
                    f"detection_name '{box.detection_name}' is not valid for "
                    f"mapping '{mapping_name}'. Valid classes: {sorted(valid_classes)}"
                )

    results = {
        sample_token: [box.to_submission_dict(sample_token) for box in boxes]
        for sample_token, boxes in boxes_per_sample.items()
    }

    submission = {
        "split":        split,
        "mapping_name": mapping_name,
        "meta": {
            "use_camera":   False,
            "use_lidar":    True,
            "use_radar":    False,
            "use_map":      False,
            "use_external": False,
        },
        "results": results,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(submission, f)

    total_boxes = sum(len(v) for v in results.values())
    print(f"Written {total_boxes} boxes across {len(results)} samples → {out_path}")
