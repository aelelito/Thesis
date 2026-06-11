"""
Class mapping loader for pseudo-label evaluation.

Reads class mapping definitions from class_mapping/*.yaml and exposes them
as Python functions used by the evaluation scripts and pseudo-label generator.

To add a new dataset mapping: add a 'mapping_<dataset>' section to each YAML.
To add a new class scheme:   add a new YAML file to class_mapping/.
"""

from pathlib import Path
from typing import Dict, List, Optional
import yaml

from nuscenes.eval.common.config import config_factory

MAPPING_DIR = Path(__file__).parent / 'class_mapping'

# ── YAML loading ───────────────────────────────────────────────────────────────

def _load_scheme(mapping_name: str) -> dict:
    path = MAPPING_DIR / f'{mapping_name}.yaml'
    with open(path) as f:
        return yaml.safe_load(f)

# ── Public constants ───────────────────────────────────────────────────────────

VALID_MAPPINGS: List[str] = [p.stem for p in sorted(MAPPING_DIR.glob('*.yaml'))]

def get_classes(mapping_name: str) -> List[str]:
    """Return the list of final class names for the given mapping scheme."""
    return _load_scheme(mapping_name)['classes']

def get_class_ranges(mapping_name: str) -> Dict[str, int]:
    """Return the evaluation distance ranges (metres) per class."""
    return _load_scheme(mapping_name)['class_ranges']

def get_nuscenes_mapping(mapping_name: str) -> Dict[str, Optional[str]]:
    """Return the nuScenes category name → final class name mapping."""
    return _load_scheme(mapping_name)['mapping_nuscenes']

def get_ecp_mapping(mapping_name: str) -> Dict[str, Optional[str]]:
    """Return the ECP category name → final class name mapping."""
    return _load_scheme(mapping_name)['mapping_ecp']

# ── Public API ─────────────────────────────────────────────────────────────────

def nuscenes_category_to_detection_name(
    category_name: str,
    mapping_name: str,
) -> Optional[str]:
    """
    Map a nuScenes ground-truth category name to the final detection class name
    for the given mapping scheme.

    Returns None for categories excluded from evaluation (e.g. stroller, barrier).
    """
    return get_nuscenes_mapping(mapping_name).get(category_name)


def make_category_fn(mapping_name: str, dataset: str = 'nuscenes'):
    """
    Return a drop-in replacement for
    nuscenes.eval.common.loaders.category_to_detection_name
    scoped to the given mapping scheme and dataset.

    dataset: 'nuscenes' (default) uses mapping_nuscenes from the YAML.
             'ecp' uses mapping_ecp — ECP categories are already in short
             form (e.g. 'car', 'pedestrian') so the mapping is identity-like.

    Intended for temporary monkey-patching around DetectionEval.__init__:

        original = nuscenes.eval.common.loaders.category_to_detection_name
        nuscenes.eval.common.loaders.category_to_detection_name = make_category_fn('8class')
        try:
            evaluator = DetectionEval(...)
        finally:
            nuscenes.eval.common.loaders.category_to_detection_name = original
    """
    if dataset == 'ecp':
        mapping = get_ecp_mapping(mapping_name)
    else:
        mapping = get_nuscenes_mapping(mapping_name)

    def category_to_detection_name(category_name: str) -> Optional[str]:
        return mapping.get(category_name)

    return category_to_detection_name


def build_detection_config(mapping_name: str):
    """
    Build a DetectionConfig restricted to the active classes for the given
    mapping scheme.

    Starts from config_factory('detection_cvpr_2019') to satisfy the devkit's
    internal DETECTION_NAMES assertion, then trims class_range and class_names
    to only the active classes afterwards.
    """
    cfg    = config_factory('detection_cvpr_2019')
    ranges = get_class_ranges(mapping_name)

    # 'vehicle' is not in the default nuScenes detection taxonomy; borrow the
    # 'car' range so the class_range dict stays valid.
    if 'vehicle' in ranges:
        cfg.class_range['vehicle'] = cfg.class_range['car']

    cfg.class_range = {k: v for k, v in cfg.class_range.items() if k in ranges}
    cfg.class_names = list(cfg.class_range.keys())
    return cfg
