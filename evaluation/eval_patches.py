"""
Shared monkey-patches for the nuScenes evaluation devkit.

Exports
-------
patched_accumulate
    Fixes TP-metric interpolation when all confidence scores are equal
    (e.g. score=1.0 for GT submissions or pipelines without real scores).

patched_summary_plot
    Adds squeeze=False to plt.subplots so single-class mappings (1class)
    don't crash on axes[0, 0].

make_patched_visualize_sample(lidar_fn=None)
    Factory that returns a visualize_sample function.
    GT is drawn in red, pseudo-labels in blue, legend always shown.
    Pass a custom lidar_fn for datasets that don't populate sample['data']
    (e.g. ECP). See make_ecp_lidar_fn() below.

make_ecp_lidar_fn(lidar_index, root)
    Returns a lidar_fn compatible with make_patched_visualize_sample that
    looks up LIDAR sample data by sample_token instead of sample['data'],
    and reads a single frame via LidarPointCloud.from_file (ECP has no
    multi-sweep index because sample['data'] is unpopulated).

eval_patches(cfg, visualize_sample_fn=None)
    Context manager that applies and restores all render/accumulate patches
    around evaluator.main().

make_scene_split_fn(eval_set, scenes)
    Returns a patched create_splits_scenes that restricts the given eval_set
    to the supplied scene list. Used by both nuScenes and ECP evaluate()
    functions around DetectionEval.__init__.
"""

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, List, Optional

# Module-level dict populated by patched_accumulate; cleared on every
# eval_patches() context-manager entry.
# Keys: (class_name, dist_th).  Values: n_gt, n_pred, n_tp, n_fp, n_fn.
_count_accumulator: Dict = {}

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import nuscenes
import nuscenes.eval.common.loaders
import nuscenes.eval.detection.evaluate
import nuscenes.eval.detection.render as _render
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.common.utils import (
    attr_acc, boxes_to_sensor, center_distance, cummean,
    scale_iou, velocity_l2, yaw_diff,
)
from nuscenes.eval.detection.data_classes import DetectionMetricData
from nuscenes.eval.detection.render import class_pr_curve, class_tp_curve, setup_axis
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.splits import create_splits_scenes


# ── Patch 1: accumulate ────────────────────────────────────────────────────────
#
# When all detection_score values are equal (e.g. 1.0 for GT submissions or
# pipelines that don't yet assign real scores), conf_ref is flat and np.interp
# gets a non-monotonic x-axis → garbage TP metrics (mATE/mASE/mAOE/mAVE).
# Fix: fall back to position-index interpolation (always strictly increasing).

def patched_accumulate(
    gt_boxes: EvalBoxes,
    pred_boxes: EvalBoxes,
    class_name: str,
    dist_fcn: Callable,
    dist_th: float,
    verbose: bool = False,
) -> DetectionMetricData:
    npos = len([1 for gt_box in gt_boxes.all if gt_box.detection_name == class_name])
    pred_boxes_list = [box for box in pred_boxes.all if box.detection_name == class_name]
    pred_confs = [box.detection_score for box in pred_boxes_list]
    n_pred = len(pred_confs)

    if verbose:
        print("Found {} GT of class {} out of {} total across {} samples.".
              format(npos, class_name, len(gt_boxes.all), len(gt_boxes.sample_tokens)))

    if npos == 0:
        _count_accumulator[(class_name, dist_th)] = {
            'n_gt': 0, 'n_pred': n_pred, 'n_tp': 0, 'n_fp': n_pred, 'n_fn': 0,
        }
        return DetectionMetricData.no_predictions()

    if verbose:
        print("Found {} PRED of class {} out of {} total across {} samples.".
              format(n_pred, class_name, len(pred_boxes.all), len(pred_boxes.sample_tokens)))

    sortind = [i for (v, i) in sorted((v, i) for (i, v) in enumerate(pred_confs))][::-1]

    tp = []
    fp = []
    conf = []
    match_data: Dict = {
        'trans_err':  [],
        'vel_err':    [],
        'scale_err':  [],
        'orient_err': [],
        'attr_err':   [],
        'conf':       [],
        'idx':        [],  # position index; used when all confs are equal
    }

    taken = set()
    for ind in sortind:
        pred_box = pred_boxes_list[ind]
        min_dist = np.inf
        match_gt_idx = None

        for gt_idx, gt_box in enumerate(gt_boxes[pred_box.sample_token]):
            if gt_box.detection_name == class_name and not (pred_box.sample_token, gt_idx) in taken:
                this_distance = dist_fcn(gt_box, pred_box)
                if this_distance < min_dist:
                    min_dist = this_distance
                    match_gt_idx = gt_idx

        is_match = min_dist < dist_th

        if is_match:
            taken.add((pred_box.sample_token, match_gt_idx))
            tp.append(1)
            fp.append(0)
            conf.append(pred_box.detection_score)

            gt_box_match = gt_boxes[pred_box.sample_token][match_gt_idx]
            match_data['trans_err'].append(center_distance(gt_box_match, pred_box))
            match_data['vel_err'].append(velocity_l2(gt_box_match, pred_box))
            match_data['scale_err'].append(1 - scale_iou(gt_box_match, pred_box))
            period = np.pi if class_name == 'barrier' else 2 * np.pi
            match_data['orient_err'].append(yaw_diff(gt_box_match, pred_box, period=period))
            match_data['attr_err'].append(1 - attr_acc(gt_box_match, pred_box))
            match_data['conf'].append(pred_box.detection_score)
            match_data['idx'].append(len(tp) - 1)
        else:
            tp.append(0)
            fp.append(1)
            conf.append(pred_box.detection_score)

    # Capture raw counts before cumsum (tp/fp are still 0/1 lists here).
    n_tp_final = int(sum(tp))
    _count_accumulator[(class_name, dist_th)] = {
        'n_gt':   npos,
        'n_pred': n_pred,
        'n_tp':   n_tp_final,
        'n_fp':   n_pred - n_tp_final,
        'n_fn':   npos - n_tp_final,
    }

    if len(match_data['trans_err']) == 0:
        return DetectionMetricData.no_predictions()

    tp   = np.cumsum(tp).astype(float)
    fp   = np.cumsum(fp).astype(float)
    conf = np.array(conf)

    prec = tp / (fp + tp)
    rec  = tp / float(npos)

    rec_interp = np.linspace(0, 1, DetectionMetricData.nelem)
    prec = np.interp(rec_interp, rec, prec, right=0)
    conf = np.interp(rec_interp, rec, conf, right=0)
    idx  = np.interp(rec_interp, rec, list(range(len(rec))), right=0)
    rec  = rec_interp

    for key in match_data.keys():
        if key in ('conf', 'idx'):
            continue
        tmp = cummean(np.array(match_data[key]))
        conf_ref = np.array(match_data['conf'])
        if np.ptp(conf_ref) < 1e-9:
            # All confidence scores are equal → use position index to avoid
            # non-monotonic x-axis in np.interp (would give garbage TP metrics).
            match_data[key] = np.interp(idx, match_data['idx'], tmp)
        else:
            match_data[key] = np.interp(conf[::-1], conf_ref[::-1], tmp[::-1])[::-1]

    return DetectionMetricData(
        recall=rec, precision=prec, confidence=conf,
        trans_err=match_data['trans_err'], vel_err=match_data['vel_err'],
        scale_err=match_data['scale_err'], orient_err=match_data['orient_err'],
        attr_err=match_data['attr_err'],
    )


# ── Patch 2: summary_plot ─────────────────────────────────────────────────────
#
# plt.subplots(nrows=1, ncols=2) returns axes of shape (2,) when squeeze=True
# (the default), causing an IndexError on axes[0, 0] when there is only 1 class.

def patched_summary_plot(
    md_list,
    metrics,
    min_precision: float,
    min_recall: float,
    dist_th_tp: float,
    savepath: str = None,
) -> None:
    """summary_plot with squeeze=False so n_classes=1 doesn't crash on axes[0,0]."""
    n_classes = len(_render.DETECTION_NAMES)
    _, axes = plt.subplots(nrows=n_classes, ncols=2, figsize=(15, 5 * n_classes), squeeze=False)
    for ind, detection_name in enumerate(_render.DETECTION_NAMES):
        title1, title2 = ('Recall vs Precision', 'Recall vs Error') if ind == 0 else (None, None)

        ax1 = setup_axis(xlim=1, ylim=1, title=title1, min_precision=min_precision,
                         min_recall=min_recall, ax=axes[ind, 0])
        ax1.set_ylabel('{} \n \n Precision'.format(_render.PRETTY_DETECTION_NAMES[detection_name]), size=20)

        ax2 = setup_axis(xlim=1, title=title2, min_recall=min_recall, ax=axes[ind, 1])
        if ind == n_classes - 1:
            ax1.set_xlabel('Recall', size=20)
            ax2.set_xlabel('Recall', size=20)

        class_pr_curve(md_list, metrics, detection_name, min_precision, min_recall, ax=ax1)
        class_tp_curve(md_list, metrics, detection_name, min_recall, dist_th_tp=dist_th_tp, ax=ax2)

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
        plt.close()


# ── Patch 3: visualize_sample ─────────────────────────────────────────────────
#
# GT drawn in red, pseudo-labels in blue, legend always shown.
# Parameterised via a lidar_fn to support datasets that don't populate
# sample['data']['LIDAR_TOP'] (e.g. ECP — see make_ecp_lidar_fn below).


def _default_lidar_fn(
    nusc: nuscenes.NuScenes,
    sample_rec: dict,
    nsweeps: int,
):
    """nuScenes-native LIDAR lookup: uses sample['data']['LIDAR_TOP']."""
    sd_record = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
    pc, _ = LidarPointCloud.from_file_multisweep(
        nusc, sample_rec, 'LIDAR_TOP', 'LIDAR_TOP', nsweeps=nsweeps
    )
    return sd_record, pc


def make_ecp_lidar_fn(lidar_index: Dict[str, str], root: Path):
    """
    Return a lidar_fn for ECP where sample['data'] is unpopulated.

    lidar_index : dict mapping sample_token → LIDAR sample_data token.
                  Build with _build_lidar_index() in evaluation_ecp.py.
    root        : ECP dataset root (used to resolve the .pcd.bin path).

    ECP has no sweep chain, so only the single key-frame is read.
    """
    def _ecp_lidar_fn(nusc: nuscenes.NuScenes, sample_rec: dict, nsweeps: int):
        token = lidar_index.get(sample_rec['token'])
        if token is None:
            return None, None
        sd_record = nusc.get('sample_data', token)
        pc = LidarPointCloud.from_file(str(root / sd_record['filename']))
        return sd_record, pc

    return _ecp_lidar_fn


def make_patched_visualize_sample(lidar_fn=None):
    """
    Return a drop-in replacement for nuscenes.eval.detection.render.visualize_sample.

    lidar_fn : callable(nusc, sample_rec, nsweeps) → (sd_record, LidarPointCloud).
               Defaults to the standard nuScenes lookup (sample['data']['LIDAR_TOP']
               + from_file_multisweep).  Pass make_ecp_lidar_fn(...) for ECP.
    """
    _lidar_fn = lidar_fn or _default_lidar_fn

    def _patched_visualize_sample(
        nusc: nuscenes.NuScenes,
        sample_token: str,
        gt_boxes: EvalBoxes,
        pred_boxes: EvalBoxes,
        nsweeps: int = 1,
        conf_th: float = 0.15,
        eval_range: float = 50,
        verbose: bool = True,
        savepath: str = None,
    ) -> None:
        sample_rec  = nusc.get('sample', sample_token)
        sd_record, pc = _lidar_fn(nusc, sample_rec, nsweeps)

        if pc is None:
            if verbose:
                print(f'No LIDAR data for sample {sample_token}, skipping visualisation')
            return

        cs_record   = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

        boxes_gt  = boxes_to_sensor(gt_boxes[sample_token],   pose_record, cs_record)
        boxes_est = boxes_to_sensor(pred_boxes[sample_token], pose_record, cs_record)

        for box_est, box_est_global in zip(boxes_est, pred_boxes[sample_token]):
            box_est.score = box_est_global.detection_score

        _, ax = plt.subplots(1, 1, figsize=(9, 9))

        points = view_points(pc.points[:3, :], np.eye(4), normalize=False)
        dists  = np.sqrt(np.sum(pc.points[:2, :] ** 2, axis=0))
        colors = np.minimum(1, dists / eval_range)
        ax.scatter(points[0, :], points[1, :], c=colors, s=0.2)
        ax.plot(0, 0, 'x', color='black')

        for box in boxes_gt:
            box.render(ax, view=np.eye(4), colors=('r', 'r', 'r'), linewidth=2)

        for box in boxes_est:
            if box.score >= conf_th:
                box.render(ax, view=np.eye(4), colors=('b', 'b', 'b'), linewidth=1)

        ax.legend(
            handles=[mpatches.Patch(color='r'), mpatches.Patch(color='b')],
            labels=['Ground Truth', 'Pseudo Label'],
            loc='upper right',
            fontsize=10,
        )

        axes_limit = eval_range + 3
        ax.set_xlim(-axes_limit, axes_limit)
        ax.set_ylim(-axes_limit, axes_limit)

        # Build a human-readable title: "scene-XXXX  frame K/N"
        scene_rec = nusc.get('scene', sample_rec['scene_token'])
        frame_idx, n_frames = 0, scene_rec['nbr_samples']
        tok = scene_rec['first_sample_token']
        while tok and tok != sample_token:
            tok = nusc.get('sample', tok)['next']
            frame_idx += 1
        scene_name = scene_rec['name']
        plt.title(
            f"{scene_name}  frame {frame_idx + 1}/{n_frames}\n"
            f"token: {sample_token}",
            fontsize=9,
        )

        if verbose:
            print('Rendering sample token %s' % sample_token)
        if savepath is not None:
            out_path = Path(savepath).parent / f"{scene_name}_frame{frame_idx + 1:02d}.png"
            plt.savefig(out_path)
            plt.close()
        else:
            plt.show()

    return _patched_visualize_sample


# ── Count helper ─────────────────────────────────────────────────────────────

def augment_metrics_summary(out_dir: Path, cfg) -> None:
    """
    Append raw TP/FP/FN counts to an already-written metrics_summary.json.

    Reads counts captured by patched_accumulate during evaluator.main() and
    injects them as 'label_counts' (keyed by class name, at cfg.dist_th_tp).
    Call this immediately after the eval_patches context manager exits.
    """
    summary_path = Path(out_dir) / 'metrics_summary.json'
    if not summary_path.exists():
        return

    with open(summary_path) as f:
        summary = json.load(f)

    dist_th_tp = cfg.dist_th_tp
    label_counts = {
        cls: _count_accumulator.get(
            (cls, dist_th_tp),
            {'n_gt': 0, 'n_pred': 0, 'n_tp': 0, 'n_fp': 0, 'n_fn': 0},
        )
        for cls in cfg.class_names
    }
    summary['label_counts']          = label_counts
    summary['label_counts_dist_th']  = dist_th_tp

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)


# ── Context manager: eval_patches ─────────────────────────────────────────────
#
# Applies and restores all render/accumulate patches around evaluator.main().
# Call with the DetectionConfig so the active class list can be derived.

@contextmanager
def eval_patches(cfg, visualize_sample_fn=None):
    """
    Context manager that applies all devkit patches around evaluator.main().

    Usage:
        vis_fn = make_patched_visualize_sample()          # or ECP variant
        with eval_patches(cfg, visualize_sample_fn=vis_fn):
            evaluator.main(plot_examples=n, render_curves=True)
    """
    from nuscenes.eval.detection.constants import PRETTY_DETECTION_NAMES

    _count_accumulator.clear()

    active_classes = list(cfg.class_names)
    _vis_fn = visualize_sample_fn or make_patched_visualize_sample()

    orig_det_names          = list(_render.DETECTION_NAMES)
    orig_pretty_names       = dict(_render.PRETTY_DETECTION_NAMES)
    orig_accumulate         = nuscenes.eval.detection.evaluate.accumulate
    orig_render_summary     = _render.summary_plot
    orig_evaluate_summary   = nuscenes.eval.detection.evaluate.summary_plot
    orig_evaluate_visualize = nuscenes.eval.detection.evaluate.visualize_sample

    _render.DETECTION_NAMES        = active_classes
    _render.PRETTY_DETECTION_NAMES = {
        k: PRETTY_DETECTION_NAMES.get(k, k.replace('_', ' ').title())
        for k in active_classes
    }
    nuscenes.eval.detection.evaluate.accumulate       = patched_accumulate
    _render.summary_plot                              = patched_summary_plot
    nuscenes.eval.detection.evaluate.summary_plot     = patched_summary_plot
    nuscenes.eval.detection.evaluate.visualize_sample = _vis_fn

    try:
        yield
    finally:
        _render.DETECTION_NAMES                           = orig_det_names
        _render.PRETTY_DETECTION_NAMES                    = orig_pretty_names
        nuscenes.eval.detection.evaluate.accumulate       = orig_accumulate
        _render.summary_plot                              = orig_render_summary
        nuscenes.eval.detection.evaluate.summary_plot     = orig_evaluate_summary
        nuscenes.eval.detection.evaluate.visualize_sample = orig_evaluate_visualize


# ── Shared utility: scene-restricted split ────────────────────────────────────

def make_scene_split_fn(eval_set: str, scenes: List[str]):
    """
    Return a replacement for create_splits_scenes that restricts the given
    eval_set to only the requested scenes. All other splits pass through
    unchanged. Used by both nuScenes (optional) and ECP (always-on) scripts.
    """
    original_splits = create_splits_scenes()

    def patched_create_splits_scenes(verbose: bool = False) -> Dict[str, List[str]]:
        splits = dict(original_splits)
        splits[eval_set] = scenes
        return splits

    return patched_create_splits_scenes
