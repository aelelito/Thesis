#!/usr/bin/env python3
"""
make_eval_report.py

Post-processing script that reads an existing evaluation output folder and
generates a PDF report + CSV metrics table.  Does NOT re-run evaluation.

Usage
-----
    python make_eval_report.py --eval_dir path/to/eval_folder
    python make_eval_report.py --eval_dir path/to/eval_folder --out_dir path/to/output_dir

Importable API
--------------
    from make_eval_report import make_report
    pdf_path, csv_path = make_report(eval_dir, out_path=None, csv_path=None)
"""

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
import numpy as np

# np.trapezoid was added in NumPy 2.0; np.trapz was removed in 2.0
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
import pandas as pd

# All pages share this size (US Letter, portrait)
_PAGE_W, _PAGE_H = 8.5, 11.0


# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _build_dataframe(summary: dict) -> pd.DataFrame:
    """Build per-class metrics DataFrame from metrics_summary.json contents."""
    label_aps       = summary.get("label_aps", {})
    mean_dist_aps   = summary.get("mean_dist_aps", {})
    label_tp_errors = summary.get("label_tp_errors", {})
    tp_errors       = summary.get("tp_errors", {})
    mean_ap         = summary.get("mean_ap", float("nan"))
    nd_score        = summary.get("nd_score", float("nan"))

    label_counts = summary.get("label_counts", {})
    counts_th    = summary.get("label_counts_dist_th", 2.0)

    classes = list(label_aps.keys())
    rows = []
    for cls in classes:
        aps = label_aps.get(cls, {})
        tp  = label_tp_errors.get(cls, {})
        cnt = label_counts.get(cls, {})
        rows.append({
            "class":  cls,
            "AP":     round(float(mean_dist_aps.get(cls, float("nan"))), 3),
            "AP@0.5": round(float(aps.get("0.5", float("nan"))), 3),
            "AP@1.0": round(float(aps.get("1.0", float("nan"))), 3),
            "AP@2.0": round(float(aps.get("2.0", float("nan"))), 3),
            "AP@4.0": round(float(aps.get("4.0", float("nan"))), 3),
            "ATE":    round(float(tp.get("trans_err",  float("nan"))), 3),
            "ASE":    round(float(tp.get("scale_err",  float("nan"))), 3),
            "AOE":    round(float(tp.get("orient_err", float("nan"))), 3),
            "AVE":    round(float(tp.get("vel_err",    float("nan"))), 3),
            "AAE":    round(float(tp.get("attr_err",   float("nan"))), 3),
            "NDS":    float("nan"),   # NDS is a global metric; filled in mean row only
            "#GT":    cnt.get("n_gt",   "—"),
            "#TP":    cnt.get("n_tp",   "—"),
            "#FP":    cnt.get("n_fp",   "—"),
            "#FN":    cnt.get("n_fn",   "—"),
        })

    # Classes with no GT — their AP is always 0 and skews the mean.
    no_gt_classes = {r["class"] for r in rows if r["#GT"] == 0}
    valid_rows    = [r for r in rows if r["class"] not in no_gt_classes]

    def _ap_mean(source_rows, col):
        vals = [r[col] for r in source_rows
                if isinstance(r[col], float) and not np.isnan(r[col])]
        return round(float(np.mean(vals)), 3) if vals else float("nan")

    def _sum_col(col):
        vals = [r[col] for r in rows if isinstance(r[col], int)]
        return sum(vals) if vals else "—"

    _tp_mean = {
        "ATE": round(float(tp_errors.get("trans_err",  float("nan"))), 3),
        "ASE": round(float(tp_errors.get("scale_err",  float("nan"))), 3),
        "AOE": round(float(tp_errors.get("orient_err", float("nan"))), 3),
        "AVE": round(float(tp_errors.get("vel_err",    float("nan"))), 3),
        "AAE": round(float(tp_errors.get("attr_err",   float("nan"))), 3),
    }

    mean_all = {
        "class":  "mean (all)",
        "AP":     round(float(mean_ap), 3),
        "AP@0.5": _ap_mean(rows, "AP@0.5"),
        "AP@1.0": _ap_mean(rows, "AP@1.0"),
        "AP@2.0": _ap_mean(rows, "AP@2.0"),
        "AP@4.0": _ap_mean(rows, "AP@4.0"),
        **_tp_mean,
        "NDS":    round(float(nd_score), 3),
        "#GT":    _sum_col("#GT"),
        "#TP":    _sum_col("#TP"),
        "#FP":    _sum_col("#FP"),
        "#FN":    _sum_col("#FN"),
    }

    # Snapshot sums before appending any mean rows to avoid double-counting
    sum_gt = _sum_col("#GT")
    sum_tp = _sum_col("#TP")
    sum_fp = _sum_col("#FP")
    sum_fn = _sum_col("#FN")

    mean_all["#GT"] = sum_gt
    mean_all["#TP"] = sum_tp
    mean_all["#FP"] = sum_fp
    mean_all["#FN"] = sum_fn

    # Attach metadata for use in rendering functions
    _build_dataframe._counts_th     = counts_th
    _build_dataframe._no_gt_classes = no_gt_classes
    _build_dataframe._valid_mean_ap = _ap_mean(valid_rows or rows, "AP")

    rows.append(mean_all)

    # Only add a second "mean (w/ GT)" row if some classes have no GT
    if no_gt_classes:
        valid_ap = _ap_mean(valid_rows, "AP")
        tp_vals  = [_tp_mean[k] for k in ("ATE", "ASE", "AOE", "AVE", "AAE")]
        tp_score_sum  = sum(1 - min(1.0, e) for e in tp_vals if not np.isnan(e))
        corrected_nds = round((5 * valid_ap + tp_score_sum) / 10, 3) \
                        if not np.isnan(valid_ap) else float("nan")
        mean_valid = {
            "class":  "mean (w/ GT)",
            "AP":     valid_ap,
            "AP@0.5": _ap_mean(valid_rows, "AP@0.5"),
            "AP@1.0": _ap_mean(valid_rows, "AP@1.0"),
            "AP@2.0": _ap_mean(valid_rows, "AP@2.0"),
            "AP@4.0": _ap_mean(valid_rows, "AP@4.0"),
            **_tp_mean,
            "NDS":    corrected_nds,
            "#GT":    sum_gt,
            "#TP":    sum_tp,
            "#FP":    sum_fp,
            "#FN":    sum_fn,
        }
        rows.append(mean_valid)

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# Rendering helpers
# ──────────────────────────────────────────────────────────────────────────────

def _fmt(v) -> str:
    """Format a cell value: round floats, replace NaN with dash."""
    if isinstance(v, float) and np.isnan(v):
        return "—"
    return str(v)


def _draw_table(ax: plt.Axes, df: pd.DataFrame, title: str,
                no_gt_classes: Optional[set] = None) -> None:
    """Render a pandas DataFrame as a matplotlib table on *ax*."""
    ax.axis("off")
    ax.set_title(title, fontsize=9, fontweight="bold", pad=3, loc="left")

    cols      = list(df.columns)
    cell_data = [[_fmt(v) for v in row] for row in df.itertuples(index=False)]
    n_rows    = len(cell_data)

    # Find the class column index for row-type identification
    cls_idx = cols.index("class") if "class" in cols else 0

    tbl = ax.table(
        cellText=cell_data,
        colLabels=cols,
        loc="upper center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.auto_set_column_width(col=list(range(len(cols))))

    # Header style
    for j in range(len(cols)):
        tbl[(0, j)].set_facecolor("#cfd8e8")
        tbl[(0, j)].set_text_props(fontweight="bold")

    # Per-row styling
    for i, row_data in enumerate(cell_data):
        row_i   = i + 1  # matplotlib table row (0 = header)
        cls_val = row_data[cls_idx]

        if cls_val in ("mean (all)", "mean (w/ GT)"):
            for j in range(len(cols)):
                tbl[(row_i, j)].set_facecolor("#e8e8d0")
                tbl[(row_i, j)].set_text_props(fontstyle="italic")
        elif no_gt_classes and cls_val in no_gt_classes:
            # Grey out classes with no GT so they are visually distinct
            for j in range(len(cols)):
                tbl[(row_i, j)].set_facecolor("#f0f0f0")
                tbl[(row_i, j)].set_text_props(color="#999999")


# ──────────────────────────────────────────────────────────────────────────────
# Page 1 – metric tables
# ──────────────────────────────────────────────────────────────────────────────

def _page1_tables(pdf: PdfPages, df: pd.DataFrame, summary: dict) -> None:
    """Page 1: global metrics table + per-class metrics table + AP bar chart."""
    n_class_rows = len(df)  # classes + 1 or 2 mean rows

    fig = plt.figure(figsize=(_PAGE_W, _PAGE_H))
    fig.suptitle("Evaluation Report — Metrics Summary", fontsize=12,
                 fontweight="bold", y=0.98)

    # 4 rows: global table | spacer | per-class table | bar chart
    # Spacer row gives independent gap control between the two tables.
    no_gt_classes_pre = getattr(_build_dataframe, "_no_gt_classes", set())
    n_global_rows = 2 if no_gt_classes_pre else 1
    gs = GridSpec(4, 1, figure=fig,
                  height_ratios=[n_global_rows, 0.6, n_class_rows, 11],
                  hspace=0.08,
                  top=0.85, bottom=0.10, left=0.06, right=0.97)

    # ── Global metrics table ──
    ax_g = fig.add_subplot(gs[0])
    ax_g.axis("off")
    ax_g.set_title("Global Metrics", fontsize=9, fontweight="bold", pad=3, loc="left")

    tp_errors     = summary.get("tp_errors", {})
    no_gt_classes = getattr(_build_dataframe, "_no_gt_classes", set())
    valid_mean_ap = getattr(_build_dataframe, "_valid_mean_ap", float("nan"))

    _te = lambda k: round(float(tp_errors.get(k, float("nan"))), 3)
    ate, ase, aoe, ave, aae = (_te("trans_err"), _te("scale_err"),
                                _te("orient_err"), _te("vel_err"), _te("attr_err"))

    if no_gt_classes:
        # Recompute NDS with corrected mAP: (5·mAP + Σ(1−min(1,err))) / 10
        tp_score_sum  = sum(1 - min(1.0, e) for e in [ate, ase, aoe, ave, aae]
                            if not np.isnan(e))
        corrected_nds = round((5 * valid_mean_ap + tp_score_sum) / 10, 3) \
                        if not np.isnan(valid_mean_ap) else float("nan")
        orig_map = round(float(summary.get("mean_ap", float("nan"))), 3)
        orig_nds = round(float(summary.get("nd_score", float("nan"))), 3)

        g_cols = ["", "mAP", "mATE", "mASE", "mAOE", "mAVE", "mAAE", "NDS"]
        g_data = [
            ["all classes",  _fmt(orig_map),      _fmt(ate), _fmt(ase),
             _fmt(aoe), _fmt(ave), _fmt(aae),  _fmt(orig_nds)],
            ["w/ GT only",   _fmt(valid_mean_ap), _fmt(ate), _fmt(ase),
             _fmt(aoe), _fmt(ave), _fmt(aae),  _fmt(corrected_nds)],
        ]
    else:
        orig_map = round(float(summary.get("mean_ap", float("nan"))), 3)
        orig_nds = round(float(summary.get("nd_score", float("nan"))), 3)
        g_cols = ["mAP", "mATE", "mASE", "mAOE", "mAVE", "mAAE", "NDS"]
        g_data = [[_fmt(orig_map), _fmt(ate), _fmt(ase),
                   _fmt(aoe), _fmt(ave), _fmt(aae), _fmt(orig_nds)]]

    tbl_g = ax_g.table(
        cellText=g_data,
        colLabels=g_cols,
        loc="upper center",
        cellLoc="center",
    )
    tbl_g.auto_set_font_size(False)
    tbl_g.set_fontsize(9)
    tbl_g.auto_set_column_width(col=list(range(len(g_cols))))
    for j in range(len(g_cols)):
        tbl_g[(0, j)].set_facecolor("#cfd8e8")
        tbl_g[(0, j)].set_text_props(fontweight="bold")
    # Style data rows
    tbl_g[(1, 0)].set_facecolor("#e8e8d0")
    if no_gt_classes:
        for j in range(len(g_cols)):
            tbl_g[(2, j)].set_facecolor("#e8e8d0")
            tbl_g[(2, j)].set_text_props(fontstyle="italic")

    # spacer row — invisible
    fig.add_subplot(gs[1]).axis("off")

    # ── Per-class table ──
    counts_th    = getattr(_build_dataframe, "_counts_th",     2.0)
    no_gt_classes = getattr(_build_dataframe, "_no_gt_classes", set())
    valid_mean_ap = getattr(_build_dataframe, "_valid_mean_ap", float("nan"))
    ax_c = fig.add_subplot(gs[2])
    _draw_table(ax_c, df,
                f"Per-Class Metrics  (AP variants · TP errors · NDS · counts at {counts_th} m)",
                no_gt_classes=no_gt_classes)

    # ── AP bar chart ──
    ax_bar   = fig.add_subplot(gs[3])
    mean_labels = {"mean (all)", "mean (w/ GT)"}
    class_df = df[~df["class"].isin(mean_labels)]
    classes  = class_df["class"].tolist()
    aps      = class_df["AP"].tolist()

    # Use the corrected mAP (excluding 0-GT classes) for the reference line
    mean_ap = valid_mean_ap

    x     = np.arange(len(classes))
    colors = ["#b0b0b0" if (no_gt_classes and c in no_gt_classes) else "#4c8cbf"
              for c in classes]
    bars  = ax_bar.bar(x, aps, color=colors, edgecolor="white", width=0.6)

    if not np.isnan(mean_ap):
        label = f"mAP = {mean_ap:.3f}"
        if no_gt_classes:
            label += "  (w/ GT only)"
        ax_bar.axhline(mean_ap, color="crimson", linestyle="--", linewidth=1.5,
                       label=label)
        ax_bar.legend(fontsize=9)

    for bar, val in zip(bars, aps):
        if not np.isnan(val):
            ax_bar.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    valid = [v for v in aps if not np.isnan(v)]
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(classes, rotation=30, ha="right", fontsize=9)
    ax_bar.set_ylabel("AP  (mean over distance thresholds)", fontsize=9)
    ax_bar.set_title("Per-Class Average Precision", fontsize=10, fontweight="bold", pad=3)
    ax_bar.set_ylim(0, min(1.05, (max(valid) if valid else 0.1) * 1.25 + 0.05))
    ax_bar.grid(axis="y", alpha=0.3)

    pdf.savefig(fig)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Page 3 – PR curves
# ──────────────────────────────────────────────────────────────────────────────

def _page3_pr_curves(
    pdf: PdfPages,
    details: dict,
    threshold: float = 2.0,
) -> None:
    """Page 3: precision–recall curves for all classes at one threshold."""
    entries = []
    for key, val in details.items():
        if ":" not in key:
            continue
        cls, th = key.rsplit(":", 1)
        try:
            if abs(float(th) - threshold) < 1e-6:
                entries.append((cls, val.get("recall", []), val.get("precision", [])))
        except ValueError:
            continue

    if not entries:
        fig, ax = plt.subplots(figsize=(_PAGE_W, _PAGE_H))
        ax.text(0.5, 0.5,
                f"No PR data found for distance threshold {threshold} m",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
        ax.set_title(f"PR Curves — threshold {threshold} m", fontsize=11, fontweight="bold")
        pdf.savefig(fig)
        plt.close(fig)
        return

    entries.sort(key=lambda e: e[0])
    n     = len(entries)
    ncols = min(2, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(_PAGE_W, _PAGE_H),
                             squeeze=False)
    fig.suptitle(
        f"Precision–Recall Curves  (distance threshold = {threshold} m)",
        fontsize=11, fontweight="bold",
    )

    for idx, (cls, recall, precision) in enumerate(entries):
        r, c = divmod(idx, ncols)
        ax   = axes[r][c]
        if recall and precision:
            ax.plot(recall, precision, color="#4c8cbf", linewidth=1.5)
            ax.fill_between(recall, precision, alpha=0.12, color="#4c8cbf")
            ap = float(_trapz(precision, recall)) if len(recall) > 1 else float("nan")
            ax.text(0.97, 0.97, f"AP={ap:.3f}",
                    ha="right", va="top", transform=ax.transAxes, fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.75))
        else:
            ax.text(0.5, 0.5, "no data",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9)

        ax.set_title(cls, fontsize=9, fontweight="bold")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
        ax.set_xlabel("Recall", fontsize=8)
        ax.set_ylabel("Precision", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Page 4 – TP error overview
# ──────────────────────────────────────────────────────────────────────────────

def _page4_tp_errors(pdf: PdfPages, df: pd.DataFrame) -> None:
    """Page 4: per-class TP errors (ATE, ASE, AOE, AVE, AAE) as stacked bars."""
    class_df = df[df["class"] != "mean"].copy()
    classes  = class_df["class"].tolist()
    x        = np.arange(len(classes))

    error_specs = [
        ("ATE", "#4c8cbf", "ATE – Translation Error (m)"),
        ("ASE", "#e07b39", "ASE – Scale Error  (1 − IoU)"),
        ("AOE", "#5aab5a", "AOE – Orientation Error (rad)"),
        ("AVE", "#a86ab5", "AVE – Velocity Error (m/s)"),
        ("AAE", "#c75454", "AAE – Attribute Error"),
    ]

    fig, axes = plt.subplots(len(error_specs), 1,
                             figsize=(_PAGE_W, _PAGE_H))
    fig.suptitle("Per-Class TP Error Overview", fontsize=11, fontweight="bold")

    for i, (col, color, label) in enumerate(error_specs):
        ax   = axes[i]
        vals = class_df[col].tolist()
        bars = ax.bar(x, vals, color=color, edgecolor="white", width=0.6)

        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.001,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        show_labels = (i == len(error_specs) - 1)
        ax.set_xticklabels(
            classes if show_labels else [""] * len(classes),
            rotation=30, ha="right", fontsize=9,
        )
        ax.set_ylabel(col, fontsize=8)
        ax.set_title(label, fontsize=8, loc="left")
        ax.grid(axis="y", alpha=0.3)

        finite = [v for v in vals if not np.isnan(v)]
        if finite:
            ax.set_ylim(0, max(finite) * 1.25 + 0.005)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def make_report(
    eval_dir: Path,
    out_path: Optional[Path] = None,
    csv_path: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """
    Generate a PDF report and a CSV metrics table from an existing eval folder.

    Parameters
    ----------
    eval_dir : Path
        Folder that contains ``metrics_summary.json`` and ``metrics_details.json``.
    out_path : Path, optional
        Destination for the PDF report.  Defaults to ``<eval_dir>/report.pdf``.
    csv_path : Path, optional
        Destination for the CSV table.  Defaults to ``<eval_dir>/metrics_table.csv``.

    Returns
    -------
    Tuple[Path, Path]
        ``(pdf_path, csv_path)`` — absolute paths of the written files.
    """
    eval_dir = Path(eval_dir)
    if out_path is None:
        out_path = eval_dir / "report.pdf"
    if csv_path is None:
        csv_path = eval_dir / "metrics_table.csv"

    summary_path = eval_dir / "metrics_summary.json"
    details_path = eval_dir / "metrics_details.json"

    if not summary_path.exists():
        raise FileNotFoundError(f"metrics_summary.json not found in {eval_dir}")
    if not details_path.exists():
        raise FileNotFoundError(f"metrics_details.json not found in {eval_dir}")

    summary = _load_json(summary_path)
    details = _load_json(details_path)

    df = _build_dataframe(summary)

    # ── CSV ──────────────────────────────────────────────────────────────────
    df.to_csv(csv_path, index=False)
    print(f"[report] CSV  → {csv_path}")

    # ── PDF (4 pages) ─────────────────────────────────────────────────────────
    with PdfPages(out_path) as pdf:
        _page1_tables(pdf, df, summary)
        _page3_pr_curves(pdf, details, threshold=2.0)
        _page4_tp_errors(pdf, df)

    print(f"[report] PDF  → {out_path}")
    return Path(out_path), Path(csv_path)


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a PDF report and CSV table from an evaluation output folder. "
            "Reads metrics_summary.json and metrics_details.json; does not re-run evaluation."
        )
    )
    parser.add_argument(
        "--eval_dir", type=Path, required=True,
        help="Path to the eval folder (must contain metrics_summary.json and metrics_details.json)",
    )
    parser.add_argument(
        "--out_dir", type=Path, default=None,
        help="Directory to write report.pdf and metrics_table.csv  [default: <eval_dir>]",
    )
    args = parser.parse_args()

    out_dir = args.out_dir if args.out_dir is not None else args.eval_dir
    pdf_path, csv_path = make_report(
        eval_dir=args.eval_dir,
        out_path=out_dir / "report.pdf",
        csv_path=out_dir / "metrics_table.csv",
    )
    print(f"\nDone.\n  Report : {pdf_path}\n  CSV    : {csv_path}")


if __name__ == "__main__":
    main()
