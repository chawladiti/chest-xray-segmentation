"""Generate QC sample visualizations: X-ray + GT segmentation overlay + label.

Selects at least `visualization.num_samples` (>=10) samples spread across
different pathologies (only rows with a non-empty mask), and for each renders:
  * the original X-ray (grayscale)
  * the ground-truth segmentation overlaid in a translucent color
  * a title with the pathology name, CXR ID, and split

Outputs individual PNGs plus a combined grid into processed/visualizations/.
This is QC only — it computes NO metrics and fabricates NO results.

Run standalone:
    python preprocessing/visualize_samples.py
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from preprocessing.config import PreprocessingContext, load_preprocessing_context


def _read_manifest(ctx: PreprocessingContext) -> List[Dict[str, Any]]:
    path = ctx.manifests_dir / ctx.prep["output"]["manifest_filename"]
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}. Run build_manifest.py first.")
    with open(path, "r", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _select_samples(rows: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    """Pick masked rows spread across pathologies (round-robin)."""
    by_path: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        if str(r.get("has_mask", "")).lower() in ("true", "1") and r.get("mask_path"):
            by_path.setdefault(r["pathology"], []).append(r)
    selected: List[Dict[str, Any]] = []
    # deterministic ordering
    buckets = [sorted(v, key=lambda x: x["image_id"]) for _, v in sorted(by_path.items())]
    idx = 0
    while len(selected) < n and any(idx < len(b) for b in buckets):
        for b in buckets:
            if idx < len(b):
                selected.append(b[idx])
                if len(selected) >= n:
                    break
        idx += 1
    return selected


def _load_downscaled(path: Path, max_side: int):
    im = Image.open(path)
    scale = max_side / max(im.size) if max(im.size) > max_side else 1.0
    if scale < 1.0:
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))))
    return im, scale


def _overlay_figure(row, viz, out_path: Path) -> None:
    max_side = int(viz["max_display_size"])
    alpha = float(viz["alpha"])
    color = np.array(viz["overlay_color"], dtype=np.float32) / 255.0

    img, _ = _load_downscaled(Path(row["image_path"]), max_side)
    img_gray = np.asarray(img.convert("L"), dtype=np.float32) / 255.0

    mask_im, _ = _load_downscaled(Path(row["mask_path"]), max_side)
    mask = (np.asarray(mask_im.convert("L")) > 127).astype(np.float32)
    # align mask to image size if rounding differs by a pixel
    if mask.shape != img_gray.shape:
        mask_im = mask_im.resize((img_gray.shape[1], img_gray.shape[0]))
        mask = (np.asarray(mask_im.convert("L")) > 127).astype(np.float32)

    rgb = np.stack([img_gray] * 3, axis=-1)
    overlay = rgb.copy()
    for c in range(3):
        overlay[..., c] = np.where(mask > 0,
                                   (1 - alpha) * rgb[..., c] + alpha * color[c],
                                   rgb[..., c])

    fig, axes = plt.subplots(1, 2, figsize=(8, 4.5))
    axes[0].imshow(rgb)
    axes[0].set_title("Original X-ray", fontsize=9)
    axes[1].imshow(overlay)
    axes[1].set_title("GT segmentation overlay", fontsize=9)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"{row['pathology']}  |  {row['image_id']}  ({row['split']})", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=int(viz["dpi"]), bbox_inches="tight")
    plt.close(fig)


def _grid_figure(rows, viz, out_path: Path) -> None:
    max_side = int(viz["max_display_size"])
    alpha = float(viz["alpha"])
    color = np.array(viz["overlay_color"], dtype=np.float32) / 255.0
    n = len(rows)
    cols = 4
    grid_rows = math.ceil(n / cols)
    fig, axes = plt.subplots(grid_rows, cols, figsize=(cols * 3, grid_rows * 3))
    axes = np.atleast_1d(axes).ravel()
    for i, row in enumerate(rows):
        img, _ = _load_downscaled(Path(row["image_path"]), max_side)
        g = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
        mim, _ = _load_downscaled(Path(row["mask_path"]), max_side)
        m = (np.asarray(mim.convert("L")) > 127).astype(np.float32)
        if m.shape != g.shape:
            mim = mim.resize((g.shape[1], g.shape[0]))
            m = (np.asarray(mim.convert("L")) > 127).astype(np.float32)
        rgb = np.stack([g] * 3, axis=-1)
        for c in range(3):
            rgb[..., c] = np.where(m > 0, (1 - alpha) * rgb[..., c] + alpha * color[c], rgb[..., c])
        axes[i].imshow(rgb)
        axes[i].set_title(f"{row['pathology']}\n{row['split']}", fontsize=7)
        axes[i].axis("off")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.suptitle("CheXlocalize GT segmentation QC samples", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=int(viz["dpi"]), bbox_inches="tight")
    plt.close(fig)


def visualize(ctx: PreprocessingContext) -> List[Path]:
    ctx.ensure_output_dirs()
    viz = ctx.prep["visualization"]
    n = max(10, int(viz["num_samples"]))
    rows = _read_manifest(ctx)
    samples = _select_samples(rows, n)
    if not samples:
        print("[viz] no non-empty masks found to visualize.")
        return []

    written: List[Path] = []
    for i, row in enumerate(samples):
        safe = row["pathology"].replace(" ", "_")
        out_path = ctx.visualizations_dir / f"sample_{i:02d}_{safe}_{row['split']}.png"
        _overlay_figure(row, viz, out_path)
        written.append(out_path)

    grid_path = ctx.visualizations_dir / "samples_grid.png"
    _grid_figure(samples, viz, grid_path)
    written.append(grid_path)

    print(f"[viz] wrote {len(written)} figures to {ctx.visualizations_dir}")
    pathologies = sorted({r["pathology"] for r in samples})
    print(f"[viz] samples span {len(pathologies)} pathologies: {pathologies}")
    return written


def main() -> int:
    ctx = load_preprocessing_context()
    visualize(ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
