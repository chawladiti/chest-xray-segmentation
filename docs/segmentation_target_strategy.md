# Segmentation target strategy (TASK 2)

**Decision: (A) one binary segmentation task per pathology — i.e. multi-label,
one independent binary channel per pathology.** A single mutually-exclusive
multi-class label map is rejected. This is a data-driven decision, justified
below.

## Evidence from the actual masks

From `docs/mask_overlap_analysis.md` (scope: all 686 images with ≥1 mask across
val + test):

- **526** images have multiple pathology masks.
- **508** of those (**96.58 %** of multi-mask images; **74.05 %** of all masked
  images) contain at least one **spatially overlapping** pathology pair.
- **3,030** overlapping pathology-mask pairs out of 3,481 checked.
- Most frequent overlaps: Airspace Opacity ↔ Enlarged Cardiomediastinum (270),
  Airspace Opacity ↔ Atelectasis (248), Cardiomegaly ↔ Enlarged Cardiomediastinum
  (243), Enlarged Cardiomediastinum ↔ Support Devices (216), …

The overlap is pervasive and clinically expected: a single pixel can simultaneously
be Airspace Opacity **and** Consolidation/Atelectasis/Edema, or lie within both
the cardiac silhouette (Cardiomegaly) and the mediastinum (Enlarged
Cardiomediastinum).

## Why NOT a single multi-class label map (rejected)

A mutually-exclusive multi-class map assigns **exactly one** class per pixel.
Because 74 % of masked images have overlapping findings, this would force an
arbitrary "winner" per pixel and **destroy ground-truth information** for the
losing class. It is therefore invalid for this dataset.

## Why per-pathology binary / multi-label (chosen)

1. **Overlap-safe:** each pathology is an independent binary mask, so a pixel can
   be positive for several pathologies at once — matching the data.
2. **Per-pair IGNORE handling:** the target trichotomy
   (`VALID_POSITIVE / VALID_NEGATIVE / IGNORE`) is defined **per (image,
   pathology)**. A per-channel representation lets the loss mask out IGNORE
   channels (positive-but-unlocalized) without dropping the whole image.
3. **Model compatibility:**
   - **Foundation models (SAM 2, MedSAM)** operate on **binary** masks per
     prompt/object — a per-pathology binary target maps directly.
   - **nnU-Net** supports either per-region binary training or multi-label
     ("region-based") training; both consume per-channel binary targets. It does
     **not** require a mutually-exclusive label map.
4. **Metric alignment:** Dice is computed per pathology (then aggregated), which
   is exactly a per-binary-channel evaluation.

### "A" vs "B" (they are the same target, different packaging)

- **(A)** one binary task per pathology and **(B)** a multi-channel/multi-label
  tensor are the **same underlying target** — B is simply A stacked along a
  channel axis (`C = 10`). We store targets as **A** (one PNG per
  image×pathology), which is the most flexible on-disk form; training code may
  stack channels into B when a model wants a single multi-channel tensor, and
  will carry a parallel **loss mask** to exclude IGNORE channels.

## Consequences for training (future, not implemented here)

- Targets live at `data/processed/targets/<split>/<image_id>/<Pathology>.png`.
- Manifest `training_targets.csv` carries `target_status`; the loss must:
  - include `VALID_POSITIVE` (mask) and `VALID_NEGATIVE` (all-zero) channels,
  - **exclude** `IGNORE` channels entirely (no false-negative supervision).
- No mutually-exclusive argmax over pathologies at any point.
