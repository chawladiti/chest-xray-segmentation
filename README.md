# chexlocalize-medical-imaging

A clean, reproducible computational medical imaging framework built on the
**CheXlocalize** chest X-ray localization/segmentation dataset.

> **Status:** Architecture initialized. No dataset, models, training, or results
> yet. See [`PROJECT_STATUS.md`](PROJECT_STATUS.md).

## Overview

This project provides a professional, reproducible foundation for chest X-ray
segmentation research: a configuration system, isolated model wrappers,
model-independent evaluation, and DEMO/FULL development modes. Components are
added incrementally; nothing is faked.

## Research question

_To be completed._ (Broadly: how do supervised segmentation and foundation
segmentation models — including appropriately fine-tuned variants — compare on
CheXlocalize chest X-ray localization, measured by Dice?)

## Dataset

**CheXlocalize** — chest X-ray localization/segmentation.

> ⚠️ **Not downloaded yet.** CheXlocalize's official validation/test images and
> annotations are used as a **localization benchmark**. This project will **not**
> train on the official test set, will **not** leak test annotations into
> training, and will **not** arbitrarily split the official test set. The real
> dataset structure will be inspected before any split/training strategy is
> chosen. _Details to be completed._

## Models

Each model is isolated under `models/` and added in its own step:

| Model         | Type                                   | Status                |
| ------------- | -------------------------------------- | --------------------- |
| nnU-Net       | Supervised segmentation (trainable)    | ⛔ Not implemented     |
| SAM 2         | Foundation model (inference / fine-tune) | ⛔ Not implemented   |
| MedSAM        | Foundation model (inference / fine-tune) | ⛔ Not implemented   |
| NVIDIA method | Published-method reproduction/adaptation | ⛔ Not implemented   |

**Terminology rule:** a model is called **fine-tuned only if its weights are
actually updated using training data.** Pretrained/zero-shot inference is never
labeled fine-tuning.

## Methodology

_To be completed._ Experiment types planned:

- Supervised model training
- Pretrained foundation-model inference
- Foundation-model fine-tuning (only where technically supported)
- Published-method reproduction/adaptation

Training contract for trainable models: max **20 epochs**, validation every
epoch, track validation Dice, save best checkpoint, configurable early-stopping
patience / learning rate / batch size, saved training history, and resume from
checkpoint. See [`configs/project.yaml`](configs/project.yaml).

## Evaluation

Model-independent evaluation (in `evaluation/`) will compute:

- **Dice coefficient**
- **Dice percentage** = Dice × 100
- **Absolute Dice improvement** = new_Dice − baseline_Dice
- **Percentage improvement** = ((new_Dice − baseline_Dice) / baseline_Dice) × 100

_Implementation to be completed._ No numerical results are invented.

## Results

_To be completed — populated only from real experiments._ DEMO and FULL results
are stored separately and never mixed.

## Reproducibility

- Central config: [`configs/project.yaml`](configs/project.yaml) +
  [`configs/loader.py`](configs/loader.py).
- No machine-specific paths are hard-coded; paths are configurable via
  environment variables (see [`.env.example`](.env.example)).
- DEMO/FULL isolation via `PROJECT_MODE`.
- Deterministic split definitions committed under `data/splits/`.

## Limitations

_To be completed._ (Will document dataset constraints, benchmark-usage caveats,
compute limits, and scope of any method reproduction/adaptation.)

## Project structure

```
chexlocalize-medical-imaging/
├── data/            # raw / processed / splits (dataset never committed)
├── configs/         # project.yaml + config loader
├── models/          # nnunet / sam2 / medsam / nvidia_method (isolated)
├── preprocessing/   # validation, image & mask/annotation processing
├── evaluation/      # Dice, aggregation, comparison, plots
├── scripts/         # CLI entry points (prep / train / infer / eval)
├── results/         # CSV/JSON results & figures (real only)
├── predictions/     # prediction masks (kept out of Git when large)
├── checkpoints/     # trained checkpoints (kept out of Git)
├── notebooks/       # exploratory analysis only
├── frontend/        # future visualization/demo interface
├── docs/            # methodology, design, reproducibility, limitations
└── tests/           # unit / data-validation / evaluation tests
```

## Installation

```bash
git clone <your-repo-url> chexlocalize-medical-imaging
cd chexlocalize-medical-imaging
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # infrastructure deps only for now
cp .env.example .env                 # optional: override paths/secrets locally
```

## Usage

Validate the project skeleton and configuration:

```bash
python scripts/validate_project.py
pytest -q
```

Switch development mode (DEMO by default):

```bash
PROJECT_MODE=full python scripts/validate_project.py
```

Further usage (data prep, training, inference, evaluation) — _to be completed._
