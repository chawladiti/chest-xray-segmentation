# nnU-Net v2 experiment (CheXlocalize)

Supervised segmentation baseline using **nnU-Net v2** on the RunPod A100. Uses the
project's already-validated preprocessing / splits / target policy unchanged.

## Label representation decision (important)

CheXlocalize's 10 pathologies overlap **arbitrarily** (any combination; ~74% mask
overlap). nnU-Net's *region-based* training only supports **nested** regions on a
single-integer label map, which cannot represent arbitrary overlap without
enumerating up to 2¹⁰ combined classes. The correct, standard representation is
therefore **one BINARY nnU-Net dataset per pathology** — each pathology is an
independent binary segmentation task, so every overlap is preserved (handled
independently). This mirrors how CheXlocalize itself scores each pathology.

## Data flow (no raw-data transfer)

Conversion runs **locally** (all source data is local) and ships only small
512×512 PNGs to the pod:

- `models/nnunet/convert_to_nnunet.py`
  - TRAIN split  -> `imagesTr` / `labelsTr` (nnU-Net trains + runs its own 5-fold CV)
  - INTERNAL_VAL -> `imagesTs` (+ ground-truth masks saved separately for our eval)
  - Official **TEST** split -> never converted or shipped (**held out**)
  - Policy preserved: VALID_POSITIVE -> real mask; VALID_NEGATIVE -> all-zero label
    (true-negative supervision); **IGNORE -> excluded from supervision**
  - images bilinear-resized; labels nearest-resized + binarized to {0,1}

## Paths (on pod)

`models/nnunet/env.sh` sets (under `NNUNET_ROOT=/workspace/nnunet`):

- `nnUNet_raw=/workspace/nnunet/nnUNet_raw`
- `nnUNet_preprocessed=/workspace/nnunet/nnUNet_preprocessed`
- `nnUNet_results=/workspace/nnunet/nnUNet_results`

## Reproduce the demo

```bash
# LOCAL: build a small balanced demo dataset
python models/nnunet/convert_to_nnunet.py --pathology "Airspace Opacity" \
  --dataset-id 501 --dataset-name Dataset501_AirspaceOpacityDemo \
  --size 512 --max-train 24 --max-val 8
# ship nnunet_data/ to the pod (tar over ssh; COPYFILE_DISABLE=1 on macOS)

# POD:
source models/nnunet/env.sh && . .envs/nnunet/bin/activate
cp -r nnunet_data/nnUNet_raw/Dataset501_AirspaceOpacityDemo "$nnUNet_raw"/
nnUNetv2_plan_and_preprocess -d 501 --verify_dataset_integrity -c 2d
nnUNetv2_train 501 2d 0 -tr nnUNetTrainer_5epochs
nnUNetv2_predict -i "$nnUNet_raw/Dataset501_AirspaceOpacityDemo/imagesTs" \
  -o /workspace/nnunet/predictions/Dataset501_AirspaceOpacityDemo \
  -d 501 -c 2d -f 0 -tr nnUNetTrainer_5epochs -chk checkpoint_best.pth
python models/nnunet/evaluate_predictions.py \
  --pred-dir /workspace/nnunet/predictions/Dataset501_AirspaceOpacityDemo \
  --gt-dir nnunet_data/internal_val_gt/Dataset501_AirspaceOpacityDemo
```

### Demo result (pipeline proof only — NOT a real metric)

5 epochs on 24 tiny cases; checkpoints `checkpoint_best.pth` + `checkpoint_final.pth`;
per-epoch validation (`progress.png`, `summary.json`). Held-out internal-val (8 cases),
scored with the **existing** `evaluation.dice`/`evaluation.iou`:
**mean Dice 0.0254 (2.54%), mean IoU 0.0134**. Deliberately low — this only proves
train → checkpoint → predict → evaluate works end-to-end.

## Full experiment (configured; run explicitly — not auto-run)

`models/nnunet/nnUNetTrainer_CheXlocalize.py` (installed into nnU-Net via
`models/nnunet/install_trainer.sh`; discovery verified). Training contract:

| Requirement | How |
| --- | --- |
| max 20 epochs | `NNUNET_NUM_EPOCHS` (default **20**) |
| validation/progress each epoch | native (`progress.png`, `training_log`, `summary.json`) |
| best checkpoint | native `checkpoint_best.pth` (best EMA pseudo-Dice) |
| resume training | native `nnUNetv2_train ... --c` |
| early stopping / patience | `NNUNET_PATIENCE` (EMA-Dice; ends loop when exceeded) |
| configurable LR | `NNUNET_LR` |
| configurable batch size | `NNUNET_BATCH` (overrides plan) |

Run one pathology end-to-end (after shipping its FULL dataset to `$nnUNet_raw`):

```bash
DID=502 PATHOLOGY="Airspace Opacity" NNUNET_NUM_EPOCHS=20 \
  bash models/nnunet/run_full.sh
```

DEMO vs FULL results are kept separate: demo -> `results/demo/`, full -> `results/full/`.

## Not done here (by design)

No SAM 2 / MedSAM / NVIDIA method. Full 20-epoch runs not started. Dataset
methodology and evaluation logic unchanged.
