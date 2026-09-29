#!/usr/bin/env bash
# Full nnU-Net v2 experiment for one CheXlocalize pathology (max 20 epochs).
#
# NOT auto-run by the demo. Configure via env, then invoke. Trains one BINARY
# dataset per pathology (arbitrary multi-label overlap -> independent binary tasks).
#
# Usage:
#   DID=502 PATHOLOGY="Airspace Opacity" bash models/nnunet/run_full.sh
#
# Env (all optional except DID + PATHOLOGY):
#   NNUNET_NUM_EPOCHS (20)  NNUNET_LR  NNUNET_BATCH  NNUNET_PATIENCE  SIZE (512)
#   FOLD (0)                RESUME (0/1 -> pass --c to continue)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source models/nnunet/env.sh >/dev/null
. .envs/nnunet/bin/activate

: "${DID:?set DID (e.g. 502)}"; : "${PATHOLOGY:?set PATHOLOGY}"
FOLD="${FOLD:-0}"; SIZE="${SIZE:-512}"
KEY=$(python -c "import re,sys;print(re.sub(r'[^a-z0-9]+','_',sys.argv[1].lower()).strip('_'))" "$PATHOLOGY")
DS="Dataset${DID}_${KEY}"

# 1) build FULL dataset locally is done on the workstation; here we assume it is
#    already present in $nnUNet_raw/$DS (ship it the same way as the demo).
echo "== plan + preprocess + integrity =="
nnUNetv2_plan_and_preprocess -d "$DID" --verify_dataset_integrity -c 2d

# 2) install custom trainer (20 epochs, configurable LR/batch/patience)
bash models/nnunet/install_trainer.sh

# 3) train (best checkpoint + per-epoch validation are native; --c resumes)
RESUME_FLAG=""; [[ "${RESUME:-0}" == "1" ]] && RESUME_FLAG="--c"
echo "== train (max ${NNUNET_NUM_EPOCHS:-20} epochs) =="
nnUNetv2_train "$DID" 2d "$FOLD" -tr nnUNetTrainer_CheXlocalize $RESUME_FLAG

# 4) predict internal_val + evaluate with the EXISTING Dice/IoU evaluator
PRED="/workspace/nnunet/predictions/$DS"
nnUNetv2_predict -i "$nnUNet_raw/$DS/imagesTs" -o "$PRED" -d "$DID" -c 2d -f "$FOLD" \
  -tr nnUNetTrainer_CheXlocalize -chk checkpoint_best.pth
mkdir -p results/full
python models/nnunet/evaluate_predictions.py \
  --pred-dir "$PRED" \
  --gt-dir "nnunet_data/internal_val_gt/$DS" \
  --out "results/full/nnunet_${KEY}.json" \
  --csv "results/full/nnunet_${KEY}_per_case.csv"
echo "DONE: results/full/nnunet_${KEY}.json"
