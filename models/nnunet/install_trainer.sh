#!/usr/bin/env bash
# Install the custom CheXlocalize trainer into the active nnU-Net package so that
# `nnUNetv2_train ... -tr nnUNetTrainer_CheXlocalize` can discover it.
# Run on the pod inside the nnunet env:  bash models/nnunet/install_trainer.sh
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$(python -c 'import nnunetv2, os; print(os.path.join(os.path.dirname(nnunetv2.__file__), "training", "nnUNetTrainer", "variants", "chexlocalize"))')"
mkdir -p "$DEST"
touch "$DEST/__init__.py"
cp "$HERE/nnUNetTrainer_CheXlocalize.py" "$DEST/"
python -c "from nnunetv2.utilities.find_class_by_name import recursive_find_python_class" 2>/dev/null || true
echo "installed -> $DEST/nnUNetTrainer_CheXlocalize.py"
