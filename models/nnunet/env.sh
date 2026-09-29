# Source this to configure nnU-Net v2 paths (on the RunPod pod).
#   source models/nnunet/env.sh
# Override the root by exporting NNUNET_ROOT before sourcing.
export NNUNET_ROOT="${NNUNET_ROOT:-/workspace/nnunet}"
export nnUNet_raw="${NNUNET_ROOT}/nnUNet_raw"
export nnUNet_preprocessed="${NNUNET_ROOT}/nnUNet_preprocessed"
export nnUNet_results="${NNUNET_ROOT}/nnUNet_results"
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"
echo "nnUNet_raw=$nnUNet_raw"
echo "nnUNet_preprocessed=$nnUNet_preprocessed"
echo "nnUNet_results=$nnUNet_results"
