#!/bin/bash
# Train FCN2 linear and erf networks, then compare predictions

set -e

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

D=${1:-10}
P=${2:-50}
N=${3:-200}
EPOCHS=${4:-2000000}
LR=${5:-1e-5}
DEVICE=${6:-cpu}

echo "Running FCN2 linear vs erf comparison experiment"
echo "  d=$D, P=$P, N=$N"
echo "  epochs=$EPOCHS, lr=$LR, device=$DEVICE"
echo ""

# Train linear network
echo "=== Training linear network ==="
python train_fcn2_linear.py \
    --d "$D" \
    --P "$P" \
    --N "$N" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --device "$DEVICE"

# Train erf network
echo ""
echo "=== Training erf network ==="
python train_fcn2_erf.py \
    --d "$D" \
    --P "$P" \
    --N "$N" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --device "$DEVICE"

# Compare outputs
echo ""
echo "=== Comparing linear vs erf predictions ==="
python compare_linear_vs_erf.py \
    --d "$D" \
    --P "$P" \
    --N "$N" \
    --device "$DEVICE"

echo ""
echo "All done!"
