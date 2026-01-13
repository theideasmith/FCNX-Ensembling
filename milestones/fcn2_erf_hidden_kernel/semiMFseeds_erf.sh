#!/bin/bash

DEVICE="cuda:0"
# if ! python - <<'PY'
# import torch
# print(torch.cuda.is_available())
# PY
# then
#   DEVICE="cpu"
# fi

ENSEMBLE_SIZE=10
EPOCHS=50000000
LR=3e-5
N=800
CHI=80
D=100
P=1200
TEMP=4.0

for SEED in 0 1 2 3; do
  python train_fcn2_erf.py \
    --d $D \
    --P $P \
    --N $N \
    --epochs $EPOCHS \
    --temperature $TEMP \
    --lr $LR \
    --chi $CHI \
    --ens $ENSEMBLE_SIZE \
    --device "$DEVICE" \
    --dataset-seed $SEED \
    --eps 0.03 \
    > train_seed${SEED}.log 2>&1 &
done

wait
echo "All training jobs finished."