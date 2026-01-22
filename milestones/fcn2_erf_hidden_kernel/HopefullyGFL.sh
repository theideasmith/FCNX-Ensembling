#!/bin/bash

DEVICE="cuda:0"
# if ! python - <<'PY'
# import torch
# print(torch.cuda.is_available())
# PY
# then
#   DEVICE="cpu"
# fi

ENSEMBLE_SIZE=2
EPOCHS=50000000
LR=3e-5
N=600
CHI=60
D=100
P=6000
TEMP=2.0

for SEED in 0 1 2; do
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
    --eps 0.00 \
    > train_P${P}_D${D}_chi${CHI}_TEMP${TEMP}_seed${SEED}.log 2>&1 &
done

wait
echo "All training jobs finished."