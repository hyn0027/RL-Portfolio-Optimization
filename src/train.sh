#!/bin/bash

python3 train.py \
    --agent MultiDQN \
    --env DiscreteRealDataEnv1 \
    --network MultiDQN_LSTM \
    --asset_codes AAPL \
    --start_date 2015-01-01 \
    --end_date 2023-01-01 \
    --interval 1d \
    --device cuda \
    --pretrain_epochs 50 \
    --train_batch_size 16