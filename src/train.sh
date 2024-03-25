#!/bin/bash

python3 train.py \
    --agent MultiDQN \
    --env DiscreteRealDataEnv1 \
    --network MultiDQN_LSTM \
    --asset_codes AAPL AMZN MSFT \
    --start_date 2015-01-01 \
    --end_date 2023-01-01 \
    --interval 1d \
    --window_size 20 \
    --device cpu \
    --pretrain_epochs 1 \
    --train_batch_size 4