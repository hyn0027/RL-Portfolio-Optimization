#!/bin/bash

python3 train.py \
    --agent MultiDQN \
    --env DiscreteRealDataEnv1 \
    --network MultiDQN_LSTM \
    --asset_codes AAPL AMZN \
    --start_date 2021-01-01 \
    --end_date 2023-01-01 \
    --interval 1d \
    --window_size 20 \
    --device cpu