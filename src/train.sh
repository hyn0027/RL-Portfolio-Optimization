#!/bin/bash

python3 train.py \
    --model MultiDQN \
    --env DiscreteRealDataEnv1 \
    --network MultiDQN_LSTM \
    --asset_codes AAPL AMZN GOOGL MSFT \
    --start_date 2024-01-01 \
    --end_date 2024-02-01 \
    --interval 1d \
    --window_size 20