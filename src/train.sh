#!/bin/bash

python3 train.py \
    --model DQN \
    --env BasicRealDataEnv \
    --asset_code AAPL AMZN GOOGL \
    --start_date 2024-01-01 \
    --end_date 2024-02-01 \
    --interval 1d