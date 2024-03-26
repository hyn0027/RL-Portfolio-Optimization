#!/bin/bash

python3 run.py \
    --agent MultiDQN \
    --env DiscreteRealDataEnv1 \
    --network MultiDQN_LSTM \
    --asset_codes NVDA MSFT \
    --start_date 2020-01-01 \
    --end_date 2023-01-01 \
    --interval 1d \
    --annual_sample 252 \
    --device cpu \
    --pretrain_epochs 50 \
    --train_batch_size 4 \
    --train_epochs 50 \
    --episode_length 252 \
    --window_size 10 \
    --mode train

# python3 run.py \
#     --agent MultiDQN \
#     --env DiscreteRealDataEnv1 \
#     --network MultiDQN_LSTM \
#     --asset_codes CAD=X \
#     --start_date 2023-01-01 \
#     --end_date 2024-01-01 \
#     --interval 1d \
#     --annual_sample 252 \
#     --device cpu \
#     --window_size 5 \
#     --mode test \
#     --model_load_path  ../model/Q_net_last_checkpoint.pth