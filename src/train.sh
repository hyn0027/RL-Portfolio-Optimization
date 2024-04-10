#!/bin/bash

# python3 run.py \
#     --agent MultiDQN \
#     --env DiscreteRealDataEnv1 \
#     --network MultiValueLSTM \
#     --asset_codes NVDA \
#     --start_date 2020-01-01 \
#     --end_date 2023-01-01 \
#     --interval 1d \
#     --annual_sample 252 \
#     --device cpu \
#     --pretrain_epochs 0 \
#     --train_batch_size 6 \
#     --train_epochs 10 \
#     --train_learning_rate 0.001 \
#     --episode_length 250 \
#     --window_size 4 \
#     --mode train \
#     --initial_balance 1000000

python3 run.py \
    --agent DQN \
    --env BasicDiscreteRealDataEnv \
    --network ValueDNN \
    --asset_codes NVDA GOOGL \
    --start_date 2020-01-01 \
    --end_date 2023-01-01 \
    --interval 1d \
    --annual_sample 252 \
    --device cpu \
    --train_batch_size 4 \
    --train_epochs 10 \
    --window_size 8 \
    --mode train \
    --initial_balance 1000000

python3 run.py \
    --agent DQN \
    --env BasicDiscreteRealDataEnv \
    --network ValueDNN \
    --asset_codes NVDA GOOGL \
    --start_date 2023-01-01 \
    --end_date 2024-01-01 \
    --interval 1d \
    --annual_sample 252 \
    --device cpu \
    --window_size 8 \
    --mode test \
    --initial_balance 1000000 \
    --model_load_path  ../model/Q_net_last_checkpoint.pth

# python3 run.py \
#     --agent MultiDQN \
#     --initial_balance 1000000 \
#     --env DiscreteRealDataEnv1 \
#     --network MultiValueLSTM \
#     --asset_codes NVDA \
#     --start_date 2023-01-01 \
#     --end_date 2024-01-01 \
#     --interval 1d \
#     --annual_sample 250 \
#     --device cpu \
#     --window_size 4 \
#     --mode test \
#     --model_load_path  ../model/Q_net_last_checkpoint.pth