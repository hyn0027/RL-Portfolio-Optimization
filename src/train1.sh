#!/bin/bash

# python3 run.py \
#     --agent MultiDQN \
#     --env DiscreteRealDataEnv1 \
#     --network MultiValueLSTM \
#     --asset_codes NVDA GOOGL MSFT \
#     --start_date 2017-11-01 \
#     --end_date 2023-01-01 \
#     --interval 1d \
#     --annual_sample 252 \
#     --device cpu \
#     --pretrain_epochs 0 \
#     --train_batch_size 32 \
#     --train_epochs 500 \
#     --train_learning_rate 1e-7 \
#     --episode_length 250 \
#     --DQN_epsilon_decay 0.9 \
#     --window_size 20 \
#     --mode train \
#     --model_save_path ../model \
#     --initial_balance 1000000

# python3 run.py \
#     --agent MultiDQN \
#     --initial_balance 1000000 \
#     --env DiscreteRealDataEnv1 \
#     --network MultiValueLSTM \
#     --asset_codes NVDA GOOGL MSFT \
#     --start_date 2022-11-01 \
#     --end_date 2024-01-01 \
#     --interval 1d \
#     --annual_sample 250 \
#     --device cuda \
#     --window_size 20 \
#     --mode test \
#     --model_load_path  ../model/MultiDQN1/Q_net_last_checkpoint.pth \
#     --evaluator_saving_path ../evaluator/MultiDQN+DiscreteRealDataEnv1.json  > test.log 2>&1

# python3 run.py \
#     --agent DQN \
#     --env BasicDiscreteRealDataEnv \
#     --network ValueDNN \
#     --asset_codes NVDA GOOGL \
#     --start_date 2020-01-01 \
#     --end_date 2023-01-01 \
#     --interval 1d \
#     --annual_sample 252 \
#     --device cpu \
#     --train_batch_size 4 \
#     --train_epochs 10 \
#     --window_size 8 \
#     --mode train \
#     --initial_balance 1000000

# python3 run.py \
#     --agent DQN \
#     --env BasicDiscreteRealDataEnv \
#     --network ValueDNN \
#     --asset_codes NVDA GOOGL \
#     --start_date 2023-01-01 \
#     --end_date 2024-01-01 \
#     --interval 1d \
#     --annual_sample 252 \
#     --device cpu \
#     --window_size 8 \
#     --mode test \
#     --initial_balance 1000000 \
#     --model_load_path  ../model/Q_net_last_checkpoint.pth \
#     --evaluator_saving_path ../evaluator/DQN+BasicDiscreteRealDataEnv.json

# python3 run.py \
#     --agent DPG \
#     --env ContinuousRealDataEnv1 \
#     --network PolicyWeightLSTM \
#     --asset_codes NVDA GOOGL MSFT AMZN EPAM WBD \
#     --start_date 2017-11-01 \
#     --end_date 2023-01-01 \
#     --train_learning_rate 0.001 \
#     --interval 1d \
#     --annual_sample 252 \
#     --device cpu \
#     --train_batch_size 1 \
#     --train_epochs 5 \
#     --window_size 50 \
#     --mode train \
#     --initial_balance 1000000 \
#     --replay_sample_distribution geometric

# python3 run.py \
#     --agent DPG \
#     --env ContinuousRealDataEnv1 \
#     --network PolicyWeightLSTM \
#     --asset_codes NVDA GOOGL MSFT AMZN EPAM WBD \
#     --start_date 2022-11-01 \
#     --end_date 2024-01-01 \
#     --interval 1d \
#     --annual_sample 252 \
#     --device cuda \
#     --window_size 50 \
#     --mode test \
#     --initial_balance 1000000 \
#     --model_load_path  ../model/model_last_checkpoint.pth \
#     --evaluator_saving_path ../evaluator/DPG+ContinuousRealDataEnv1.json

# python3 run.py \
#     --agent DPG \
#     --env BasicContinuousRealDataEnv \
#     --network PolicyCNN \
#     --train_learning_rate 0.0001 \
#     --asset_codes NVDA GOOGL MSFT AMZN \
#     --start_date 2020-01-01 \
#     --end_date 2023-01-01 \
#     --interval 1d \
#     --annual_sample 252 \
#     --device cpu \
#     --train_batch_size 4 \
#     --train_epochs 10 \
#     --window_size 50 \
#     --mode train \
#     --initial_balance 1000000

# python3 run.py \
#     --agent DPG \
#     --env BasicContinuousRealDataEnv \
#     --network PolicyCNN \
#     --asset_codes NVDA GOOGL MSFT AMZN \
#     --start_date 2023-01-01 \
#     --end_date 2024-01-01 \
#     --interval 1d \
#     --annual_sample 252 \
#     --device cpu \
#     --window_size 50 \
#     --mode test \
#     --initial_balance 1000000 \
#     --model_load_path  ../model/model_last_checkpoint.pth \
#     --evaluator_saving_path ../evaluator/DPG+BasicContinuousRealDataEnv.json