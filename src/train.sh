#!/bin/bash

python3 run.py \
    --agent MultiDQN \
    --env DiscreteRealDataEnv1 \
    --network MultiValueLSTM \
    --asset_codes ^GSPC ^DJI ^RUT \
    --start_date 2014-01-01 \
    --end_date 2023-01-01 \
    --interval 1d \
    --annual_sample 252 \
    --device cpu \
    --pretrain_epochs 85 \
    --train_batch_size 32 \
    --train_epochs 800 \
    --pretrain_learning_rate 1e-3 \
    --train_learning_rate 0.0000013 \
    --episode_length 229 \
    --DQN_epsilon_decay 0.9998 \
    --replay_sample_unique \
    --window_size 20 \
    --risk_free_return 0.06 \
    --mode train \
    --model_save_path ../model/MultiDQN+^GSPC+^DJI+^RUT2 \
    --replay_window 2000 \
    --trading_size 1000 \
    --initial_balance 1000000 > ../logs/train_MultiDQN+^GSPC+^DJI+^RUT2.log 2>&1


# python3 run.py \
#     --agent MultiDQN \
#     --env DiscreteRealDataEnv1 \
#     --network MultiValueLSTM \
#     --asset_codes ^GSPC ^DJI ^RUT \
#     --start_date 2014-01-01 \
#     --end_date 2023-01-01 \
#     --interval 1d \
#     --annual_sample 252 \
#     --device cpu \
#     --pretrain_epochs 100 \
#     --train_batch_size 32 \
#     --train_epochs 500 \
#     --pretrain_learning_rate 1e-3 \
#     --train_learning_rate 8e-7 \
#     --episode_length 229 \
#     --DQN_epsilon_decay 0.9998 \
#     --replay_sample_unique \
#     --window_size 20 \
#     --risk_free_return 0.06 \
#     --mode train \
#     --model_save_path ../model/MultiDQN+^GSPC+^DJI+^RUT \
#     --replay_window 2000 \
#     --trading_size 1000 \
#     --initial_balance 1000000 > ../logs/train_MultiDQN+^GSPC+^DJI+^RUT.log 2>&1


# python3 run.py \
#     --agent MultiDQN \
#     --env DiscreteRealDataEnv1 \
#     --network MultiValueLSTM \
#     --asset_codes BTC-USD ETH-USD MATIC-USD \
#     --start_date 2023-08-01 \
#     --end_date 2024-01-01 \
#     --interval 1h \
#     --annual_sample 8760 \
#     --device cpu \
#     --pretrain_epochs 50 \
#     --train_batch_size 32 \
#     --train_epochs 500 \
#     --pretrain_learning_rate 1e-3 \
#     --train_learning_rate 8e-7 \
#     --episode_length 195 \
#     --DQN_epsilon_decay 0.999 \
#     --replay_sample_unique \
#     --window_size 20 \
#     --risk_free_return 0.06 \
#     --mode train \
#     --model_save_path ../model/MultiDQN+BTC-USD+ETH-USD+MATIC-USD \
#     --replay_window 2000 \
#     --trading_size 1000 \
#     --initial_balance 1000000 > ../logs/train_MultiDQN+BTC-USD+ETH-USD+MATIC-USD.log 2>&1


# python3 run.py \
#     --agent MultiDQN \
#     --initial_balance 1000000 \
#     --env DiscreteRealDataEnv1 \
#     --network MultiValueLSTM \
#     --asset_codes AMZN EPAM WBD \
#     --start_date 2022-11-01 \
#     --end_date 2024-01-01 \
#     --interval 1d \
#     --annual_sample 250 \
#     --device cuda \
#     --window_size 20 \
#     --mode test \
#     --model_load_path  ../model/MultiDQN+BTC-USD+ETH-USD+MATIC-USD/Q_net_last_checkpoint.pth \
#     --evaluator_saving_path ../evaluator/MultiDQN+DiscreteRealDataEnv1+BTC-USD+ETH-USD+MATIC-USD.json > ../logs/test_MultiDQN+BTC-USD+ETH-USD+MATIC-USD.log 2>&1

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
#     --network PolicyWeightCNN \
#     --asset_codes NVDA GOOGL MSFT AMZN EPAM WBD \
#     --start_date 2017-11-01 \
#     --end_date 2023-01-01 \
#     --train_learning_rate 0.001 \
#     --interval 1d \
#     --annual_sample 252 \
#     --device cuda \
#     --train_batch_size 1 \
#     --train_epochs 5 \
#     --window_size 50 \
#     --mode train \
#     --initial_balance 1000000 \
#     --replay_sample_distribution geometric

# python3 run.py \
#     --agent DPG \
#     --env ContinuousRealDataEnv1 \
#     --network PolicyWeightCNN \
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