#!/bin/bash


python3 run.py \
    --agent MultiDPG \
    --env BasicContinuousRealDataEnv \
    --network PolicyTransformer \
    --asset_codes ^GSPC ^DJI ^RUT\
    --start_date 2014-01-01 \
    --end_date 2023-01-01 \
    --train_learning_rate 8e-5 \
    --DPG_update_window_size 50 \
    --interval 1d \
    --annual_sample 252 \
    --device cuda \
    --train_batch_size 64 \
    --train_epochs 2000 \
    --window_size 50 \
    --risk_free_return 0.04 \
    --mode train \
    --model_save_path ../model/DPG+Transformer+^GSPC+^DJI+^RUT1 \
    --initial_balance 1000000 \
    --replay_window 2000 \
    --trading_size 1000 \
    --replay_sample_unique > ../logs/train_DPG+Transformer+^GSPC+^DJI+^RUT1.log 2>&1


# python3 run.py \
#     --agent MultiDPG \
#     --env BasicContinuousRealDataEnv \
#     --network PolicyTransformer \
#     --asset_codes ^GSPC ^DJI ^RUT ^IXIC ^NYA ^XAX \
#     --start_date 2014-01-01 \
#     --end_date 2023-01-01 \
#     --train_learning_rate 2e-6 \
#     --DPG_update_window_size 50 \
#     --interval 1d \
#     --annual_sample 252 \
#     --device cuda \
#     --train_batch_size 8 \
#     --train_epochs 2000 \
#     --window_size 50 \
#     --risk_free_return 0.04 \
#     --mode train \
#     --model_save_path ../model/DPG+Transformer+^GSPC+^DJI+^RUT+^IXIC+^NYA+^XAX \
#     --initial_balance 1000000 \
#     --replay_window 2000 \
#     --trading_size 1000 \
#     --replay_sample_unique > ../logs/train_DPG+Transformer+^GSPC+^DJI+^RUT+^IXIC+^NYA+^XAX.log 2>&1
