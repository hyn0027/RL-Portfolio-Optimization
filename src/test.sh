#!/bin/bash

BASE_CMD="python3 run.py \
    --agent DPG \
    --initial_balance 1000000 \
    --env ContinuousRealDataEnv1 \
    --network PolicyWeightLSTM \
    --asset_codes ^GSPC ^DJI ^RUT \
    --start_date 2022-11-17 \
    --end_date 2024-01-01 \
    --interval 1d \
    --DPG_update_window_size 30 \
    --annual_sample 252 \
    --train_learning_rate 5e-4 \
    --train_batch_size 1 \
    --device cpu \
    --risk_free_return 0.04 \
    --replay_window 31 \
    --window_size 50 \
    --mode test "

# LOG_PATH_BASE="../logs/test_MultiDQN+^GSPC+^DJI+^RUT"
# MODEL_PATH_BASE="../model/MultiDQN+^GSPC+^DJI+^RUT"
# EVALUATOR_PATH_BASE="../evaluator/MultiDQN+DiscreteRealDataEnv1+^GSPC+^DJI+^RUT"

LOG_PATH_BASE="../logs/test_DPG+LSTM+^GSPC+^DJI+^RUT"
MODEL_PATH_BASE="../model/DPG+LSTM+^GSPC+^DJI+^RUT"
EVALUATOR_PATH_BASE="../evaluator/DPG+LSTM+DiscreteRealDataEnv1+^GSPC+^DJI+^RUT"

mkdir -p $LOG_PATH_BASE

MODEL_LOAD_PATH=$MODEL_PATH_BASE"/model_last_checkpoint.pth"
EVALUATOR_SAVING_PATH=$EVALUATOR_PATH_BASE"/model_last_checkpoint"
LOG_PATH=$LOG_PATH_BASE"/model_last_checkpoint.log"

mkdir -p $EVALUATOR_SAVING_PATH

$BASE_CMD --model_load_path $MODEL_LOAD_PATH --evaluator_saving_path $EVALUATOR_SAVING_PATH > $LOG_PATH 2>&1


# for i in $(seq 134 164)
for i in $(seq 15 15)
do
    # Update the model_load_path for each iteration
    MODEL_LOAD_PATH=$MODEL_PATH_BASE"/DPG_epoch${i}.pth"
    
    EVALUATOR_SAVING_PATH=$EVALUATOR_PATH_BASE"/model${i}"

    mkdir -p $EVALUATOR_SAVING_PATH

    LOG_PATH=$LOG_PATH_BASE"/model${i}.log"

    # Run the command with updated model_load_path and log file
    $BASE_CMD --test_model_only --model_load_path $MODEL_LOAD_PATH --evaluator_saving_path $EVALUATOR_SAVING_PATH > $LOG_PATH 2>&1
    
    # Optional: print status to console
    echo "Completed run for model_load_path epoch${i}"
done

