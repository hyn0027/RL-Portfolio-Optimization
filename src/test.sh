#!/bin/bash

BASE_CMD="python3 run.py \
    --agent MultiDQN \
    --initial_balance 1000000 \
    --env DiscreteRealDataEnv1 \
    --network MultiValueLSTM \
    --asset_codes AMZN EPAM WBD \
    --start_date 2022-11-01 \
    --end_date 2024-01-01 \
    --interval 1d \
    --annual_sample 250 \
    --device cpu \
    --window_size 20 \
    --mode test "

LOG_PATH_BASE="../logs/test_MultiDQN+AMZN+EPAM+WBD"
MODEL_PATH_BASE="../model/MultiDQN+AMZN+EPAM+WBD"
EVALUATOR_PATH_BASE="../evaluator/MultiDQN+DiscreteRealDataEnv1+AMZN+EPAM+WBD"

mkdir -p $LOG_PATH_BASE

MODEL_LOAD_PATH=$MODEL_PATH_BASE"/Q_net_last_checkpoint.pth"
EVALUATOR_SAVING_PATH=$EVALUATOR_PATH_BASE"/model_last_checkpoint"
LOG_PATH=$LOG_PATH_BASE"/model_last_checkpoint.log"

mkdir -p $EVALUATOR_SAVING_PATH

$BASE_CMD --model_load_path $MODEL_LOAD_PATH --evaluator_saving_path $EVALUATOR_SAVING_PATH > $LOG_PATH 2>&1

# Loop from 0 to 100

for i in $(seq 9 13)
do
    # Update the model_load_path for each iteration
    MODEL_LOAD_PATH=$MODEL_PATH_BASE"/Q_net_epoch${i}.pth"
    
    EVALUATOR_SAVING_PATH=$EVALUATOR_PATH_BASE"/model${i}"

    mkdir -p $EVALUATOR_SAVING_PATH

    LOG_PATH=$LOG_PATH_BASE"/model${i}.log"

    # Run the command with updated model_load_path and log file
    $BASE_CMD --test_model_only --model_load_path $MODEL_LOAD_PATH --evaluator_saving_path $EVALUATOR_SAVING_PATH > $LOG_PATH 2>&1
    
    # Optional: print status to console
    echo "Completed run for model_load_path epoch${i}"
done

