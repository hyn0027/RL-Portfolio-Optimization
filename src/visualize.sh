#!/bin/bash

# python visualize.py --visualize_mode json \
#     --visualize_single_path ../evaluator/MultiDQN+DiscreteRealDataEnv1+AMZN+EPAM+WBD \
#     --output_path ../visualization/MultiDQN+DiscreteRealDataEnv1+AMZN+EPAM+WBD

# python visualize.py --visualize_mode json \
#     --visualize_single_path ../evaluator/MultiDQN+DiscreteRealDataEnv1+BTC-USD+ETH-USD+MATIC-USD \
#     --output_path ../visualization/MultiDQN+DiscreteRealDataEnv1+BTC-USD+ETH-USD+MATIC-USD


# python visualize.py --visualize_mode single \
#     --visualize_single_path ../evaluator/MultiDQN+DiscreteRealDataEnv1+^GSPC+^DJI+^RUT \
#     --output_path ../visualization/MultiDQN+DiscreteRealDataEnv1+^GSPC+^DJI+^RUT


# python visualize.py --visualize_mode single \
#     --visualize_single_path ../evaluator/MultiDQN+DiscreteRealDataEnv1+^GSPC+^DJI+^RUT2 \
#     --output_path ../visualization/MultiDQN+DiscreteRealDataEnv1+^GSPC+^DJI+^RUT2


# python visualize.py --visualize_mode json \
#     --visualize_single_path ../evaluator/MultiDQN+DiscreteRealDataEnv1+^GSPC+^DJI+^RUT2 \
#     --output_path ../visualization/MultiDQN+DiscreteRealDataEnv1+^GSPC+^DJI+^RUT2

    

python visualize.py --visualize_mode single \
    --visualize_single_path ../evaluator/DPG+LSTM+DiscreteRealDataEnv1+^GSPC+^DJI+^RUT \
    --output_path ../visualization/DPG+LSTM+DiscreteRealDataEnv1+^GSPC+^DJI+^RUT
