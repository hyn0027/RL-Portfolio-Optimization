#!/bin/bash

python visualize.py --visualize_mode single \
    --visualize_single_path ../evaluator/MultiDQN+DiscreteRealDataEnv1+AMZN+EPAM+WBD \
    --output_path ../visualization/MultiDQN+DiscreteRealDataEnv1+AMZN+EPAM+WBD

python visualize.py --visualize_mode single \
    --visualize_single_path ../evaluator/DPG+CNN+AMZN+EPAM+WBD \
    --output_path ../visualization/DPG+CNN+AMZN+EPAM+WBD