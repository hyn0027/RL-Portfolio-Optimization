#!/bin/bash

# python visualize.py --visualize_mode single \
#     --visualize_single_path ../evaluator/MultiDQN+DiscreteRealDataEnv1+^GSPC+^DJI+^RUT2 \
#     --output_path ../visualization/MultiDQN+DiscreteRealDataEnv1+^GSPC+^DJI+^RUT2 \
#     --window_size 50


# python visualize.py --visualize_mode single \
#     --visualize_single_path ../evaluator/DPG+LSTM+DiscreteRealDataEnv1+^GSPC+^DJI+^RUT \
#     --output_path ../visualization/DPG+LSTM+DiscreteRealDataEnv1+^GSPC+^DJI+^RUT \
#     --window_size 20

# python visualize.py --visualize_mode single \
#     --visualize_single_path ../evaluator/DPG+CNN+DiscreteRealDataEnv1+^GSPC+^DJI+^RUT \
#     --output_path ../visualization/DPG+CNN+DiscreteRealDataEnv1+^GSPC+^DJI+^RUT \
#     --window_size 20

# python visualize.py --visualize_mode single \
#     --visualize_single_path ../evaluator/DPG+RNN+DiscreteRealDataEnv1+^GSPC+^DJI+^RUT \
#     --output_path ../visualization/DPG+RNN+DiscreteRealDataEnv1+^GSPC+^DJI+^RUT \
#     --window_size 20




# python visualize.py --visualize_mode single \
#     --visualize_single_path ../evaluator/DPG+LSTM+DiscreteRealDataEnv1+^GSPC+^DJI+^RUT+^IXIC+^NYA+^XAX  \
#     --output_path ../visualization/DPG+LSTM+DiscreteRealDataEnv1+^GSPC+^DJI+^RUT+^IXIC+^NYA+^XAX  \
#     --window_size 20

# python visualize.py --visualize_mode single \
#     --visualize_single_path ../evaluator/DPG+CNN+DiscreteRealDataEnv1+^GSPC+^DJI+^RUT+^IXIC+^NYA+^XAX  \
#     --output_path ../visualization/DPG+CNN+DiscreteRealDataEnv1+^GSPC+^DJI+^RUT+^IXIC+^NYA+^XAX  \
#     --window_size 20

# python visualize.py --visualize_mode single \
#     --visualize_single_path ../evaluator/DPG+RNN+DiscreteRealDataEnv1+^GSPC+^DJI+^RUT+^IXIC+^NYA+^XAX  \
#     --output_path ../visualization/DPG+RNN+DiscreteRealDataEnv1+^GSPC+^DJI+^RUT+^IXIC+^NYA+^XAX  \
#     --window_size 20


python visualize.py --visualize_mode single \
    --visualize_single_path ../evaluator/DPG+Transformer+^GSPC+^DJI+^RUT1  \
    --output_path ../visualization/DPG+Transformer+^GSPC+^DJI+^RUT  \
    --window_size 5


python visualize.py --visualize_mode single \
    --visualize_single_path ../evaluator/DPG+Transformer+^GSPC+^DJI+^RUT+^IXIC+^NYA+^XAX1  \
    --output_path ../visualization/DPG+Transformer+^GSPC+^DJI+^RUT+^IXIC+^NYA+^XAX  \
    --window_size 5



python visualize.py --visualize_mode json \
    --output_path ../visualization/^GSPC+^DJI+^RUT \
    --visualization_json_path visualization_config_low.json \
    --window_size 20


# python visualize.py --visualize_mode json \
#     --output_path ../visualization/^GSPC+^DJI+^RUT+^IXIC+^NYA+^XAX \
#     --visualization_json_path visualization_config.json \
#     --window_size 20