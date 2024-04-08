@echo off
set PYTHON=python
set SCRIPT=run.py

%PYTHON% %SCRIPT% ^
    --agent MultiDQN ^
    --env DiscreteRealDataEnv1 ^
    --network MultiDiscreteLSTM ^
    --asset_codes NVDA AAPL GOOGL ^
    --start_date 2020-01-01 ^
    --end_date 2023-01-01 ^
    --interval 1d ^
    --device cpu ^
    --pretrain_epochs 1 ^
    --train_batch_size 4 ^
    --train_epochs 1
