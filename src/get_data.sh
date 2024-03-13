#!/bin/bash

python3 data.py \
    --asset_code AAPL AMZN GOOGL MSFT TSLA NVDA INTC CSCO \
    --start_date 2024-01-01 \
    --end_date 2024-02-01 \
    --interval 1d
