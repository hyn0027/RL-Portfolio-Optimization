# RL-Portfolio-Optimization

This project is inspired by [Deep-Reinforcement-Stock-Trading](https://github.com/Albert-Z-Guo/Deep-Reinforcement-Stock-Trading). This project intends to provide a unified framework for implementing, training, testing, and evaluating reinforcement learning algorithms, especially DRL methods, for portfolio optimization problem on multiple types of assets, including stocks, futures, ETFs, and crypto currencies.

This project is part of my bachelor's thesis and is still developing. We plan to implement all basic feature before May 2024. We will reproduce the algorithms of some important recent studies on DRL for portfolio optimization before July 2024.

# Environments

python >= 3.9

```bash
pip install -r requirements.txt
```

# Documentation and Usage

For API documentation, please refer to this [online doc](https://hyn0027.github.io/RL-Portfolio-Optimization/).

To build documentations locally:
```bash
sh generate_doc.sh
```

For quick usage:

```bash
cd src
sh train.sh
```

For more parameter explanations, see:

```bash
cd src
python run.py -h
```
or you can check the online doc.