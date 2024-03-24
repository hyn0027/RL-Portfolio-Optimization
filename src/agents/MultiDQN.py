import argparse
from utils.logging import get_logger

from agents import register_agent
from agents.BaseAgent import BaseAgent
from envs.BaseEnv import BaseEnv
from networks import registered_networks

import torch
import torch.nn as nn

logger = get_logger("MultiDQN")


@register_agent("MultiDQN")
class MultiDQN(BaseAgent):
    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        super(MultiDQN, MultiDQN).add_args(parser)

    def __init__(self, args: argparse.Namespace, env: BaseEnv) -> None:
        super().__init__(args, env)
        logger.info("Initializing MultiDQN")
        self.Q_network: nn.Module = registered_networks[args.network](args)
        self.target_Q_network = registered_networks[args.network](args)
        logger.info(self.Q_network)
        total_params = sum(p.numel() for p in self.Q_network.parameters())
        logger.info(f"Total number of parameters: {total_params}")

    def train(self) -> None:
        self.pretrain()
        self.multiDQN_train()

    def pretrain(self) -> None:
        pass

    def multiDQN_train(self) -> None:
        pass
