import argparse
from utils.logging import get_logger
from typing import Optional

from networks import registered_networks
from utils.replay import Replay
from agents import register_agent
from agents.BaseAgent import BaseAgent
from envs.BaseEnv import BaseEnv

import torch
import torch.nn as nn

logger = get_logger("DPG")


@register_agent("DPG")
class DPG(BaseAgent):
    """The DPG class is a subclass of BaseAgent and implements the DPG algorithm.

    Raises:
        ValueError: missing model_load_path for testing
    """

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        super(DPG, DPG).add_args(parser)

    def __init__(
        self,
        args: argparse.Namespace,
        env: BaseEnv,
        device: Optional[str] = None,
        test_mode: bool = False,
    ) -> None:
        """the constructor for the DPG agent

        Args:
            args (argparse.Namespace): arguments
            env (BaseEnv): the trading environment
            device (Optional[str], optional): torch device. Defaults to None, which means the device is automatically selected.
            test_mode (bool, optional): test or train mode. Defaults to False.
        """
        logger.info("Initialize DPG agent")
        super().__init__(args, env, device, test_mode)
        if not self.test_mode:
            self.model: nn.Module = registered_networks[args.network](args)
            if self.fp16:
                self.model = self.model.half()
            self.model.to(self.device)

            logger.info(self.model)
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Total number of parameters: {total_params}")

            self.replay = Replay(args)

            self.train_optimizer = torch.optim.Adam(
                self.model.parameters(), lr=args.train_learning_rate
            )
            self.train_optimizer.zero_grad()
        else:
            self.model: nn.Module = registered_networks[args.network](args)
            logger.info(self.model)

            if not args.model_load_path:
                raise ValueError("model_load_path is required for testing")
            self.model.load_state_dict(
                torch.load(args.model_load_path, map_location=self.device)
            )
            logger.info(f"model loaded from {args.model_load_path}")

            self.model.to(self.device)

        logger.info("DPG agent initialized")

    def train(self) -> None:
        """train the DPG agent"""
