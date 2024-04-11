import argparse
from typing import Optional, Dict, Tuple

from utils.data import Data
from utils.logging import get_logger

from envs import register_env
from envs.BasicRealDataEnv import BasicRealDataEnv
from envs.BaseEnv import BaseEnv

import torch

logger = get_logger("BasicDiscreteRealDataEnv")


@register_env("BasicContinuousRealDataEnv")
class BasicContinuousRealDataEnv(BasicRealDataEnv):
    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        super(BasicContinuousRealDataEnv, BasicContinuousRealDataEnv).add_args(parser)

    def __init__(
        self,
        args: argparse.Namespace,
        data: Data,
        device: Optional[torch.device] = None,
    ) -> None:
        logger.info("Initializing BasicContinuousRealDataEnv")
        super().__init__(args, data, device)
        logger.info("BasicContinuousRealDataEnv initialized")

    def to(self, device: str) -> None:
        super().to(device)

    def act(
        self, action: torch.Tensor
    ) -> Tuple[Dict[str, Optional[torch.Tensor]], torch.Tensor, bool]:
        """
        perform an action (the trading size)

        Args:
            action (torch.Tensor): the action to perform

        Returns:
            Tuple[Dict[str, Optional[torch.Tensor]], torch.Tensor, bool]: the new state, reward, and whether the episode is done
        """
        (
            new_portfolio_weight,
            new_portfolio_weight_next_day,
            new_rf_weight,
            new_rf_weight_next_day,
            new_portfolio_value,
            static_portfolio_value,
        ) = self._get_new_portfolio_weight_and_value(action)

        reward = (
            (new_portfolio_value - self.portfolio_value) / self.portfolio_value * 100
        )

        done = self.time_index == self.data.time_dimension() - 2

        new_state = {
            "price": (self._get_price_tensor_in_window(self.time_index + 1)),
            "Portfolio_Weight_Today": new_portfolio_weight,
        }

        return new_state, reward, done

    def update(self, action: torch.Tensor) -> None:
        """
        update the environment

        Args:
            action (torch.Tensor): the action to perform
        """
        BaseEnv.update(self, action)
