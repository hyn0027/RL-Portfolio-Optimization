import argparse
from typing import Optional, Dict, List

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

    def state_dimension(self) -> Dict[str, torch.Size]:
        """the dimension of the state tensors.

        Returns:
            Dict[str, torch.Size]: the dimension of the state tensors
        """
        return {"price": torch.Size([self.window_size, self.asset_num])}

    def state_tensor_names(self) -> List[str]:
        """the names of the state tensors

        Returns:
            List[str]: the names of the state tensors
        """
        return ["price"]

    def get_state(
        self,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """get the state tensors at the current time.

        Returns:
            Dict[str, torch.Tensor]: the state tensors
        """
        return {
            "price": self._get_price_tensor_in_window(self.time_index),
            "time_index": self.time_index,
            "portfolio_value": self.portfolio_value,
        }

    def act(self, action: torch.Tensor, state: Dict) -> torch.Tensor:
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
            new_portfolio_value_next_day,
            static_portfolio_value,
        ) = self._get_new_portfolio_weight_and_value(action, state["time_index"])

        reward = (
            (new_portfolio_value_next_day - state["portfolio_value"])
            / state["portfolio_value"]
            * 100
        )

        new_state = {
            "Portfolio_Weight_Today": new_portfolio_weight,
        }
        done = self.time_index == self.data.time_dimension() - 2

        return new_state, reward, done

    def update(self, action: torch.Tensor) -> None:
        """
        update the environment

        Args:
            action (torch.Tensor): the action to perform
        """
        BaseEnv.update(self, action)
