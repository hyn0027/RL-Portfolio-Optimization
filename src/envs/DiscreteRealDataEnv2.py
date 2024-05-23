import argparse
from typing import List, Optional, Dict, Tuple, Union
from itertools import product
import random

from utils.data import Data
from utils.logging import get_logger
from envs import register_env
from envs.BasicDiscreteRealDataEnv import BasicDiscreteRealDataEnv
from envs.BaseEnv import BaseEnv

import torch

logger = get_logger("DiscreteRealDataEnv2")


@register_env("DiscreteRealDataEnv2")
class DiscreteRealDataEnv2(BasicDiscreteRealDataEnv):
    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        super(DiscreteRealDataEnv2, DiscreteRealDataEnv2).add_args(parser)

    def __init__(
        self,
        args: argparse.Namespace,
        data: Data,
        device: Optional[torch.device] = None,
    ) -> None:
        """initialize the environment

        Args:
            args (argparse.Namespace): arguments
            data (Data): data,
            device (Optional[str], optional): device to run the environment. Defaults to None, which means to use the GPU if available.
        """
        logger.info("Initializing DiscreteRealDataEnv2")
        super().__init__(args, data, device)
        logger.info("DiscreteRealDataEnv2 initialized")

    def action_is_valid(
        self, action: torch.Tensor, state: Optional[Dict[str, torch.Tensor]] = None
    ) -> bool:
        """check if the action is valid

        Args:
            action (torch.Tensor): the action
            state (Optional[Dict[str, torch.Tensor]], optional): the current state. Defaults to None.

        Returns:
            bool: whether the action is valid
        """
        if state is None:
            portfolio_weight = self.portfolio_weight
            portfolio_value = self.portfolio_value
            rf_weight = self.rf_weight
        else:
            portfolio_weight = state["portfolio_weight"]
            portfolio_value = state["portfolio_value"]
            rf_weight = state["rf_weight"]
        return not self._asset_shortage(
            action * self.trading_size, portfolio_weight, portfolio_value
        ) and not self._cash_shortage(
            action * self.trading_size, portfolio_value, rf_weight
        )

    def possible_actions(
        self, state: Dict[str, torch.Tensor] = None
    ) -> List[torch.Tensor]:
        """get the possible actions

        Args:
            state (Dict[str, torch.Tensor], optional): the current state. Defaults to None.

        Returns:
            List[torch.Tensor]: the possible actions
        """
        return_action = []
        for action in self.all_actions:
            if self.action_is_valid(action, state):
                return_action.append(action)
        return return_action

    def get_momentum_action(self) -> torch.Tensor:
        """get the momentum action

        Returns:
            torch.Tensor: the momentum action
        """
        current_price = self._get_price_tensor(self.time_index)
        prev_price = self._get_price_tensor(self.time_index - 1)
        action = torch.zeros(self.asset_num, dtype=torch.int32, device=self.device)
        for asset_index in range(self.asset_num):
            if current_price[asset_index] > prev_price[asset_index]:
                action[asset_index] = 1
            elif (
                current_price[asset_index] < prev_price[asset_index]
                and self.portfolio_weight[asset_index] * self.portfolio_value
                > self.trading_size
            ):
                action[asset_index] = -1
        while self._cash_shortage(action * self.trading_size):
            for asset_index in range(self.asset_num):
                if action[asset_index] == 1:
                    action[asset_index] = 0
                    break

        return action

    def get_reverse_momentum_action(self) -> torch.Tensor:
        """get the reverse momentum action

        Returns:
            torch.Tensor: the reverse momentum action
        """
        current_price = self._get_price_tensor(self.time_index)
        prev_price = self._get_price_tensor(self.time_index - 1)
        action = torch.zeros(self.asset_num, dtype=torch.int32, device=self.device)
        for asset_index in range(self.asset_num):
            if current_price[asset_index] < prev_price[asset_index]:
                action[asset_index] = 1
            elif (
                current_price[asset_index] > prev_price[asset_index]
                and self.portfolio_weight[asset_index] * self.portfolio_value
                > self.trading_size
            ):
                action[asset_index] = -1
        while self._cash_shortage(action * self.trading_size):
            for asset_index in range(self.asset_num):
                if action[asset_index] == 1:
                    action[asset_index] = 0
                    break

        return action
