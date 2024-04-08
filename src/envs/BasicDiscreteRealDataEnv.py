import argparse
from typing import List, Optional, Dict, Tuple
from itertools import product
import random

from utils.data import Data
from utils.logging import get_logger
from envs import register_env
from envs.BasicRealDataEnv import BasicRealDataEnv
from envs.BaseEnv import BaseEnv

import torch

logger = get_logger("BasicDiscreteRealDataEnv")


@register_env("BasicDiscreteRealDataEnv")
class BasicDiscreteRealDataEnv(BasicRealDataEnv):
    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        super(BasicDiscreteRealDataEnv, BasicDiscreteRealDataEnv).add_args(parser)
        parser.add_argument(
            "--trading_size",
            type=float,
            default=1e4,
            help="the size of each trading in terms of currency",
        )

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
        logger.info("Initializing BasicDiscreteRealDataEnv")
        super().__init__(args, data, device)

        self.trading_size = torch.tensor(
            args.trading_size, dtype=self.dtype, device=self.device
        )

        self.all_actions = []
        action_number = range(-1, 2)  # -1, 0, 1
        for action in product(action_number, repeat=self.asset_num):
            self.all_actions.append(
                torch.tensor(action, dtype=self.dtype, device=self.device)
            )

        logger.info("BasicDiscreteRealDataEnv initialized")

    def to(self, device: str) -> None:
        """move the environment to the given device

        Args:
            device (torch.device): the device to move to
        """
        super().to(device)
        self.trading_size = self.trading_size.to(self.device)
        self.all_actions = [a.to(self.device) for a in self.all_actions]

    def find_action_index(self, action: torch.Tensor) -> int:
        """given an action, find the index of the action in all_actions

        Args:
            action (torch.Tensor): the trading decision of each asset

        Returns:
            int: the index of the action in all_actions, -1 if not found
        """
        for i, a in enumerate(self.all_actions):
            if torch.equal(a, action):
                return i
        return -1

    def act(
        self, action: torch.tensor
    ) -> Tuple[Dict[str, Optional[torch.Tensor]], torch.Tensor, bool]:
        """update the environment with the given action at the given time

        Args:
            action (torch.tensor): the action to take

        Returns:
            Tuple[Dict[str, torch.tensor], float, bool]: the new state, the reward, and whether the episode is done
        """
        if self.find_action_index(action) == -1:
            raise ValueError(f"Invalid action: {action}")
        action = action * self.trading_size
        (
            new_portfolio_weight,
            new_portfolio_weight_next_day,
            new_rf_weight,
            new_portfolio_value,
            static_portfolio_value,
        ) = self._get_new_portfolio_weight_and_value(action)

        reward = new_portfolio_value - self.portfolio_value

        done = self.time_index == self.data.time_dimension() - 2

        new_state = {
            "price": (self._get_price_tensor_in_window(self.time_index + 1)),
        }

        return new_state, reward, done

    def update(self, action: torch.Tensor) -> None:
        if self.find_action_index(action) == -1:
            raise ValueError(f"Invalid action: {action}")
        action = action * self.trading_size
        BaseEnv.update(self, action)

    def select_random_action(self) -> torch.Tensor:
        """select a random action

        Returns:
            torch.Tensor: the random action
        """
        return self.all_actions[random.randint(0, len(self.all_actions) - 1)]

    def possible_actions(
        self, state: Dict[str, torch.Tensor] = None
    ) -> List[torch.Tensor]:
        """get the possible actions

        Args:
            state (Dict[str, torch.Tensor], optional): the current state. Defaults to None.

        Returns:
            List[torch.Tensor]: the possible actions
        """
        return self.all_actions
