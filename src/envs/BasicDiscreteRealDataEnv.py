import argparse
from typing import List, Optional, Dict, Tuple, Union
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
        self,
        action: torch.Tensor,
        state: Optional[Dict[str, Union[torch.Tensor, int]]] = None,
    ) -> Tuple[Dict[str, Optional[Union[torch.Tensor, int]]], torch.Tensor, bool]:
        """update the environment with the given action at the given time

        Args:
            action (torch.tensor): the action to take
            state (Optional[Dict[str, Union[torch.Tensor, int]]], optional): the state tensors. Defaults to None.

        Returns:
            Tuple[Dict[str, Optional[Union[torch.Tensor, int]]], torch.Tensor, bool]: the new state, reward, and whether the episode is done
        """
        if self.find_action_index(action) == -1:
            raise ValueError(f"Invalid action: {action}")

        if state is None:
            time_index = self.time_index
            portfolio_value = self.portfolio_value
        else:
            time_index: int = state["time_index"]
            portfolio_value: torch.Tensor = state["portfolio_value"]

        new_state = self.update(action, state, modify_inner_state=False)
        reward = (
            (new_state["portfolio_value"] - portfolio_value)
            / torch.abs(portfolio_value)
            * 100
        )
        done = time_index == self.data.time_dimension() - 2

        return new_state, reward, done

    def update(
        self,
        action: torch.Tensor = None,
        state: Optional[Dict[str, Union[torch.Tensor, int]]] = None,
        modify_inner_state: Optional[bool] = None,
    ) -> Dict[str, Union[torch.Tensor, int]]:
        """
        update the environment

        Args:
            action (torch.Tensor): the action to perform. Defaults to None.
            state (Optional[Dict[str, Union[torch.Tensor, int]]], optional): the state tensors. Defaults to None.
            modify_inner_state (Optional[bool], optional): whether to modify the inner state. Defaults to None.

        returns:
            Dict[str, Union[torch.Tensor, int]]: the new state
        """
        if modify_inner_state is None:
            modify_inner_state = state is None
        if action is None:
            action = torch.zeros(self.asset_num, dtype=self.dtype, device=self.device)
        if self.find_action_index(action) == -1:
            raise ValueError(f"Invalid action: {action}")
        action = action * self.trading_size
        new_state = BaseEnv.update(self, action, state, modify_inner_state)
        ret_state = self.get_state(new_state)
        ret_state["new_portfolio_weight_prev_day"] = new_state[
            "new_portfolio_weight_prev_day"
        ]
        ret_state["prev_price"] = new_state["prev_price"]
        return ret_state

    def select_random_action(self) -> torch.Tensor:
        """select a random action

        Returns:
            torch.Tensor: the random action
        """
        possible_actions = self.possible_actions()
        return random.choice(possible_actions)

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
            elif current_price[asset_index] < prev_price[asset_index]:
                action[asset_index] = -1
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
            if current_price[asset_index] > prev_price[asset_index]:
                action[asset_index] = -1
            elif current_price[asset_index] < prev_price[asset_index]:
                action[asset_index] = 1
        return action
