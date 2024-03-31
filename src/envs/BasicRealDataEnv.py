import argparse
from typing import List, Optional, Dict, Tuple
from itertools import product

from utils.data import Data
from utils.logging import get_logger
from envs.BaseEnv import BaseEnv

import torch

logger = get_logger("BasicRealDataEnv")


class BasicRealDataEnv(BaseEnv):
    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        super(BasicRealDataEnv, BasicRealDataEnv).add_args(parser)
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
        logger.info("Initializing BasicRealDataEnv")
        self.data = data
        self.asset_codes = data.asset_codes
        self.time_zone: str = args.time_zone
        self.asset_num = len(self.asset_codes)

        super().__init__(args, device)

        self.trading_size = torch.tensor(
            args.trading_size, dtype=self.dtype, device=self.device
        )

        price_list = []
        for time_index in range(0, self.data.time_dimension()):
            new_price = []
            for asset_code in self.asset_codes:
                asset_data = self.data.get_asset_hist_at_time(
                    asset_code, self.data.time_list[time_index]
                )
                new_price.append(asset_data["Close"])
            price_list.append(
                torch.tensor(new_price, dtype=self.dtype, device=self.device)
            )
        self.price_matrix = torch.stack(price_list, dim=1)
        self.price_change_matrix = self.price_matrix[:, 1:] / self.price_matrix[:, :-1]

        self.all_actions = []
        action_number = range(-1, 2)  # -1, 0, 1
        for action in product(action_number, repeat=len(self.asset_codes)):
            self.all_actions.append(
                torch.tensor(action, dtype=torch.int8, device=self.device)
            )

        logger.info("BasicRealDataEnv initialized")

    def to(self, device: str) -> None:
        """move the environment to the given device

        Args:
            device (torch.device): the device to move to
        """
        super().to(device)
        self.trading_size = self.trading_size.to(self.device)
        self.price_matrix = self.price_matrix.to(self.device)
        self.price_change_matrix = self.price_change_matrix.to(self.device)
        self.all_actions = [a.to(self.device) for a in self.all_actions]

    def get_asset_num(self) -> int:
        """get the number of assets, excluding risk-free asset

        Returns:
            int: the number of assets
        """
        return self.asset_num

    def train_time_range(self) -> range:
        """the range of time indices

        Returns:
            range: the range of time indices
        """
        return range(0, self.data.time_dimension() - 1)

    def test_time_range(self) -> range:
        """the range of time indices

        Returns:
            range: the range of time indices
        """
        return range(0, self.data.time_dimension())

    def _get_price_change_ratio_tensor(
        self, time_index: Optional[int] = None
    ) -> torch.tensor:
        """get the price change ratio tensor at a given time

        Args:
            time_index (Optional[int], optional):
                the time index to get the price change ratio.
                Defaults to None, which means to get the price change ratio at the current time.

        Returns:
            torch.tensor: the price change ratio tensor
        """
        if time_index is None:
            time_index = self.time_index
        return self.price_change_matrix[:, time_index - 1]

    def _get_price_tensor(self, time_index: Optional[int] = None) -> torch.tensor:
        """get the price tensor at a given time

        Args:
            time_index (Optional[int], optional):
                the time index to get the price.
                Defaults to None, which means to get the price at the current time.

        Returns:
            torch.tensor: the price tensor
        """
        if time_index is None:
            time_index = self.time_index
        return self.price_matrix[:, time_index]

    def _get_price_tensor_in_window(self, time_index: int) -> torch.tensor:
        """get the price tensor in a window centered at a given time

        Args:
            time_index (int): the time index to get the price
            window_size (int): the window size

        Returns:
            torch.tensor: the price tensor in the window
        """
        return self.price_matrix[: time_index - self.window_size + 1, time_index + 1]

    def state_dimension(self) -> Dict[str, torch.Size]:
        """the dimension of the state tensors.

        Returns:
            Dict[str, torch.Size]: the dimension of the state tensors
        """
        return {
            "price": torch.Size([self.asset_num, self.window_size]),
        }

    def state_tensor_names(self) -> List[str]:
        """the names of the state tensors

        Returns:
            List[str]: the names of the state tensors
        """
        return ["price"]

    def action_dimension(self) -> torch.Size:
        """the dimension of the action the agent can take

        Returns:
            torch.Size: the dimension of the action the agent can take
        """
        return torch.Size([len(self.asset_codes)])

    def get_state(
        self,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """get the state tensors at the current time.

        Returns:
            Dict[str, torch.tensor]: the state tensors
        """
        if self.time_index < self.window_size - 1:
            raise None
        return {
            "price": self._get_price_tensor_in_window(self.time_index),
        }

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
            "price": (
                self._get_price_tensor_in_window(self.time_index + 1)
                if self.time_index + 2 >= self.window_size
                else None
            ),
        }

        return new_state, reward, done

    def reset(self) -> None:
        """reset the environment."""
        logger.info("Resetting BasicRealDataEnv")
        self.time_index = 0
        BaseEnv.initialize_weight(self)

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
        return self.all_actions[torch.randint(0, len(self.all_actions))]

    def possible_actions(self) -> List[torch.Tensor]:
        """get the possible actions

        Returns:
            List[torch.Tensor]: the possible actions
        """
        return self.all_actions
