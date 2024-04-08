import argparse
from typing import List, Optional, Dict, Tuple
from itertools import product
import random

from utils.data import Data
from utils.logging import get_logger
from envs import register_env
from envs.BaseEnv import BaseEnv

import torch

logger = get_logger("BasicRealDataEnv")


class BasicRealDataEnv(BaseEnv):
    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        super(BasicRealDataEnv, BasicRealDataEnv).add_args(parser)

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

        logger.info("BasicRealDataEnv initialized")

    def to(self, device: str) -> None:
        """move the environment to the given device

        Args:
            device (torch.device): the device to move to
        """
        super().to(device)
        self.price_matrix = self.price_matrix.to(self.device)
        self.price_change_matrix = self.price_change_matrix.to(self.device)

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
        return range(0, self.data.time_dimension() - 1)

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
        # return self.price_matrix[:, time_index - self.window_size + 1 : time_index + 1]
        start_index = time_index - self.window_size + 1
        if start_index < 0:
            padding = -start_index
            start_index = 0
        else:
            padding = 0

        slice = self.price_matrix[:, start_index : time_index + 1]
        if padding > 0:
            padding_tensor = self.price_matrix[:, 0].unsqueeze(1).repeat(1, padding)
            slice = torch.cat((padding_tensor, slice), dim=1)
        return slice

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

    def get_state(
        self,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """get the state tensors at the current time.

        Returns:
            Dict[str, torch.tensor]: the state tensors
        """
        return {
            "price": self._get_price_tensor_in_window(self.time_index),
        }

    def reset(self) -> None:
        """reset the environment."""
        logger.info("Resetting BasicRealDataEnv")
        self.time_index = 0
        BaseEnv.initialize_weight(self)
