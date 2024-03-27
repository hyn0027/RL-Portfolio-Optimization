import argparse
from typing import List, Optional, Tuple
import torch

from utils.data import Data
from utils.logging import get_logger
from envs.BaseEnv import BaseEnv

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
        super().__init__(args, device)
        self.data = data
        self.asset_codes = data.asset_codes
        self.time_zone: str = args.time_zone
        self.asset_num = len(self.asset_codes)

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
        self.device = device
        self.price_matrix = self.price_matrix.to(device)
        self.price_change_matrix = self.price_change_matrix.to(device)

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
        return range(0, self.data.time_dimension())

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
