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

    def get_asset_num(self) -> int:
        return self.asset_num

    def train_time_range(self) -> range:
        return range(0, self.data.time_dimension())

    def test_time_range(self) -> range:
        return range(0, self.data.time_dimension())

    def _get_price_change_ratio_tensor(
        self, time_index: Optional[int] = None
    ) -> torch.tensor:
        if time_index is None:
            time_index = self.time_index
        return self.price_change_matrix[:, time_index - 1]
