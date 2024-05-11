import argparse
from typing import List, Optional, Dict, Union

from utils.data import Data
from utils.logging import get_logger
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
        open_price_list = []
        high_price_list = []
        low_price_list = []

        for time_index in range(0, self.data.time_dimension()):
            new_price = []
            open_price = []
            high_price = []
            low_price = []
            for asset_code in self.asset_codes:
                asset_data = self.data.get_asset_hist_at_time(
                    asset_code, self.data.time_list[time_index]
                )
                new_price.append(asset_data["Close"])
                open_price.append(asset_data["Open"])
                high_price.append(asset_data["High"])
                low_price.append(asset_data["Low"])
            price_list.append(
                torch.tensor(new_price, dtype=self.dtype, device=self.device)
            )
            open_price_list.append(
                torch.tensor(open_price, dtype=self.dtype, device=self.device)
            )
            high_price_list.append(
                torch.tensor(high_price, dtype=self.dtype, device=self.device)
            )
            low_price_list.append(
                torch.tensor(low_price, dtype=self.dtype, device=self.device)
            )
        self.price_matrix = torch.stack(price_list, dim=1)
        self.open_price_matrix = torch.stack(open_price_list, dim=1)
        self.high_price_matrix = torch.stack(high_price_list, dim=1)
        self.low_price_matrix = torch.stack(low_price_list, dim=1)
        self.price_change_matrix = self.price_matrix[:, 1:] / self.price_matrix[:, :-1]
        self.train_window_interval_end = getattr(args, "DPG_update_window_size", 0)

        logger.info("BasicRealDataEnv initialized")

    def to(self, device: str) -> None:
        """move the environment to the given device

        Args:
            device (torch.device): the device to move to
        """
        super().to(device)
        self.price_matrix = self.price_matrix.to(self.device)
        self.open_price_matrix = self.open_price_matrix.to(self.device)
        self.high_price_matrix = self.high_price_matrix.to(self.device)
        self.low_price_matrix = self.low_price_matrix.to(self.device)
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
        return range(
            self.window_size - 1,
            self.data.time_dimension() - 1 - self.train_window_interval_end,
        )

    def test_time_range(self) -> range:
        """the range of time indices

        Returns:
            range: the range of time indices
        """
        return range(self.window_size - 1, self.data.time_dimension() - 1)

    def _get_price_change_ratio_tensor(
        self, time_index: Optional[int] = None
    ) -> torch.Tensor:
        """get the price change ratio tensor at a given time

        Args:
            time_index (Optional[int], optional):
                the time index to get the price change ratio.
                Defaults to None, which means to get the price change ratio at the current time.

        Returns:
            torch.Tensor: the price change ratio tensor
        """
        if time_index is None:
            time_index = self.time_index
        return self.price_change_matrix[:, time_index - 1]

    def _get_price_tensor(self, time_index: Optional[int] = None) -> torch.Tensor:
        """get the price tensor at a given time

        Args:
            time_index (Optional[int], optional):
                the time index to get the price.
                Defaults to None, which means to get the price at the current time.

        Returns:
            torch.Tensor: the price tensor
        """
        if time_index is None:
            time_index = self.time_index
        return self.price_matrix[:, time_index]

    def _get_price_tensor_in_window(self, time_index: int) -> torch.Tensor:
        """get the price tensor in a window centered at a given time

        Args:
            time_index (int): the time index to get the price
            window_size (int): the window size

        Returns:
            torch.Tensor: the price tensor in the window
        """
        return self._get_tensor_in_window(self.price_matrix, time_index)

    def _get_high_price_tensor_in_window(self, time_index: int) -> torch.Tensor:
        """get the high price tensor in a window centered at a given time

        Args:
            time_index (int): the time index to get the high price

        Returns:
            torch.Tensor: the high price tensor in the window
        """
        return self._get_tensor_in_window(self.high_price_matrix, time_index)

    def _get_low_price_tensor_in_window(self, time_index: int) -> torch.Tensor:
        """get the low price tensor in a window centered at a given time

        Args:
            time_index (int): the time index to get the low price

        Returns:
            torch.Tensor: the low price tensor in the window
        """
        return self._get_tensor_in_window(self.low_price_matrix, time_index)

    def _get_open_price_tensor_in_window(self, time_index: int) -> torch.Tensor:
        """get the open price tensor in a window centered at a given time

        Args:
            time_index (int): the time index to get the open price

        Returns:
            torch.Tensor: the open price tensor in the window
        """
        return self._get_tensor_in_window(self.open_price_matrix, time_index)

    def _get_tensor_in_window(
        self, tensor: torch.Tensor, time_index: int
    ) -> torch.Tensor:
        """get the tensor in a window centered at a given time

        Args:
            tensor (torch.Tensor): the tensor
            time_index (int): the time index to get the tensor

        Returns:
            torch.Tensor: the tensor in the window
        """
        start_index = time_index - self.window_size + 1
        if start_index < 0:
            padding = -start_index
            start_index = 0
        else:
            padding = 0

        slice = tensor[:, start_index : time_index + 1]
        if padding > 0:
            padding_tensor = tensor[:, 0].unsqueeze(1).repeat(1, padding)
            slice = torch.cat((padding_tensor, slice), dim=1)
        return slice

    def state_dimension(self) -> Dict[str, torch.Size]:
        """the dimension of the state tensors.

        Returns:
            Dict[str, torch.Size]: the dimension of the state tensors
        """
        return {
            "price": torch.Size([self.asset_num, self.window_size]),
            "time_index": torch.Size([1]),
            "portfolio_weight": torch.Size([self.asset_num]),
            "rf_weight": torch.Size([1]),
            "portfolio_value": torch.Size([1]),
        }

    def state_tensor_names(self) -> List[str]:
        """the names of the state tensors

        Returns:
            List[str]: the names of the state tensors
        """
        return [
            "price",
            "time_index",
            "portfolio_weight",
            "rf_weight",
            "portfolio_value",
        ]

    def get_state(
        self,
        state: Optional[Dict[str, Union[torch.Tensor, int]]] = None,
    ) -> Optional[Dict[str, Union[torch.Tensor, int]]]:
        """get the state tensors at the current time.

        Args:
            state (Optional[Dict[str, Union[torch.Tensor, int]]], optional): the state tensors. Defaults to None.


        Returns:
            Dict[str, Union[torch.Tensor, int]]: the state tensors
        """

        if state is None:
            time_index = self.time_index
            portfolio_weight = self.portfolio_weight
            rf_weight = self.rf_weight
            portfolio_value = self.portfolio_value
        else:
            time_index: int = state["time_index"]
            portfolio_weight: torch.Tensor = state["portfolio_weight"]
            rf_weight: torch.Tensor = state["rf_weight"]
            portfolio_value: torch.Tensor = state["portfolio_value"]
        return {
            "price": self._get_price_tensor_in_window(time_index),
            "time_index": time_index,
            "portfolio_weight": portfolio_weight,
            "rf_weight": rf_weight,
            "portfolio_value": portfolio_value,
        }

    def reset(self) -> None:
        """reset the environment."""
        logger.info("Resetting BasicRealDataEnv")
        self.time_index = self.window_size - 1
        BaseEnv.initialize_weight(self)
