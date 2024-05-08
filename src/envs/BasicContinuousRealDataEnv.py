import argparse
from typing import Optional, Dict, List, Tuple, Union

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
        return {
            "price": torch.Size([self.window_size, self.asset_num]),
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
            Dict[str, torch.Tensor]: the state tensors
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
        price = self._get_price_tensor_in_window(time_index).transpose(0, 1)
        price = price.unsqueeze(0)
        return {
            "price": price,
            "time_index": time_index,
            "portfolio_weight": portfolio_weight,
            "rf_weight": rf_weight,
            "portfolio_value": portfolio_value,
        }

    def act(
        self,
        action: torch.Tensor,
        state: Optional[Dict[str, Union[torch.Tensor, int]]] = None,
    ) -> Tuple[Dict[str, Optional[Union[torch.Tensor, int]]], torch.Tensor, bool]:
        """
        perform an action (the trading size)

        Args:
            action (torch.Tensor): the action to perform
            state (Optional[Dict[str, Union[torch.Tensor, int]]], optional): the state tensors. Defaults to None.

        Returns:
            Tuple[Dict[str, Optional[Union[torch.Tensor, int]]], torch.Tensor, bool]: the new state, reward, and whether the episode is done
        """
        if state is None:
            time_index = self.time_index
            portfolio_value = self.portfolio_value
        else:
            time_index: int = state["time_index"]
            portfolio_value = state["portfolio_value"]
        new_state = self.update(action, state=state, modify_inner_state=False)
        reward = (new_state["portfolio_value"] - portfolio_value) / torch.abs(
            portfolio_value
        )
        done = time_index == self.data.time_dimension() - 2

        return new_state, reward, done

    def update(
        self,
        action: torch.Tensor,
        state: Optional[Dict[str, Union[torch.Tensor, int]]] = None,
        modify_inner_state: Optional[bool] = None,
    ) -> Dict[str, Union[torch.Tensor, int]]:
        """
        update the environment

        Args:
            action (torch.Tensor): the action to perform
            state (Optional[Dict[str, Union[torch.Tensor, int]]], optional): the state tensors. Defaults to None.
            modify_inner_state (Optional[bool], optional): whether to modify the inner state. Defaults to None.

        returns:
            Dict[str, Union[torch.Tensor, int]]: the new state
        """
        if modify_inner_state is None:
            modify_inner_state = state is None
        if state is None:
            portfolio_value = self.portfolio_value
            portfolio_weight = self.portfolio_weight
            rf_weight = self.rf_weight
        else:
            portfolio_value: torch.Tensor = state["portfolio_value"]
            portfolio_weight: torch.Tensor = state["portfolio_weight"]
            rf_weight: torch.Tensor = state["rf_weight"]
        for i in range(self.asset_num):
            if (
                action[i] < 0
                and portfolio_weight[i] * self.portfolio_value < -action[i]
            ):
                action[i] = -portfolio_weight[i] * self.portfolio_value
        while self._cash_shortage(action, portfolio_value, rf_weight):
            for i in range(self.asset_num):
                if action[i] > 0:
                    action[i] = 0
                    break
        new_state = BaseEnv.update(self, action, state, modify_inner_state)
        new_state.pop("new_rf_weight_prev_day", None)
        new_state.pop("new_portfolio_value_prev_day", None)
        new_state.pop("static_portfolio_value", None)
        ret_state = self.get_state(new_state)
        ret_state["new_portfolio_weight_prev_day"] = new_state[
            "new_portfolio_weight_prev_day"
        ]
        ret_state["prev_price"] = new_state["prev_price"]
        return ret_state
