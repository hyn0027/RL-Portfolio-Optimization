import argparse
from typing import Dict, Optional, Tuple, List, Union

from utils.logging import get_logger

import torch


from envs import register_env
from utils.data import Data
from envs.BasicContinuousRealDataEnv import BasicContinuousRealDataEnv
from envs.BaseEnv import BaseEnv

logger = get_logger("ContinuousRealDataEnv1")


@register_env("ContinuousRealDataEnv1")
class ContinuousRealDataEnv1(BasicContinuousRealDataEnv):
    """
    Reference:
        https://arxiv.org/abs/1706.10059
    """

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        super(ContinuousRealDataEnv1, ContinuousRealDataEnv1).add_args(parser)

    def __init__(
        self,
        args: argparse.Namespace,
        data: Data,
        device: Optional[torch.device] = None,
    ) -> None:
        logger.info("Initializing ContinuousRealDataEnv1")
        super().__init__(args, data, device)
        self.previous_weight = self.portfolio_weight
        self.previous_value = self.portfolio_value
        logger.info("ContinuousRealDataEnv1 initialized")

    def to(self, device: str) -> None:
        super().to(device)

    def state_dimension(self) -> Dict[str, torch.Size]:
        """the dimension of the state tensors.

        Returns:
            Dict[str, torch.Size]: the dimension of the state tensors
        """
        return {
            "Xt": torch.Size([3, self.window_size, self.asset_num]),
            "Wt_ 1": torch.size([self.asset_num]),
            "previous_weight": torch.Size([self.asset_num]),
            "previous_value": torch.Size([1]),
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
            "Xt",
            "Wt_1",
            "previous_weight",
            "previous_value",
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
            state (Optional[Dict[str, torch.Tensor]], optional): the state tensors. Defaults to None.

        Returns:
            Dict[str, torch.Tensor]: the state tensors
        """
        if state is None:
            previous_weight = self.previous_weight
            previous_value = self.previous_value
            time_index = self.time_index
            portfolio_weight = self.portfolio_weight
            rf_weight = self.rf_weight
            portfolio_value = self.portfolio_value
        else:
            previous_weight: torch.Tensor = state["previous_weight"]
            previous_value: torch.Tensor = state["previous_value"]
            time_index: int = state["time_index"]
            portfolio_weight: torch.Tensor = state["portfolio_weight"]
            rf_weight: torch.Tensor = state["rf_weight"]
            portfolio_value: torch.Tensor = state["portfolio_value"]

        vt = self._get_price_tensor_in_window(time_index).transpose(0, 1)
        vt_hi = self._get_high_price_tensor_in_window(time_index).transpose(0, 1)
        vt_lo = self._get_low_price_tensor_in_window(time_index).transpose(0, 1)

        vt_hi = vt_hi / vt[0]
        vt_lo = vt_lo / vt[0]
        vt = vt / vt[0]

        Xt = torch.stack((vt, vt_hi, vt_lo), dim=0)
        Wt_1 = previous_weight

        return {
            "Xt": Xt,
            "Wt_1": Wt_1,
            "previous_value": previous_value,
            "previous_weight": previous_weight,
            "time_index": time_index,
            "portfolio_weight": portfolio_weight,
            "rf_weight": rf_weight,
            "portfolio_value": portfolio_value,
        }

    def act(
        self,
        action_weight: torch.Tensor,
        state: Optional[Dict[str, Union[torch.Tensor, int]]] = None,
    ) -> Tuple[Dict[str, Optional[Union[torch.Tensor, int]]], torch.Tensor, bool]:
        """
        perform an action (the portfolio weight)

        Args:
            action_weight (torch.Tensor): the action to perform
            state (Optional[Dict[str, Union[torch.Tensor, int]]], optional): the state tensors. Defaults to None.

        Returns:
            Tuple[Dict[str, Optional[Union[torch.Tensor, int]]], torch.Tensor, bool]: the new state, reward, and whether the episode is done
        """
        if state is None:
            time_index = self.time_index
            previous_value = self.previous_value
            portfolio_weight = self.portfolio_weight
            portfolio_value = self.portfolio_value
        else:
            time_index: int = state["time_index"]
            previous_value: torch.Tensor = state["previous_value"]
            portfolio_weight: torch.Tensor = state["portfolio_weight"]
            portfolio_value: torch.Tensor = state["portfolio_value"]

        action, mu = BaseEnv._get_trading_size_according_to_weight_after_trade(
            self,
            portfolio_weight,
            action_weight,
            portfolio_value,
        )

        new_state = self.update(action_weight, state=state, modify_inner_state=False)
        reward = torch.log(new_state["previous_value"] / previous_value)
        done = time_index == self.data.time_dimension() - 2

        return new_state, reward, done

    def update(
        self,
        action_weight: torch.Tensor,
        state: Optional[Dict[str, Union[torch.Tensor, int]]],
        modify_inner_state: Optional[bool] = None,
    ) -> Dict[str, Union[torch.Tensor, int]]:
        """
        update the environment

        Args:
            action_weight (torch.Tensor): the action to perform, means the weight after trade
            state (Optional[Dict[str, Union[torch.Tensor, int]]], optional): the state tensors. Defaults to None.
            modify_inner_state (Optional[bool], optional): whether to modify the inner state. Defaults to None.

        returns:
            Dict[str, Union[torch.Tensor, int]]: the new state
        """
        if state is None:
            portfolio_weight = self.portfolio_weight
            portfolio_value = self.portfolio_value
        else:
            portfolio_weight: torch.Tensor = state["portfolio_weight"]
            portfolio_value: torch.Tensor = state["portfolio_value"]
        if modify_inner_state is None:
            modify_inner_state = state is None

        action, mu = BaseEnv._get_trading_size_according_to_weight_after_trade(
            self, portfolio_weight, action_weight, portfolio_value
        )
        # print("111")
        # print(portfolio_weight)
        # print(action_weight)
        # print(portfolio_value)
        # print(action)
        # print("222")
        new_state = BaseEnv.update(self, action, state, modify_inner_state)
        if modify_inner_state:
            self.previous_weight = new_state["new_portfolio_weight_prev_day"]
            self.previous_value = new_state["new_portfolio_value_prev_day"]
        new_state["previous_value"] = new_state["new_portfolio_value_prev_day"]
        new_state["previous_weight"] = new_state["new_portfolio_weight_prev_day"]
        new_state.pop("new_rf_weight_prev_day", None)
        new_state.pop("static_portfolio_value", None)
        ret_state = self.get_state(new_state)
        ret_state["new_portfolio_weight_prev_day"] = new_state[
            "new_portfolio_weight_prev_day"
        ]
        ret_state["prev_price"] = new_state["prev_price"]
        return ret_state
