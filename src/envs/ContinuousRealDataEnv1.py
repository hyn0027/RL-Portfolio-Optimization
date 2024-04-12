import argparse
from typing import Dict, Optional, Tuple, List

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
        }

    def state_tensor_names(self) -> List[str]:
        """the names of the state tensors

        Returns:
            List[str]: the names of the state tensors
        """
        return ["Xt", "Wt_1"]

    def get_state(
        self,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """get the state tensors at the current time.

        Returns:
            Dict[str, torch.Tensor]: the state tensors
        """
        vt = self._get_price_tensor_in_window(self.time_index).transpose(0, 1)
        vt_hi = self._get_high_price_tensor_in_window(self.time_index).transpose(0, 1)
        vt_lo = self._get_low_price_tensor_in_window(self.time_index).transpose(0, 1)

        vt_hi = vt_hi / vt[0]
        vt_lo = vt_lo / vt[0]
        vt = vt / vt[0]

        Xt = torch.stack((vt, vt_hi, vt_lo), dim=0)
        Wt_1 = self.previous_weight

        return {
            "Xt": Xt,
            "Wt_1": Wt_1,
            "time_index": self.time_index,
            "portfolio_value": self.portfolio_value,
            "portfolio_weight": self.portfolio_weight,
            "previous_value": self.previous_value,
        }

    def act(self, action_weight: torch.Tensor, state: Dict) -> torch.Tensor:
        """
        perform an action (the trading size)

        Args:
            action_weight (torch.Tensor): the action to perform, means the weight after trade

        Returns:
            Tuple[Dict[str, Optional[torch.Tensor]], torch.Tensor, bool]: the new state, reward, and whether the episode is done
        """
        action, mu = BaseEnv._get_trading_size_according_to_weight_after_trade(
            self,
            state["portfolio_weight"],
            action_weight,
            state["portfolio_value"],
        )
        (
            new_portfolio_weight,
            new_portfolio_weight_next_day,
            new_rf_weight,
            new_rf_weight_next_day,
            new_portfolio_value,
            new_portfolio_value_next_day,
            static_portfolio_value,
        ) = self._get_new_portfolio_weight_and_value(
            action, time_index=state["time_index"]
        )

        reward = torch.log(new_portfolio_value / state["previous_value"])

        return reward

    def update(self, action_weight: torch.Tensor) -> None:
        """
        update the environment

        Args:
            action_weight (torch.Tensor): the action to perform, means the weight after trade
        """
        action, mu = BaseEnv._get_trading_size_according_to_weight_after_trade(
            self, self.portfolio_weight, action_weight, self.portfolio_value
        )
        (
            new_portfolio_weight,
            new_portfolio_weight_next_day,
            new_rf_weight,
            new_rf_weight_next_day,
            new_portfolio_value,
            new_portfolio_value_next_day,
            static_portfolio_value,
        ) = self._get_new_portfolio_weight_and_value(action)
        BaseEnv.update(self, action)
        self.previous_weight = new_portfolio_weight
        self.previous_value = new_portfolio_value
