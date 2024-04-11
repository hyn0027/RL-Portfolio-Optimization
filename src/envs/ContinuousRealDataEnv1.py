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
            "Wt_ 1": torch.size([self.asset_num + 1]),
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

        Xt = torch.stack((vt, vt_hi, vt_lo), dim=0)
        wt = self.previous_weight
        Wt_1 = torch.cat((torch.tensor([1.0]) - torch.sum(wt), wt), dim=0)

        return {
            "Xt": Xt,
            "Wt_1": Wt_1,
        }

    def act(
        self, action_weight: torch.Tensor
    ) -> Tuple[Dict[str, Optional[torch.Tensor]], torch.Tensor, bool]:
        """
        perform an action (the trading size)

        Args:
            action_weight (torch.Tensor): the action to perform, means the weight after trade

        Returns:
            Tuple[Dict[str, Optional[torch.Tensor]], torch.Tensor, bool]: the new state, reward, and whether the episode is done
        """
        action = BaseEnv._get_trading_size_according_to_weight_after_trade(
            self, self.portfolio_weight, action_weight, self.portfolio_value
        )
        (
            new_portfolio_weight,
            new_portfolio_weight_next_day,
            new_rf_weight,
            new_rf_weight_next_day,
            new_portfolio_value,
            static_portfolio_value,
        ) = self._get_new_portfolio_weight_and_value(action)

        reward = (
            (new_portfolio_value - self.portfolio_value) / self.portfolio_value * 100
        )

        done = self.time_index == self.data.time_dimension() - 2

        vt_new = self._get_price_tensor_in_window(self.time_index + 1).transpose(0, 1)
        vt_hi_new = self._get_high_price_tensor_in_window(
            self.time_index + 1
        ).transpose(0, 1)
        vt_lo_new = self._get_low_price_tensor_in_window(self.time_index + 1).transpose(
            0, 1
        )

        Xt_new = torch.stack((vt_new, vt_hi_new, vt_lo_new), dim=0)
        Wt_1_new = torch.cat()

        return (
            {
                "Xt": Xt_new,
                # TODO
                "Portfolio_Weight_Today": new_portfolio_weight,
            },
            reward,
            done,
        )

    def update(self, action_weight: torch.Tensor) -> None:
        """
        update the environment

        Args:
            action_weight (torch.Tensor): the action to perform, means the weight after trade
        """
        action = BaseEnv._get_trading_size_according_to_weight_after_trade(
            self, self.portfolio_weight, action_weight, self.portfolio_value
        )
        (
            new_portfolio_weight,
            new_portfolio_weight_next_day,
            new_rf_weight,
            new_rf_weight_next_day,
            new_portfolio_value,
            static_portfolio_value,
        ) = self._get_new_portfolio_weight_and_value(action)
        BaseEnv.update(self, action)
        self.previous_weight = new_portfolio_weight
