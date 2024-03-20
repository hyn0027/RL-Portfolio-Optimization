import argparse
from typing import Dict, Optional, Tuple, List
import copy


from envs import register_env
from data import Data
from envs.BasicRealDataEnv import BasicRealDataEnv
import pandas as pd
import torch


@register_env("DiscreteRealDataEnv1")
class DiscreteRealDataEnv1(BasicRealDataEnv):
    """
    reference: https://arxiv.org/abs/1907.03665
    The environment for discrete action space, used by DQN in the paper.
    """

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        super(DiscreteRealDataEnv1, DiscreteRealDataEnv1).add_args(parser)
        parser.add_argument(
            "--initial_balance",
            type=float,
            default=1e6,
            help="the initial balance",
        )
        parser.add_argument(
            "--trading_size",
            type=float,
            default=1e4,
            help="the size of each trading",
        )

    def __init__(self, args: argparse.Namespace, data: Data) -> None:
        super().__init__(args, data)
        self.time_index = 1
        self.portfolio_value = args.initial_balance

        self.portfolio_weight = torch.cat(
            (torch.tensor([1.0]), torch.zeros(len(self.asset_codes)))
        )
        self.kc_window: List[torch.Tensor] = []
        self.ko_window: List[torch.Tensor] = []
        self.kh_window: List[torch.Tensor] = []
        self.kl_window: List[torch.Tensor] = []
        self.kv_window: List[torch.Tensor] = []

    def time_dimension(self) -> int:
        return self.data.time_dimension() - 1

    def state_dimension(self) -> Dict[str, torch.Size]:
        return {
            "Kc_Matrix": torch.Size([len(self.asset_codes), self.window_size]),
            "Ko_Matrix": torch.Size([len(self.asset_codes), self.window_size]),
            "Kh_Matrix": torch.Size([len(self.asset_codes), self.window_size]),
            "Kl_Matrix": torch.Size([len(self.asset_codes), self.window_size]),
            "Kv_Matrix": torch.Size([len(self.asset_codes), self.window_size]),
            "Portfolio_Weight": torch.Size([len(self.asset_codes)]),
        }

    def state_tensor_names(self):
        return [
            "Kc_Matrix",
            "Ko_Matrix",
            "Kh_Matrix",
            "Kl_Matrix",
            "Kv_Matrix",
            "Portfolio_Weight",
        ]

    def action_dimension(self) -> torch.Size:
        return torch.Size([len(self.asset_codes)])

    def get_state(self) -> Optional[Dict[str, torch.tensor]]:
        if self.time_index < self.window_size - 1:
            return None
        return {
            "Kc_Matrix": self.__get_Kx_Matrix(self.kc_window),
            "Ko_Matrix": self.__get_Kx_Matrix(self.ko_window),
            "Kh_Matrix": self.__get_Kx_Matrix(self.kh_window),
            "Kl_Matrix": self.__get_Kx_Matrix(self.kl_window),
            "Kv_Matrix": self.__get_Kx_Matrix(self.kv_window),
            "Portfolio_Weight": self.portfolio_weight,
        }

    def __get_Kx_Matrix(self, kx_window: List[torch.Tensor]) -> torch.tensor:
        return torch.stack(kx_window, dim=1)

    def __update_Kx_window(
        self, kx_window: List[torch.Tensor], new_kx: torch.Tensor
    ) -> List[torch.Tensor]:
        if len(kx_window) == self.window_size:
            kx_window.pop(0)
        kx_window.append(new_kx)

    def step(
        self, action: torch.tensor, update: bool
    ) -> Tuple[Dict[str, torch.tensor], float, bool]:
        if action.size() != self.action_dimension():
            raise ValueError("action dimension not match")

        # calculate
        # TODO: implement step
        # calculate new portfolio_weight
        new_kc_window = copy.deepcopy(self.kc_window)
        new_ko_window = copy.deepcopy(self.ko_window)
        new_kh_window = copy.deepcopy(self.kh_window)
        new_kl_window = copy.deepcopy(self.kl_window)
        new_kv_window = copy.deepcopy(self.kv_window)

        new_kc = []
        new_ko = []
        new_kh = []
        new_kl = []
        new_kv = []

        for asset_code in self.asset_codes:
            asset_data = self.data.get_asset_hist_at_time(
                asset_code, self.data.time_list[self.time_index]
            )
            asset_previous_data = self.data.get_asset_hist_at_time(
                asset_code, self.data.time_list[self.time_index - 1]
            )
            new_kc.append(
                (asset_data["Close"] - asset_previous_data["Close"])
                / asset_previous_data["Close"]
            )
            new_ko.append(
                (asset_data["Open"] - asset_previous_data["Close"])
                / asset_previous_data["Close"]
            )
            new_kh.append(
                (asset_data["Close"] - asset_data["High"]) / asset_data["High"]
            )
            new_kl.append((asset_data["Close"] - asset_data["Low"]) / asset_data["Low"])
            new_kv.append(
                (asset_data["Volume"] - asset_previous_data["Volume"])
                / asset_previous_data["Volume"]
            )
        new_kc = torch.tensor(new_kc)
        new_ko = torch.tensor(new_ko)
        new_kh = torch.tensor(new_kh)
        new_kl = torch.tensor(new_kl)
        new_kv = torch.tensor(new_kv)

        new_kc_window = self.__update_Kx_window(new_kc_window, new_kc)
        new_ko_window = self.__update_Kx_window(new_ko_window, new_ko)
        new_kh_window = self.__update_Kx_window(new_kh_window, new_kh)
        new_kl_window = self.__update_Kx_window(new_kl_window, new_kl)

        # update
        if update:
            self.time_index += 1
            self.kc_window = new_kc_window
            self.ko_window = new_ko_window
            self.kh_window = new_kh_window
            self.kl_window = new_kl_window
            self.kv_window = new_kv_window
            # update portfolio_weight
            # update portfolio_value

    def reset(self) -> None:
        raise NotImplementedError("reset not implemented")
