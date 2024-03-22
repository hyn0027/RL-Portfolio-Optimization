import argparse
from typing import Dict, Optional, Tuple, List
import copy
from itertools import product
from utils.logging import get_logger


from envs import register_env
from data import Data
from envs.BasicRealDataEnv import BasicRealDataEnv
import torch

logger = get_logger("DiscreteRealDataEnv1")


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
            help="the size of each trading in terms of currency",
        )

    def __init__(
        self,
        args: argparse.Namespace,
        data: Data,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(args, data, device)
        logger.info("initializing DiscreteRealDataEnv1")

        self.time_index = 1
        self.portfolio_value = torch.tensor([args.initial_balance], device=self.device)
        self.trading_size = torch.tensor(args.trading_size, device=self.device)

        self.portfolio_weight = torch.zeros(len(self.asset_codes), device=self.device)
        self.cash_weight = torch.tensor([1.0], device=self.device)

        # compute all Kx in advance
        kc_list, ko_list, kh_list, kl_list, kv_list = [], [], [], [], []
        price_list = []
        for time_index in self.time_range():
            new_kc, new_ko, new_kh, new_kl, new_kv = [], [], [], [], []
            new_price = []
            for asset_code in self.asset_codes:
                asset_data = self.data.get_asset_hist_at_time(
                    asset_code, self.data.time_list[time_index]
                )
                asset_previous_data = self.data.get_asset_hist_at_time(
                    asset_code, self.data.time_list[time_index - 1]
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
                new_kl.append(
                    (asset_data["Close"] - asset_data["Low"]) / asset_data["Low"]
                )
                new_kv.append(
                    (asset_data["Volume"] - asset_previous_data["Volume"])
                    / asset_previous_data["Volume"]
                )
                new_price.append(asset_data["Close"])
            kc_list.append(torch.tensor(new_kc, device=self.device))
            ko_list.append(torch.tensor(new_ko, device=self.device))
            kh_list.append(torch.tensor(new_kh, device=self.device))
            kl_list.append(torch.tensor(new_kl, device=self.device))
            kv_list.append(torch.tensor(new_kv, device=self.device))
            price_list.append(torch.tensor(new_price, device=self.device))
        self.full_kc_matrix = torch.stack(kc_list, dim=1)
        self.full_ko_matrix = torch.stack(ko_list, dim=1)
        self.full_kh_matrix = torch.stack(kh_list, dim=1)
        self.full_kl_matrix = torch.stack(kl_list, dim=1)
        self.full_kv_matrix = torch.stack(kv_list, dim=1)
        self.full_price_matrix = torch.stack(price_list, dim=1)
        self.price_change_matrix = (
            self.full_price_matrix[:, 1:] / self.full_price_matrix[:, :-1]
        )

        # compute all actions
        self.all_actions = []
        action_number = range(-1, 2)  # -1, 0, 1
        for action in product(action_number, repeat=len(self.asset_codes)):
            self.all_actions.append(
                torch.tensor(action, dtype=torch.int8, device=self.device)
            )

        logger.info("DiscreteRealDataEnv1 initialized")

    def time_range(self) -> range:
        return range(1, self.data.time_dimension())

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

    def get_state(self) -> Optional[Dict[str, torch.Tensor]]:
        if self.time_index < self.window_size - 1:
            return None
        return {
            "Kc_Matrix": self.__get_Kx_State(self.kc_matrix),
            "Ko_Matrix": self.__get_Kx_State(self.ko_matrix),
            "Kh_Matrix": self.__get_Kx_State(self.kh_matrix),
            "Kl_Matrix": self.__get_Kx_State(self.kl_matrix),
            "Kv_Matrix": self.__get_Kx_State(self.kv_matrix),
            "Portfolio_Weight": self.portfolio_weight,
        }

    def __get_Kx_State(
        self, kx_matrix: torch.Tensor, time_index: Optional[int] = None
    ) -> torch.Tensor:
        if time_index is None:
            time_index = self.time_index
        return kx_matrix[:, time_index - self.window_size + 1 : time_index]

    def __get_price_tensor(self, time_index: Optional[int] = None) -> torch.Tensor:
        if time_index is None:
            time_index = self.time_index
        return self.full_price_matrix[:, time_index]

    def __get_price_change_tensor(
        self, time_index: Optional[int] = None
    ) -> torch.tensor:
        if time_index is None:
            time_index = self.time_index
        return self.price_change_matrix[:, time_index - 1]

    def act(self, action: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], float, bool]:
        if action.size() != self.action_dimension():
            raise ValueError("action dimension not match")
        if action not in self.possible_actions():
            raise ValueError("action not valid")

        # calculate
        # TODO: implement step
        # calculate new portfolio_weight
        new_portfolio_weight = copy.deepcopy(self.portfolio_weight)

        new_state = {
            "Kc_Matrix": self.__get_Kx_State(self.kc_matrix, self.time_index + 1),
            "Ko_Matrix": self.__get_Kx_State(self.ko_matrix, self.time_index + 1),
            "Kh_Matrix": self.__get_Kx_State(self.kh_matrix, self.time_index + 1),
            "Kl_Matrix": self.__get_Kx_State(self.kl_matrix, self.time_index + 1),
            "Kv_Matrix": self.__get_Kx_State(self.kv_matrix, self.time_index + 1),
            "Portfolio_Weight": new_portfolio_weight,
        }

    def update(self, action: torch.Tensor) -> None:
        self.time_index += 1
        # TODO
        pass

    def reset(self) -> None:
        raise NotImplementedError("reset not implemented")

    def __action_validity(self, action: torch.Tensor) -> bool:
        # for all action[i] < 0, check if the portfolio_weight[i] * portfolio_value >= abs(action[i]) * trading_size
        if torch.any(action < 0):
            if not torch.all(
                self.portfolio_weight[action < 0] * self.portfolio_value
                >= torch.abs(action[action < 0]) * self.trading_size
            ):
                return False
        # for all action, total cost += action[i] * trading_size
        total_cost = torch.sum(action * self.trading_size)
        return total_cost <= self.portfolio_value * self.cash_weight

    def possible_actions(self) -> List[torch.Tensor]:
        return [action for action in self.all_actions if self.__action_validity(action)]
