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
        parser.add_argument(
            "--episode_length",
            type=int,
            default=200,
            help="the length of each episode",
        )
        parser.add_argument(
            "--distribution_beta",
            type=float,
            default=0.5,
            help="the beta parameter for the distribution, range from 0 to 1",
        )

    def __init__(
        self,
        args: argparse.Namespace,
        data: Data,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(args, data, device)
        logger.info("initializing DiscreteRealDataEnv1")

        self.episode_range = []
        self.episode_length = args.episode_length
        for end_time_index in range(
            self.data.time_dimension() - 1, 1, -self.episode_length
        ):
            start_time_index = end_time_index - self.episode_length
            self.episode_range.append(
                {"start_time_index": start_time_index, "end_time_index": end_time_index}
            )
        self.episode_range.reverse()
        self.episode_num = len(self.episode_range)
        if self.episode_num == 0:
            raise ValueError("no valid episode range found")

        self.time_index = self.episode_range[0]["start_time_index"]
        self.start_time_index = self.episode_range[0]["start_time_index"]
        self.end_time_index = self.episode_range[0]["end_time_index"]
        self.episode = 0

        beta = args.distribution_beta
        self.accumulated_prob = []
        for episode in range(0, self.episode_num):
            prob = (
                beta
                * (1 - beta) ** (self.episode_num - episode - 1)
                / (1 - (1 - beta) ** self.episode_num)
            )
            if episode == 0:
                self.accumulated_prob.append(prob)
            else:
                self.accumulated_prob.append(prob + self.accumulated_prob[-1])
        self.accumulated_prob = torch.tensor(
            self.accumulated_prob, dtype=torch.float32, device=self.device
        )

        self.portfolio_value = torch.tensor(args.initial_balance, device=self.device)
        self.trading_size = torch.tensor(args.trading_size, device=self.device)

        self.portfolio_weight = torch.zeros(len(self.asset_codes), device=self.device)
        self.cash_weight = torch.tensor(1.0, device=self.device)

        # compute all Kx in advance
        kc_list, ko_list, kh_list, kl_list, kv_list = [], [], [], [], []
        price_list = []
        for time_index in range(1, self.data.time_dimension()):
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
        # self.kc_matrix = torch.stack(kc_list, dim=1)
        # self.ko_matrix = torch.stack(ko_list, dim=1)
        # self.kh_matrix = torch.stack(kh_list, dim=1)
        # self.kl_matrix = torch.stack(kl_list, dim=1)
        # self.kv_matrix = torch.stack(kv_list, dim=1)

        kc_matrix = torch.stack(kc_list, dim=1)
        ko_matrix = torch.stack(ko_list, dim=1)
        kh_matrix = torch.stack(kh_list, dim=1)
        kl_matrix = torch.stack(kl_list, dim=1)
        kv_matrix = torch.stack(kv_list, dim=1)
        self.Xt_matrix = torch.stack(
            [kc_matrix, ko_matrix, kh_matrix, kl_matrix, kv_matrix], dim=0
        )

        price_matrix = torch.stack(price_list, dim=1)
        self.price_change_matrix = price_matrix[:, 1:] / price_matrix[:, :-1]

        # compute all actions
        self.all_actions = []
        action_number = range(-1, 2)  # -1, 0, 1
        for action in product(action_number, repeat=len(self.asset_codes)):
            self.all_actions.append(
                torch.tensor(action, dtype=torch.int8, device=self.device)
            )

        logger.info("DiscreteRealDataEnv1 initialized")

    def to(self, device: str) -> None:
        logger.info("Changing device to %s", device)
        self.device = torch.device(device)
        self.accumulated_prob = self.accumulated_prob.to(self.device)
        self.portfolio_value = self.portfolio_value.to(self.device)
        self.trading_size = self.trading_size.to(self.device)
        self.portfolio_weight = self.portfolio_weight.to(self.device)
        self.cash_weight = self.cash_weight.to(self.device)
        # self.kc_matrix = self.kc_matrix.to(self.device)
        # self.ko_matrix = self.ko_matrix.to(self.device)
        # self.kh_matrix = self.kh_matrix.to(self.device)
        # self.kl_matrix = self.kl_matrix.to(self.device)
        # self.kv_matrix = self.kv_matrix.to(self.device)
        self.Xt_matrix = self.Xt_matrix.to(self.device)
        self.price_change_matrix = self.price_change_matrix.to(self.device)
        self.all_actions = [a.to(self.device) for a in self.all_actions]

    def sample_distribution_and_set_episode(self) -> int:
        # sample according to self.accumulated_prob
        prob = torch.rand(1, device=self.device)
        for episode in range(0, self.episode_num):
            if prob < self.accumulated_prob[episode]:
                self.set_episode(episode)
                return episode

    def set_episode(self, episode: int) -> None:
        self.episode = episode
        self.time_index = self.episode_range[episode]["start_time_index"]
        self.start_time_index = self.episode_range[episode]["start_time_index"]
        self.end_time_index = self.episode_range[episode]["end_time_index"]

    def time_range(self) -> range:
        return range(self.start_time_index, self.end_time_index)

    def state_dimension(self) -> Dict[str, torch.Size]:
        return {
            # "Kc_Matrix": torch.Size([len(self.asset_codes), self.window_size]),
            # "Ko_Matrix": torch.Size([len(self.asset_codes), self.window_size]),
            # "Kh_Matrix": torch.Size([len(self.asset_codes), self.window_size]),
            # "Kl_Matrix": torch.Size([len(self.asset_codes), self.window_size]),
            # "Kv_Matrix": torch.Size([len(self.asset_codes), self.window_size]),
            "Xt_Matrix": torch.Size([5, len(self.asset_codes), self.window_size]),
            "Portfolio_Weight": torch.Size([len(self.asset_codes) + 1]),
        }

    def state_tensor_names(self):
        return [
            # "Kc_Matrix",
            # "Ko_Matrix",
            # "Kh_Matrix",
            # "Kl_Matrix",
            # "Kv_Matrix",
            "Xt_Matrix",
            "Portfolio_Weight",
        ]

    def action_dimension(self) -> torch.Size:
        return torch.Size([len(self.asset_codes)])

    def get_state(self) -> Optional[Dict[str, torch.Tensor]]:
        if self.time_index - self.start_time_index < self.window_size - 1:
            return None
        return {
            # "Kc_Matrix": self.__get_Kx_State(self.kc_matrix),
            # "Ko_Matrix": self.__get_Kx_State(self.ko_matrix),
            # "Kh_Matrix": self.__get_Kx_State(self.kh_matrix),
            # "Kl_Matrix": self.__get_Kx_State(self.kl_matrix),
            # "Kv_Matrix": self.__get_Kx_State(self.kv_matrix),
            "Xt_Matrix": self.__get_Xt_State(),
            "Portfolio_Weight": self.__concat_weight(
                self.portfolio_weight, self.cash_weight
            ),
        }

    def __concat_weight(
        self, portfolio_weight: torch.Tensor, cash_weight: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat((cash_weight.unsqueeze(0), portfolio_weight), dim=0)

    def __get_Kx_State(
        self, kx_matrix: torch.Tensor, time_index: Optional[int] = None
    ) -> torch.Tensor:
        if time_index is None:
            time_index = self.time_index
        return kx_matrix[:, time_index - self.window_size + 1 : time_index]

    def __get_Xt_State(self, time_index: Optional[int] = None) -> torch.Tensor:
        if time_index is None:
            time_index = self.time_index
        return self.Xt_matrix[:, :, time_index - self.window_size + 1 : time_index]

    def __get_price_change_ratio_tensor(
        self, time_index: Optional[int] = None
    ) -> torch.tensor:
        if time_index is None:
            time_index = self.time_index
        return self.price_change_matrix[:, time_index - 1]

    def act(
        self, action: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, bool]:
        if action.size() != self.action_dimension():
            raise ValueError("action dimension not match")
        if self.find_action_index(action) == -1:
            raise ValueError("action not valid")

        (
            new_portfolio_weight,
            new_cash_weight,
            new_portfolio_value,
            static_portfolio_value,
        ) = self.get_new_portfolio_weight_and_value(action)

        reward = (new_portfolio_value - static_portfolio_value) / static_portfolio_value

        done = self.time_index == self.end_time_index - 1

        new_state = {
            # "Kc_Matrix": self.__get_Kx_State(self.kc_matrix, self.time_index + 1),
            # "Ko_Matrix": self.__get_Kx_State(self.ko_matrix, self.time_index + 1),
            # "Kh_Matrix": self.__get_Kx_State(self.kh_matrix, self.time_index + 1),
            # "Kl_Matrix": self.__get_Kx_State(self.kl_matrix, self.time_index + 1),
            # "Kv_Matrix": self.__get_Kx_State(self.kv_matrix, self.time_index + 1),
            "Xt_Matrix": self.__get_Xt_State(self.time_index + 1),
            "Portfolio_Weight": self.__concat_weight(
                new_portfolio_weight, new_cash_weight
            ),
        }

        return new_state, reward, done

    def get_new_portfolio_weight_and_value(
        self, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # for all action[i], weight[i] += action[i] * trading_size / portfolio_value
        new_portfolio_weight = (
            self.portfolio_weight + action * self.trading_size / self.portfolio_value
        )
        new_cash_weight = torch.tensor(1.0, device=self.device) - torch.sum(
            new_portfolio_weight
        )
        # changing to the next day
        # portfolio_value = value * (price change vec * portfolio_weight + cash_weight)
        price_change_rate = self.__get_price_change_ratio_tensor(self.time_index + 1)
        new_portfolio_value = self.portfolio_value * (
            torch.sum(price_change_rate * new_portfolio_weight) + new_cash_weight
        )
        static_portfolio_value = self.portfolio_value * (
            torch.sum(price_change_rate * self.portfolio_weight) + self.cash_weight
        )
        # adjust weight based on new value
        new_portfolio_weight = (
            price_change_rate
            * new_portfolio_weight
            * self.portfolio_value
            / new_portfolio_value
        )
        new_cash_weight = torch.tensor(1.0, device=self.device) - torch.sum(
            new_portfolio_weight
        )
        return (
            new_portfolio_weight,
            new_cash_weight,
            new_portfolio_value,
            static_portfolio_value,
        )

    def update(self, action: torch.Tensor) -> None:
        self.time_index += 1
        self.portfolio_weight, self.cash_weight, self.portfolio_value, _ = (
            self.get_new_portfolio_weight_and_value(action)
        )

    def reset(self, args: argparse.Namespace) -> None:
        logger.info("resetting DiscreteRealDataEnv1")
        self.time_index = self.start_time_index
        self.portfolio_value = torch.tensor(args.initial_balance, device=self.device)
        self.portfolio_weight = torch.zeros(len(self.asset_codes), device=self.device)
        self.cash_weight = torch.tensor(1.0, device=self.device)

    def __cash_shortage(self, action: torch.Tensor) -> bool:
        return (
            torch.sum(action * self.trading_size)
            > self.portfolio_value * self.cash_weight
        )

    def __asset_shortage(self, action: torch.Tensor) -> bool:
        return torch.any(
            self.portfolio_weight[action < 0] * self.portfolio_value
            < torch.abs(action[action < 0]) * self.trading_size
        )

    def __action_validity(self, action: torch.Tensor) -> bool:
        return not self.__cash_shortage(action) and not self.__asset_shortage(action)

    def find_action_index(self, action: torch.Tensor) -> int:
        for i, a in enumerate(self.all_actions):
            if torch.equal(a, action):
                return i
        return -1

    def possible_action_index(self) -> torch.Tensor:
        possible_action_index = []
        for idx, action in enumerate(self.all_actions):
            if self.__action_validity(action):
                possible_action_index.append(idx)
        return torch.tensor(
            possible_action_index, dtype=torch.int32, device=self.device
        )

    def action_mapping(
        self, action: torch.Tensor, Q_Values: torch.Tensor
    ) -> torch.Tensor:
        if self.find_action_index(action) == -1:
            raise ValueError("action not valid")
        if self.__asset_shortage(action):
            return self.__action_mapping_rule2(action, Q_Values)
        elif self.__cash_shortage(action):
            return self.__action_mapping_rule1(action)
        return copy.deepcopy(action)

    def __action_mapping_rule1(
        self, action: torch.Tensor, Q_Values: torch.Tensor
    ) -> torch.Tensor:
        possible_action_index = []
        for idx, new_action in enumerate(self.all_actions):
            if (
                torch.all(new_action[action == 1] == 1)
                and torch.all(action[new_action == 1] == 1)
                and torch.all(new_action[action == 0] == 0)
                and self.__action_validity(new_action)
            ):
                possible_action_index.append(idx)

        possible_values = Q_Values[possible_action_index]
        max_index = torch.argmax(possible_values)
        return copy.deepcopy(self.all_actions[possible_action_index[max_index]])

    def __action_mapping_rule2(
        self, action: torch.Tensor, Q_Values: torch.Tensor
    ) -> torch.Tensor:
        new_action = copy.deepcopy(action)
        condition = (
            self.portfolio_weight[new_action < 0] * self.portfolio_value
            < torch.abs(new_action[new_action < 0]) * self.trading_size
        )
        new_action[condition] = 0
        if self.__cash_shortage(new_action):
            return self.__action_mapping_rule1(new_action, Q_Values)
        return new_action
