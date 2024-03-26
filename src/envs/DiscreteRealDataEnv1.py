import argparse
from typing import Dict, Optional, Tuple, List
import copy
from itertools import product
from utils.logging import get_logger
import torch
import random


from envs import register_env
from utils.data import Data
from envs.BasicRealDataEnv import BasicRealDataEnv

logger = get_logger("DiscreteRealDataEnv1")


@register_env("DiscreteRealDataEnv1")
class DiscreteRealDataEnv1(BasicRealDataEnv):
    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        super(DiscreteRealDataEnv1, DiscreteRealDataEnv1).add_args(parser)
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
            default=0.3,
            help="the beta parameter for the distribution, range from 0 to 1",
        )

    def __init__(
        self,
        args: argparse.Namespace,
        data: Data,
        device: Optional[torch.device] = None,
    ) -> None:
        logger.info("initializing DiscreteRealDataEnv1")

        super().__init__(args, data, device)

        self.episode_range = []
        self.episode_length: int = args.episode_length
        for end_time_index in range(
            self.data.time_dimension() - 1, 1, -self.episode_length
        ):
            start_time_index = end_time_index - self.episode_length
            if start_time_index < self.window_size:
                break
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

        beta: float = args.distribution_beta
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
            self.accumulated_prob, dtype=self.dtype, device=self.device
        )

        self.trading_size = torch.tensor(
            args.trading_size, dtype=self.dtype, device=self.device
        )

        self.initialize_weight()

        # compute all Kx in advance
        kc_list, ko_list, kh_list, kl_list, kv_list = [], [], [], [], []
        for time_index in range(1, self.data.time_dimension()):
            new_kc, new_ko, new_kh, new_kl, new_kv = [], [], [], [], []
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
            kc_list.append(torch.tensor(new_kc, dtype=self.dtype, device=self.device))
            ko_list.append(torch.tensor(new_ko, dtype=self.dtype, device=self.device))
            kh_list.append(torch.tensor(new_kh, dtype=self.dtype, device=self.device))
            kl_list.append(torch.tensor(new_kl, dtype=self.dtype, device=self.device))
            kv_list.append(torch.tensor(new_kv, dtype=self.dtype, device=self.device))

        kc_matrix = torch.stack(kc_list, dim=1)
        ko_matrix = torch.stack(ko_list, dim=1)
        kh_matrix = torch.stack(kh_list, dim=1)
        kl_matrix = torch.stack(kl_list, dim=1)
        kv_matrix = torch.stack(kv_list, dim=1)
        self.Xt_matrix = torch.stack(
            [kc_matrix, ko_matrix, kh_matrix, kl_matrix, kv_matrix], dim=0
        )
        self.Xt_matrix[torch.isnan(self.Xt_matrix)] = 0

        # compute all actions
        self.all_actions = []
        action_number = range(-1, 2)  # -1, 0, 1
        for action in product(action_number, repeat=len(self.asset_codes)):
            self.all_actions.append(
                torch.tensor(action, dtype=torch.int8, device=self.device)
            )

        logger.info("DiscreteRealDataEnv1 initialized")

    def to(self, device: str) -> None:
        super().to(device)
        self.accumulated_prob = self.accumulated_prob.to(self.device)
        self.trading_size = self.trading_size.to(self.device)
        self.Xt_matrix = self.Xt_matrix.to(self.device)
        self.price_change_matrix = self.price_change_matrix.to(self.device)
        self.all_actions = [a.to(self.device) for a in self.all_actions]

    def sample_distribution_and_set_episode(self) -> int:
        # sample according to self.accumulated_prob
        prob = torch.rand(1, dtype=self.dtype, device=self.device)
        for episode in range(0, self.episode_num):
            if prob < self.accumulated_prob[episode]:
                self.set_episode(episode)
                return episode

    def set_episode(self, episode: int) -> None:
        self.episode = episode
        self.time_index = self.episode_range[episode]["start_time_index"]
        self.start_time_index = self.episode_range[episode]["start_time_index"]
        self.end_time_index = self.episode_range[episode]["end_time_index"]

    def set_episode_for_testing(self) -> None:
        self.episode = -1
        self.start_time_index = self.window_size
        self.end_time_index = self.data.time_dimension() - 1
        self.time_index = self.start_time_index

    def train_time_range(self) -> range:
        return range(self.start_time_index, self.end_time_index)

    def test_time_range(self) -> range:
        return range(self.start_time_index, self.end_time_index)

    def pretrain_train_time_range(self, shuffle: bool = True) -> List:
        range_list = list(range(self.window_size + 100, self.data.time_dimension() - 1))
        if shuffle:
            random.shuffle(range_list)
        return range_list

    def pretrain_eval_time_range(self) -> filter:
        return range(self.window_size, self.window_size + 100)

    def state_dimension(self) -> Dict[str, torch.Size]:
        return {
            "Xt_Matrix": torch.Size([5, len(self.asset_codes), self.window_size]),
            "Portfolio_Weight": torch.Size([len(self.asset_codes) + 1]),
        }

    def state_tensor_names(self):
        return [
            "Xt_Matrix",
            "Portfolio_Weight",
        ]

    def action_dimension(self) -> torch.Size:
        return torch.Size([len(self.asset_codes)])

    def get_state(self) -> Optional[Dict[str, torch.Tensor]]:
        if self.time_index < self.window_size:
            return None
        return {
            "Xt_Matrix": self._get_Xt_state(),
            "Portfolio_Weight": self._concat_weight(
                self.portfolio_weight, self.rf_weight
            ),
        }

    def _get_Xt_state(self, time_index: Optional[int] = None) -> torch.Tensor:
        if time_index is None:
            time_index = self.time_index
        return self.Xt_matrix[:, :, time_index - self.window_size : time_index]

    def get_Xt_state_and_pretrain_target(
        self, time_index: int
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if time_index < self.window_size:
            return None
        return self._get_Xt_state(time_index), self.Xt_matrix[:, :, time_index]

    def act(
        self, action: int
    ) -> Tuple[Dict[str, Optional[torch.Tensor]], torch.Tensor, bool]:
        if action < 0 or action >= len(self.all_actions):
            raise ValueError("action not valid")
        action = self.all_actions[action]
        if not self._action_validity(action):
            raise ValueError("action not valid")

        (
            new_portfolio_weight,
            new_portfolio_weight_next_day,
            new_rf_weight,
            new_portfolio_value,
            static_portfolio_value,
        ) = self._get_new_portfolio_weight_and_value(action)

        reward = (
            (new_portfolio_value - static_portfolio_value)
            / static_portfolio_value
            * 100
        )

        done = self.time_index == self.end_time_index - 1

        new_state = {
            "Xt_Matrix": (
                self._get_Xt_state(self.time_index + 1)
                if self.time_index + 1 >= self.window_size
                else None
            ),
            "Portfolio_Weight": self._concat_weight(
                new_portfolio_weight_next_day, new_rf_weight
            ),
            "Portfolio_Weight_Today": new_portfolio_weight,
        }

        return new_state, reward, done

    def _get_new_portfolio_weight_and_value(
        self, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # for all action[i], weight[i] += action[i] * trading_size / portfolio_value
        trading_size = action * self.trading_size
        return super()._get_new_portfolio_weight_and_value(trading_size)

    def update(self, action: int) -> None:
        if action < 0 or action >= len(self.all_actions):
            raise ValueError("action not valid")
        action = self.all_actions[action]
        trading_size = action * self.trading_size
        super().update(trading_size)

    def reset(self) -> None:
        logger.info("resetting DiscreteRealDataEnv1")
        self.time_index = self.start_time_index
        super().initialize_weight()

    def _cash_shortage(self, action: torch.Tensor) -> bool:
        return super()._cash_shortage(action * self.trading_size)

    def _asset_shortage(self, action: torch.Tensor) -> bool:
        return super()._asset_shortage(action * self.trading_size)

    def _action_validity(self, action: torch.Tensor) -> bool:
        return not self._cash_shortage(action) and not self._asset_shortage(action)

    def find_action_index(self, action: torch.Tensor) -> int:
        for i, a in enumerate(self.all_actions):
            if torch.equal(a, action):
                return i
        return -1

    def possible_action_indexes(self) -> torch.Tensor:
        possible_action_indexes = []
        for idx, action in enumerate(self.all_actions):
            if self._action_validity(action):
                possible_action_indexes.append(idx)
        return torch.tensor(
            possible_action_indexes, dtype=torch.int32, device=self.device
        )

    def action_mapping(self, action_index: int, Q_Values: torch.Tensor) -> int:
        if action_index < 0 or action_index >= len(self.all_actions):
            raise ValueError("action not valid")
        action = self.all_actions[action_index]
        if self._asset_shortage(action):
            return self._action_mapping_rule2(action, Q_Values)
        elif self._cash_shortage(action):
            return self._action_mapping_rule1(action, Q_Values)
        return action_index

    def _action_mapping_rule1(
        self, action: torch.Tensor, Q_Values: torch.Tensor
    ) -> int:
        possible_action_indexes = []
        for idx, new_action in enumerate(self.all_actions):
            if (
                torch.all(new_action[action == -1] == -1)
                and torch.all(action[new_action == -1] == -1)
                and torch.all(new_action[action == 0] == 0)
                and self._action_validity(new_action)
            ):
                possible_action_indexes.append(idx)

        possible_values = Q_Values[possible_action_indexes]
        max_index = torch.argmax(possible_values)
        return possible_action_indexes[max_index]

    def _action_mapping_rule2(
        self, action: torch.Tensor, Q_Values: torch.Tensor
    ) -> int:
        new_action = copy.deepcopy(action)

        condition = (new_action < 0) & (
            self.portfolio_weight * self.portfolio_value
            < torch.abs(new_action) * self.trading_size
        )
        new_action[condition] = 0
        if self._cash_shortage(new_action):
            return self._action_mapping_rule1(new_action, Q_Values)
        return self.find_action_index(new_action)
