import argparse
from typing import Dict, Optional, Tuple, List, Union
import copy
from utils.logging import get_logger
import torch
import random


from envs import register_env
from utils.data import Data
from envs.BasicDiscreteRealDataEnv import BasicDiscreteRealDataEnv
from envs.BaseEnv import BaseEnv

logger = get_logger("DiscreteRealDataEnv1")


@register_env("DiscreteRealDataEnv1")
class DiscreteRealDataEnv1(BasicDiscreteRealDataEnv):
    """
    Reference:
        original paper: https://arxiv.org/abs/1907.03665
    """

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        super(DiscreteRealDataEnv1, DiscreteRealDataEnv1).add_args(parser)
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

        logger.info("DiscreteRealDataEnv1 initialized")

    def to(self, device: str) -> None:
        super().to(device)
        self.accumulated_prob = self.accumulated_prob.to(self.device)
        self.Xt_matrix = self.Xt_matrix.to(self.device)
        self.price_change_matrix = self.price_change_matrix.to(self.device)

    def sample_distribution_and_set_episode(self) -> int:
        """sample a distribution and set the episode accordingly
        please refer to paper https://arxiv.org/abs/1907.03665 for more details

        Returns:
            int: the episode index
        """
        # sample according to self.accumulated_prob
        prob = torch.rand(1, dtype=self.dtype, device=self.device)
        for episode in range(0, self.episode_num):
            if prob < self.accumulated_prob[episode]:
                self.set_episode(episode)
                return episode

    def set_episode(self, episode: int) -> None:
        """set the episode given the episode index

        Args:
            episode (int): the episode index
        """
        self.episode = episode
        self.time_index = self.episode_range[episode]["start_time_index"]
        self.start_time_index = self.episode_range[episode]["start_time_index"]
        self.end_time_index = self.episode_range[episode]["end_time_index"]

    def set_episode_for_testing(self) -> None:
        """special function to set the episode for testing"""
        self.episode = -1
        self.start_time_index = self.window_size
        self.end_time_index = self.data.time_dimension() - 1
        self.time_index = self.start_time_index

    def train_time_range(self) -> range:
        return range(self.start_time_index, self.end_time_index)

    def test_time_range(self) -> range:
        return range(self.start_time_index, self.end_time_index)

    def pretrain_train_time_range(self, shuffle: bool = True) -> List:
        """the list of time indices for pretraining

        Args:
            shuffle (bool, optional): whether to shuffle. Defaults to True.

        Returns:
            List: the list of time indices
        """
        range_list = list(range(self.window_size + 100, self.data.time_dimension() - 1))
        if shuffle:
            random.shuffle(range_list)
        return range_list

    def pretrain_eval_time_range(self) -> range:
        """the list of time indices for pretraining evaluation

        Returns:
            range: the list of time indices
        """
        return range(self.window_size, self.window_size + 100)

    def state_dimension(self) -> Dict[str, torch.Size]:
        """the dimension of the state tensors, including Xt_Matrix and Portfolio_Weight

        Returns:
            Dict[str, torch.Size]: the dimension of the state tensors
        """
        return {
            "Xt_Matrix": torch.Size([5, len(self.asset_codes), self.window_size]),
            "Portfolio_Weight": torch.Size([len(self.asset_codes) + 1]),
            "time_index": torch.Size([1]),
            "portfolio_weight": torch.Size([self.asset_num]),
            "rf_weight": torch.Size([1]),
            "portfolio_value": torch.Size([1]),
        }

    def state_tensor_names(self):
        """the names of the state tensors, including Xt_Matrix and Portfolio_Weight

        Returns:
            List[str]: the names of the state tensors
        """
        return [
            "Xt_Matrix",
            "Portfolio_Weight",
            "time_index",
            "portfolio_weight",
            "rf_weight",
            "portfolio_value",
        ]

    def get_state(
        self,
        state: Optional[Dict[str, Union[torch.Tensor, int]]] = None,
    ) -> Optional[Dict[str, Union[torch.Tensor, int]]]:
        """get the state tensors at the current time, including Xt_Matrix and Portfolio_Weight

        Args:
            state (Optional[Dict[str, Union[torch.Tensor, int]]], optional): the state tensors. Defaults to None.

        Returns:
            Dict[str, torch.Tensor]: the state tensors
        """
        if self.time_index < self.window_size:
            return None

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
            "Xt_Matrix": self._get_Xt_state(time_index=time_index),
            "Portfolio_Weight": self._concat_weight(portfolio_weight, rf_weight),
            "time_index": time_index,
            "portfolio_weight": portfolio_weight,
            "rf_weight": rf_weight,
            "portfolio_value": portfolio_value,
        }

    def _get_Xt_state(self, time_index: Optional[int] = None) -> torch.Tensor:
        """get the Xt state tensor at a given time

        Args:
            time_index (Optional[int], optional): the time_index.
                Defaults to None, which means to get the Xt state tensor at the current time.

        Returns:
            torch.Tensor: the Xt state tensor
        """
        if time_index is None:
            time_index = self.time_index
        return self.Xt_matrix[:, :, time_index - self.window_size : time_index]

    def get_pretrain_input_and_target(
        self, time_index: int
    ) -> Optional[Tuple[Dict[str, torch.Tensor], torch.Tensor]]:
        """get the Xt state tensor and the pretrain target at a given time

        Args:
            time_index (int): the time index

        Returns:
            Optional[Tuple[Dict[str, torch.Tensor], torch.Tensor]]:
                the input state and the pretrain target
        """
        if time_index < self.window_size:
            return None
        return {
            "Xt_Matrix": self._get_Xt_state(time_index),
            "Portfolio_Weight": torch.empty(0, device=self.device),
        }, self.Xt_matrix[:, :, time_index]

    def act(
        self,
        action_idx: int,
        state: Optional[Dict[str, Union[torch.Tensor, int]]] = None,
    ) -> Tuple[Dict[str, Optional[Union[torch.Tensor, int]]], torch.Tensor, bool]:
        """update the environment with the given action at the given time

        Args:
            action_idx (int): the id of the action to take
            state (Optional[Dict[str, Union[torch.Tensor, int]]], optional): the state tensors. Defaults to None.

        Raises:
            ValueError: action not valid

        Returns:
            Tuple[Dict[str, Optional[Union[torch.Tensor, int]]], torch.Tensor, bool]: the new state, reward, and whether the episode is done
        """

        if action_idx < 0 or action_idx >= len(self.all_actions):
            raise ValueError("action not valid")

        new_state = self.update(action_idx, state, modify_inner_state=False)
        reward = (
            (new_state["portfolio_value"] - new_state["static_portfolio_value"])
            / new_state["static_portfolio_value"]
            * 100
        )
        new_state.pop("static_portfolio_value", None)
        done = self.time_index == self.end_time_index - 1

        return new_state, reward, done

    def update(
        self,
        action: int,
        state: Optional[Dict[str, Union[torch.Tensor, int]]] = None,
        modify_inner_state: Optional[bool] = None,
    ) -> Dict[str, Union[torch.Tensor, int]]:
        """update the environment with the given action index

        Args:
            action (int): the id of the action to take
            state (Optional[Dict[str, Union[torch.Tensor, int]]], optional): the state tensors. Defaults to None.
            modify_inner_state (Optional[bool], optional): whether to modify the inner state. Defaults to None.

        Raises:
            ValueError: the action is not valid

        returns:
            Dict[str, Union[torch.Tensor, int]]: the new state
        """
        if action < 0 or action >= len(self.all_actions):
            raise ValueError("action not valid")
        action = self.all_actions[action]
        if not self._action_validity(action):
            raise ValueError("action not valid")

        if modify_inner_state is None:
            modify_inner_state = state is None

        trading_size = action * self.trading_size
        new_state = BaseEnv.update(self, trading_size, state, modify_inner_state)

        ret_state = self.get_state(new_state)
        ret_state["new_portfolio_weight_prev_day"] = new_state[
            "new_portfolio_weight_prev_day"
        ]
        ret_state["prev_price"] = new_state["prev_price"]
        ret_state["static_portfolio_value"] = new_state["static_portfolio_value"]
        return ret_state

    def reset(self) -> None:
        """reset the environment to the initial state"""
        logger.info("resetting DiscreteRealDataEnv1")
        self.time_index = self.start_time_index
        BaseEnv.initialize_weight(self)

    def _cash_shortage(
        self,
        action: torch.Tensor,
        portfolio_value: Optional[torch.Tensor] = None,
        rf_weight: Optional[torch.Tensor] = None,
    ) -> bool:
        """assert whether there is cash shortage after trading

        Args:
            action (torch.Tensor): the trading decision of each asset
            portfolio_value (Optional[torch.Tensor], optional): the portfolio value. Defaults to None.
            rf_weight (Optional[torch.Tensor], optional): the risk free weight. Defaults to None.

        Returns:
            bool: whether there is cash shortage after trading
        """
        return BaseEnv._cash_shortage(
            self, action * self.trading_size, portfolio_value, rf_weight
        )

    def _asset_shortage(
        self,
        action: torch.Tensor,
        portfolio_weight: Optional[torch.Tensor] = None,
        portfolio_value: Optional[torch.Tensor] = None,
    ) -> bool:
        """assert whether there is asset shortage after trading

        Args:
            action (torch.Tensor): the trading decision of each asset
            portfolio_weight (Optional[torch.Tensor], optional): the portfolio weight. Defaults to None.
            portfolio_value (Optional[torch.Tensor], optional): the portfolio value. Defaults to None.

        Returns:
            bool: whether there is asset shortage after trading
        """
        return BaseEnv._asset_shortage(
            self, action * self.trading_size, portfolio_weight, portfolio_value
        )

    def _action_validity(
        self,
        action: torch.Tensor,
        portfolio_weight: Optional[torch.Tensor] = None,
        portfolio_value: Optional[torch.Tensor] = None,
        rf_weight: Optional[torch.Tensor] = None,
    ) -> bool:
        """assert whether the action is valid

        Args:
            action (torch.Tensor): the trading decision of each asset
            portfolio_weight (Optional[torch.Tensor], optional): the portfolio weight. Defaults to None.
            portfolio_value (Optional[torch.Tensor], optional): the portfolio value. Defaults to None.
            rf_weight (Optional[torch.Tensor], optional): the risk free weight. Defaults to None.

        Returns:
            bool: whether the action is valid
        """
        return not self._cash_shortage(
            action, portfolio_value, rf_weight
        ) and not self._asset_shortage(action, portfolio_weight, portfolio_value)

    def possible_actions(self) -> torch.Tensor:
        """get all possible action indexes

        Returns:
            torch.Tensor: all possible action indexes
        """
        possible_action_indexes = []
        for idx, action in enumerate(self.all_actions):
            if self._action_validity(action):
                possible_action_indexes.append(idx)
        return torch.tensor(
            possible_action_indexes, dtype=torch.int32, device=self.device
        )

    def action_mapping(
        self,
        action_index: int,
        Q_Values: torch.Tensor,
        state: Dict[str, torch.Tensor] = {},
    ) -> int:
        """perform action mapping based on the Q values

        Args:
            action_index (int): the index of the action to map
            Q_Values (torch.Tensor): the Q values of all actions
            state (Dict[str, torch.Tensor]): the state tensors. Defaults to {}.

        Raises:
            ValueError: action not valid

        Returns:
            int: the index of the mapped action
        """
        if len(state) > 0:
            portfolio_weight = state["portfolio_weight"]
            portfolio_value = state["portfolio_value"]
            rf_weight = state["rf_weight"]
        else:
            portfolio_weight = None
            portfolio_value = None
            rf_weight = None
        if action_index < 0 or action_index >= len(self.all_actions):
            raise ValueError("action not valid")
        action = self.all_actions[action_index]
        if self._asset_shortage(action, portfolio_weight, portfolio_value):
            action_index = self._action_mapping_rule2(action)
            action = self.all_actions[action_index]
        if self._cash_shortage(action, portfolio_value, rf_weight):
            return self._action_mapping_rule1(
                action, Q_Values, portfolio_value, rf_weight
            )
        return action_index

    def _action_mapping_rule1(
        self,
        action: torch.Tensor,
        Q_Values: torch.Tensor,
        portfolio_value: Optional[torch.Tensor] = None,
        rf_weight: Optional[torch.Tensor] = None,
    ) -> int:
        """action mapping rule 1: if there is cash shortage,
        find the subset action with the highest Q value

        Args:
            action (torch.Tensor): the trading decision of each asset
            Q_Values (torch.Tensor): the Q values of all actions
            portfolio_value (Optional[torch.Tensor], optional): the portfolio value. Defaults to None.
            rf_weight (Optional[torch.Tensor], optional): the risk free weight. Defaults to None.

        Returns:
            int: the index of the mapped action
        """
        try:
            possible_action_indexes = []
            for idx, new_action in enumerate(self.all_actions):
                if (
                    torch.all(new_action[action == -1] == -1)
                    and torch.all(action[new_action == -1] == -1)
                    and torch.all(new_action[action == 0] == 0)
                    and not self._cash_shortage(new_action, portfolio_value, rf_weight)
                ):
                    possible_action_indexes.append(idx)

            possible_values = Q_Values[possible_action_indexes]
            max_index = torch.argmax(possible_values)
            return possible_action_indexes[max_index]
        except:
            print("action:", action)
            for idx, new_action in enumerate(self.all_actions):
                print("new_action: ", new_action)
                if (
                    torch.all(new_action[action == -1] == -1)
                    and torch.all(action[new_action == -1] == -1)
                    and torch.all(new_action[action == 0] == 0)
                    and not self._cash_shortage(new_action, portfolio_value, rf_weight)
                ):
                    print("valid")
                else:
                    print("invalid")
                print("condition1:", torch.all(new_action[action == -1] == -1))
                print("condition2:", torch.all(action[new_action == -1] == -1))
                print("condition3:", torch.all(new_action[action == 0] == 0))
                print(
                    "condition4:",
                    not self._cash_shortage(new_action, portfolio_value, rf_weight),
                )

            exit(-1)

    def _action_mapping_rule2(self, action: torch.Tensor) -> int:
        """the action mapping rule 2: if there is asset shortage,
        don't trade the asset with shortage

        Args:
            action (torch.Tensor): the trading decision of each asset

        Returns:
            int: the index of the mapped action
        """
        new_action = copy.deepcopy(action)

        condition = (new_action < 0) & (
            self.portfolio_weight * self.portfolio_value
            < torch.abs(new_action) * self.trading_size
        )
        new_action[condition] = 0
        return self.find_action_index(new_action)

    def select_random_action(self) -> int:
        """select a random valid action, return its index

        Returns:
            int: the index of the selected action
        """
        possible_action_indexes = self.possible_actions()
        action_index = int(
            possible_action_indexes[
                random.randint(0, len(possible_action_indexes) - 1)
            ].item()
        )
        return action_index
