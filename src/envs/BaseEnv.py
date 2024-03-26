import argparse
from typing import Dict, Tuple, List, Optional
from utils.logging import get_logger

import torch

logger = get_logger("BaseEnv")


class BaseEnv:
    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        """add arguments to the parser

        Args:
            parser (argparse.ArgumentParser): the parser to add arguments to
        """
        parser.add_argument(
            "--initial_balance",
            type=float,
            default=1e6,
            help="the initial balance",
        )
        parser.add_argument(
            "--risk_free_return",
            type=float,
            default=0.0,
            help="risk free return"
        )
        parser.add_argument(
            "--transaction_cost_rate",
            type=float,
            default=0.0025,
            help="the transaction cost rate for each trading",
        )
        parser.add_argument(
            "--transaction_cost_base",
            type=float,
            default=0.0,
            help="the transaction cost base (bias) for each trading",
        )

    def __init__(
        self,
        args: argparse.Namespace,
        device: Optional[str] = None,
    ) -> None:
        """initialize the environment

        Args:
            args (argparse.Namespace): arguments
            data (Data): data
        """
        logger.info("Initializing BaseEnv")
        self.args = args
        self.window_size = args.window_size
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device is None
            else torch.device(device)
        )
        self.fp16 = args.fp16
        self.dtype = torch.float16 if args.fp16 else torch.float32
        self.time_index = 0
        
        self.rf_return = torch.tensor(args.risk_free_return, dtype=self.dtype, device=self.device)
        
        self.transaction_cost_rate = torch.tensor(
            args.transaction_cost_rate, dtype=self.dtype, device=self.device
        )
        self.transaction_cost_base = torch.tensor(
            args.transaction_cost_base, dtype=self.dtype, device=self.device
        )
        logger.info("BaseEnv Initialized")

    def initialize_weight(self) -> None:
        self.portfolio_value = torch.tensor(
            self.args.initial_balance, dtype=self.dtype, device=self.device
        )
        self.portfolio_weight = torch.zeros(
            self.get_asset_num(), dtype=self.dtype, device=self.device
        )
        self.rf_weight = torch.tensor(1.0, dtype=self.dtype, device=self.device)
        
    def get_asset_num(self) -> int:
        raise NotImplementedError("asset num not implemented")

    def to(self, device: str) -> None:
        """move the environment to the given device

        Args:
            device (torch.device): the device to move to
        """
        logger.info("Changing device to %s", device)
        self.device = torch.device(device)
        self.portfolio_value = self.portfolio_value.to(self.device)
        self.portfolio_weight = self.portfolio_weight.to(self.device)
        self.rf_weight = self.rf_weight.to(self.device)
        self.transaction_cost_rate = self.transaction_cost_rate.to(self.device)
        self.transaction_cost_base = self.transaction_cost_base.to(self.device)

    def train_time_range(self) -> range:
        """the range of time indices

        Returns:
            range: the range of time indices
        """
        raise NotImplementedError("time_range not implemented")

    def test_time_range(self) -> range:
        """the range of time indices

        Returns:
            range: the range of time indices
        """
        raise NotImplementedError("time_range not implemented")

    def state_dimension(self) -> Dict[str, torch.Size]:
        """the dimension of the state tensors

        Returns:
            Dict[str, torch.Size]: the dimension of the state tensors
        """
        raise NotImplementedError("state_dimension not implemented")

    def state_tensor_names(self) -> List[str]:
        """the names of the state tensors

        Returns:
            List[str]: the names of the state tensors
        """
        raise NotImplementedError("state_tensor_names not implemented")

    def action_dimension(self) -> torch.Size:
        """the dimension of the action the agent can take

        Returns:
            torch.Size: the dimension of the action the agent can take
        """
        raise NotImplementedError("action_dimension not implemented")

    def get_state(
        self,
    ) -> Dict[str, torch.tensor]:
        """get the state tensors at a given time

        Returns:
            Dict[str, torch.tensor]: the state tensors
        """
        raise NotImplementedError("get_state not implemented")

    def act(self, action: torch.tensor) -> Tuple[Dict[str, torch.tensor], float, bool]:
        """update the environment with the given action at the given time

        Args:
            action (torch.tensor): the action to take
            time_index (int): the time index to take the action at
            update (bool): whether to update the environment


        Returns:
            Tuple[Dict[str, torch.tensor], float, bool]: the new state, the reward, and whether the episode is done
        """
        raise NotImplementedError("act not implemented")

    def reset(self) -> None:
        """reset the environment"""
        raise NotImplementedError("reset not implemented")

    def update(self, trading_size: torch.Tensor) -> None:
        """update the environment with the given action

        Args:
            action (torch.tensor): the action to take
        """
        _, self.portfolio_weight, self.rf_weight, self.portfolio_value, _ = (
            BaseEnv._get_new_portfolio_weight_and_value(self, trading_size)
        )
        self.time_index += 1

    def _concat_weight(
        self, portfolio_weight: torch.Tensor, rf_weight: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat((rf_weight.unsqueeze(0), portfolio_weight), dim=0)
    
    def _get_price_change_ratio_tensor(
        self, time_index: Optional[int] = None
    ) -> torch.tensor:
        raise NotImplementedError("_get_price_change_ratio_tensor not implemented")
    
    def _transaction_cost(self, trading_size: torch.Tensor) -> torch.Tensor:
        return (
            torch.sum(torch.abs(trading_size)) * self.transaction_cost_rate + 
            torch.nonzero(trading_size).size(0) * self.transaction_cost_base
        )
        
    def _get_new_portfolio_weight_and_value(
        self, trading_size: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # get portfolio weight after trading
        new_portfolio_weight = (
            self.portfolio_weight + trading_size / self.portfolio_value
        )
        # add transaction cost
        transaction_cost = self._transaction_cost(trading_size)
        new_portfolio_value = self.portfolio_value - transaction_cost
        new_portfolio_weight = (
            new_portfolio_weight * self.portfolio_value / new_portfolio_value
        )
        new_rf_weight = torch.tensor(
            1.0, dtype=self.dtype, device=self.device
        ) - torch.sum(new_portfolio_weight)

        # changing to the next day
        # portfolio_value = value * (price change vec * portfolio_weight + rf_weight * (rf + 1))
        price_change_rate = self._get_price_change_ratio_tensor(self.time_index + 1)
        new_portfolio_value_next_day = new_portfolio_value * (
            torch.sum(price_change_rate * new_portfolio_weight) + new_rf_weight * (self.rf_return + 1.0)
        )
        # adjust weight based on new value
        new_portfolio_weight_next_day = (
            price_change_rate
            * new_portfolio_weight
            * new_portfolio_value
            / new_portfolio_value_next_day
        )
        new_rf_weight_next_day = torch.tensor(
            1.0, dtype=self.dtype, device=self.device
        ) - torch.sum(new_portfolio_weight_next_day)
        
        static_portfolio_value = self.portfolio_value * (
            torch.sum(price_change_rate * self.portfolio_weight) + self.rf_weight * (self.rf_return + 1.0)
        )
        return (
            new_portfolio_weight,
            new_portfolio_weight_next_day,
            new_rf_weight_next_day,
            new_portfolio_value_next_day,
            static_portfolio_value,
        )
    
    def _cash_shortage(self, trading_size: torch.Tensor) -> bool:
        return (
            torch.sum(trading_size) + self._transaction_cost(trading_size)
            > self.portfolio_value * self.rf_weight
        )
    
    def _asset_shortage(self, trading_size: torch.Tensor) -> bool:
        return torch.any(
            self.portfolio_weight[trading_size < 0] * self.portfolio_value
            < -trading_size[trading_size < 0]
        )