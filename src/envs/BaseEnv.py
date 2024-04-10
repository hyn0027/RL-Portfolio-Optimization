import argparse
from typing import Dict, Tuple, List, Optional
from utils.logging import get_logger

import torch

logger = get_logger("BaseEnv")


class BaseEnv:
    """the base environment class, should be overridden by specific environments"""

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        """add arguments to the parser

            to add arguments to the parser, modify the method as follows:

            .. code-block:: python

                @staticmethod
                def add_args(parser: argparse.ArgumentParser) -> None:
                    parser.add_argument(
                        ...
                    )


            then add arguments to the parser

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
            help="risk free return (per time index)",
        )
        parser.add_argument(
            "--transaction_cost_rate_buy",
            type=float,
            default=0.0025,
            help="the transaction cost rate for each buy action",
        )
        parser.add_argument(
            "--transaction_cost_rate_sell",
            type=float,
            default=0.0025,
            help="the transaction cost rate for each sell action",
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
            device (Optional[str], optional): device to run the environment. Defaults to None, which means to use the GPU if available.
        """
        logger.info("Initializing BaseEnv")
        self.args = args
        self.window_size: int = args.window_size
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device is None
            else torch.device(device)
        )
        self.fp16: bool = args.fp16
        self.dtype = torch.float16 if args.fp16 else torch.float32
        self.time_index = 0

        self.rf_return = torch.tensor(
            args.risk_free_return, dtype=self.dtype, device=self.device
        )

        self.transaction_cost_rate_buy = torch.tensor(
            args.transaction_cost_rate_buy, dtype=self.dtype, device=self.device
        )

        self.transaction_cost_rate_sell = torch.tensor(
            args.transaction_cost_rate_sell, dtype=self.dtype, device=self.device
        )
        self.transaction_cost_base = torch.tensor(
            args.transaction_cost_base, dtype=self.dtype, device=self.device
        )

        self.initialize_weight()

        logger.info("BaseEnv Initialized")

    def initialize_weight(self) -> None:
        """initialize the portfolio weight, risk free asset weight, and value"""
        self.portfolio_value = torch.tensor(
            self.args.initial_balance, dtype=self.dtype, device=self.device
        )
        self.portfolio_weight = torch.zeros(
            self.get_asset_num(), dtype=self.dtype, device=self.device
        )
        self.rf_weight = torch.tensor(1.0, dtype=self.dtype, device=self.device)

    def get_asset_num(self) -> int:
        """get the number of assets, should be overridden by specific environments

        Raises:
            NotImplementedError: asset num not implemented

        Returns:
            int: the number of assets
        """
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
        self.transaction_cost_rate_buy = self.transaction_cost_rate_buy.to(self.device)
        self.transaction_cost_rate_sell = self.transaction_cost_rate_sell.to(
            self.device
        )
        self.transaction_cost_base = self.transaction_cost_base.to(self.device)

    def train_time_range(self) -> range:
        """the range of time indices, should be overridden by specific environments

        Raises:
            NotImplementedError: time_range not implemented

        Returns:
            range: the range of time indices
        """
        raise NotImplementedError("time_range not implemented")

    def test_time_range(self) -> range:
        """the range of time indices, should be overridden by specific environments

        Raises:
            NotImplementedError: time_range not implemented

        Returns:
            range: the range of time indices
        """
        raise NotImplementedError("time_range not implemented")

    def state_dimension(self) -> Dict[str, torch.Size]:
        """the dimension of the state tensors, should be overridden by specific environments

        Raises:
            NotImplementedError: state_dimension not implemented

        Returns:
            Dict[str, torch.Size]: the dimension of the state tensors
        """
        raise NotImplementedError("state_dimension not implemented")

    def state_tensor_names(self) -> List[str]:
        """the names of the state tensors, should be overridden by specific environments

        Raises:
            NotImplementedError: state_tensor_names not implemented

        Returns:
            List[str]: the names of the state tensors
        """
        raise NotImplementedError("state_tensor_names not implemented")

    def action_dimension(self) -> torch.Size:
        """the dimension of the action the agent can take

        Returns:
            torch.Size: the dimension of the action the agent can take
        """
        return torch.Size(self.get_asset_num())

    def get_state(
        self,
    ) -> Dict[str, torch.tensor]:
        """get the state tensors at the current time, should be overridden by specific environments

        Raises:
            NotImplementedError: get_state not implemented

        Returns:
            Dict[str, torch.tensor]: the state tensors
        """
        raise NotImplementedError("get_state not implemented")

    def act(
        self, action: torch.tensor
    ) -> Tuple[Dict[str, torch.tensor], torch.Tensor, bool]:
        """update the environment with the given action at the given time, should be overridden by specific environments

        Args:
            action (torch.tensor): the action to take

        Raises:
            NotImplementedError: act not implemented

        Returns:
            Tuple[Dict[str, torch.tensor], torch.Tensor, bool]: the new state, the reward, and whether the episode is done
        """
        raise NotImplementedError("act not implemented")

    def reset(self) -> None:
        """reset the environment, should be overridden by specific environments

        Raises:
            NotImplementedError: reset not implemented
        """
        raise NotImplementedError("reset not implemented")

    def update(self, trading_size: torch.Tensor) -> None:
        """update the environment with the given trading size of each tensor

        Args:
            trading_size (torch.Tensor): the trading size of each asset
        """
        _, self.portfolio_weight, self.rf_weight, self.portfolio_value, _ = (
            BaseEnv._get_new_portfolio_weight_and_value(self, trading_size)
        )
        self.time_index += 1

    def _concat_weight(
        self, portfolio_weight: torch.Tensor, rf_weight: torch.Tensor
    ) -> torch.Tensor:
        """concat the portfolio weight and risk free weight, the risk free weight is at the first position

        Args:
            portfolio_weight (torch.Tensor): the portfolio weight
            rf_weight (torch.Tensor): the risk free weight

        Returns:
            torch.Tensor: the concatenated weight
        """
        return torch.cat((rf_weight.unsqueeze(0), portfolio_weight), dim=0)

    def _get_price_change_ratio_tensor(
        self, time_index: Optional[int] = None
    ) -> torch.tensor:
        """get the price change ratio tensor at a given time, should be overridden by specific environments

        Args:
            time_index (Optional[int], optional):
                the time index to get the price change ratio.
                Defaults to None, which means to get the price change ratio at the current time.

        Raises:
            NotImplementedError: _get_price_change_ratio_tensor not implemented

        Returns:
            torch.tensor: the price change ratio tensor
        """
        raise NotImplementedError("_get_price_change_ratio_tensor not implemented")

    def _transaction_cost(self, trading_size: torch.Tensor) -> torch.Tensor:
        """compute the transaction cost of the trading

        .. code-block:: python

                transaction_cost = sum(abs(trading_size) for trading_size > 0) * transaction_cost_rate_buy
                                 + sum(abs(trading_size) for trading_size < 0) * transaction_cost_rate_sell
                                 + count(trading_size != 0) * transaction_cost_base


        Args:
            trading_size (torch.Tensor): the trading size of each asset

        Returns:
            torch.Tensor: the transaction cost
        """
        buy_trading_size = torch.clamp(trading_size, min=0)
        sell_trading_size = torch.clamp(trading_size, max=0)

        buy_cost = (
            torch.sum(torch.abs(buy_trading_size)) * self.transaction_cost_rate_buy
            + torch.nonzero(buy_trading_size).size(0) * self.transaction_cost_base
        )
        rate_sell = self.transaction_cost_rate_sell / (
            1 + self.transaction_cost_rate_sell
        )
        sell_cost = (
            torch.sum(torch.abs(sell_trading_size)) * rate_sell
            + torch.nonzero(sell_trading_size).size(0) * self.transaction_cost_base
        )

        return buy_cost + sell_cost

    def _find_trading_size_according_to_weight_after_trade(
        self,
        portfolio_weight_before_trade: torch.Tensor,
        rf_weight_before_trade: torch.Tensor,
        portfolio_weight_after_trade: torch.Tensor,
        rf_weight_after_trade: torch.Tensor,
        portfolio_value_before_trade: torch.Tensor,
    ) -> torch.Tensor:
        """find the trading size according to the weight before and after trading (don't change day)

        Reference: https://arxiv.org/abs/1706.10059 (Section 2.3)

        Args:
            portfolio_weight_before_trade (torch.Tensor): the weight before trading
            rf_weight_before_trade (torch.Tensor): the risk free weight before trading
            portfolio_weight_after_trade (torch.Tensor): the weight after trading
            rf_weight_after_trade (torch.Tensor): the risk free weight after trading
            portfolio_value_before_trade (torch.Tensor): the portfolio value before trading

        Returns:
            torch.Tensor: the trading size
        """
        cp = self.transaction_cost_rate_buy
        cs = self.transaction_cost_rate_sell

        def f(mu: torch.Tensor) -> torch.Tensor:
            factor = 1 / (1 - cp * rf_weight_after_trade)
            total = (
                1
                - cp * rf_weight_before_trade
                - (cs + cp - cs * cp)
                * torch.sum(
                    torch.relu(
                        portfolio_weight_before_trade
                        - mu * portfolio_weight_after_trade
                    )
                )
            )
            return factor * total

        mu = (
            (cp + cs)
            / 2
            * torch.sum(
                torch.abs(portfolio_weight_after_trade - portfolio_weight_before_trade)
            )
        )
        prev_mu = torch.tensor(float("inf"), dtype=self.dtype, device=self.device)

        threshold = 1e-6
        max_iter = 100

        while torch.abs(mu - prev_mu) > threshold and max_iter > 0:
            prev_mu = mu
            mu = f(mu)
            max_iter -= 1

        trading_size = portfolio_value_before_trade * (
            portfolio_weight_after_trade * mu - portfolio_weight_before_trade
        )

        return trading_size

    def _get_new_portfolio_weight_and_value(
        self, trading_size: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """get the new portfolio weight and value after trading and transitioning to the next day

        Args:
            trading_size (torch.Tensor): the trading size of each asset

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                the new portfolio weight, the new portfolio weight at the next day,
                the new risk free weight at the next day, the new portfolio value at the next day,
                and the portfolio value at the next day with static weight
        """
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
            torch.sum(price_change_rate * new_portfolio_weight)
            + new_rf_weight * (self.rf_return + 1.0)
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
            torch.sum(price_change_rate * self.portfolio_weight)
            + self.rf_weight * (self.rf_return + 1.0)
        )
        return (
            new_portfolio_weight,
            new_portfolio_weight_next_day,
            new_rf_weight_next_day,
            new_portfolio_value_next_day,
            static_portfolio_value,
        )

    def _cash_shortage(
        self,
        trading_size: torch.Tensor,
        portfolio_value: Optional[torch.Tensor] = None,
        rf_weight: Optional[torch.Tensor] = None,
    ) -> bool:
        """assert whether there is cash shortage after trading

        Args:
            trading_size (torch.Tensor): the trading size of each asset
            portfolio_value (Optional[torch.Tensor], optional): the portfolio value. Defaults to None.
            rf_weight (Optional[torch.Tensor], optional): the risk free weight. Defaults to None.

        Returns:
            bool: whether there is cash shortage after trading
        """
        if portfolio_value is None:
            portfolio_value = self.portfolio_value
        if rf_weight is None:
            rf_weight = self.rf_weight
        return (
            torch.sum(trading_size) + self._transaction_cost(trading_size)
            > portfolio_value * rf_weight
        )

    def _asset_shortage(
        self,
        trading_size: torch.Tensor,
        portfolio_weight: Optional[torch.Tensor] = None,
        portfolio_value: Optional[torch.Tensor] = None,
    ) -> bool:
        """assert whether there is asset shortage after trading

        Args:
            trading_size (torch.Tensor): the trading size of each asset
            portfolio_weight (torch.Tensor): the portfolio weight. default to None (Use the current portfolio weight)
            portfolio_value (torch.Tensor): the portfolio value. default to None (Use the current portfolio value)

        Returns:
            bool: whether there is asset shortage after trading
        """
        if portfolio_weight is None:
            portfolio_weight = self.portfolio_weight
        if portfolio_value is None:
            portfolio_value = self.portfolio_value
        return torch.any(
            portfolio_weight[trading_size < 0] * portfolio_value
            < -trading_size[trading_size < 0]
        )

    def select_random_action(self) -> torch.Tensor:
        """select a random action, should be overridden by specific environments

        Raises:
            NotImplementedError: select_random_action not implemented

        Returns:
            torch.Tensor: the random action
        """
        raise NotImplementedError("select_random_action not implemented")
