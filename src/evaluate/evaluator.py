import argparse
from typing import Tuple, Optional
from math import sqrt

from utils.logging import get_logger

import torch

logger = get_logger("Evaluator")


class Evaluator:
    """the evaluator to evaluate the portfolio performance"""

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
            "--disable_CR",
            action="store_true",
            help="Disable the CR metric",
        )
        parser.add_argument(
            "--disable_SR",
            action="store_true",
            help="Disable the SR metric",
        )
        parser.add_argument(
            "--disable_SteR",
            action="store_true",
            help="Disable the SterR metric",
        )
        parser.add_argument(
            "--disable_AT",
            action="store_true",
            help="Disable the AT metric",
        )
        parser.add_argument(
            "--disable_MDD",
            action="store_true",
            help="Disable the MDD metric",
        )
        parser.add_argument(
            "--annual_sample",
            type=int,
            default=252,
            help="The number of samples in a year",
        )

    def __init__(self, args: argparse.Namespace) -> None:
        """initialize the evaluator

        Args:
            args (argparse.Namespace): the arguments
        """
        logger.info("Initializing Evaluator")
        self.disable_CR: bool = args.disable_CR
        self.disable_SR: bool = args.disable_SR
        self.disable_SteR: bool = args.disable_SteR
        self.disable_AT: bool = args.disable_AT
        self.disable_MDD: bool = args.disable_MDD
        self.annual_sample: int = args.annual_sample
        self.risk_free_return: float = args.risk_free_return
        self.portfolio_value_list = []
        self.portfolio_weight_list = []
        self.return_rate = []
        self.previous_portfolio_weight = args.initial_balance
        self.asset_prices = []
        logger.info("Evaluator initialized")

    def reset(self, initial_balance: float) -> None:
        """reset the evaluator

        Args:
            initial_balance (float): the initial balance
        """
        logger.info("Resetting Evaluator")
        self.portfolio_value_list = []
        self.portfolio_weight_list = []
        self.return_rate = []
        self.previous_portfolio_weight = initial_balance
        self.asset_prices = []

    def push(
        self,
        portfolio_value: float,
        portfolio_weight: Tuple[torch.Tensor, torch.Tensor],
        current_price: torch.Tensor,
    ) -> None:
        """push the portfolio value and weight to the evaluator

        Args:
            portfolio_value (float): the portfolio value
            portfolio_weight (Tuple[torch.Tensor, torch.Tensor]): the portfolio weight before and after trading
            current_price (torch.Tensor): the current price of the considered assets
        """
        self.portfolio_value_list.append(portfolio_value)
        self.portfolio_weight_list.append(portfolio_weight)
        self.return_rate.append(
            (portfolio_value - self.previous_portfolio_weight)
            / self.previous_portfolio_weight
        )
        self.previous_portfolio_weight = portfolio_value
        self.asset_prices.append(current_price)

    def output_record_to_json(self, path: str) -> None:
        """output the record to a json file

        Args:
            path (str): the path to the output file
        """
        import json

        record = {
            "portfolio_value_list": self.portfolio_value_list,
            "portfolio_weight_list": self.portfolio_weight_list,
            "return_rate": self.return_rate,
            "asset_prices": self.asset_prices,
        }
        with open(path, "w") as f:
            json.dump(record, f)

    def evaluate(
        self,
    ) -> None:
        """evaluate the portfolio

        Raises:
            ValueError: portfolio_value_list is empty
            ValueError: portfolio_weight_list is empty
        """
        if len(self.portfolio_value_list) == 0:
            raise ValueError("portfolio_value_list is empty")
        if not self.disable_AT and len(self.portfolio_weight_list) == 0:
            raise ValueError("portfolio_weight_list is empty")

        logger.info("Evaluating portfolio")
        if not self.disable_CR:
            logger.info(f"CR: {self.calculate_CR()}")
        if not self.disable_SR:
            logger.info(f"SR: {self.calculate_SR()}")
        if not self.disable_SteR:
            logger.info(f"SteR: {self.calculate_SteR()}")
        if not self.disable_AT:
            logger.info(f"AT: {self.calculate_AT()}")
        if not self.disable_MDD:
            logger.info(f"MDD: {self.calculate_MDD()}")

    def calculate_CR(self) -> float:
        """evaluate the CR metric

        Returns:
            float: the CR metric
        """
        if self.disable_CR:
            return None
        if len(self.portfolio_value_list) == 0:
            raise ValueError("portfolio_value_list is empty")
        return (
            self.portfolio_value_list[-1] - self.portfolio_value_list[0]
        ) / self.portfolio_value_list[0]

    def calculate_SR(self) -> float:
        """evaluate the SR metric

        Returns:
            float: the SR metric
        """
        if self.disable_SR:
            return None
        if len(self.portfolio_value_list) == 0:
            raise ValueError("portfolio_value_list is empty")
        average_return = sum(self.return_rate) / len(self.return_rate)
        std_return = (
            sum([(x - average_return) ** 2 for x in self.return_rate])
            / len(self.return_rate)
        ) ** 0.5
        if std_return == 0:
            return 0
        return (
            (average_return - self.risk_free_return)
            / std_return
            * sqrt(self.annual_sample)
        )

    def calculate_SteR(self) -> float:
        """evaluate the SteR metric

        Returns:
            float: the SteR metric
        """
        if self.disable_SteR:
            return None
        if len(self.portfolio_value_list) == 0:
            raise ValueError("portfolio_value_list is empty")
        average_return = sum(self.return_rate) / len(self.return_rate)
        # get the sum of min(return_rate, 0)**2
        neg_return = sum([min(x, 0) ** 2 for x in self.return_rate])
        if neg_return == 0:
            return 0
        return (
            (average_return - self.risk_free_return)
            / sqrt(neg_return / len(self.return_rate))
            * sqrt(self.annual_sample)
        )

    def calculate_AT(self) -> float:
        """evaluate the AT metric

        Returns:
            float: the AT metric
        """
        if self.disable_AT:
            return None
        if len(self.portfolio_weight_list) == 0:
            raise ValueError("portfolio_weight_list is empty")
        return sum(
            [
                torch.sum(torch.abs(w1 - w2)).item()
                for w1, w2 in self.portfolio_weight_list
            ]
        ) / (2 * len(self.portfolio_weight_list))

    def calculate_MDD(self) -> float:
        """evaluate the MDD metric

        Returns:
            float: the MDD metric
        """
        if self.disable_MDD:
            return None
        if len(self.portfolio_value_list) == 0:
            raise ValueError("portfolio_value_list is empty")
        mdd = 0
        peak = self.portfolio_value_list[0]
        for value in self.portfolio_value_list:
            if value > peak:
                peak = value
            mdd = min(mdd, (value - peak) / peak)
        return -mdd
