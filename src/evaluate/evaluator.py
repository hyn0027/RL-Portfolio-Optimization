import argparse
from typing import Optional, Tuple

import torch


class Evaluator:
    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        """add arguments to the parser

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

    def __init__(self, args: argparse.Namespace) -> None:
        self.disable_CR = args.disable_CR
        self.disable_SR = args.disable_SR
        self.disable_SteR = args.disable_SteR
        self.disable_AT = args.disable_AT
        self.portfolio_value_list = []
        self.portfolio_weight_list = []

    def reset(self) -> None:
        """reset the evaluator"""
        self.portfolio_value_list = []
        self.portfolio_weight_list = []

    def push(
        self,
        portfolio_value: float,
        portfolio_weight: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        """push the portfolio value and weight to the evaluator

        Args:
            portfolio_value (float): the portfolio value
            portfolio_weight (float): the portfolio weight
        """
        if self.disable_AT and portfolio_weight is None:
            raise ValueError("portfolio_weight is required when AT is enabled")
        self.portfolio_value_list.append(portfolio_value)
        self.portfolio_weight_list.append(portfolio_weight)

    def evaluate(self) -> Tuple[float, float, float, float]:
        """evaluate the portfolio

        Returns:
            Tuple[float, float, float, float]: the metrics
        """
        if len(self.portfolio_value_list) == 0:
            raise ValueError("portfolio_value_list is empty")
        if not self.disable_AT and len(self.portfolio_weight_list) == 0:
            raise ValueError("portfolio_weight_list is empty")

    def evaluate_CR(self) -> float:
        """evaluate the CR metric

        Returns:
            float: the CR metric
        """
        if self.disable_CR:
            return None
        # TODO
        pass
