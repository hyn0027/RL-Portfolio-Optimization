import argparse

from typing import Dict
from utils.logging import get_logger
from networks import register_network


import torch
import torch.nn as nn

logger = get_logger("PolicyWeightLSTM")


@register_network("PolicyWeightLSTM")
class PolicyWeightLSTM(nn.Module):
    """
    The PolicyWeightLSTM model

    Reference:
        original paper: https://arxiv.org/abs/1706.10059
    """

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
            "--feature_num",
            type=int,
            default=3,
            help="number of features",
        )

    def __init__(self, args: argparse.Namespace) -> None:
        """initialize the PolicyWeightLSTM model

        Args:
            args (argparse.Namespace): the arguments
        """
        super().__init__()
        self.LSTM = nn.LSTM(args.feature_num, 20, 1)
        self.conv = nn.Conv2d(21, 1, (1, 1))
        self.softmax = nn.Softmax(dim=0)
        self.rf_bias = nn.Parameter(torch.tensor([1], dtype=torch.float32))

    def forward(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """forward pass of the PolicyWeightLSTM model

        Args:
            x (Dict[str, torch.Tensor]): the input tensors

        Returns:
            torch.Tensor: the output tensor
        """
        x = state["Xt"]  # 3 * window_size * asset_num
        w = state["Wt_1"]  # asset_num

        x = x.permute(1, 2, 0)  # window_size * asset_num * 3
        _, (hn, _) = self.LSTM(x)  # 1 * asset_num * 20
        w = w.view(1, -1, 1)  # 1 * asset_num * 1
        x = torch.cat((hn, w), dim=2)  # 1 * asset_num * 21
        x = x.permute(2, 1, 0)  # 21 * asset_num * 1
        x = self.conv(x)  # 1 * asset_num * 1
        x = x.view(-1)  # asset_num
        x = torch.cat((self.rf_bias, x), 0)
        x = self.softmax(x)
        return x[1:]
