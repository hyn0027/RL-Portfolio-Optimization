import argparse

from typing import Dict
from utils.logging import get_logger
from networks import register_network


import torch
import torch.nn as nn

logger = get_logger("PolicyWeightCNN")


@register_network("PolicyWeightCNN")
class PolicyWeightCNN(nn.Module):
    """
    The PolicyWeightCNN model

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
        """initialize the PolicyWeightCNN model

        Args:
            args (argparse.Namespace): the arguments
        """
        super().__init__()
        self.layer1 = nn.Conv2d(args.feature_num, 2, (1, 3))
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Conv2d(2, 20, (1, args.window_size - 2))
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Conv2d(21, 1, (1, 1))
        self.softmax = nn.Softmax(dim=0)

        self.rf_bias = nn.Parameter(torch.randn(1))

    def forward(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """forward pass of the PolicyWeightCNN model

        Args:
            x (Dict[str, torch.Tensor]): the input tensors

        Returns:
            torch.Tensor: the output tensor
        """
        x = state["Xt"].transpose(1, 2)
        w = state["Wt_1"]

        x = self.layer1(x)
        x = self.relu1(x)

        x = self.layer2(x)
        x = self.relu2(x)

        w = w.view(1, -1, 1)
        x = torch.cat((x, w), 0)

        x = self.layer3(x)
        x = x.view(-1)
        x = torch.cat((self.rf_bias, x), 0)
        x = self.softmax(x)
        return x[1:]
