import argparse

from typing import Dict
from utils.logging import get_logger
from networks import register_network


import torch
import torch.nn as nn

logger = get_logger("PolicyCNN")


@register_network("PolicyCNN")
class PolicyCNN(nn.Module):
    """
    The PolicyCNN model
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
            default=1,
            help="number of features",
        )

    def __init__(self, args: argparse.Namespace) -> None:
        """initialize the PolicyCNN model

        Args:
            args (argparse.Namespace): the arguments
        """
        super().__init__()
        self.layer1 = nn.Conv2d(args.feature_num, 4, (1, 3))
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Conv2d(4, 32, (1, args.window_size - 2))
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Conv2d(32, 1, (1, 1))

    def forward(self, state: Dict[str, torch.Tensor]) -> torch.Tensor:
        """forward pass of the PolicyCNN model

        Args:
            x (Dict[str, torch.Tensor]): the input tensors

        Returns:
            torch.Tensor: the output tensor
        """
        x = state["price"].transpose(1, 2)

        x = self.layer1(x)
        x = self.relu1(x)

        x = self.layer2(x)
        x = self.relu2(x)

        x = self.layer3(x)
        x = x.view(-1) * 100
        return x
