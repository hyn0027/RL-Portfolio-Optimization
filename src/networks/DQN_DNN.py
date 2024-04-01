import argparse

from typing import Dict
from utils.logging import get_logger
from networks import register_network


import torch
import torch.nn as nn

logger = get_logger("DQN_DNN")


@register_network("DQN_DNN")
class DQN_DNN(nn.Module):
    """The DQN_DNN model"""

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
            "--hidden_size1",
            type=int,
            default=128,
            help="hidden size of the first layer",
        )
        parser.add_argument(
            "--hidden_size2",
            type=int,
            default=128,
            help="hidden size of the second layer",
        )

    def __init__(self, args: argparse.Namespace) -> None:
        """initialize the DQN_DNN model

        Args:
            args (argparse.Namespace): the arguments
        """
        super().__init__()
        self.fc1 = nn.Linear(args.asset_num * args.window_size, args.hidden_size1)
        self.fc_action = nn.Linear(args.asset_num, args.hidden_size1)
        self.fc2 = nn.Linear(args.hidden_size1 * 2, args.hidden_size2)
        self.fc3 = nn.Linear(args.hidden_size2, 1)

    def forward(self, state: Dict[str, torch.Tensor], action: torch.Tensor):
        """forward pass

        Args:
            state (Dict[str, torch.Tensor]): the state
            action (torch.Tensor): the action

        Returns:
            torch.Tensor: the output
        """
        x = torch.relu(self.fc1(state["price"].reshape(-1)))
        y = torch.relu(self.fc_action(action))
        z = torch.cat((x, y), 0)
        z = torch.relu(self.fc2(z))
        return self.fc3(z)
