import argparse


from utils.logging import get_logger
from networks import register_network


import torch
import torch.nn as nn

logger = get_logger("MultiDQN_LSTM")


@register_network("MultiDQN_LSTM")
class MultiDQN_LSTM(nn.Module):
    """
    The MultiDQN_LSTM model

    references:
        https://arxiv.org/abs/1907.03665

        https://github.com/Jogima-cyber/portfolio-manager
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
            "--LSTM_layers",
            type=int,
            default=1,
            help="number of LSTM layers",
        )
        parser.add_argument(
            "--LSTM_hidden_size",
            type=int,
            default=128,
            help="hidden size of LSTM",
        )
        parser.add_argument(
            "--LSTM_output_size",
            type=int,
            default=20,
            help="output size of LSTM",
        )
        parser.add_argument(
            "--DNN_hidden_size_1",
            type=int,
            default=64,
            help="hidden size of first DNN layer 1",
        )
        parser.add_argument(
            "--DNN_hidden_size_2",
            type=int,
            default=32,
            help="hidden size of first DNN layer 2",
        )
        parser.add_argument(
            "--decoder_hidden_size",
            type=int,
            default=64,
            help="hidden size of decoder",
        )

    def __init__(self, args: argparse.Namespace) -> None:
        """initialize the MultiDQN_LSTM model

        Args:
            args (argparse.Namespace): the arguments
        """
        super().__init__()
        self.encoder = LSTMEncoder(args)
        self.decoder = Decoder(args)
        self.asset_num = len(args.asset_codes)
        self.DNN = DNN(args)

    def forward(
        self, Xt: torch.Tensor, wt: torch.Tensor, pretrain: bool
    ) -> torch.Tensor:
        """the overridden forward method

        Args:
            Xt (torch.Tensor): the state tensor Xt
            wt (torch.Tensor): the weight tensor wt
            pretrain (bool): is this a pretrain step or not

        Returns:
            torch.Tensor: the output tensor
        """
        hn = self.encoder(Xt)
        # hn has size [asset_num, LSTM_output_size]
        if pretrain:
            return self.decoder(hn)
        return self.DNN(hn, wt)


class LSTMEncoder(nn.Module):
    """the LSTM encoder for the MultiDQN_LSTM model"""

    def __init__(self, args: argparse.Namespace) -> None:
        """initialize the LSTM encoder

        Args:
            args (argparse.Namespace): the arguments
        """
        super().__init__()
        self.LSTM = nn.LSTM(
            input_size=5,
            hidden_size=args.LSTM_hidden_size,
            proj_size=args.LSTM_output_size,
            num_layers=args.LSTM_layers,
        )

    def forward(self, Xt: torch.Tensor) -> torch.Tensor:
        """the overridden forward method

        Args:
            Xt (torch.Tensor): the state tensor Xt

        Returns:
            torch.Tensor: the output tensor
        """
        Xt = Xt.permute(2, 1, 0)
        _, (hn, _) = self.LSTM(Xt)
        return hn.squeeze(0)


class Decoder(nn.Module):
    """the decoder for the MultiDQN_LSTM model on pretraining"""

    def __init__(self, args: argparse.Namespace) -> None:
        """initialize the decoder

        Args:
            args (argparse.Namespace): the arguments
        """
        super().__init__()
        self.fc1 = nn.Linear(args.LSTM_output_size, args.decoder_hidden_size)
        self.fc2 = nn.Linear(args.decoder_hidden_size, 5)

    def forward(self, hn: torch.Tensor) -> torch.Tensor:
        """the overridden forward method

        Args:
            hn (torch.Tensor): the hidden state tensor hn from the LSTM encoder

        Returns:
            torch.Tensor: the output tensor
        """
        x = self.fc1(hn)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


class DNN(nn.Module):
    """the DNN for the MultiDQN_LSTM model on training"""

    def __init__(self, args: argparse.Namespace) -> None:
        """initialize the DNN

        Args:
            args (argparse.Namespace): the arguments
        """
        super().__init__()
        self.asset_num = len(args.asset_codes)
        self.fc1 = nn.Linear(
            args.LSTM_output_size * self.asset_num + self.asset_num + 1,
            args.DNN_hidden_size_1,
        )
        self.fc2 = nn.Linear(args.DNN_hidden_size_1, args.DNN_hidden_size_2)
        self.fc3 = nn.Linear(args.DNN_hidden_size_2, 3**self.asset_num)

    def forward(self, hn: torch.Tensor, wt: torch.Tensor) -> torch.Tensor:
        """the overridden forward method

        Args:
            hn (torch.Tensor): the hidden state tensor hn from the LSTM encoder
            wt (torch.Tensor): the weight tensor wt

        Returns:
            torch.Tensor: the output tensor
        """
        x = self.fc1(
            torch.cat(
                (
                    hn.view(-1),
                    wt.view(-1),
                ),
                dim=0,
            )
        )
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        return x
