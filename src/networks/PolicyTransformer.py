import argparse

from typing import Dict, List
from utils.logging import get_logger
from networks import register_network
import math


import torch
import torch.nn as nn
import torch.nn.functional as F

logger = get_logger("PolicyTransformer")


@register_network("PolicyTransformer")
class PolicyTransformer(nn.Module):
    """
    The PolicyTransformer model
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
            "--d_model",
            type=int,
            default=64,
            help="number of features",
        )
        parser.add_argument(
            "--dropout",
            type=float,
            default=0.1,
            help="dropout probability",
        )
        parser.add_argument(
            "--nhead",
            type=int,
            default=8,
            help="number of heads of the multiheadattention models",
        )
        parser.add_argument(
            "--num_layers",
            type=int,
            default=4,
            help="number of layers of the transformer model",
        )
        parser.add_argument(
            "--dim_feedforward",
            type=int,
            default=256,
            help="dimension of the feedforward network model",
        )

    def __init__(self, args: argparse.Namespace) -> None:
        """initialize the PolicyTransformer model

        Args:
            args (argparse.Namespace): the arguments
        """
        super().__init__()
        self.asset_num = args.asset_num
        self.d_model = args.d_model
        self.dropout = args.dropout if args.mode == "train" else 0.0
        self.nhead = args.nhead
        self.num_layers = args.num_layers
        self.window_size = args.window_size

        self.price_layer_norm = nn.LayerNorm([self.window_size, self.asset_num, 3])
        self.price_feature_embedding = nn.Linear(3 * self.asset_num, self.d_model)
        self.positional_encoding = PositionalEncoding(self.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dropout=self.dropout,
            dim_feedforward=args.dim_feedforward,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=self.num_layers
        )

        self.output_layer = nn.Linear(self.d_model + self.asset_num, self.asset_num + 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, states: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
        if isinstance(states, dict):
            states = [states]
            batched = False
        else:
            batched = True
        batch_size = len(states)

        # Extract features from each state in the batch

        price = torch.stack(
            [state["price"] for state in states]
        )  # batch_size * window_size * asset_num
        high = torch.stack(
            [state["price_high"] for state in states]
        )  # batch_size * window_size * asset_num
        low = torch.stack(
            [state["price_low"] for state in states]
        )  # batch_size * window_size * asset_num
        portfolio_weight = torch.stack(
            [state["portfolio_weight"] for state in states]
        )  # batch_size * asset_num
        portfolio_value = torch.stack(
            [state["portfolio_value"] for state in states]
        )  # batch_size * 1

        price_feature = torch.stack(
            [price, high, low], dim=3
        )  # batch_size * window_size * asset_num * 3
        price_feature = price_feature / price_feature[:, -1, :, 0].unsqueeze(
            1
        ).unsqueeze(
            3
        )  # batch_size * window_size * asset_num * 3
        # layernorm over window_size
        price_feature = self.price_layer_norm(price_feature)
        price_feature = price_feature.reshape(batch_size, self.window_size, -1)

        price_embedded = self.price_feature_embedding(
            price_feature
        )  # batch_size * window_size * d_model
        # add position embedding
        price_embedded = self.positional_encoding(price_embedded)
        x = self.transformer_encoder(
            price_embedded
        )  # batch_size * window_size * d_model
        x = x[:, -1, :]  # batch_size * d_model
        x = torch.cat([x, portfolio_weight], dim=1)
        x = self.output_layer(x).squeeze(1)  # batch_size * (asset_num + 1)
        x = self.softmax(x)
        output = x[:, :-1]

        # Calculate trading weight and trading size
        trading_weight = output - portfolio_weight  # batch_size * asset_num
        trading_size = (
            trading_weight * portfolio_value.unsqueeze(1) * 0.95
        )  # batch_size * asset_num
        if batched:
            return trading_size
        else:
            return trading_size[0]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return x
