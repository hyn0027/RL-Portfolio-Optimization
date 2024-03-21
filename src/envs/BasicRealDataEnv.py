import argparse
from typing import List
import torch

from data import Data
from envs.BaseEnv import BaseEnv


class BasicRealDataEnv(BaseEnv):
    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        super(BasicRealDataEnv, BasicRealDataEnv).add_args(parser)

    def __init__(self, args: argparse.Namespace, data: Data) -> None:
        super().__init__(args)
        self.data = data
        self.asset_codes = data.asset_codes
        self.time_zone = args.time_zone

    def possible_actions(self) -> List[torch.tensor]:
        raise NotImplementedError("possible_action not implemented")
