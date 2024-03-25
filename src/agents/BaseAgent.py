import argparse
from typing import Optional
from envs.BaseEnv import BaseEnv

import torch


class BaseAgent:
    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        pass

    def __init__(
        self,
        args: argparse.Namespace,
        env: BaseEnv,
        device: Optional[str] = None,
    ) -> None:
        self.asset_codes = args.asset_codes
        self.env = env
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device is None
            else torch.device(device)
        )
        self.fp16 = args.fp16
        self.args = args

    def train(self) -> None:
        raise NotImplementedError("train method not implemented")
