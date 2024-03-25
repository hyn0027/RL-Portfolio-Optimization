import argparse
from typing import Optional
from envs.BaseEnv import BaseEnv

import torch


class BaseAgent:
    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--model_load_path",
            type=str,
            required=False,
            help="Path to load model from",
        )

    def __init__(
        self,
        args: argparse.Namespace,
        env: BaseEnv,
        device: Optional[str] = None,
        test_mode: bool = False,
    ) -> None:
        torch.set_num_threads(args.num_threads)
        self.asset_codes = args.asset_codes
        self.env = env
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device is None
            else torch.device(device)
        )
        self.fp16 = args.fp16
        self.dtype = torch.float16 if args.fp16 else torch.float32
        self.args = args
        self.test_mode = test_mode

    def train(self) -> None:
        raise NotImplementedError("train method not implemented")
