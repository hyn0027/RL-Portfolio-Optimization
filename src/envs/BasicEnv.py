import argparse
from typing import Dict, Optional

from envs import register_env
from data import Data
from envs.BaseEnv import BaseEnv
import pandas as pd
import torch


@register_env("BasicEnv")
class BasicEnv(BaseEnv):
    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        super(BasicEnv, BasicEnv).add_args(parser)

    def __init__(self, args: argparse.Namespace, data: Data) -> None:
        super().__init__(args, data)

    def state_dimension(self) -> Dict[str, torch.Size]:
        return {
            "price": torch.Size([len(self.asset_codes), self.window_size]),
        }

    def state_tensor_names(self):
        return ["price"]

    def action_dimension(self) -> int:
        return len(self.asset_codes)

    def get_state(self, time: pd.Timestamp) -> Optional[Dict[str, torch.tensor]]:
        time_index = self.data.get_time_index(time)
        if time_index < self.window_size - 1:
            return None
        timestamps = self.data.timestamps[
            time_index - self.window_size + 1 : time_index + 1
        ]
