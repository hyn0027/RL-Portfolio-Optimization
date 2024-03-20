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
        super().__init__(args)
        self.data = data
        self.asset_codes = data.asset_codes
        self.time_zone = args.time_zone

    def time_dimension(self) -> int:
        return self.data.time_dimension()

    def state_dimension(self) -> Dict[str, torch.Size]:
        return {
            "price": torch.Size([len(self.asset_codes), self.window_size]),
        }

    def state_tensor_names(self):
        return ["price"]

    def action_dimension(self) -> int:
        return len(self.asset_codes)

    def get_state(self, time_index: int) -> Optional[Dict[str, torch.tensor]]:
        if time_index < self.window_size - 1:
            return None
        timestamps = self.data.get_time_list()[
            time_index - self.window_size + 1 : time_index + 1
        ]
        for asset_code in self.asset_codes:
            for timestamp in timestamps:
                asset_hist_info = self.data.get_asset_hist_at_time(
                    asset_code, timestamp
                )
                print(asset_hist_info)
        print(timestamps)
