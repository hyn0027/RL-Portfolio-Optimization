import argparse

from utils.logging import get_logger
from data import Data
import pandas as pd
import numpy as np

logger = get_logger("BaseEnv")


class BaseEnv:
    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        pass

    def __init__(self, args, data: Data) -> None:
        self.data = data
        self.asset_codes = args.asset_codes
        self.data_index_type = self.check_data_index()

    def check_data_index(self) -> pd._typing.DtypeObj:
        data_index_type = None
        for asset_code in self.asset_codes:
            if asset_code not in self.data.index():
                raise ValueError(f"Asset code {asset_code} not in data index")
            if not data_index_type:
                data_index_type = self.data.get_asset_hist_index_type(asset_code)
            elif data_index_type != self.data.get_asset_hist_index_type(asset_code):
                raise ValueError("Data index types do not match")
        logger.info(f"Data index type: {data_index_type}")
        return data_index_type

    def state_dimension(self) -> int:
        raise NotImplementedError("state_dimension not implemented")

    def action_dimension(self) -> int:
        raise NotImplementedError("action_dimension not implemented")

    def get_state(self, time) -> np.ndarray:
        pass
