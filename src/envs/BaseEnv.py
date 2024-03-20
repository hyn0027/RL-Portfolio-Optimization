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

    def state_dimension(self) -> int:
        raise NotImplementedError("state_dimension not implemented")

    def action_dimension(self) -> int:
        raise NotImplementedError("action_dimension not implemented")

    def get_state(
        self,
        time: pd.DatetimeTZDtype,
    ) -> np.ndarray:
        pass
