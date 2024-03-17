import argparse
from typing import Dict, Any, Tuple
import pandas as pd


class BaseAgent:
    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        pass

    def __init__(self, args: argparse.Namespace) -> None:
        self.asset_codes = args.asset_codes

    def set_data(self, data: Dict[str, Dict[str, Any]]) -> None:
        self.data = data

    def get_asset_data(self, asset_code: str) -> Dict[str, Any]:
        return self.data[asset_code]

    def get_asset_info(self, asset_code: str) -> Dict[str, Any]:
        return self.data[asset_code]["info"]

    def get_asset_hist(self, asset_code: str) -> pd.DataFrame:
        return self.data[asset_code]["hist"]

    def get_asset_option_dates(self, asset_code: str) -> Tuple:
        return self.data[asset_code]["option_dates"]

    def get_asset_calls(self, asset_code: str, date: str) -> pd.DataFrame:
        return self.data[asset_code]["calls"][date]

    def get_asset_puts(self, asset_code: str, date: str) -> pd.DataFrame:
        return self.data[asset_code]["puts"][date]

    def train(self) -> None:
        raise NotImplementedError("train method not implemented")
