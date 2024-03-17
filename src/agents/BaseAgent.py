import argparse
from data import Data


class BaseAgent:
    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        pass

    def __init__(self, args: argparse.Namespace, data: Data) -> None:
        self.asset_codes = args.asset_codes
        self.data = data

    def train(self) -> None:
        raise NotImplementedError("train method not implemented")
