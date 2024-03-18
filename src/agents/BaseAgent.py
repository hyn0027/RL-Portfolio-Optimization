import argparse


class BaseAgent:
    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        pass

    def __init__(self, args: argparse.Namespace) -> None:
        self.asset_codes = args.asset_codes

    def train(self) -> None:
        raise NotImplementedError("train method not implemented")
