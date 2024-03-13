from argparse import ArgumentParser


class BaseAgent:
    @staticmethod
    def add_args(parser: ArgumentParser) -> None:
        pass

    def __init__(self, args) -> None:
        self.asset_codes = args.asset_codes

    def set_data(self, data) -> None:
        pass

    def train(self) -> None:
        raise NotImplementedError("train method not implemented")
