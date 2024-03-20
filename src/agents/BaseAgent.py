import argparse
from envs.BaseEnv import BaseEnv


class BaseAgent:
    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        pass

    def __init__(self, args: argparse.Namespace, env: BaseEnv) -> None:
        self.asset_codes = args.asset_codes
        self.env = env

    def train(self) -> None:
        raise NotImplementedError("train method not implemented")
