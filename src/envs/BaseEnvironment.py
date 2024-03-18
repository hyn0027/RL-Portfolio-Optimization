import argparse

from envs import register_env
from data import Data


@register_env("BaseEnv")
class BaseEnv:
    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        pass

    def __init__(self, args, data: Data) -> None:
        self.data = data

    def reset(self) -> None:
        raise NotImplementedError("reset method not implemented")
