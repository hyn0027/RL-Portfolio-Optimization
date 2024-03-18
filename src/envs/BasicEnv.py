import argparse

from envs import register_env
from data import Data
from envs.BaseEnv import BaseEnv


@register_env("BasicEnv")
class BasicEnv(BaseEnv):
    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        super(BasicEnv, BasicEnv).add_args(parser)
        pass

    def __init__(self, args, data: Data) -> None:
        super().__init__(args, data)
