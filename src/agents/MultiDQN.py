import argparse

from agents import register_agent
from agents.BaseAgent import BaseAgent
from envs.BaseEnv import BaseEnv


@register_agent("MultiDQN")
class MultiDQN(BaseAgent):
    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        super(MultiDQN, MultiDQN).add_args(parser)

    def __init__(self, args: argparse.Namespace, env: BaseEnv) -> None:
        super().__init__(args, env)

    def train(self) -> None:
        pass
