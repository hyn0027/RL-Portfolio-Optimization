import argparse

from agents import register_agent
from agents.BaseAgent import BaseAgent


@register_agent("DQN")
class DQN(BaseAgent):
    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        super(DQN, DQN).add_args(parser)
        parser.add_argument("--batch_size", type=int, default=32)

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
