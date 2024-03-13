from agents import register_agent
from agents.BaseAgent import BaseAgent
from argparse import ArgumentParser


@register_agent("DQN")
class DQN(BaseAgent):
    @staticmethod
    def add_args(parser: ArgumentParser) -> None:
        super(DQN, DQN).add_args(parser)
        parser.add_argument("--batch_size", type=int, default=32)

    def __init__(self, args) -> None:
        super().__init__(args)
        # todo: implement the constructor
        pass
