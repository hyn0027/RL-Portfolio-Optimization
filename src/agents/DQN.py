from agents import register_agent
from agents.BaseAgent import BaseAgent
from argparse import ArgumentParser


@register_agent("DQN")
class DQN(BaseAgent):
    @staticmethod
    def add_args(parser: ArgumentParser) -> None:
        parser.add_argument("--batch_size", type=int, default=32)
