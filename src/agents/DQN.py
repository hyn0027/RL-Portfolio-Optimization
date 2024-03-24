import argparse
from typing import Optional

from agents import register_agent
from agents.BaseAgent import BaseAgent
from envs.BaseEnv import BaseEnv


@register_agent("DQN")
class DQN(BaseAgent):
    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        super(DQN, DQN).add_args(parser)

    def __init__(
        self,
        args: argparse.Namespace,
        env: BaseEnv,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(args, env, device)

    def train(self) -> None:
        pass
