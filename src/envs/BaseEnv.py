import argparse
from typing import Dict, Tuple, List
from utils.logging import get_logger

from data import Data
import pandas as pd
import torch

logger = get_logger("BaseEnv")


class BaseEnv:
    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        """add arguments to the parser

        Args:
            parser (argparse.ArgumentParser): the parser to add arguments to
        """

    def __init__(self, args: argparse.Namespace) -> None:
        """initialize the environment

        Args:
            args (argparse.Namespace): arguments
            data (Data): data
        """
        self.window_size = args.window_size

    def time_dimension(self) -> int:
        """the time dimension of the environment

        Returns:
            int: the time dimension of the environment
        """
        raise NotImplementedError("time_dimension not implemented")

    def state_dimension(self) -> Dict[str, torch.Size]:
        """the dimension of the state tensors

        Returns:
            Dict[str, torch.Size]: the dimension of the state tensors
        """
        raise NotImplementedError("state_dimension not implemented")

    def state_tensor_names(self) -> List[str]:
        """the names of the state tensors

        Returns:
            List[str]: the names of the state tensors
        """
        raise NotImplementedError("state_tensor_names not implemented")

    def action_dimension(self) -> torch.Size:
        """the dimension of the action the agent can take

        Returns:
            torch.Size: the dimension of the action the agent can take
        """
        raise NotImplementedError("action_dimension not implemented")

    def get_state(
        self,
    ) -> Dict[str, torch.tensor]:
        """get the state tensors at a given time

        Returns:
            Dict[str, torch.tensor]: the state tensors
        """
        raise NotImplementedError("get_state not implemented")

    def step(
        self, action: torch.tensor, update: bool
    ) -> Tuple[Dict[str, torch.tensor], float, bool]:
        """update the environment with the given action at the given time

        Args:
            action (torch.tensor): the action to take
            time_index (int): the time index to take the action at
            update (bool): whether to update the environment


        Returns:
            Tuple[Dict[str, torch.tensor], float, bool]: the new state, the reward, and whether the episode is done
        """
        raise NotImplementedError("act not implemented")

    def reset(self) -> None:
        """reset the environment"""
        raise NotImplementedError("reset not implemented")
