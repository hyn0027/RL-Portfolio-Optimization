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
        pass

    def __init__(self, args: argparse.Namespace, data: Data) -> None:
        """initialize the environment

        Args:
            args (argparse.Namespace): arguments
            data (Data): data
        """
        self.data = data
        self.asset_codes = args.asset_codes
        self.time_zone = args.time_zone

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

    def action_dimension(self) -> int:
        """the dimension of the action the agent can take

        Returns:
            int: the dimension of the action the agent can take
        """
        raise NotImplementedError("action_dimension not implemented")

    def get_state(
        self,
        time: pd.Timestamp,
    ) -> Dict[str, torch.tensor]:
        """get the state tensors at a given time

        Args:
            time (pd.Timestamp): the time to get the state at

        Returns:
            Dict[str, torch.tensor]: the state tensors
        """
        raise NotImplementedError("get_state not implemented")

    def act(
        self,
        action: torch.tensor,
        time: pd.Timestamp,
    ) -> Tuple[Dict[str, torch.tensor], float, bool]:
        """update the environment with the given action at the given time

        Args:
            action (torch.tensor): the action to take
            time (pd.Timestamp): the time to take the action at


        Returns:
            Tuple[Dict[str, torch.tensor], float, bool]: the new state, the reward, and whether the episode is done
        """
        raise NotImplementedError("act not implemented")

    def reset(self) -> None:
        """reset the environment"""
        raise NotImplementedError("reset not implemented")
