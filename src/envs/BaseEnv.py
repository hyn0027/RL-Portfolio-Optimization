import argparse
from typing import Dict, Tuple, List, Optional
from utils.logging import get_logger

import torch

logger = get_logger("BaseEnv")


class BaseEnv:
    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        """add arguments to the parser

        Args:
            parser (argparse.ArgumentParser): the parser to add arguments to
        """

    def __init__(
        self,
        args: argparse.Namespace,
        device: Optional[str] = None,
    ) -> None:
        """initialize the environment

        Args:
            args (argparse.Namespace): arguments
            data (Data): data
        """
        self.window_size = args.window_size
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device is None
            else torch.device(device)
        )

    def to(self, device: str) -> None:
        """move the environment to the given device

        Args:
            device (torch.device): the device to move to
        """
        raise NotImplementedError("to not implemented")

    def time_range(self) -> range:
        """the range of time indices

        Returns:
            range: the range of time indices
        """
        raise NotImplementedError("time_range not implemented")

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

    def act(self, action: torch.tensor) -> Tuple[Dict[str, torch.tensor], float, bool]:
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

    def update(self, action: torch.tensor) -> None:
        """update the environment with the given action

        Args:
            action (torch.tensor): the action to take
        """
        raise NotImplementedError("update not implemented")
