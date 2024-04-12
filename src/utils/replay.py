from utils.logging import get_logger
import argparse
from typing import Any
import numpy as np
from collections import deque
import random
import copy

logger = get_logger("Replay")


class Replay:
    """The replay memory"""

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        """add arguments to the parser

            to add arguments to the parser, modify the method as follows:

            .. code-block:: python

                @staticmethod
                def add_args(parser: argparse.ArgumentParser) -> None:
                    parser.add_argument(
                        ...
                    )


            then add arguments to the parser

        Args:
            parser (argparse.ArgumentParser): the parser to add arguments to
        """
        parser.add_argument(
            "--replay_sample_distribution",
            type=str,
            choices=["uniform", "geometric"],
            default="uniform",
            help="replay sample distribution",
        )
        parser.add_argument(
            "--replay_sample_geometric_p",
            type=float,
            default=0.5,
            help="geometric distribution parameter",
        )
        parser.add_argument(
            "--replay_sample_unique",
            action="store_true",
            help="sample unique experiences",
        )
        parser.add_argument(
            "--replay_window",
            type=int,
            default=2000,
            help="replay window size",
        )
        parser.add_argument(
            "--train_batch_size",
            type=int,
            default=32,
            help="batch size for training",
        )

    def __init__(
        self,
        args: argparse.Namespace,
    ) -> None:
        """initialize the replay memory

        Args:
            args (argparse.Namespace): the arguments
        """
        self.batch_size = args.train_batch_size
        self.memory = (
            [] if args.replay_window == 0 else deque(maxlen=args.replay_window)
        )
        self.sample_distribution = args.replay_sample_distribution
        self.sample_geometric_p = args.replay_sample_geometric_p
        self.sample_unique = args.replay_sample_unique
        logger.info("Replay initialized")

    def remember(self, experience: Any) -> None:
        """remember an experience

        Args:
            experience (Any): the experience to remember
        """
        self.memory.append(experience)

    def sample(self) -> Any:
        """randomly sample a batch of experiences

        Returns:
            Any: the batch of experiences
        """
        if self.sample_distribution == "uniform":
            if self.sample_unique:
                return random.sample(self.memory, self.batch_size)
            else:
                return random.choices(self.memory, k=self.batch_size)
        if self.sample_distribution == "geometric":
            if self.sample_unique:
                return_list = []
                memory_copy = copy.deepcopy(self.memory)
                memory_len = len(memory_copy)
                indices = np.random.geometric(self.sample_geometric_p, self.batch_size)
                for idx in indices:
                    index = memory_len - idx
                    if index < 0:
                        index = memory_len - 1
                    return_list.append(memory_copy.pop(index))
                    memory_len -= 1
                    memory_copy = memory_copy[:index] + memory_copy[index + 1 :]
                return return_list
            else:
                indices = np.random.geometric(self.sample_geometric_p, self.batch_size)
                memory_len = len(self.memory)
                for i in range(len(indices)):
                    indices[i] = memory_len - indices[i]
                    if indices[i] < 0:
                        indices[i] = memory_len - 1
                return [self.memory[i] for i in indices]
        raise NotImplementedError(
            f"sample distribution {self.sample_distribution} not implemented"
        )

    def reset(self) -> None:
        """reset the replay memory to an empty state"""
        self.memory.clear()
        logger.info("Replay memory cleared")

    def __len__(self) -> int:
        return len(self.memory)

    def has_enough_samples(self) -> bool:
        """return if the replay memory has enough samples

        Returns:
            bool: whether the replay memory has enough samples
        """
        if self.sample_unique:
            return len(self.memory) >= self.batch_size
        else:
            return len(self.memory) >= 1
