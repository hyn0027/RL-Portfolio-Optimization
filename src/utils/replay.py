from utils.logging import get_logger
from typing import Optional, Any
from collections import deque
import random

logger = get_logger("Replay")


class Replay:
    """The replay memory"""

    def __init__(self, batch_size: int, max_len: Optional[int] = None) -> None:
        """initialize the replay memory

        Args:
            batch_size (int): the batch size
            max_len (Optional[int], optional): the replay buffer size. Defaults to None.
        """
        self.batch_size = batch_size
        self.memory = [] if max_len is None else deque(maxlen=max_len)
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
        return random.sample(self.memory, self.batch_size)

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
        return len(self.memory) >= self.batch_size
