from utils.logging import get_logger
from typing import Optional, Any
from collections import deque
import random

logger = get_logger("Replay")


class Replay:
    def __init__(self, batch_size: int, max_len: Optional[int] = None) -> None:
        self.batch_size = batch_size
        self.memory = [] if max_len is None else deque(maxlen=max_len)
        logger.info("Replay initialized")

    def remember(self, experience: Any) -> None:
        self.memory.append(experience)

    def sample(self) -> Any:
        return random.sample(self.memory, self.batch_size)

    def reset(self) -> None:
        self.memory.clear()
        logger.info("Replay memory cleared")
