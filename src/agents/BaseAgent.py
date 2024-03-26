import argparse
import os
from utils.logging import get_logger
from typing import Optional, Generic, TypeVar
from datetime import datetime

from utils.file import create_path_recursively

import torch

BaseEnv = TypeVar('BaseEnv')

logger = get_logger("BaseAgent")

class BaseAgent(Generic[BaseEnv]):
    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--model_load_path",
            type=str,
            required=False,
            help="Path to load model from",
        )
        parser.add_argument(
            "--train_batch_size",
            type=int,
            default=32,
            help="batch size for training",
        )
        parser.add_argument(
            "--train_epochs",
            type=int,
            default=500,
            help="number of epochs for training",
        )
        parser.add_argument(
            "--train_learning_rate",
            type=float,
            default=0.001,
            help="learning rate for training",
        )
        parser.add_argument(
            "--loss_min",
            type=float,
            default=0.0001,
            help="minimal value of loss"
        )

    def __init__(
        self,
        args: argparse.Namespace,
        env: BaseEnv,
        device: Optional[str] = None,
        test_mode: bool = False,
    ) -> None:
        logger.info("Initializing BaseAgent")
        torch.set_num_threads(args.num_threads)
        self.env = env
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device is None
            else torch.device(device)
        )
        self.fp16 = args.fp16
        self.dtype = torch.float16 if args.fp16 else torch.float32
        self.args = args
        self.test_mode = test_mode
        
        if not self.test_mode:
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.model_save_path = os.path.join(
                args.model_save_path, "MultiDQN", current_time
            )
            create_path_recursively(self.model_save_path)
            self.train_epochs: int = args.train_epochs
            self.train_batch_size: int = args.train_batch_size
            self.train_learning_rate: float = args.train_learning_rate
            
            self.loss_scale = 1
            self.loss_min = torch.tensor(args.loss_min, dtype=self.dtype, device=self.device)
        logger.info("BaseAgent initialized")

    def train(self) -> None:
        raise NotImplementedError("train method not implemented")
