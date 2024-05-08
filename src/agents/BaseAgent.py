import argparse
from utils.logging import get_logger
from typing import Optional, Generic, TypeVar
from datetime import datetime
from evaluate.evaluator import Evaluator

from utils.file import create_path_recursively

import torch

BaseEnv = TypeVar("BaseEnv")

logger = get_logger("BaseAgent")


class BaseAgent(Generic[BaseEnv]):
    """the base class for all agents

    Args:
        Generic (TypeVar): the base type of the environment

    Raises:
        NotImplementedError: train method not implemented
    """

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
            "--model_load_path",
            type=str,
            required=False,
            help="Path to load model from",
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
            default=0.01,
            help="learning rate for training",
        )
        parser.add_argument(
            "--loss_min", type=float, default=0.0001, help="minimal value of loss"
        )
        parser.add_argument(
            "--evaluator_saving_path",
            type=str,
            required=False,
            help="Path to save the evaluator",
        )

    def __init__(
        self,
        args: argparse.Namespace,
        env: BaseEnv,
        device: Optional[str] = None,
        test_mode: bool = False,
    ) -> None:
        """the constructor for the BaseAgent

        Args:
            args (argparse.Namespace): arguments
            env (BaseEnv): the trading environment
            device (Optional[str], optional): torch device. Defaults to None, which means the device is automatically selected.
            test_mode (bool, optional): test or train mode. Defaults to False.
        """
        logger.info("Initializing BaseAgent")
        torch.set_num_threads(args.num_threads)
        self.env = env
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device is None
            else torch.device(device)
        )
        self.fp16: bool = args.fp16
        self.dtype = torch.float16 if args.fp16 else torch.float32
        self.args = args
        self.test_mode = test_mode

        if not self.test_mode:
            self.model_save_path = args.model_save_path
            create_path_recursively(self.model_save_path)
            self.train_epochs: int = args.train_epochs
            self.train_learning_rate: float = args.train_learning_rate

            self.loss_scale = 1
            self.loss_min = torch.tensor(
                args.loss_min, dtype=self.dtype, device=self.device
            )
        else:
            self.evaluator = Evaluator(args)
        logger.info("BaseAgent initialized")

    def train(self) -> None:
        """train the agent. Must be implemented by the subclass

        Raises:
            NotImplementedError: train method not implemented
        """
        raise NotImplementedError("train method not implemented")

    def test(self) -> None:
        """test the agent. Must be implemented by the subclass

        Raises:
            NotImplementedError: test method not implemented
        """
        raise NotImplementedError("test method not implemented")
