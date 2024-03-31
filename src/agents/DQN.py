import argparse
from utils.logging import get_logger
from typing import Optional, TypeVar
from tqdm import tqdm
import os
import random

from agents import register_agent
from agents.BaseAgent import BaseAgent
from envs.BaseEnv import BaseEnv
from networks import registered_networks
from evaluate.evaluator import Evaluator
from utils.replay import Replay

import torch
import torch.nn as nn


BaseEnv = TypeVar("BaseEnv")
logger = get_logger("DQN")


@register_agent("DQN")
class DQN(BaseAgent[BaseEnv]):
    """The DQN class is a subclass of BaseAgent and implements the DQN algorithm.

    Args:
        BaseAgent (TypeVar): the type of the environment

    Raises:
        ValueError: missing model_load_path for testing
    """

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        super(DQN, DQN).add_args(parser)
        parser.add_argument(
            "--replay_window",
            type=int,
            default=2000,
            help="replay window size",
        )
        parser.add_argument(
            "--DQN_gamma",
            type=float,
            default=0.9,
            help="discount factor for DQN",
        )
        parser.add_argument(
            "--DQN_epsilon",
            type=float,
            default=1.0,
            help="epsilon for DQN",
        )
        parser.add_argument(
            "--DQN_epsilon_decay",
            type=float,
            default=0.999,
            help="epsilon decay for DQN",
        )
        parser.add_argument(
            "--DQN_epsilon_min",
            type=float,
            default=0.01,
            help="minimum epsilon for DQN",
        )

    def __init__(
        self,
        args: argparse.Namespace,
        env: BaseEnv,
        device: Optional[str] = None,
        test_mode: bool = False,
    ) -> None:
        """the constructor for the DQN agent

        Args:
            args (argparse.Namespace): arguments
            env (BaseEnv): the trading environment
            device (Optional[str], optional): torch device. Defaults to None, which means the device is automatically selected.
            test_mode (bool, optional): test or train mode. Defaults to False.
        """
        logger.info("Initializing DQN")

        super().__init__(args, env, device, test_mode)
        if not self.test_mode:
            self.Q_network: nn.Module = registered_networks[args.network](args)
            self.target_Q_network = registered_networks[args.network](args)
            if self.fp16:
                self.Q_network.half()
                self.target_Q_network.half()

            self.Q_network.to(self.device)
            self.target_Q_network.to(self.device)

            logger.info(self.Q_network)
            total_params = sum(p.numel() for p in self.Q_network.parameters())
            logger.info(f"Total number of parameters: {total_params}")

            self.gamma: float = args.DQN_gamma
            self.epsilon: float = args.DQN_epsilon
            self.epsilon_decay: float = args.DQN_epsilon_decay
            self.epsilon_min: float = args.DQN_epsilon_min
            self.replay = Replay(args.train_batch_size, args.replay_window)
        else:
            self.Q_network: nn.Module = registered_networks[args.network](args)
            logger.info(self.Q_network)

            if not args.model_load_path:
                raise ValueError("model_load_path is required for testing")
            logger.info(f"loading model from {args.model_load_path}")
            self.Q_network.load_state_dict(
                torch.load(args.model_load_path, map_location=self.device)
            )
            logger.info(f"model loaded from {args.model_load_path}")

            self.Q_network.to(self.device)
            self.evaluate = Evaluator(args)

        logger.info("DQN initialized")

    def train(self) -> None:
        """train the DQN agent"""
        self.Q_network.train()
        self.target_Q_network.eval()
        self.replay.reset()
        self.target_Q_network.load_state_dict(self.Q_network.state_dict())
        for epoch in range(self.train_epochs):
            self.env.reset()
            time_indices = self.env.train_time_range()
            progress_bar = tqdm(total=len(time_indices), position=0, leave=True)
            for _ in time_indices:
                state = self.env.get_state()
                if state is None:
                    action = self.env.select_random_action()
                else:
                    if random.random() < self.epsilon:
                        action = self.env.select_random_action()
                    else:
                        max_Q_value = torch.tensor(
                            float("-inf"), device=self.device, dtype=self.dtype
                        )
                        best_action = None
                        for action in self.env.possible_actions():
                            Q_value = self.Q_network(state, action)
                            if Q_value > max_Q_value:
                                max_Q_value = Q_value
                                best_action = action
                        action = best_action
                new_state, reward, done = self.env.act(action)
                if not done:
                    self.replay.remember((state, action, reward, new_state))
                self.env.update(action)
                self._update_Q_network()
                self._update_epsilon()
                progress_bar.update(1)
            progress_bar.close()
            self._update_target_network()
            logger.info(
                f"Finish epoch {epoch+1}/{self.train_epochs}, epsilon: {self.epsilon:.5f}, portfolio value: {self.env.portfolio_value:.5f}"
            )
            save_path = os.path.join(self.model_save_path, f"Q_net_last_checkpoint.pth")
            logger.info(f"Saving model to {save_path}")
            torch.save(
                self.Q_network.state_dict(),
                save_path,
            )
            logger.info(f"Model saved to {save_path}")

    def _update_epsilon(self) -> None:
        """update the epsilon"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def _update_Q_network(self) -> float:
        """random sample multiple experiences
        from replay buffer and update Q network

        Returns:
            float: the training loss
        """
        self.Q_network.train()
        if not self.replay.has_enough_samples():
            return float("nan")
        experiences = self.replay.sample()
        loss = torch.tensor(0.0, dtype=self.dtype, device=self.device)
        for experience in experiences:
            state, action, reward, new_state = experience
            with torch.no_grad():
                max_Q_value = torch.tensor(
                    float("-inf"), device=self.device, dtype=self.dtype
                )
                for new_action in self.env.possible_actions(new_state):
                    Q_value = self.target_Q_network(new_state, new_action)
                    if Q_value > max_Q_value:
                        max_Q_value = Q_value
                target = reward + self.gamma * max_Q_value
            q_value = self.Q_network(state, action)
            loss += (q_value - target).pow(2) * self.loss_scale
        if loss < self.loss_min:
            self.loss_scale *= 2
        loss.backward()
        self.train_optimizer.step()
        self.train_optimizer.zero_grad()
        self.Q_network.eval()
        return loss.item()

    def _update_target_network(self) -> None:
        """update the target network with the Q network weights"""
        self.target_Q_network.load_state_dict(self.Q_network.state_dict())
        logger.info("Target network updated")
