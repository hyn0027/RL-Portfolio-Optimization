import argparse
from utils.logging import get_logger
from typing import Optional
from tqdm import tqdm
from copy import deepcopy
import os

from networks import registered_networks
from utils.replay import Replay
from agents import register_agent
from agents.DPG import DPG
from envs.BaseEnv import BaseEnv

import torch
import torch.nn as nn

logger = get_logger("MultiDPG")


@register_agent("MultiDPG")
class MultiDPG(DPG):
    """The MultiDPG class is a subclass of BaseAgent and implements the DPG algorithm.

    Raises:
        ValueError: missing model_load_path for testing
    """

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        super(MultiDPG, MultiDPG).add_args(parser)

    def __init__(
        self,
        args: argparse.Namespace,
        env: BaseEnv,
        device: Optional[str] = None,
        test_mode: bool = False,
    ) -> None:
        """the constructor for the MultiDPG agent

        Args:
            args (argparse.Namespace): arguments
            env (BaseEnv): the trading environment
            device (Optional[str], optional): torch device. Defaults to None, which means the device is automatically selected.
            test_mode (bool, optional): test or train mode. Defaults to False.
        """
        logger.info("Initialize MultiDPG agent")
        super().__init__(args, env, device, test_mode)
        self.train_batch_size = args.train_batch_size
        logger.info("MultiDPG agent initialized")

    def train(self) -> None:
        """train the MultiDPG agent"""
        self.model.eval()
        self.replay.reset()
        save_path = os.path.join(self.model_save_path, f"DPG_epoch0.pth")
        logger.info(f"Saving model to {save_path}")
        torch.save(
            self.model.state_dict(),
            save_path,
        )
        logger.info(f"Model saved to {save_path}")
        for epoch in range(self.train_epochs):
            self.env.reset()
            time_indices = self.env.train_time_range()
            progress_bar = tqdm(
                total=len(time_indices) // self.train_batch_size, position=0, leave=True
            )
            for _ in time_indices:
                with torch.no_grad():
                    state = self.env.get_state()
                    action = self.model(state)
                    self.env.update(action, state=state, modify_inner_state=True)
                    self.replay.remember(state)
                if _ % self.train_batch_size == 0:
                    self._update_model()
                    progress_bar.update(1)
            progress_bar.close()
            lr = self.train_scheduler.get_last_lr()[0]
            logger.info(
                f"Finish epoch {epoch + 1}/{self.train_epochs}, portfolio value: {self.env.portfolio_value:.5f}, learning rate: {lr}"
            )
            save_path = os.path.join(self.model_save_path, f"model_last_checkpoint.pth")
            logger.info(f"Saving model to {save_path}")
            torch.save(
                self.model.state_dict(),
                save_path,
            )
            logger.info(f"Model saved to {save_path}")
            save_path = os.path.join(self.model_save_path, f"DPG_epoch{epoch + 1}.pth")
            logger.info(f"Saving model to {save_path}")
            torch.save(
                self.model.state_dict(),
                save_path,
            )
            logger.info(f"Model saved to {save_path}")

    def _update_model(self) -> float:
        """update the model"""
        # torch.autograd.set_detect_anomaly(True)
        if not self.replay.has_enough_samples(interval=self.update_window_size):
            return float("nan")
        self.model.train()
        self.train_optimizer.zero_grad()
        states = self.replay.sample(interval=self.update_window_size)
        states = [deepcopy(states)]
        loss = torch.tensor(0.0, dtype=self.dtype, device=self.device)
        for _ in range(self.update_window_size):
            actions = self.model(states[-1])
            new_states = []
            for i in range(len(states[-1])):
                action = actions[i]
                state = states[-1][i]
                new_state, reward, _ = self.env.act(action, state)
                loss += -reward
                new_states.append(new_state)
            states.append(new_states)
        loss.backward()
        self.train_optimizer.step()
        self.train_scheduler.step()
        self.model.eval()
        return loss.item()

    def test(self) -> None:
        # model
        logger.info("Testing Model")
        self.model.eval()
        self.env.reset()
        self.replay.reset()
        self.evaluator.reset()
        time_indices = self.env.test_time_range()
        progress_bar = tqdm(total=len(time_indices), position=0, leave=True)
        for _ in time_indices:
            with torch.no_grad():
                state = self.env.get_state()
                self.replay.remember(state)
                action = self.model(state)
                portfolio_value = self.env.portfolio_value.item()
                portfolio_weight_before_trade = self.env.portfolio_weight
                new_state = self.env.update(action)
                portfolio_weight_after_trade = new_state[
                    "new_portfolio_weight_prev_day"
                ]
                self.evaluator.push(
                    portfolio_value,
                    (portfolio_weight_before_trade, portfolio_weight_after_trade),
                    new_state["prev_price"],
                )
            self._update_model()
            progress_bar.update(1)
        progress_bar.close()
        logger.info("Model Results:")
        self.evaluator.evaluate()
        if not os.path.exists(self.evaluator_save_path):
            os.makedirs(self.evaluator_save_path)
        self.evaluator.output_record_to_json(
            os.path.join(self.evaluator_save_path, "model.json")
        )
        if self.test_model_only:
            return
