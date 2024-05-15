import argparse
from utils.logging import get_logger
from typing import Optional
from tqdm import tqdm
import os

from networks import registered_networks
from utils.replay import Replay
from agents import register_agent
from agents.BaseAgent import BaseAgent
from envs.BaseEnv import BaseEnv

import torch
import torch.nn as nn

logger = get_logger("DPG")


@register_agent("DPG")
class DPG(BaseAgent):
    """The DPG class is a subclass of BaseAgent and implements the DPG algorithm.

    Raises:
        ValueError: missing model_load_path for testing
    """

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        super(DPG, DPG).add_args(parser)
        parser.add_argument(
            "--DPG_update_window_size",
            type=int,
            default=10,
            help="the update window size",
        )

    def __init__(
        self,
        args: argparse.Namespace,
        env: BaseEnv,
        device: Optional[str] = None,
        test_mode: bool = False,
    ) -> None:
        """the constructor for the DPG agent

        Args:
            args (argparse.Namespace): arguments
            env (BaseEnv): the trading environment
            device (Optional[str], optional): torch device. Defaults to None, which means the device is automatically selected.
            test_mode (bool, optional): test or train mode. Defaults to False.
        """
        logger.info("Initialize DPG agent")
        super().__init__(args, env, device, test_mode)
        if not self.test_mode:
            self.model: nn.Module = registered_networks[args.network](args)
            if self.fp16:
                self.model = self.model.half()
            self.model.to(self.device)

            logger.info(self.model)
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Total number of parameters: {total_params}")

            self.replay = Replay(args)

            self.train_optimizer = torch.optim.Adam(
                self.model.parameters(), lr=args.train_learning_rate
            )
            self.update_window_size: int = args.DPG_update_window_size
            self.train_optimizer.zero_grad()
        else:
            self.model: nn.Module = registered_networks[args.network](args)
            logger.info(self.model)

            if not args.model_load_path:
                raise ValueError("model_load_path is required for testing")
            self.model.load_state_dict(
                torch.load(args.model_load_path, map_location=self.device)
            )
            logger.info(f"model loaded from {args.model_load_path}")

            if not args.evaluator_saving_path:
                raise ValueError("evaluator_saving_path is required for testing")
            self.evaluator_save_path = args.evaluator_saving_path

            self.model.to(self.device)

        logger.info("DPG agent initialized")

    def train(self) -> None:
        """train the DPG agent"""
        self.model.eval()
        self.replay.reset()
        for epoch in range(self.train_epochs):
            self.env.reset()
            time_indices = self.env.train_time_range()
            progress_bar = tqdm(total=len(time_indices), position=0, leave=True)
            for _ in time_indices:
                with torch.no_grad():
                    state = self.env.get_state()
                    action = self.model(state)
                    self.env.update(action, state=state, modify_inner_state=True)
                    self.replay.remember(state)
                self._update_model()
                progress_bar.update(1)
            progress_bar.close()
            logger.info(
                f"Finish epoch {epoch + 1}/{self.train_epochs}, portfolio value: {self.env.portfolio_value:.5f}"
            )
            save_path = os.path.join(self.model_save_path, f"model_last_checkpoint.pth")
            logger.info(f"Saving model to {save_path}")
            torch.save(
                self.model.state_dict(),
                save_path,
            )
            logger.info(f"Model saved to {save_path}")
            save_path = os.path.join(self.model_save_path, f"DPG_epoch{epoch}.pth")
            logger.info(f"Saving model to {save_path}")
            torch.save(
                self.model.state_dict(),
                save_path,
            )
            logger.info(f"Model saved to {save_path}")

    def _update_model(self) -> float:
        """update the model"""
        self.model.train()
        self.train_optimizer.zero_grad()
        if not self.replay.has_enough_samples(interval=self.update_window_size):
            return float("nan")
        states = self.replay.sample(interval=self.update_window_size)
        loss = torch.tensor(
            self.update_window_size, dtype=self.dtype, device=self.device
        )
        for state in states:
            for _ in range(self.update_window_size):
                action = self.model(state)
                state, reward, _ = self.env.act(action, state)
                loss += -reward
        loss.backward()
        self.train_optimizer.step()
        self.model.eval()
        return loss.item()

    def test(self) -> None:
        # model
        logger.info("Testing Model")
        self.model.eval()
        self.env.reset()
        self.evaluator.reset()
        time_indices = self.env.test_time_range()
        progress_bar = tqdm(total=len(time_indices), position=0, leave=True)
        for _ in time_indices:
            state = self.env.get_state()
            action = self.model(state)
            portfolio_value = self.env.portfolio_value.item()
            portfolio_weight_before_trade = self.env.portfolio_weight
            new_state = self.env.update(action)
            portfolio_weight_after_trade = new_state["new_portfolio_weight_prev_day"]
            self.evaluator.push(
                portfolio_value,
                (portfolio_weight_before_trade, portfolio_weight_after_trade),
                new_state["prev_price"],
            )
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

        # buy and hold
        logger.info("Testing B&H")
        self.env.reset()
        self.evaluator.reset()
        time_indices = self.env.test_time_range()
        progress_bar = tqdm(total=len(time_indices), position=0, leave=True)
        for _ in time_indices:
            portfolio_value = self.env.portfolio_value.item()
            portfolio_weight_before_trade = self.env.portfolio_weight
            new_state = self.env.update()
            portfolio_weight_after_trade = new_state["new_portfolio_weight_prev_day"]
            self.evaluator.push(
                portfolio_value,
                (portfolio_weight_before_trade, portfolio_weight_after_trade),
                new_state["prev_price"],
            )
            progress_bar.update(1)
        progress_bar.close()
        logger.info("B&H Results:")
        self.evaluator.evaluate()
        self.evaluator.output_record_to_json(
            os.path.join(self.evaluator_save_path, "B&H.json")
        )

        # random
        logger.info("Testing Random")
        self.env.reset()
        self.evaluator.reset()
        time_indices = self.env.test_time_range()
        progress_bar = tqdm(total=len(time_indices), position=0, leave=True)
        for _ in time_indices:
            action = self.env.select_random_action()
            portfolio_value = self.env.portfolio_value.item()
            portfolio_weight_before_trade = self.env.portfolio_weight
            new_state = self.env.update(action)
            portfolio_weight_after_trade = new_state["new_portfolio_weight_prev_day"]
            self.evaluator.push(
                portfolio_value,
                (portfolio_weight_before_trade, portfolio_weight_after_trade),
                new_state["prev_price"],
            )
            progress_bar.update(1)
        progress_bar.close()
        logger.info("Random Results:")
        self.evaluator.evaluate()
        self.evaluator.output_record_to_json(
            os.path.join(self.evaluator_save_path, "random.json")
        )

        # testing momentum
        logger.info("Testing Momentum")
        self.env.reset()
        self.evaluator.reset()
        time_indices = self.env.test_time_range()
        progress_bar = tqdm(total=len(time_indices), position=0, leave=True)
        for _ in time_indices:
            action = self.env.get_momentum_action()
            portfolio_value = self.env.portfolio_value.item()
            portfolio_weight_before_trade = self.env.portfolio_weight
            new_state = self.env.update(action)
            portfolio_weight_after_trade = new_state["new_portfolio_weight_prev_day"]
            self.evaluator.push(
                portfolio_value,
                (portfolio_weight_before_trade, portfolio_weight_after_trade),
                new_state["prev_price"],
            )
            progress_bar.update(1)
        progress_bar.close()
        logger.info("Momentum Results:")
        self.evaluator.evaluate()
        self.evaluator.output_record_to_json(
            os.path.join(self.evaluator_save_path, "momentum.json")
        )

        # testing reverse momentum
        logger.info("Testing Reverse Momentum")
        self.env.reset()
        self.evaluator.reset()
        time_indices = self.env.test_time_range()
        progress_bar = tqdm(total=len(time_indices), position=0, leave=True)
        for _ in time_indices:
            action = self.env.get_reverse_momentum_action()
            portfolio_value = self.env.portfolio_value.item()
            portfolio_weight_before_trade = self.env.portfolio_weight
            new_state = self.env.update(action)
            portfolio_weight_after_trade = new_state["new_portfolio_weight_prev_day"]
            self.evaluator.push(
                portfolio_value,
                (portfolio_weight_before_trade, portfolio_weight_after_trade),
                new_state["prev_price"],
            )
            progress_bar.update(1)
        progress_bar.close()
        logger.info("Reverse Momentum Results:")
        self.evaluator.evaluate()
        self.evaluator.output_record_to_json(
            os.path.join(self.evaluator_save_path, "reverse_momentum.json")
        )
