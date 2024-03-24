import argparse
from utils.logging import get_logger
from typing import Optional

from agents import register_agent
from agents.BaseAgent import BaseAgent
from envs.DiscreteRealDataEnv1 import DiscreteRealDataEnv1
from networks import registered_networks
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

logger = get_logger("MultiDQN")


@register_agent("MultiDQN")
class MultiDQN(BaseAgent):
    """
    references:
        https://arxiv.org/abs/1907.03665
        https://github.com/Jogima-cyber/portfolio-manager
    """

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        super(MultiDQN, MultiDQN).add_args(parser)
        parser.add_argument(
            "--pretrain_batch_size",
            type=int,
            default=32,
            help="batch size for pretraining",
        )
        parser.add_argument(
            "--pretrain_epochs",
            type=int,
            default=50,
            help="number of epochs for pretraining",
        )
        parser.add_argument(
            "--pretrain_learning_rate",
            type=float,
            default=0.002,
            help="learning rate for pretraining",
        )

    def __init__(
        self,
        args: argparse.Namespace,
        env: DiscreteRealDataEnv1,
        device: Optional[str] = None,
    ) -> None:
        logger.info("Initializing MultiDQN")

        super().__init__(args, env, device)
        self.env = env

        self.Q_network: nn.Module = registered_networks[args.network](args)
        self.target_Q_network = registered_networks[args.network](args)
        self.Q_network.to(self.device)
        self.target_Q_network.to(self.device)

        logger.info(self.Q_network)
        total_params = sum(p.numel() for p in self.Q_network.parameters())
        logger.info(f"Total number of parameters: {total_params}")

        self.pretrain_epoch = args.pretrain_epochs
        self.pretrain_batch_size = args.pretrain_batch_size
        self.pretrain_learning_rate = args.pretrain_learning_rate

    def train(self) -> None:
        self.pretrain()
        self.multiDQN_train()

    def pretrain(self) -> None:
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            self.Q_network.parameters(), lr=self.pretrain_learning_rate
        )
        logger.info("Starting pretraining")
        for epoch in range(self.pretrain_epoch):
            logger.info(f"Epoch {epoch+1}/{self.pretrain_epoch}")
            self.Q_network.train()
            optimizer.zero_grad()
            running_loss = torch.tensor(0.0, device=self.device)
            batch_cnt = 0
            total_loss = torch.tensor(0.0, device=self.device)
            logger.info("Training")
            for time_index in tqdm(self.env.pretrain_time_range()):
                data = self.env.get_Xt_state_and_pretrain_target(time_index)
                if data is None:
                    continue
                Xt_state, pretrain_target = data
                out = self.Q_network(Xt_state, torch.empty(0, device=self.device), True)
                loss = criterion(out, pretrain_target.permute(1, 0))
                running_loss += loss
                total_loss += loss
                batch_cnt += 1
                if batch_cnt % self.pretrain_batch_size == 0:
                    running_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    running_loss = torch.tensor(0.0, device=self.device)
            if batch_cnt % self.pretrain_batch_size != 0:
                running_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            train_avg_loss = total_loss / batch_cnt
            self.Q_network.eval()
            total_loss = torch.tensor(0.0, device=self.device)
            batch_cnt = 0
            logger.info("Evaluating")
            for time_index in tqdm(self.env.pretrain_eval_time_range()):
                data = self.env.get_Xt_state_and_pretrain_target(time_index)
                if data is None:
                    continue
                Xt_state, pretrain_target = data
                out = self.Q_network(Xt_state, torch.empty(0, device=self.device), True)
                loss = criterion(out, pretrain_target.permute(1, 0))
                total_loss += loss
                batch_cnt += 1
            eval_avg_loss = total_loss / batch_cnt
            logger.info(
                f"Epoch {epoch+1}/{self.pretrain_epoch}, Train Loss: {train_avg_loss:.5f}, Eval Loss: {eval_avg_loss:.5f}"
            )

    def multiDQN_train(self) -> None:
        pass
