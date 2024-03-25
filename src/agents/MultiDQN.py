import argparse
from utils.logging import get_logger
from typing import Optional
from datetime import datetime
import os
import random
from utils.file import create_path_recursively

from agents import register_agent
from agents.BaseAgent import BaseAgent
from envs.DiscreteRealDataEnv1 import DiscreteRealDataEnv1
from networks import registered_networks
from tqdm import tqdm
from utils.replay import Replay

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
        env: DiscreteRealDataEnv1,
        device: Optional[str] = None,
    ) -> None:
        logger.info("Initializing MultiDQN")

        super().__init__(args, env, device)
        self.env = env

        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.model_save_path = os.path.join(
            args.model_save_path, "MultiDQN", current_time
        )
        create_path_recursively(self.model_save_path)

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

        self.pretrain_epochs: int = args.pretrain_epochs
        self.pretrain_batch_size: int = args.pretrain_batch_size
        self.pretrain_learning_rate: float = args.pretrain_learning_rate
        self.train_epochs: int = args.train_epochs
        self.train_batch_size: int = args.train_batch_size
        self.train_learning_rate: float = args.train_learning_rate
        self.gamma: float = args.DQN_gamma
        self.epsilon: float = args.DQN_epsilon
        self.epsilon_decay: float = args.DQN_epsilon_decay
        self.epsilon_min: float = args.DQN_epsilon_min
        self.replay = Replay(args.train_batch_size, args.replay_window)
        self.train_optimizer = optim.Adam(
            self.Q_network.parameters(), lr=self.train_learning_rate
        )
        self.train_optimizer.zero_grad()
        self.loss_scale = 1
        self.loss_min = torch.tensor(0.0001, dtype=self.dtype, device=self.device)

    def train(self) -> None:
        self.pretrain()
        self.multiDQN_train()

    def pretrain(self) -> None:
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            self.Q_network.parameters(), lr=self.pretrain_learning_rate
        )

        save_path = os.path.join(self.model_save_path, "pretrain_model_best.pth")
        logger.info("Starting pretraining")
        lowest_loss = float("inf")
        for epoch in range(self.pretrain_epochs):
            logger.info(f"Epoch {epoch+1}/{self.pretrain_epochs}")
            self.Q_network.train()
            optimizer.zero_grad()
            running_loss = torch.tensor(0.0, dtype=self.dtype, device=self.device)
            batch_cnt = 0
            total_loss = torch.tensor(0.0, dtype=self.dtype, device=self.device)
            logger.info("Training")
            for time_index in tqdm(self.env.pretrain_train_time_range(shuffle=True)):
                data = self.env.get_Xt_state_and_pretrain_target(time_index)
                if data is None:
                    continue
                Xt_state, pretrain_target = data
                out = self.Q_network(
                    Xt_state, torch.empty(0, dtype=self.dtype, device=self.device), True
                )
                loss = criterion(out, pretrain_target.permute(1, 0))
                running_loss += loss
                total_loss += loss
                batch_cnt += 1
                if batch_cnt % self.pretrain_batch_size == 0:
                    running_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    running_loss = torch.tensor(
                        0.0, dtype=self.dtype, device=self.device
                    )
            if batch_cnt % self.pretrain_batch_size != 0:
                running_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            train_avg_loss = total_loss / batch_cnt
            self.Q_network.eval()
            total_loss = torch.tensor(0.0, dtype=self.dtype, device=self.device)
            batch_cnt = 0
            logger.info("Evaluating")
            for time_index in tqdm(self.env.pretrain_eval_time_range()):
                data = self.env.get_Xt_state_and_pretrain_target(time_index)
                if data is None:
                    continue
                Xt_state, pretrain_target = data
                out = self.Q_network(
                    Xt_state, torch.empty(0, dtype=self.dtype, device=self.device), True
                )
                loss = criterion(out, pretrain_target.permute(1, 0))
                total_loss += loss
                batch_cnt += 1
            eval_avg_loss = total_loss / batch_cnt
            if eval_avg_loss < lowest_loss:
                lowest_loss = eval_avg_loss
                logger.info(f"Saving best model to {save_path}")
                torch.save(
                    self.Q_network.state_dict(),
                    save_path,
                )
                logger.info(f"Best model saved to {save_path}")
            logger.info(
                f"Epoch {epoch+1}/{self.pretrain_epochs}, Train Loss: {train_avg_loss:.5f}, Eval Loss: {eval_avg_loss:.5f}, Best Eval Loss: {lowest_loss:.5f}"
            )
        logger.info("Pretraining complete")

        # load best model
        logger.info(f"Loading best model from {save_path}")
        self.Q_network.load_state_dict(
            torch.load(
                os.path.join(self.model_save_path, "pretrain_model_best.pth"),
                map_location=self.device,
            )
        )
        logger.info(f"Best model loaded from {save_path}")

    def multiDQN_train(self) -> None:
        self.Q_network.train()
        self.train_optimizer.zero_grad()
        self.replay.reset()
        self.target_Q_network.load_state_dict(self.Q_network.state_dict())
        self.Q_network.train()
        for epoch in range(self.train_epochs):
            episode = self.env.sample_distribution_and_set_episode()
            self.env.set_episode(episode)
            self.env.reset(self.args)
            time_indices = self.env.train_time_range()
            progress_bar = tqdm(total=len(time_indices), position=0, leave=True)
            for _ in time_indices:
                state = self.env.get_state()
                if state is None:
                    possible_action_indexes = self.env.possible_action_indexes()
                    action_index = int(
                        possible_action_indexes[
                            random.randint(0, len(possible_action_indexes) - 1)
                        ].item()
                    )
                else:
                    Xt = state["Xt_Matrix"]
                    wt = state["Portfolio_Weight"]

                    experience_list = []
                    for possible_action_index in self.env.possible_action_indexes():
                        new_state, reward, done = self.env.act(possible_action_index)
                        if done:
                            break
                        experience_list.append(
                            (state, possible_action_index, reward, new_state)
                        )
                    if len(experience_list) > 0:
                        self.replay.remember(experience_list)

                    if random.random() < self.epsilon:
                        possible_action_indexes = self.env.possible_action_indexes()
                        action_index = int(
                            possible_action_indexes[
                                random.randint(0, len(possible_action_indexes) - 1)
                            ].item()
                        )
                    else:
                        action_q_value = self.Q_network(Xt, wt, False)
                        action_index = int(torch.argmax(action_q_value).item())
                        action_index = self.env.action_mapping(
                            action_index, action_q_value
                        )
                new_state, reward, done = self.env.act(action_index)
                self.env.update(action_index)
                self.update_Q_network()
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
                progress_bar.update(1)
            progress_bar.close()
            self.update_target_network()
            logger.info(
                f"Finish epoch {epoch+1}/{self.train_epochs}, epsilon: {self.epsilon:.5f}, portfolio value: {self.env.portfolio_value:.5f}"
            )
            save_path = os.path.join(self.model_save_path, f"Q_net_{epoch}.pth")
            logger.info(f"Saving model to {save_path}")
            torch.save(
                self.Q_network.state_dict(),
                save_path,
            )
            logger.info(f"Model saved to {save_path}")

    def update_Q_network(self) -> float:
        self.Q_network.train()
        self.target_Q_network.eval()
        if not self.replay.has_enough_samples():
            return float("nan")
        K = self.replay.sample()
        loss = torch.tensor(0.0, dtype=self.dtype, device=self.device)
        for L in K:
            for state, action_index, reward, new_state in L:
                Xt = state["Xt_Matrix"]
                wt = state["Portfolio_Weight"]
                new_Xt = new_state["Xt_Matrix"]
                new_wt = new_state["Portfolio_Weight"]
                with torch.no_grad():
                    target_q_values = self.target_Q_network(new_Xt, new_wt, False)
                    best_action_index = int(torch.argmax(target_q_values).item())
                    best_action_index = self.env.action_mapping(
                        best_action_index, target_q_values
                    )
                    target_q_value = (
                        reward + self.gamma * target_q_values[best_action_index]
                    )
                q_value = self.Q_network(Xt, wt, False)[action_index]
                loss += ((q_value - target_q_value) ** 2) * self.loss_scale
        if loss < self.loss_min:
            self.loss_scale *= 2
        loss.backward()
        self.train_optimizer.step()
        self.train_optimizer.zero_grad()
        return loss.item()

    def update_target_network(self) -> None:
        self.target_Q_network.load_state_dict(self.Q_network.state_dict())
        logger.info("Target network updated")
