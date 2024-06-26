import argparse
from utils.logging import get_logger
from typing import Optional
import os
import random
from tqdm import tqdm

from agents import register_agent
from agents.DQN import DQN
from envs.DiscreteRealDataEnv1 import DiscreteRealDataEnv1


import torch
import torch.nn as nn
import torch.optim as optim

logger = get_logger("MultiDQN")


@register_agent("MultiDQN")
class MultiDQN(DQN[DiscreteRealDataEnv1]):
    """
    The MultiDQN class is a subclass of DQN and implements the MultiDQN algorithm.
    It outputs the Q value of all possible actions simultaneously.
    It takes environment DiscreteRealDataEnv1 as it's own env.

    references:
        original paper: https://arxiv.org/abs/1907.03665

        reference implementation: https://github.com/Jogima-cyber/portfolio-manager
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
            default=0.0002,
            help="learning rate for pretraining",
        )

    def __init__(
        self,
        args: argparse.Namespace,
        env: DiscreteRealDataEnv1,
        device: Optional[str] = None,
        test_mode: bool = False,
    ) -> None:
        """the constructor for the MultiDQN agent

        Args:
            args (argparse.Namespace): arguments
            env (BaseEnv): the trading environment
            device (Optional[str], optional): torch device. Defaults to None, which means the device is automatically selected.
            test_mode (bool, optional): test or train mode. Defaults to False.
        """

        logger.info("Initializing MultiDQN")

        super().__init__(args, env, device, test_mode)

        if not self.test_mode:
            self.pretrain_epochs: int = args.pretrain_epochs
            self.pretrain_batch_size: int = args.pretrain_batch_size
            self.pretrain_learning_rate: float = args.pretrain_learning_rate
        logger.info("MultiDQN initialized")

    def train(self) -> None:
        """the train for multiDQN is composed of two steps:
        1. pretrain the Q network
        2. train the Q network using multiDQN
        """
        if self.pretrain_epochs > 0:
            self._pretrain()
        self._multiDQN_train()

    def _pretrain(self) -> None:
        """the pretraining step of the multiDQN algorithm"""
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            self.Q_network.parameters(), lr=self.pretrain_learning_rate
        )
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

        save_path = os.path.join(self.model_save_path, "pretrain_best_checkpoint.pth")
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
                data = self.env.get_pretrain_input_and_target(time_index)
                if data is None:
                    continue
                input, pretrain_target = data
                out = self.Q_network(input, True)
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
                scheduler.step()
                optimizer.zero_grad()
            train_avg_loss = total_loss / batch_cnt
            # self.Q_network.eval()
            # total_loss = torch.tensor(0.0, dtype=self.dtype, device=self.device)
            # batch_cnt = 0
            # logger.info("Evaluating")
            # for time_index in tqdm(self.env.pretrain_eval_time_range()):
            #     data = self.env.get_pretrain_input_and_target(time_index)
            #     if data is None:
            #         continue
            #     input, pretrain_target = data
            #     out = self.Q_network(input, True)
            #     loss = criterion(out, pretrain_target.permute(1, 0))
            #     total_loss += loss
            #     batch_cnt += 1
            # eval_avg_loss = total_loss / batch_cnt
            # if eval_avg_loss < lowest_loss:
            #     lowest_loss = eval_avg_loss
            # logger.info(f"Saving best model to {save_path}")
            # torch.save(
            #     self.Q_network.state_dict(),
            #     save_path,
            # )
            # logger.info(f"Best model saved to {save_path}")
            # logger.info(
            #     f"Epoch {epoch+1}/{self.pretrain_epochs}, Train Loss: {train_avg_loss:.5f}, Eval Loss: {eval_avg_loss:.5f}, Best Eval Loss: {lowest_loss:.5f}"
            # )
            learning_rate = optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch {epoch+1}/{self.pretrain_epochs}, Train Loss: {train_avg_loss:.5f}, learning rate: {learning_rate:.5f}"
            )
        logger.info("Pretraining complete")

        # # load best model
        # logger.info(f"Loading best model from {save_path}")
        # self.Q_network.load_state_dict(
        #     torch.load(
        #         save_path,
        #         map_location=self.device,
        #     )
        # )
        # logger.info(f"Best model loaded from {save_path}")

    def _multiDQN_train(self) -> None:
        """the training step of the multiDQN algorithm"""
        self.Q_network.eval()
        self.target_Q_network.load_state_dict(self.Q_network.state_dict())
        self.target_Q_network.eval()
        self.replay.reset()
        save_path = os.path.join(self.model_save_path, "Q_net_epoch0.pth")
        logger.info(f"Saving model to {save_path}")
        torch.save(
            self.Q_network.state_dict(),
            save_path,
        )
        logger.info(f"Model saved to {save_path}")
        for epoch in range(self.train_epochs):
            episode = self.env.sample_distribution_and_set_episode()
            self.env.set_episode(episode)
            self.env.reset()
            time_indices = self.env.train_time_range()
            progress_bar = tqdm(total=len(time_indices), position=0, leave=True)
            for _ in time_indices:
                state = self.env.get_state()
                if state is None:
                    action_index = self.env.select_random_action()
                else:
                    experience_list = []
                    for possible_action_index in self.env.possible_actions():
                        new_state, reward, done = self.env.act(possible_action_index)
                        experience_list.append(
                            (possible_action_index, reward, new_state)
                        )
                    if len(experience_list) > 0:
                        self.replay.remember(
                            {
                                "initial_state": state,
                                "experience": experience_list,
                                "done": done,
                            }
                        )

                    if random.random() < self.epsilon:
                        action_index = self.env.select_random_action()
                    else:
                        action_q_value = self.Q_network(state, False)
                        action_index = int(torch.argmax(action_q_value).item())
                        action_index = self.env.action_mapping(
                            action_index, action_q_value
                        )
                # new_state, reward, done = self.env.act(action_index)
                self.env.update(action_index)
                self._update_Q_network()
                self._update_epsilon()
                progress_bar.update(1)
            progress_bar.close()
            self._update_target_network()
            lr = self.train_optimizer.param_groups[0]["lr"]
            logger.info(
                f"Finish epoch {epoch+1}/{self.train_epochs}, portfolio value: {self.env.portfolio_value:.5f}, learning rate: {lr}"
            )
            save_path = os.path.join(self.model_save_path, f"Q_net_last_checkpoint.pth")
            logger.info(f"Saving model to {save_path}")
            torch.save(
                self.Q_network.state_dict(),
                save_path,
            )
            logger.info(f"Model saved to {save_path}")
            save_path = os.path.join(
                self.model_save_path, f"Q_net_epoch{epoch + 1}.pth"
            )
            logger.info(f"Saving model to {save_path}")
            torch.save(
                self.Q_network.state_dict(),
                save_path,
            )
            logger.info(f"Model saved to {save_path}")

    def _update_Q_network(self) -> float:
        """random sample multiple experience lists
        from replay buffer and update Q network

        Returns:
            float: the training loss
        """
        if not self.replay.has_enough_samples():
            return float("nan")
        self.train_optimizer.zero_grad()
        self.Q_network.train()
        K = self.replay.sample()
        mse_loss = nn.MSELoss()
        input = []
        target = []
        for L in K:
            initial_state = L["initial_state"]
            new_state = L["experience"][0][2]
            done = L["done"]
            q_values = self.Q_network(initial_state, False)
            if not done:
                with torch.no_grad():
                    hn = self.target_Q_network(new_state, False, only_LSTM=True)
            for action_index, reward, new_state in L["experience"]:
                if not done:
                    new_state["hn"] = hn
                    with torch.no_grad():
                        target_q_values = self.target_Q_network(
                            new_state, False, no_LSTM=True
                        )
                    best_action_index = int(torch.argmax(target_q_values).item())
                    best_action_index = self.env.action_mapping(
                        best_action_index, target_q_values, new_state
                    )
                    input.append(q_values[action_index])
                    target.append(
                        reward + self.gamma * target_q_values[best_action_index]
                    )
                else:
                    input.append(q_values[action_index])
                    target.append(reward)
        input = torch.stack(input, dim=0)
        target = torch.stack(target, dim=0)
        loss = mse_loss(input, target)
        loss.backward()
        self.train_optimizer.step()
        self.train_scheduler.step()
        self.Q_network.eval()
        return loss.item()

    def test(self) -> None:
        """test the MultiDQN agent"""
        # model
        logger.info("Testing Model")
        self.Q_network.eval()
        self.env.set_episode_for_testing()
        self.env.reset()
        self.replay.reset()
        self.evaluator.reset()
        time_indices = self.env.test_time_range()
        progress_bar = tqdm(total=len(time_indices), position=0, leave=True)
        for _ in time_indices:
            state = self.env.get_state()
            if state is None:
                action_index = self.env.select_random_action()
            else:
                experience_list = []
                for possible_action_index in self.env.possible_actions():
                    new_state, reward, done = self.env.act(possible_action_index)
                    experience_list.append((possible_action_index, reward, new_state))
                if len(experience_list) > 0:
                    self.replay.remember(
                        {
                            "initial_state": state,
                            "experience": experience_list,
                            "done": done,
                        }
                    )
                action_q_value = self.Q_network(state, False)
                action_index = int(torch.argmax(action_q_value).item())
                action_index = self.env.action_mapping(action_index, action_q_value)
            portfolio_value = self.env.portfolio_value.item()
            portfolio_weight_before_trade = self.env.portfolio_weight
            new_state = self.env.update(action_index)
            portfolio_weight_after_trade = new_state["new_portfolio_weight_prev_day"]
            self.evaluator.push(
                portfolio_value,
                (portfolio_weight_before_trade, portfolio_weight_after_trade),
                new_state["prev_price"],
            )
            self._update_Q_network()
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
        self.env.set_episode_for_testing()
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
        self.env.set_episode_for_testing()
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
        self.env.set_episode_for_testing()
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
        self.env.set_episode_for_testing()
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
