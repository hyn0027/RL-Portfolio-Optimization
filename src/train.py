import importlib
import logging
import time
import yaml
import sys

from utils.utils import *

logging.basicConfig(
    format="[%(asctime)s.%(msecs)03d %(filename)s:%(lineno)3s] %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def read_config():
    with open("config.yaml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            logging.error(exc)
            sys.exit(0)

    logging.info(config)
    model_list = config.get("model_list")
    if model_list is None:
        logging.error("No model_list found")
        sys.exit(0)
    if not isinstance(model_list, list):
        logging.error("model_list should be a list")
        sys.exit(0)
    if len(model_list) == 0:
        logging.error("model_list should not be empty")
        sys.exit(0)

    stock_list = config.get("stock_list")
    if stock_list is None:
        logging.error("No stock_list found")
        sys.exit(0)
    if not isinstance(stock_list, list):
        logging.error("stock_list should be a list")
        sys.exit(0)
    if len(stock_list) == 0:
        logging.error("stock_list should not be empty")
        sys.exit(0)

    model_name = config.get("model", model_list[0])
    if model_name not in model_list:
        logging.error(f"model_name should be one of {model_list}")
        sys.exit(0)
    stock_name = config.get("stock", stock_list[0])
    if stock_name not in stock_list:
        logging.error(f"stock_name should be one of {stock_list}")
        sys.exit(0)
    window_size = config.get("window_size", 10)
    if not isinstance(window_size, int) or window_size <= 0:
        logging.error("window_size should be an integer greater than 0")
        sys.exit(0)
    num_episode = config.get("num_episode", 10)
    if not isinstance(num_episode, int) or num_episode <= 0:
        logging.error("num_episode should be an integer greater than 0")
        sys.exit(0)
    initial_balance = config.get("initial_balance", 20000.0)
    if not isinstance(initial_balance, (int, float)) or initial_balance <= 0:
        logging.error("initial_balance should be a number greater than 0")
        sys.exit(0)
    initial_balance = float(initial_balance)

    return model_name, stock_name, window_size, num_episode, initial_balance


def main():
    # read config
    model_name, stock_name, window_size, num_episode, initial_balance = read_config()
    exit(-1)
    stock_prices = stock_close_prices(stock_name)
    trading_period = len(stock_prices) - 1

    # select learning model
    model = importlib.import_module(f"agents.{model_name}")
    agent = model.Agent(state_dim=window_size + 3, balance=initial_balance)

    logging.info(f"Trading Object:           {stock_name}")
    logging.info(f"Trading Period:           {trading_period} days")
    logging.info(f"Window Size:              {window_size} days")
    logging.info(f"Training Episode:         {num_episode}")
    logging.info(f"Model Name:               {model_name}")
    logging.info("Initial Portfolio Value: ${:,}".format(initial_balance))

    train(model_name, agent, window_size, num_episode, stock_prices, trading_period)


def train(
    model_name,
    agent,
    window_size,
    num_episode,
    stock_prices,
    trading_period,
):
    logging.info("Training...")
    returns_across_episodes = []
    num_experience_replay = 0
    action_dict = {0: "Hold", 1: "Buy", 2: "Sell"}

    def hold(actions):
        # encourage selling for profit and liquidity
        next_probable_action = np.argsort(actions)[1]
        if next_probable_action == 2 and len(agent.inventory) > 0:
            max_profit = stock_prices[t] - min(agent.inventory)
            if max_profit > 0:
                sell(t)
                actions[next_probable_action] = (
                    1  # reset this action's value to the highest
                )
                return "Hold", actions

    def buy(t):
        if agent.balance > stock_prices[t]:
            agent.balance -= stock_prices[t]
            agent.inventory.append(stock_prices[t])
            return "Buy: ${:.2f}".format(stock_prices[t])

    def sell(t):
        if len(agent.inventory) > 0:
            agent.balance += stock_prices[t]
            bought_price = agent.inventory.pop(0)
            profit = stock_prices[t] - bought_price
            global reward
            reward = profit
            return "Sell: ${:.2f} | Profit: ${:.2f}".format(stock_prices[t], profit)

    start_time = time.time()
    for e in range(1, num_episode + 1):
        logging.info(f"\nEpisode: {e}/{num_episode}")

        agent.reset()  # reset to initial balance and hyperparameters
        state = generate_combined_state(
            0, window_size, stock_prices, agent.balance, len(agent.inventory)
        )

        for t in range(1, trading_period + 1):
            if t % 100 == 0:
                logging.info(
                    f"\n-------------------Period: {t}/{trading_period}-------------------"
                )

            reward = 0
            next_state = generate_combined_state(
                t, window_size, stock_prices, agent.balance, len(agent.inventory)
            )
            previous_portfolio_value = (
                len(agent.inventory) * stock_prices[t] + agent.balance
            )

            if model_name == "DDPG":
                actions = agent.act(state, t)
                action = np.argmax(actions)
            else:
                with agent.suppress_stdout():
                    actions = agent.model.predict(state)[0]
                action = agent.act(state)

            # execute position
            logging.info(
                "Step: {}\tHold signal: {:.4} \tBuy signal: {:.4} \tSell signal: {:.4}".format(
                    t, actions[0], actions[1], actions[2]
                )
            )
            if action != np.argmax(actions):
                logging.info(f"\t\t'{action_dict[action]}' is an exploration.")
            if action == 0:  # hold
                execution_result = hold(actions)
            if action == 1:  # buy
                execution_result = buy(t)
            if action == 2:  # sell
                execution_result = sell(t)

            # check execution result
            if execution_result is None:
                reward -= (
                    treasury_bond_daily_return_rate() * agent.balance
                )  # missing opportunity
            else:
                if isinstance(execution_result, tuple):  # if execution_result is 'Hold'
                    actions = execution_result[1]
                    execution_result = execution_result[0]
                logging.info(execution_result)

            # calculate reward
            current_portfolio_value = (
                len(agent.inventory) * stock_prices[t] + agent.balance
            )
            unrealized_profit = current_portfolio_value - agent.initial_portfolio_value
            reward += unrealized_profit

            agent.portfolio_values.append(current_portfolio_value)
            agent.return_rates.append(
                (current_portfolio_value - previous_portfolio_value)
                / previous_portfolio_value
            )

            done = True if t == trading_period else False
            agent.remember(state, actions, reward, next_state, done)

            # update state
            state = next_state

            # experience replay
            if len(agent.memory) > agent.buffer_size:
                num_experience_replay += 1
                loss = agent.experience_replay()
                logging.info(
                    "Episode: {}\tLoss: {:.2f}\tAction: {}\tReward: {:.2f}\tBalance: {:.2f}\tNumber of Stocks: {}".format(
                        e,
                        loss,
                        action_dict[action],
                        reward,
                        agent.balance,
                        len(agent.inventory),
                    )
                )
                agent.tensorboard.on_batch_end(
                    num_experience_replay,
                    {"loss": loss, "portfolio value": current_portfolio_value},
                )

            if done:
                portfolio_return = evaluate_portfolio_performance(agent, logging)
                returns_across_episodes.append(portfolio_return)

        # save models periodically
        if e % 5 == 0:
            if model_name == "DQN":
                agent.model.save("saved_models/DQN_ep" + str(e) + ".h5")
            elif model_name == "DDPG":
                agent.actor.model.save_weights(
                    "saved_models/DDPG_ep{}_actor.h5".format(str(e))
                )
                agent.critic.model.save_weights(
                    "saved_models/DDPG_ep{}_critic.h5".format(str(e))
                )
            logging.info("model saved")

    logging.info(
        "total training time: {0:.2f} min".format((time.time() - start_time) / 60)
    )
    plot_portfolio_returns_across_episodes(model_name, returns_across_episodes)


if __name__ == "__main__":
    main()
