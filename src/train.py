import argparse

from utils.logging import set_up_logging, get_logger
from agents import registered_agents
from envs import registered_envs
from data import load_data_object

logger = get_logger("train")


def parse_args() -> argparse.Namespace:
    """parse command line arguments

    Returns:
        argparse.Namespace: parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Trainer of RL for Portfolio Optimization"
    )
    parser.add_argument(
        "--verbose",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="DQN",
        choices=registered_agents.keys(),
        help="Name of the model to use",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="BasicEnv",
        choices=registered_envs.keys(),
        help="Name of the environment to use",
    )
    parser.add_argument(
        "--asset_codes",
        required=True,
        nargs="+",
        type=str,
        help="List of asset codes to use",
    )
    parser.add_argument(
        "--start_date",
        type=str,
        required=False,
        help="Start date to use data. Format: YYYY-MM-DD",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        required=False,
        help="End date to use data. Format: YYYY-MM-DD",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="1d",
        choices=[
            "1m",
            "2m",
            "5m",
            "15m",
            "30m",
            "60m",
            "90m",
            "1h",
            "1d",
            "5d",
            "1wk",
            "1mo",
            "3mo",
        ],
        help="Interval to use data.",
    )
    parser.add_argument(
        "--period",
        type=str,
        required=False,
        choices=[
            "1d",
            "5d",
            "1mo",
            "3mo",
            "6mo",
            "1y",
            "2y",
            "5y",
            "10y",
            "ytd",
            "max",
        ],
        help="Period to use data.",
    )
    parser.add_argument(
        "--base_data_path",
        type=str,
        default="../data",
        help="Base data path to save the asset data",
    )
    parser.add_argument(
        "--reload_data",
        action="store_true",
        help="Reload the data from the source",
    )
    parser.add_argument(
        "--time_zone",
        type=str,
        default="America/New_York",
        choices=[
            "America/Argentina/Buenos_Aires",  # Argentina/BUE
            "Australia/Sydney",  # Australia/ASX
            "Europe/Amsterdam",  # Austria/VIE
            "Europe/Brussels",  # Belgium/BRU
            "America/Sao_Paulo",  # Brazil/SAO
            "America/Toronto",  # Canada/CNQ Canada/TOR Canada/VAN
            "Asia/Shanghai",  # China/SHH China/SHZ
            "Europe/Copenhagen",  # Denmark/CPH
            "Europe/Tallinn",  # Estonia/TAL
            "Europe/Helsinki",  # Finland/HEL
            "Europe/Paris",  # France/ENX France/PAR
            "Europe/Berlin",  # Germany/FRA Germany/BER Germany/DUS Germany/GER Germany/HAM
            # Germany/HAN Germany/MUN Germany/STU
            "Europe/Athens",  # Greece/ATN
            "Asia/Hong_Kong",  # Hong_Kong/HKG
            "Europe/London",  # Iceland/ICE United_Kingdom/IOB United_Kingdom/LSE
            "Asia/Kolkata",  # India/BSE India/NSI
            "Asia/Jakarta",  # Indonesia/JKT
            "Europe/Dublin",  # Ireland/ISE
            "Asia/Jerusalem",  # Israel/TLV
            "Europe/Rome",  # Italy/MIL Italy/TLO
            "Europe/Riga",  # Latvia/RIS
            "Europe/Vilnius",  # Lithuania/LIT
            "Asia/Kuala_Lumpur",  # Malaysia/KLS
            "America/Mexico_City",  # Mexico/MEX
            "Europe/Amsterdam",  # Netherlands/AMS
            "Pacific/Auckland",  # New_Zealand/NZE
            "Europe/Oslo",  # Norway/OSL
            "Europe/Lisbon",  # Portugal/LIS
            "Asia/Qatar",  # Qatar/DOH
            "Europe/Moscow",  # Russia/MCX
            "Asia/Singapore",  # Singapore/SES
            "Asia/Seoul",  # South_Korea/KOE South_Korea/KSC
            "Europe/Madrid",  # Spain/MCE
            "Europe/Stockholm",  # Sweden/STO
            "Europe/Zurich",  # Switzerland/EBS
            "Asia/Taipei",  # Taiwan/TAI Taiwan/TWO
            "Asia/Bangkok",  # Thailand/SET
            "Europe/Istanbul",  # Turkey/IST
            "America/New_York",  # United_States/ASE United_States/NMS United_States/NCM
            # United_States/NGM United_States/NYQ United_States/PNK United_States/PCX
            "America/Caracas",  # Venezuela/CCS
        ],
        help="Time zone to use for the data. Default: America/New_York. Time zone reference:"
        "America/Argentina/Buenos_Aires - Argentina/BUE; "
        "Australia/Sydney - Australia/ASX; "
        "Europe/Amsterdam - Austria/VIE; "
        "Europe/Brussels - Belgium/BRU; "
        "America/Sao_Paulo - Brazil/SAO; "
        "America/Toronto - Canada/CNQ Canada/TOR Canada/VAN; "
        "Asia/Shanghai - China/SHH China/SHZ; "
        "Europe/Copenhagen - Denmark/CPH; "
        "Europe/Tallinn - Estonia/TAL; "
        "Europe/Helsinki - Finland/HEL; "
        "Europe/Paris - France/ENX France/PAR; "
        "Europe/Berlin - Germany/FRA Germany/BER Germany/DUS Germany/GER Germany/HAM Germany/HAN Germany/MUN Germany/STU; "
        "Europe/Athens - Greece/ATN; "
        "Asia/Hong_Kong - Hong_Kong/HKG; "
        "Europe/London - Iceland/ICE United_Kingdom/IOB United_Kingdom/LSE; "
        "Asia/Kolkata - India/BSE India/NSI; "
        "Asia/Jakarta - Indonesia/JKT; "
        "Europe/Dublin - Ireland/ISE; "
        "Asia/Jerusalem - Israel/TLV; "
        "Europe/Rome - Italy/MIL Italy/TLO; "
        "Europe/Riga - Latvia/RIS; "
        "Europe/Vilnius - Lithuania/LIT; "
        "Asia/Kuala_Lumpur - Malaysia/KLS; "
        "America/Mexico_City - Mexico/MEX; "
        "Europe/Amsterdam - Netherlands/AMS; "
        "Pacific/Auckland - New_Zealand/NZE; "
        "Europe/Oslo - Norway/OSL; "
        "Europe/Lisbon - Portugal/LIS; "
        "Asia/Qatar - Qatar/DOH; "
        "Europe/Moscow - Russia/MCX; "
        "Asia/Singapore - Singapore/SES; "
        "Asia/Seoul - South_Korea/KOE South_Korea/KSC; "
        "Europe/Madrid - Spain/MCE; "
        "Europe/Stockholm - Sweden/STO; "
        "Europe/Zurich - Switzerland/EBS; "
        "Asia/Taipei - Taiwan/TAI Taiwan/TWO; "
        "Asia/Bangkok - Thailand/SET; "
        "Europe/Istanbul - Turkey/IST; "
        "America/New_York - United_States/ASE United_States/NMS United_States/NCM United_States/NGM United_States/NYQ United_States/PNK United_States/PCX; "
        "America/Caracas - Venezuela/CCS",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=5,
        help="The number of time steps to look back at",
    )

    args, _ = parser.parse_known_args()

    if args.model:
        agent_cls = registered_agents[args.model]
        agent_cls.add_args(parser)

    if args.env:
        env_cls = registered_envs[args.env]
        env_cls.add_args(parser)

    return parser.parse_args()


def main() -> None:
    set_up_logging()

    # Parse command line arguments
    args = parse_args()
    logger.info(args)

    logger.info(f"Agent:            {args.model}")
    logger.info(f"Trading Assets:   {args.asset_codes}")
    logger.info(f"Start Date:       {args.start_date}")
    logger.info(f"End Date:         {args.end_date}")
    logger.info(f"Interval:         {args.interval}")
    logger.info(f"Period:           {args.period}")

    # Load data
    data = load_data_object(
        args.base_data_path,
        args.asset_codes,
        args.start_date,
        args.end_date,
        args.interval,
        args.period,
        args.reload_data,
    )
    data.uniform_time(args.time_zone)

    # set up environment
    env = registered_envs[args.env](args, data)

    agent = registered_agents[args.model](args, env)
    agent.train()

    exit(-1)

    # read config
    # model_name, stock_name, window_size, num_episode, initial_balance = read_config()
    # stock_prices = stock_close_prices(stock_name)
    # trading_period = len(stock_prices) - 1

    # # select learning model
    # model = importlib.import_module(f"agents.{model_name}")
    # agent = model.Agent(state_dim=window_size + 3, balance=initial_balance)

    # train(model_name, agent, window_size, num_episode, stock_prices, trading_period)


# def train(
#     model_name,
#     agent,
#     window_size,
#     num_episode,
#     stock_prices,
#     trading_period,
# ):
#     logger.info("Training...")
#     returns_across_episodes = []
#     num_experience_replay = 0
#     action_dict = {0: "Hold", 1: "Buy", 2: "Sell"}

#     def hold(actions):
#         # encourage selling for profit and liquidity
#         next_probable_action = np.argsort(actions)[1]
#         if next_probable_action == 2 and len(agent.inventory) > 0:
#             max_profit = stock_prices[t] - min(agent.inventory)
#             if max_profit > 0:
#                 sell(t)
#                 actions[next_probable_action] = (
#                     1  # reset this action's value to the highest
#                 )
#                 return "Hold", actions

#     def buy(t):
#         if agent.balance > stock_prices[t]:
#             agent.balance -= stock_prices[t]
#             agent.inventory.append(stock_prices[t])
#             return "Buy: ${:.2f}".format(stock_prices[t])

#     def sell(t):
#         if len(agent.inventory) > 0:
#             agent.balance += stock_prices[t]
#             bought_price = agent.inventory.pop(0)
#             profit = stock_prices[t] - bought_price
#             global reward
#             reward = profit
#             return "Sell: ${:.2f} | Profit: ${:.2f}".format(stock_prices[t], profit)

#     start_time = time.time()
#     for e in range(1, num_episode + 1):
#         logger.info(f"\nEpisode: {e}/{num_episode}")

#         agent.reset()  # reset to initial balance and hyperparameters
#         state = generate_combined_state(
#             0, window_size, stock_prices, agent.balance, len(agent.inventory)
#         )

#         for t in range(1, trading_period + 1):
#             if t % 100 == 0:
#                 logger.info(
#                     f"\n-------------------Period: {t}/{trading_period}-------------------"
#                 )

#             reward = 0
#             next_state = generate_combined_state(
#                 t, window_size, stock_prices, agent.balance, len(agent.inventory)
#             )
#             previous_portfolio_value = (
#                 len(agent.inventory) * stock_prices[t] + agent.balance
#             )

#             if model_name == "DDPG":
#                 actions = agent.act(state, t)
#                 action = np.argmax(actions)
#             else:
#                 with agent.suppress_stdout():
#                     actions = agent.model.predict(state)[0]
#                 action = agent.act(state)

#             # execute position
#             logger.info(
#                 "Step: {}\tHold signal: {:.4} \tBuy signal: {:.4} \tSell signal: {:.4}".format(
#                     t, actions[0], actions[1], actions[2]
#                 )
#             )
#             if action != np.argmax(actions):
#                 logger.info(f"\t\t'{action_dict[action]}' is an exploration.")
#             if action == 0:  # hold
#                 execution_result = hold(actions)
#             if action == 1:  # buy
#                 execution_result = buy(t)
#             if action == 2:  # sell
#                 execution_result = sell(t)

#             # check execution result
#             if execution_result is None:
#                 reward -= (
#                     treasury_bond_daily_return_rate() * agent.balance
#                 )  # missing opportunity
#             else:
#                 if isinstance(execution_result, tuple):  # if execution_result is 'Hold'
#                     actions = execution_result[1]
#                     execution_result = execution_result[0]
#                 logger.info(execution_result)

#             # calculate reward
#             current_portfolio_value = (
#                 len(agent.inventory) * stock_prices[t] + agent.balance
#             )
#             unrealized_profit = current_portfolio_value - agent.initial_portfolio_value
#             reward += unrealized_profit

#             agent.portfolio_values.append(current_portfolio_value)
#             agent.return_rates.append(
#                 (current_portfolio_value - previous_portfolio_value)
#                 / previous_portfolio_value
#             )

#             done = True if t == trading_period else False
#             agent.remember(state, actions, reward, next_state, done)

#             # update state
#             state = next_state

#             # experience replay
#             if len(agent.memory) > agent.buffer_size:
#                 num_experience_replay += 1
#                 loss = agent.experience_replay()
#                 logger.info(
#                     "Episode: {}\tLoss: {:.2f}\tAction: {}\tReward: {:.2f}\tBalance: {:.2f}\tNumber of Stocks: {}".format(
#                         e,
#                         loss,
#                         action_dict[action],
#                         reward,
#                         agent.balance,
#                         len(agent.inventory),
#                     )
#                 )
#                 agent.tensorboard.on_batch_end(
#                     num_experience_replay,
#                     {"loss": loss, "portfolio value": current_portfolio_value},
#                 )

#             if done:
#                 portfolio_return = evaluate_portfolio_performance(agent, logging)
#                 returns_across_episodes.append(portfolio_return)

#         # save models periodically
#         if e % 5 == 0:
#             if model_name == "DQN":
#                 agent.model.save("saved_models/DQN_ep" + str(e) + ".h5")
#             elif model_name == "DDPG":
#                 agent.actor.model.save_weights(
#                     "saved_models/DDPG_ep{}_actor.h5".format(str(e))
#                 )
#                 agent.critic.model.save_weights(
#                     "saved_models/DDPG_ep{}_critic.h5".format(str(e))
#                 )
#             logger.info("model saved")

#     logger.info(
#         "total training time: {0:.2f} min".format((time.time() - start_time) / 60)
#     )
#     plot_portfolio_returns_across_episodes(model_name, returns_across_episodes)


if __name__ == "__main__":
    main()
