import argparse
import os

from utils.logging import set_up_logging, get_logger
from agents import registered_agents
from envs import registered_envs
from networks import registered_networks
from utils.data import load_data_object
from utils.replay import Replay
from evaluate.evaluator import Evaluator

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
        "--agent",
        type=str,
        default="DQN",
        choices=registered_agents.keys(),
        help="Name of the agent to use",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="DiscreteRealDataEnv1",
        choices=registered_envs.keys(),
        help="Name of the environment to use",
    )

    parser.add_argument(
        "--network",
        type=str,
        default="MultiValueLSTM",
        choices=registered_networks.keys(),
        help="Name of the network to use",
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
        "--model_save_path",
        type=str,
        default="../model",
        help="Path to save the model",
    )
    parser.add_argument(
        "--reload_data",
        action="store_true",
        help="Reload the data from the source",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "test"],
        default="train",
        help="Mode to run the agent in",
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
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=[
            "cpu",
            "cuda",
            "ipu",
            "xpu",
            "mkldnn",
            "opengl",
            "opencl",
            "ideep",
            "hip",
            "ve",
            "fpga",
            "ort",
            "xla",
            "lazy",
            "vulkan",
            "mps",
            "meta",
            "hpu",
            "mtia",
        ],
        help="The device to run the model on",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use mixed precision training",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=10,
        help="Number of threads to use for data loading",
    )

    Evaluator.add_args(parser)
    Replay.add_args(parser)

    args, _ = parser.parse_known_args()

    if args.agent:
        agent_cls = registered_agents[args.agent]
        agent_cls.add_args(parser)

    if args.env:
        env_cls = registered_envs[args.env]
        env_cls.add_args(parser)

    if args.network:
        network_cls = registered_networks[args.network]
        network_cls.add_args(parser)

    return parser.parse_args()


def main() -> None:
    """the main function to run the training or testing of the agent"""
    set_up_logging()

    # Parse command line arguments
    args = parse_args()
    logger.info(args)

    logger.info(f"Agent:            {args.agent}")
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

    args.asset_num = len(args.asset_codes)

    # set up environment
    env = registered_envs[args.env](args, data, args.device)

    if args.mode == "train":
        agent = registered_agents[args.agent](args, env, args.device, False)
        agent.train()
    else:
        agent = registered_agents[args.agent](args, env, args.device, True)
        agent.test()


if __name__ == "__main__":
    main()
