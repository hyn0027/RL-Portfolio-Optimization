import argparse
import json
import os
from matplotlib import pyplot as plt
import numpy as np
from typing import Dict

from utils.logging import set_up_logging, get_logger


logger = get_logger("visualize")


def parse_args() -> argparse.Namespace:
    """parse command line arguments

    Returns:
        argparse.Namespace: parsed arguments
    """
    parser = argparse.ArgumentParser(description="Visualizing evaluation data")

    parser.add_argument(
        "--visualization_json_path",
        default="visualization_config.json",
        type=str,
        help="path to the list of visualizations",
    )
    parser.add_argument(
        "--visualize_single_path",
        type=str,
        help="path to the evaluators file",
    )
    parser.add_argument(
        "--output_path",
        default="../visualization",
        type=str,
        help="output path for the visualizations",
    )
    parser.add_argument(
        "--visualize_mode",
        choices=["json", "single"],
        default="json",
        help="choose the mode to visualize the data",
    )
    parser.add_argument(
        "--model_name",
        default="Model",
        type=str,
        help="the name of the model",
    )
    parser.add_argument(
        "--window_size",
        default=20,
        type=int,
        help="the window size for moving average",
    )
    return parser.parse_args()


def get_visualization_list(visualization_json_path: str) -> Dict[str, str]:
    """get the list of visualizations

    Args:
        visualization_json_path (str): the path to the visualization list

    Returns:
        Dict[str, str]: the list of visualizations
    """
    with open(visualization_json_path, "r") as f:
        visualization_list = json.load(f)
    return visualization_list


def visualize_list(visualization_list: Dict[str, str], save_path: str) -> None:
    """visualize the list of visualizations

    Args:
        visualization_list (Dict[str, str]): the list of visualizations
        save_path (str): the path to save the visualization
    """
    value_dict = {}
    for visualization_name, visualization_path in visualization_list.items():
        logger.info(f"visualizing {visualization_name} at {visualization_path}")
        # visualize the data
        with open(visualization_path, "r") as f:
            data = json.load(f)
            value_list = data["portfolio_value_list"]
            value_dict[visualization_name] = value_list
            CR = data["CR"]
            SR = data["SR"]
            SteR = data["SteR"]
            AT = data["AT"]
            MDD = data["MDD"]
            logger.info(
                f"CR: {CR * 100 :.3f}, SR: {SR:.3f}, AT: {AT:.4f}, MDD: {MDD * 100:.3f}"
            )

    logger.info("visualizing the data")
    # draw value_dict
    plt.clf()
    fig = plt.figure(figsize=(8, 6), dpi=500)
    for visualization_name, value_list in value_dict.items():
        plt.plot(value_list, label=visualization_name, linewidth=0.7)
    plt.legend()
    plt.savefig(save_path)


def visualize_across_epoch(base_path: str, save_path: str, window_size: int) -> None:
    """visualize the data across epochs

    Args:
        base_path (str): the base path to the data
        save_path (str): the path to save the visualization
    """
    epoch_num = 0
    CR_list = []
    SR_list = []
    SteR_list = []
    AT_list = []
    MDD_list = []
    while True:
        epoch_path = os.path.join(base_path, f"model{epoch_num}", "model.json")
        if not os.path.exists(epoch_path):
            break
        logger.info(f"visualizing epoch {epoch_num} from {epoch_path}")
        with open(epoch_path, "r") as f:
            data = json.load(f)
            CR_list.append(data["CR"])
            SR_list.append(data["SR"])
            SteR_list.append(data["SteR"])
            AT_list.append(data["AT"])
            MDD_list.append(data["MDD"])
        epoch_num += 1

    def moving_average(data):
        return np.convolve(data, np.ones(window_size) / window_size, mode="valid")

    plt.clf()
    fig = plt.figure(figsize=(6, 4.5), dpi=500)
    plt.plot(CR_list, label="CR", linewidth=1)
    moving_avg = moving_average(CR_list)
    x = np.arange(len(CR_list))
    plt.plot(
        x[window_size - 1 :],
        moving_avg,
        label=f"Moving Average (window={window_size})",
        linewidth=1,
    )
    plt.legend()
    plt.savefig(os.path.join(save_path, "CR.png"))

    plt.clf()
    fig = plt.figure(figsize=(6, 4.5), dpi=500)
    plt.plot(SR_list, label="SR", linewidth=1)
    moving_avg = moving_average(SR_list)
    x = np.arange(len(SR_list))
    plt.plot(
        x[window_size - 1 :],
        moving_avg,
        label=f"Moving Average (window={window_size})",
        linewidth=1,
    )
    plt.legend()
    plt.savefig(os.path.join(save_path, "SR.png"))

    plt.clf()
    fig = plt.figure(figsize=(6, 4.5), dpi=500)
    plt.plot(SteR_list, label="SteR", linewidth=1)
    moving_avg = moving_average(SteR_list)
    x = np.arange(len(SteR_list))
    plt.plot(
        x[window_size - 1 :],
        moving_avg,
        label=f"Moving Average (window={window_size})",
        linewidth=1,
    )
    plt.legend()
    plt.savefig(os.path.join(save_path, "SteR.png"))

    plt.clf()
    fig = plt.figure(figsize=(6, 4.5), dpi=500)
    plt.plot(AT_list, label="AT", linewidth=1)
    moving_avg = moving_average(AT_list)
    x = np.arange(len(AT_list))
    plt.plot(
        x[window_size - 1 :],
        moving_avg,
        label=f"Moving Average (window={window_size})",
        linewidth=1,
    )
    plt.legend()
    plt.savefig(os.path.join(save_path, "AT.png"))

    plt.clf()
    fig = plt.figure(figsize=(6, 4.5), dpi=500)
    plt.plot(MDD_list, label="MDD", linewidth=1)
    moving_avg = moving_average(MDD_list)
    x = np.arange(len(MDD_list))
    plt.plot(
        x[window_size - 1 :],
        moving_avg,
        label=f"Moving Average (window={window_size})",
        linewidth=1,
    )
    plt.legend()
    plt.savefig(os.path.join(save_path, "MDD.png"))

    logger.info(f"highest CR at epoch {np.argmax(CR_list)}")
    logger.info(f"highest SR at epoch {np.argmax(SR_list)}")
    logger.info(f"highest SteR at epoch {np.argmax(SteR_list)}")


def main() -> None:
    """the main function to visualize the data"""
    set_up_logging()

    # Parse command line arguments
    args = parse_args()
    logger.info(args)

    if args.visualize_mode == "single":
        if args.visualize_single_path is None:
            raise ValueError("visualize_single_path is not provided")
        logger.info(f"visualizing {args.visualize_single_path}")
        last_checkpoint_path = os.path.join(
            args.visualize_single_path, "model_last_checkpoint"
        )
        visualization_list = {
            args.model_name: os.path.join(last_checkpoint_path, "model.json"),
            "BuyAndHold": os.path.join(last_checkpoint_path, "B&H.json"),
            "Momentum": os.path.join(last_checkpoint_path, "momentum.json"),
            "Reversal": os.path.join(last_checkpoint_path, "reverse_momentum.json"),
            "Random": os.path.join(last_checkpoint_path, "random.json"),
        }
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)
        try:
            visualize_list(
                visualization_list, os.path.join(args.output_path, "result.png")
            )
        except:
            logger.warning("Failed to visualize the data list")
        visualize_across_epoch(
            args.visualize_single_path,
            args.output_path,
            args.window_size,
        )
    elif args.visualize_mode == "json":
        logger.info("get the visualization list")
        visualization_list = get_visualization_list(args.visualization_json_path)
        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)
        visualize_list(visualization_list, os.path.join(args.output_path, "result.png"))


if __name__ == "__main__":
    main()
