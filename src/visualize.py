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

    logger.info("visualizing the data")
    # draw value_dict
    plt.clf()
    for visualization_name, value_list in value_dict.items():
        plt.plot(value_list, label=visualization_name)
    plt.legend()
    plt.savefig(save_path)


def visualize_across_epoch(base_path: str, save_path: str) -> None:
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

    plt.clf()
    plt.plot(CR_list, label="CR")
    x = np.arange(len(CR_list))
    coefficients = np.polyfit(x, CR_list, 3)
    quadratic_fit = np.poly1d(coefficients)
    x_fit = np.linspace(x.min(), x.max(), 500)
    y_fit = quadratic_fit(x_fit)
    plt.plot(x_fit, y_fit, label="Fit", linestyle="--")
    plt.legend()
    plt.savefig(os.path.join(save_path, "CR.png"))

    plt.clf()
    plt.plot(SR_list, label="SR")
    x = np.arange(len(SR_list))
    coefficients = np.polyfit(x, SR_list, 3)
    quadratic_fit = np.poly1d(coefficients)
    x_fit = np.linspace(x.min(), x.max(), 500)
    y_fit = quadratic_fit(x_fit)
    plt.plot(x_fit, y_fit, label="Fit", linestyle="--")
    plt.legend()
    plt.savefig(os.path.join(save_path, "SR.png"))

    plt.clf()
    plt.plot(SteR_list, label="SteR")
    x = np.arange(len(SteR_list))
    coefficients = np.polyfit(x, SteR_list, 3)
    quadratic_fit = np.poly1d(coefficients)
    x_fit = np.linspace(x.min(), x.max(), 500)
    y_fit = quadratic_fit(x_fit)
    plt.plot(x_fit, y_fit, label="Fit", linestyle="--")
    plt.legend()
    plt.savefig(os.path.join(save_path, "SteR.png"))

    plt.clf()
    plt.plot(AT_list, label="AT")
    x = np.arange(len(AT_list))
    coefficients = np.polyfit(x, AT_list, 3)
    quadratic_fit = np.poly1d(coefficients)
    x_fit = np.linspace(x.min(), x.max(), 500)
    y_fit = quadratic_fit(x_fit)
    plt.plot(x_fit, y_fit, label="Fit", linestyle="--")
    plt.legend()
    plt.savefig(os.path.join(save_path, "AT.png"))

    plt.clf()
    plt.plot(MDD_list, label="MDD")
    x = np.arange(len(MDD_list))
    coefficients = np.polyfit(x, MDD_list, 3)
    quadratic_fit = np.poly1d(coefficients)
    x_fit = np.linspace(x.min(), x.max(), 500)
    y_fit = quadratic_fit(x_fit)
    plt.plot(x_fit, y_fit, label="Fit", linestyle="--")
    plt.legend()
    plt.savefig(os.path.join(save_path, "MDD.png"))


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
        visualize_list(visualization_list, os.path.join(args.output_path, "result.png"))
        visualize_across_epoch(
            args.visualize_single_path,
            args.output_path,
        )
    elif args.visualize_mode == "json":
        logger.info("get the visualization list")
        visualization_list = get_visualization_list(args.visualization_json_path)
        visualize_list(visualization_list, os.path.join(args.output_path, "result.png"))


if __name__ == "__main__":
    main()
