import os
import json
import argparse
from utils.logging import set_up_logging, get_logger
from typing import Optional, Dict, Tuple, List, Any

import yfinance as yf
import pandas as pd

logger = get_logger("data")


def parse_args() -> argparse.Namespace:
    """parse command line arguments

    Returns:
        argparse.Namespace: parsed arguments
    """

    parser = argparse.ArgumentParser(description="asset data retriever")
    parser.add_argument(
        "--verbose",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )
    parser.add_argument(
        "--asset_codes",
        nargs="+",
        type=str,
        help="A list of asset code to fetch data for. E.g. AAPL, TSLA, MSFT, etc.",
    )
    parser.add_argument(
        "--start_date",
        type=str,
        required=False,
        help="Start date to fetch data from. Format: YYYY-MM-DD",
    )
    parser.add_argument(
        "--end_date",
        type=str,
        required=False,
        help="End date to fetch data from. Format: YYYY-MM-DD",
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
        help="Interval to fetch data from.",
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
        help="Period to fetch data from.",
    )
    parser.add_argument(
        "--base_data_path",
        type=str,
        default="../data",
        help="Base data path to save the asset data",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Whether or not to reload data if the data already exists.",
    )
    return parser.parse_args()


class Data:
    def __init__(self, data: Dict[str, Dict[str, Any]] = {}) -> None:
        self.data = data

    def add_asset_data(self, asset_code: str, data: Dict[str, Any]) -> None:
        self.data[asset_code] = data

    def get_asset_data(self, asset_code: str) -> Dict[str, Any]:
        return self.data[asset_code]

    def get_asset_info(self, asset_code: str) -> Dict[str, Any]:
        return self.data[asset_code]["info"]

    def get_asset_hist(self, asset_code: str) -> pd.DataFrame:
        return self.data[asset_code]["hist"]

    def get_asset_option_dates(self, asset_code: str) -> Tuple:
        return self.data[asset_code]["option_dates"]

    def get_asset_calls(self, asset_code: str, date: str) -> pd.DataFrame:
        return self.data[asset_code]["calls"][date]

    def get_asset_puts(self, asset_code: str, date: str) -> pd.DataFrame:
        return self.data[asset_code]["puts"][date]


def load_data_object(
    base_data_path: str,
    asset_codes: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    interval: str = "1d",
    period: Optional[str] = None,
    reload: bool = False,
) -> Dict[str, Dict[str, Any]]:
    """load data for the given asset codes

    Args:
        base_data_path (str): the base data path to save the asset data
        asset_codes (List[str]): the list of asset codes to use
        start_date (Optional[str], optional): start date. Defaults to None.
        end_date (Optional[str], optional): end date. Defaults to None.
        interval (str, optional): data interval, select from \
            ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]. \
            Defaults to "1d".
        period (Optional[str], optional): fetch date period, select from \
            ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]. \
            Defaults to None.
        reload (bool, optional): whether or not to reload data. Defaults to False.


    Returns:
        Dict[str, Dict[str, Any]]: the loaded data
    """
    logger.info(f"Loading data for asset codes: {asset_codes}")

    data = {}
    for asset_code in asset_codes:
        info, hist, option_dates, calls, puts, base_path = get_and_save_asset_data(
            base_data_path,
            asset_code,
            start_date=start_date,
            end_date=end_date,
            interval=interval,
            period=period,
            reload=reload,
        )
        data[asset_code] = {
            "info": info,
            "hist": hist,
            "option_dates": option_dates,
            "calls": calls,
            "puts": puts,
            "base_path": base_path,
        }

    logger.info("All data loaded")

    return Data(data)


def get_asset_data(
    base_data_path: str,
    asset_code: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    interval: str = "1d",
    period: Optional[str] = None,
    reload: bool = False,
) -> Tuple[Dict, pd.DataFrame, Tuple, Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """get asset data from Yahoo Finance

    Args:
        base_data_path (str): base data path to save the asset data
        asset_code (str): asset code to fetch data for. E.g. AAPL, TSLA, MSFT, etc.
        start_date (Optional[str], optional): start date. Defaults to None.
        end_date (Optional[str], optional): end date. Defaults to None.
        interval (str, optional): data interval, select from \
            ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]. \
            Defaults to "1d".
        period (Optional[str], optional): fetch date period, select from \
            ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]. \
            Defaults to None.
        reload (bool, optional): whether or not to reload data. Defaults to False.

    Raises:
        ValueError: start_date and end_date must be used together
        ValueError: start_date and end_date cannot be used with period
        ValueError: at least one of start_date, end_date, or period must be used
        ValueError: no data found for asset_code

    Returns:
        Tuple[Dict, pd.DataFrame, tuple, Dict, Dict]: \
        asset info, asset history, options dates, calls, puts
    """

    base_path = _data_path(
        base_data_path, asset_code, start_date, end_date, interval, period
    )

    if os.path.exists(base_path) and not reload:
        return load_single_data_from_local(
            base_data_path, asset_code, start_date, end_date, interval, period
        )

    if (start_date is not None and end_date is None) or (
        start_date is None and end_date is not None
    ):
        raise ValueError("start_date and end_date must be used together")
    if period is not None and (start_date is not None or end_date is not None):
        raise ValueError("start_date and end_date cannot be used with period")
    if period is None and start_date is None and end_date is None:
        raise ValueError("At least one of start_date, end_date, or period must be used")

    if interval not in [
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
    ]:
        raise ValueError(
            "Invalid interval. Supported intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo"
        )

    if period is not None and period not in [
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
    ]:
        raise ValueError(
            "Invalid period. Supported periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max"
        )

    logger.info(
        f"Fetching data for asset_code: {asset_code}, start_date: {start_date}, end_date: {end_date}, interval: {interval}, period: {period}"
    )

    asset = yf.Ticker(asset_code)
    info = asset.info

    if info["trailingPegRatio"] is None:
        raise ValueError(f"No data found for asset_code: {asset_code}")

    hist = asset.history(
        start=start_date, end=end_date, interval=interval, period=period
    )

    options_dates = asset.options

    calls = {date: asset.option_chain(date).calls for date in options_dates}
    puts = {date: asset.option_chain(date).puts for date in options_dates}

    logger.info(f"Successfully fetched data for asset_code: {asset_code}")

    return info, hist, options_dates, calls, puts


def get_and_save_asset_data(
    base_data_path: str,
    asset_code: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    interval: str = "1d",
    period: Optional[str] = None,
    reload: bool = False,
) -> Tuple[
    Dict, pd.DataFrame, Tuple, Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], str
]:
    """get and save asset data

    Args:
        base_data_path (str): base data path to save the asset data
        asset_code (str): asset code to fetch data for. E.g. AAPL, TSLA, MSFT, etc.
        start_date (Optional[str], optional): start date . Defaults to None.
        end_date (Optional[str], optional): end date. Defaults to None.
        interval (str, optional): data interval, select from \
            ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]. \
            Defaults to "1d".
        period (Optional[str], optional): fetch date period, select from \
            ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]. \
            Defaults to None.
        reload (bool, optional): whether or not to reload data. Defaults to False.

    Returns:
        Tuple[Dict, pd.DataFrame, Tuple, Dict, Dict, str]: \
        asset info, asset history, options dates, calls, puts, base path
    """
    base_path = _data_path(
        base_data_path, asset_code, start_date, end_date, interval, period
    )

    if os.path.exists(base_path) and not reload:
        info, hist, option_dates, calls, puts = load_single_data_from_local(
            base_data_path, asset_code, start_date, end_date, interval, period
        )
        return info, hist, option_dates, calls, puts, base_path

    info, hist, option_dates, calls, puts = get_asset_data(
        base_data_path, asset_code, start_date, end_date, interval, period, reload
    )

    if not os.path.exists(base_path):
        os.makedirs(base_path)

    _save_asset_info(base_path, info)
    _save_asset_hist(base_path, hist)
    _save_asset_options_dates(base_path, option_dates)
    _save_asset_calls(base_path, calls)
    _save_asset_puts(base_path, puts)

    logger.info(f"Saved data to: {base_path}")

    return info, hist, option_dates, calls, puts, base_path


def save_asset_data(
    base_data_path: str,
    asset_code: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    interval: str = "1d",
    period: Optional[str] = None,
    reload: bool = False,
) -> str:
    """save asset data

    Args:
        base_data_path (str): base data path to save the asset data
        asset_code (str): asset code to fetch data for. E.g. AAPL, TSLA, MSFT, etc.
        start_date (Optional[str], optional): start date . Defaults to None.
        end_date (Optional[str], optional): end date. Defaults to None.
        interval (str, optional): data interval, select from \
            ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]. \
            Defaults to "1d".
        period (Optional[str], optional): fetch date period, select from \
            ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]. \
            Defaults to None.
        reload (bool, optional): whether or not to reload data. Defaults to False.

    Returns:
        str: base path
    """
    base_path = _data_path(
        base_data_path, asset_code, start_date, end_date, interval, period
    )

    if os.path.exists(base_path) and not reload:
        return base_path

    info, hist, option_dates, calls, puts = get_asset_data(
        base_data_path, asset_code, start_date, end_date, interval, period, reload
    )

    if not os.path.exists(base_path):
        os.makedirs(base_path)

    _save_asset_info(base_path, info)
    _save_asset_hist(base_path, hist)
    _save_asset_options_dates(base_path, option_dates)
    _save_asset_calls(base_path, calls)
    _save_asset_puts(base_path, puts)

    logger.info(f"Saved data to: {base_path}")

    return base_path


def load_single_data_from_local(
    base_data_path: str,
    asset_code: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    interval: str = "1d",
    period: Optional[str] = None,
) -> Tuple[Dict, pd.DataFrame, Tuple, Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """load data from local file

    Args:
        base_data_path (str): base data path to save the asset data
        asset_code (str): asset code to fetch data for. E.g. AAPL, TSLA, MSFT, etc.
        start_date (Optional[str], optional): start date . Defaults to None.
        end_date (Optional[str], optional): end date. Defaults to None.
        interval (str, optional): data interval, select from \
            ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]. \
            Defaults to "1d".
        period (Optional[str], optional): fetch date period, select from \
            ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]. \
            Defaults to None.

    Returns:
        Tuple[ Dict, pd.DataFrame, Tuple, Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]: \
        asset info, asset history, options dates, calls, puts
    """
    base_path = _data_path(
        base_data_path, asset_code, start_date, end_date, interval, period
    )

    if not os.path.exists(base_path):
        raise ValueError(f"Data not found at {base_path}.")

    logger.info(f"Loading data from: {base_path}")

    info = _get_asset_info(base_path)
    hist = _get_asset_hist(base_path)
    option_dates = _get_asset_options_dates(base_path)
    calls = _get_asset_calls(base_path, option_dates)
    puts = _get_asset_puts(base_path, option_dates)

    logger.info(f"Successfully loaded data from: {base_path}")

    return info, hist, option_dates, calls, puts


def _save_asset_info(base_path: str, info: Dict) -> None:
    path = f"{base_path}/info.json"
    with open(path, "w") as json_file:
        json.dump(info, json_file, indent=4)
    logger.debug(f"Saved asset info to: {path}")


def _get_asset_info(base_path: str) -> Dict:
    path = f"{base_path}/info.json"
    with open(path, "r") as json_file:
        info = json.load(json_file)
    logger.debug(f"Loaded asset info from: {path}")
    return info


def _save_asset_hist(base_path: str, hist: pd.DataFrame) -> None:
    path = f"{base_path}/hist.csv"
    hist.to_csv(path)
    logger.debug(f"Saved asset history to: {path}")


def _get_asset_hist(base_path: str) -> pd.DataFrame:
    path = f"{base_path}/hist.csv"
    hist = pd.read_csv(path)
    logger.debug(f"Loaded asset history from: {path}")
    return hist


def _save_asset_options_dates(base_path: str, option_dates: Tuple) -> None:
    path = f"{base_path}/option_dates.json"
    with open(path, "w") as json_file:
        json.dump(option_dates, json_file, indent=4)
    logger.debug(f"Saved asset option dates to: {path}")


def _get_asset_options_dates(base_path: str) -> Tuple:
    path = f"{base_path}/option_dates.json"
    with open(path, "r") as json_file:
        option_dates = json.load(json_file)
    logger.debug(f"Loaded asset option dates from: {path}")
    return option_dates


def _save_asset_calls(base_path: str, calls: Dict[str, pd.DataFrame]) -> None:
    for date, call in calls.items():
        path = f"{base_path}/calls_{date}.csv"
        call.to_csv(path)
        logger.debug(f"Saved asset calls of date {date} to: {path}")


def _get_asset_calls(base_path: str, dates: Tuple) -> Dict[str, pd.DataFrame]:
    calls = {}
    for date in dates:
        path = f"{base_path}/calls_{date}.csv"
        call = pd.read_csv(path)
        calls[date] = call
        logger.debug(f"Loaded asset calls of date {date} from: {path}")
    return calls


def _save_asset_puts(base_path: str, puts: Dict[str, pd.DataFrame]) -> None:
    for date, put in puts.items():
        path = f"{base_path}/puts_{date}.csv"
        put.to_csv(path)
        logger.debug(f"Saved asset puts of date {date} to: {path}")


def _get_asset_puts(base_path: str, dates: Tuple) -> Dict[str, pd.DataFrame]:
    puts = {}
    for date in dates:
        path = f"{base_path}/puts_{date}.csv"
        put = pd.read_csv(path)
        puts[date] = put
        logger.debug(f"Loaded asset puts of date {date} from: {path}")
    return puts


def _data_path(
    base_data_path: str,
    asset_code: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    interval: str = "1d",
    period: Optional[str] = None,
) -> str:
    """get data base path to save the asset data

    Args:
        base_data_path (str): base data path to save the asset data
        asset_code (str): asset code to fetch data for. E.g. AAPL, TSLA, MSFT, etc.
        start_date (Optional[str], optional): start date . Defaults to None.
        end_date (Optional[str], optional): end date. Defaults to None.
        interval (str, optional): data interval, select from \
            ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]. \
            Defaults to "1d".
        period (Optional[str], optional): fetch date period, select from \
            ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]. \
            Defaults to None.

    Returns:
        str: data base path to save the asset data
    """
    return os.path.join(
        base_data_path,
        f"{asset_code}_{start_date}_{end_date}_{interval}_{period}",
    )


def main() -> None:
    args = parse_args()

    set_up_logging(args.verbose)
    logger.info(args)

    for asset_code in args.asset_codes:
        _ = save_asset_data(
            args.base_data_path,
            asset_code,
            args.start_date,
            args.end_date,
            args.interval,
            args.period,
            args.reload,
        )


if __name__ == "__main__":
    main()
