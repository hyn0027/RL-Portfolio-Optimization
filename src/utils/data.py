import os
import json
import argparse
from utils.logging import set_up_logging, get_logger
from typing import Optional, Dict, Tuple, List, Any
from utils.file import create_path_recursively

import yfinance as yf
import pandas as pd

logger = get_logger("data")


class SingleData:
    """the single data class"""

    def __init__(
        self,
        asset_code: str,
        info: Dict,
        hist: pd.DataFrame,
        option_dates: Tuple,
        calls: Dict[str, pd.DataFrame],
        puts: Dict[str, pd.DataFrame],
    ) -> None:
        """initialize the single data class

        Args:
            asset_code (str): the asset code
            info (Dict): the asset info
            hist (pd.DataFrame): the asset history data
            option_dates (Tuple): the option dates for the asset
            calls (Dict[str, pd.DataFrame]): the calls data
            puts (Dict[str, pd.DataFrame]): the puts data
        """
        self.asset_code = asset_code
        self.info = info
        self.hist = hist
        self.option_dates = option_dates
        self.calls = calls
        self.puts = puts


class Data:
    """the Data manager for this project"""

    def __init__(self, data: Dict[str, SingleData] = {}) -> None:
        """initialize the Data manager

        Args:
            data (Dict[str, SingleData], optional): a dict of SingleData. Defaults to {}.
        """
        self.data = data
        self.asset_codes = list(data.keys())
        self.time_list = self._get_time_list()
        self.check_time_list()
        logger.info(f"Total {len(self.time_list)} time points found.")

    def check_time_list(self) -> None:
        """check the time list"""
        # if a time stamp is not found in one asset, delete it from all assets
        for asset_code in self.asset_codes:
            for time in self.time_list:
                if time not in self.data[asset_code].hist.index:
                    self.delete_from_time(time)

    def delete_from_time(self, time: pd.Timestamp) -> None:
        """delete data from the given time

        Args:
            time (pd.Timestamp): the time to delete data from
        """
        logger.info(f"Deleting data from time {time}.")
        for asset_code in self.asset_codes:
            self.data[asset_code].hist = self.data[asset_code].hist.drop(
                time, errors="ignore"
            )
            for option_date in self.get_asset_option_dates(asset_code):
                self.data[asset_code].calls[option_date] = (
                    self.data[asset_code].calls[option_date].drop(time, errors="ignore")
                )
                self.data[asset_code].puts[option_date] = (
                    self.data[asset_code].puts[option_date].drop(time, errors="ignore")
                )
        self.time_list = self._get_time_list()
        logger.info(f"Data deleted from time {time}.")

    def uniform_time(self, time_zone: str = "America/New_York") -> None:
        """uniform all data time to the given time zone

        Args:
            time_zone (str, optional): the time zone. Defaults to "America/New_York".
        """
        logger.info(f"Changing all data timezone to {time_zone}.")
        for asset_code in self.asset_codes:
            self.data[asset_code].hist = self.data[asset_code].hist.tz_convert(
                time_zone
            )
            for option_date in self.get_asset_option_dates(asset_code):
                self.data[asset_code].calls[option_date]["lastTradeDate"] = (
                    self.data[asset_code]
                    .calls[option_date]["lastTradeDate"]
                    .dt.tz_convert(time_zone)
                )
                self.data[asset_code].puts[option_date]["lastTradeDate"] = (
                    self.data[asset_code]
                    .puts[option_date]["lastTradeDate"]
                    .dt.tz_convert(time_zone)
                )
        logger.info(f"All data timezone changed to {time_zone}.")

    def _get_time_list(self) -> List[pd.Timestamp]:
        """get the list of all time index

        Returns:
            List[pd.Timestamp]: the list of all time index
        """
        time_set = set()
        for asset_code in self.asset_codes:
            time_set = time_set.union(set(self.data[asset_code].hist.index))
        time_list = list(time_set)
        time_list.sort()
        return time_list

    def get_time_index(self, time: pd.Timestamp) -> int:
        """get the index of the given time

        Args:
            time (pd.Timestamp): the given timestamp

        Returns:
            int: the index of the given time
        """
        return self.time_list.index(time)

    def time_dimension(self) -> int:
        """the number of time points

        Returns:
            int: the number of time points
        """
        return len(self.time_list)

    def asset_dimension(self) -> int:
        """the number of assets

        Returns:
            int: the number of assets
        """
        return len(self.asset_codes)

    def add_asset_data(self, asset_code: str, data: SingleData) -> None:
        """add asset data to the data manager

        Args:
            asset_code (str): the new asset code
            data (SingleData): the asset data info
        """
        self.data[asset_code] = data
        self.asset_codes.append(asset_code)
        self.time_list = self._get_time_list()
        logger.info(f"Total {len(self.time_list)} time points found.")

    def get_asset_data(self, asset_code: str) -> SingleData:
        """get the asset data

        Args:
            asset_code (str): the asset code to get data for

        Returns:
            SingleData: the asset data
        """
        return self.data[asset_code]

    def get_asset_info(self, asset_code: str) -> Dict[str, Any]:
        """get the asset info

        Args:
            asset_code (str): the asset code to get info for

        Returns:
            Dict[str, Any]: the asset info
        """
        return self.data[asset_code].info

    def get_asset_hist(self, asset_code: str) -> pd.DataFrame:
        """get the asset history data

        Args:
            asset_code (str): the asset code to get history data for

        Returns:
            pd.DataFrame: the asset history data
        """
        return self.data[asset_code].hist

    def get_asset_hist_at_time(
        self, asset_code: str, time: pd.Timestamp
    ) -> Optional[pd.Series]:
        """the asset history data at the given time

        Args:
            asset_code (str): the asset code to get history data for
            time (pd.Timestamp): the timestamp to get history data for

        Returns:
            Optional[pd.Series]: the asset history data at the given time
        """
        try:
            return self.data[asset_code].hist.loc[time]
        except KeyError:
            return None

    def get_asset_option_dates(self, asset_code: str) -> Tuple:
        """get the asset option dates

        Args:
            asset_code (str): the asset code to get option dates for

        Returns:
            Tuple: the asset option dates
        """
        return self.data[asset_code].option_dates

    def get_asset_calls(self, asset_code: str, date: str) -> pd.DataFrame:
        """get the asset calls data

        Args:
            asset_code (str): the asset code to get calls data for
            date (str): the date to get calls data for

        Returns:
            pd.DataFrame: the asset calls data
        """
        return self.data[asset_code].calls[date]

    def get_asset_puts(self, asset_code: str, date: str) -> pd.DataFrame:
        """get the asset puts data

        Args:
            asset_code (str): the asset code to get puts data for
            date (str): the date to get puts data for

        Returns:
            pd.DataFrame: the asset puts data
        """
        return self.data[asset_code].puts[date]


def load_data_object(
    base_data_path: str,
    asset_codes: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    interval: str = "1d",
    period: Optional[str] = None,
    reload: bool = False,
) -> Data:
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
        data[asset_code] = SingleData(asset_code, info, hist, option_dates, calls, puts)

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

    if len(info) < 2:
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

    create_path_recursively(base_path)

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

    create_path_recursively(base_path)

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
    """save asset info to local file

    Args:
        base_path (str): the base path to save the asset data
        info (Dict): the asset info to save
    """
    path = f"{base_path}/info.json"
    with open(path, "w") as json_file:
        json.dump(info, json_file, indent=4)
    logger.debug(f"Saved asset info to: {path}")


def _get_asset_info(base_path: str) -> Dict:
    """get asset info from local file

    Args:
        base_path (str): the base path to save the asset data

    Returns:
        Dict: the asset info
    """
    path = f"{base_path}/info.json"
    with open(path, "r") as json_file:
        info = json.load(json_file)
    logger.debug(f"Loaded asset info from: {path}")
    return info


def _save_asset_hist(base_path: str, hist: pd.DataFrame) -> None:
    """save asset history to local file

    Args:
        base_path (str): the base path to save the asset data
        hist (pd.DataFrame): the asset history to save
    """
    path = f"{base_path}/hist.csv"
    hist.to_csv(path)
    logger.debug(f"Saved asset history to: {path}")


def _get_asset_hist(base_path: str) -> pd.DataFrame:
    """get asset history from local file

    Args:
        base_path (str): the base path to save the asset data

    Returns:
        pd.DataFrame: the asset history
    """
    path = f"{base_path}/hist.csv"
    hist = pd.read_csv(path, index_col=0, parse_dates=True)
    if hist.index.dtype != "datetime64[ns]":
        hist.index = pd.to_datetime(hist.index, utc=True)
    logger.debug(f"Loaded asset history from: {path}")
    return hist


def _save_asset_options_dates(base_path: str, option_dates: Tuple) -> None:
    """save asset option dates to local file

    Args:
        base_path (str): the base path to save the asset data
        option_dates (Tuple): the asset option dates to save
    """
    path = f"{base_path}/option_dates.json"
    with open(path, "w") as json_file:
        json.dump(option_dates, json_file, indent=4)
    logger.debug(f"Saved asset option dates to: {path}")


def _get_asset_options_dates(base_path: str) -> Tuple:
    """get asset option dates from local file

    Args:
        base_path (str): the base path to save the asset data

    Returns:
        Tuple: the asset option dates
    """
    path = f"{base_path}/option_dates.json"
    with open(path, "r") as json_file:
        option_dates = json.load(json_file)
    logger.debug(f"Loaded asset option dates from: {path}")
    return option_dates


def _save_asset_calls(base_path: str, calls: Dict[str, pd.DataFrame]) -> None:
    """save asset calls to local file

    Args:
        base_path (str): the base path to save the asset data
        calls (Dict[str, pd.DataFrame]): the asset calls to save
    """
    for date, call in calls.items():
        path = f"{base_path}/calls_{date}.csv"
        call.to_csv(path)
        logger.debug(f"Saved asset calls of date {date} to: {path}")


def _get_asset_calls(base_path: str, dates: Tuple) -> Dict[str, pd.DataFrame]:
    """get asset calls from local file

    Args:
        base_path (str): the base path to save the asset data
        dates (Tuple): the dates to get calls data for

    Returns:
        Dict[str, pd.DataFrame]: the asset calls
    """
    calls = {}
    for date in dates:
        path = f"{base_path}/calls_{date}.csv"
        call = pd.read_csv(path, parse_dates=["lastTradeDate"])
        calls[date] = call
        logger.debug(f"Loaded asset calls of date {date} from: {path}")
    return calls


def _save_asset_puts(base_path: str, puts: Dict[str, pd.DataFrame]) -> None:
    """save asset puts to local file

    Args:
        base_path (str): the base path to save the asset data
        puts (Dict[str, pd.DataFrame]): the asset puts to save
    """
    for date, put in puts.items():
        path = f"{base_path}/puts_{date}.csv"
        put.to_csv(path)
        logger.debug(f"Saved asset puts of date {date} to: {path}")


def _get_asset_puts(base_path: str, dates: Tuple) -> Dict[str, pd.DataFrame]:
    """get asset puts from local file

    Args:
        base_path (str): the base path to save the asset data
        dates (Tuple): the dates to get puts data for

    Returns:
        Dict[str, pd.DataFrame]: the asset puts
    """
    puts = {}
    for date in dates:
        path = f"{base_path}/puts_{date}.csv"
        put = pd.read_csv(path, parse_dates=["lastTradeDate"])
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
