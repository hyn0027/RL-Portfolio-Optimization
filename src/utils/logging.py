import logging


class ColoredFormatter(logging.Formatter):
    """A logging formatter that colors the output"""

    COLORS = {
        "DEBUG": "\033[94m",  # Blue
        "INFO": "\033[92m",  # Green
        "WARNING": "\033[93m",  # Yellow
        "ERROR": "\033[91m",  # Red
        "CRITICAL": "\033[95m",  # Magenta
        "ENDC": "\033[0m",  # Reset
    }
    """the colors for different log levels"""

    def format(self, record: logging.LogRecord) -> str:
        """format the log record

        Args:
            record (logging.LogRecord): the log record

        Returns:
            str: the formatted log record
        """
        color_format = (
            self.COLORS.get(record.levelname, "")
            + f"[%(asctime)s %(levelname)s %(module)s:%(lineno)d]"
            + self.COLORS["ENDC"]
            + "%(message)s"
        )
        self._style._fmt = color_format
        return super().format(record)


def set_up_logging(level: str = "INFO") -> None:
    """set up basic logging configuration

    Args:
        level (str, optional): the verbose level. Defaults to "INFO".
    """
    logging.basicConfig(level=level)
    formatter = ColoredFormatter()
    for handler in logging.getLogger().handlers:
        handler.setFormatter(formatter)


def get_logger(name: str) -> logging.Logger:
    """get a logger

    Args:
        name (str): the name of the logger

    Returns:
        logging.Logger: the logger
    """
    return logging.getLogger(name)
