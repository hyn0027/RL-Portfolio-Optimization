import logging


def set_up_logging(level: str = "INFO"):
    logging.basicConfig(
        format="%(levelname)s - %(module)s:%(lineno)d - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=level,
    )


def get_logger(name: str):
    return logging.getLogger(name)
