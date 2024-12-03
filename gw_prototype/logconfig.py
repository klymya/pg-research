import logging

from config import get_config


def configure_logging() -> None:
    config = get_config()
    logging.basicConfig(
        level=config["logging"]["level"].upper(),
        format="%(process)s - %(asctime)s - [%(levelname)s] - %(name)s - (%(filename)s).%(funcName)s(%(lineno)d) - "\
            "%(message)s"
    )
