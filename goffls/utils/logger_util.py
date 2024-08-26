from logging import FileHandler, Formatter, getLevelName, Logger, StreamHandler
from pathlib import Path
from typing import Union


def load_logger(logging_settings: dict,
                logger_name: str) -> Union[Logger, None]:
    """
    Loads a Logger instance.

    Args:
        logging_settings (dict): the logging settings to be used by the logger.
        logger_name (str): the logger name.

    Returns:
        Union[Logger, None]: the Logger instance.
    """
    logger = None
    enable_logging = logging_settings["enable_logging"]
    log_to_file = logging_settings["log_to_file"]
    log_to_console = logging_settings["log_to_console"]
    file_name = logging_settings["file_name"]
    if file_name is not None:
        file_name = Path(logging_settings["file_name"]).absolute()
    file_mode = logging_settings["file_mode"]
    encoding = logging_settings["encoding"]
    level = logging_settings["level"]
    format_str = logging_settings["format_str"]
    date_format = logging_settings["date_format"]
    if enable_logging:
        logger = Logger(name=logger_name, level=level)
        formatter = Formatter(fmt=format_str, datefmt=date_format)
        if log_to_file:
            file_name.parent.mkdir(exist_ok=True, parents=True)
            file_handler = FileHandler(filename=file_name, mode=file_mode, encoding=encoding)
            file_handler.setLevel(logger.getEffectiveLevel())
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        if log_to_console:
            console_handler = StreamHandler()
            console_handler.setLevel(logger.getEffectiveLevel())
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
    return logger


def log_message(logger: Logger,
                message: str,
                message_level: str) -> None:
    """
    Logs a message using a Logger instance.

    Args:
        logger (Logger): the Logger instance.
        message (str): the message to be logged.
        message_level (str): the message level.

    Returns:
        None
    """
    if logger:
        logger_level = getLevelName(logger.getEffectiveLevel())
        if logger_level != "NOTSET":
            if message_level == "DEBUG":
                logger.debug(msg=message)
            elif message_level == "INFO":
                logger.info(msg=message)
            elif message_level == "WARNING":
                logger.warning(msg=message)
            elif message_level == "ERROR":
                logger.error(msg=message)
            elif message_level == "CRITICAL":
                logger.critical(msg=message)
