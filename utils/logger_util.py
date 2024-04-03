from logging import FileHandler, Formatter, getLevelName, Logger, StreamHandler
from pathlib import Path
from re import findall
from typing import Optional


def load_logger(logging_settings: dict,
                logger_name: str) -> Optional[Logger]:
    logger = None
    if logging_settings["enable_logging"]:
        logger = Logger(name=logger_name,
                        level=logging_settings["level"])
        formatter = Formatter(fmt=logging_settings["format"],
                              datefmt=logging_settings["date_format"])
        if logging_settings["log_to_file"]:
            file_parents_path = findall("(.*/)", logging_settings["file_name"])
            if file_parents_path:
                Path(file_parents_path[0]).mkdir(parents=True, exist_ok=True)
            file_handler = FileHandler(filename=logging_settings["file_name"],
                                       mode=logging_settings["file_mode"],
                                       encoding=logging_settings["encoding"])
            file_handler.setLevel(logger.getEffectiveLevel())
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        if logging_settings["log_to_console"]:
            console_handler = StreamHandler()
            console_handler.setLevel(logger.getEffectiveLevel())
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
    return logger


def log_message(logger: Logger,
                message: str,
                message_level: str) -> None:
    if logger and getLevelName(logger.getEffectiveLevel()) != "NOTSET":
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
