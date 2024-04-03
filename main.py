from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path
from sys import argv
from time import perf_counter

from clients.flower.client_launcher import FlowerClientLauncher
from servers.flower.server_launcher import FlowerServerLauncher
from utils.logger_util import load_logger, log_message

# List of implemented actions.
__AVAILABLE_ACTIONS = [{"name": "launch_server",
                        "description": "Launches a FL server instance.",
                        "implemented_servers": ["flower"]},
                       {"name": "launch_client",
                        "description": "Launches a FL client instance.",
                        "implemented_clients": ["flower"]}]


def _get_available_actions() -> list:
    actions = []
    for action in __AVAILABLE_ACTIONS:
        actions.append(action["name"])
    return actions


def _verify_if_action_is_valid(action: str) -> None:
    available_actions = _get_available_actions()
    if action not in available_actions:
        available_actions = ", ".join(available_actions)
        error_message = ("'{0}' is not a valid action.\nAvailable actions: {1}"
                         .format(action, available_actions))
        raise ValueError(error_message)


def main() -> None:
    # Begin.
    # Start the performance counter.
    perf_counter_start = perf_counter()
    # Parse the GOFFLS arguments.
    description = '''Generic Optimization Framework for Federated Learning Schedules (GOFFLS)\n---------------'''
    epilog = ""
    ap = ArgumentParser(description=description,
                        epilog=epilog,
                        formatter_class=RawDescriptionHelpFormatter)
    ap.add_argument("action",
                    type=str,
                    help="Action to perform.")
    if "launch_server" in argv:
        ap.add_argument("--implementation",
                        type=str,
                        required=True,
                        help="Server Implementation (no default)")
        ap.add_argument("--id",
                        type=int,
                        required=True,
                        help="Server ID (no default)")
        ap.add_argument("--config_file",
                        type=Path,
                        required=True,
                        help="Server Config File (no default)")
    elif "launch_client" in argv:
        ap.add_argument("--implementation",
                        type=str,
                        required=True,
                        help="Client Implementation (no default)")
        ap.add_argument("--id",
                        type=int,
                        required=True,
                        help="Client ID (no default)")
        ap.add_argument("--config_file",
                        type=Path,
                        required=True,
                        help="Client Config File (no default)")
    parsed_args = ap.parse_args()
    # Get the user-provided arguments.
    action = str(parsed_args.action)
    # Verify if the user-provided action is valid.
    _verify_if_action_is_valid(action)
    # Set the logger.
    logging_settings = {"enable_logging": True,
                        "level": "INFO",
                        "format": "%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
                        "date_format": "%Y/%m/%d %H:%M:%S",
                        "log_to_file": False,
                        "log_to_console": True}
    logger_name = "GOFFLS_Logger"
    logger = load_logger(logging_settings, logger_name)
    if action == "launch_server":
        server_id = int(parsed_args.id)
        server_config_file = Path(parsed_args.config_file)
        server_implementation = str(parsed_args.implementation)
        if server_implementation == "flower":
            fs = FlowerServerLauncher(server_id, server_config_file)
            fs.launch_server()
    elif action == "launch_client":
        client_id = int(parsed_args.id)
        client_config_file = Path(parsed_args.config_file)
        client_implementation = str(parsed_args.implementation)
        if client_implementation == "flower":
            fc = FlowerClientLauncher(client_id, client_config_file)
            fc.launch_client()
    # Stop the performance counter.
    perf_counter_stop = perf_counter()
    # Log a 'elapsed time in seconds' message.
    elapsed_time_seconds = round((perf_counter_stop - perf_counter_start), 2)
    message = "Elapsed time: {0} {1}.".format(elapsed_time_seconds,
                                              "seconds" if elapsed_time_seconds != 1 else "second")
    log_message(logger, message, "INFO")
    # End.
    exit(0)


if __name__ == "__main__":
    main()
