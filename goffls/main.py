from argparse import ArgumentParser, RawTextHelpFormatter
from pathlib import Path
from sys import argv
from time import perf_counter

from goffls.client_launcher.flower_client_launcher import FlowerClientLauncher
from goffls.server_launcher.flower_server_launcher import FlowerServerLauncher
from goffls.util.logger_util import load_logger, log_message
from goffls.util.setup_tools_util import get_version

# Paths.
__BASE_PATH = Path(__file__).parent.resolve()
__VERSION_FILE = __BASE_PATH.joinpath("VERSION")
__FLOWER_CLIENT_CONFIG_FILE = __BASE_PATH.joinpath("client/config/flower_client.cfg")
__FLOWER_SERVER_CONFIG_FILE = __BASE_PATH.joinpath("client/config/flower_server.cfg")

# List of implemented tools.
__AVAILABLE_TOOLS = [{"name": "GOFFLS",
                      "description": "",
                      "actions": [{"launch_server": "launches a FL server instance"},
                                  {"launch_client": "launches a FL client instance"}]}]


def _get_tools_info() -> list:
    tools_info = []
    for available_tool in __AVAILABLE_TOOLS:
        tool_actions = ""
        for action in available_tool["actions"]:
            tool_action = list(action.keys())[0]
            tool_action_description = list(action.values())[0]
            tool_actions += tool_action.ljust(25) + tool_action_description + "\n"
        tool_info = "{0}".format(tool_actions)
        tools_info.append(tool_info)
    return tools_info


def _get_tools_names() -> list:
    tools_names = []
    for available_tool in __AVAILABLE_TOOLS:
        tool_name = available_tool["name"]
        tools_names.append(tool_name)
    return tools_names


def _verify_if_tool_name_is_valid(tool_name: str) -> None:
    tools_names = _get_tools_names()
    if tool_name not in tools_names:
        error_message = "'{0}' is not a valid tool.\nAvailable tools: {1}".format(tool_name, ", ".join(tools_names))
        raise ValueError(error_message)


def _get_tool_actions(tool_name: str) -> list:
    tool_actions = []
    for available_tool in __AVAILABLE_TOOLS:
        if available_tool["name"] == tool_name:
            for action in available_tool["actions"]:
                tool_actions.extend(list(action.keys()))
            break
    return tool_actions


def _verify_if_tool_action_is_valid(tool_name: str,
                                    tool_action: str) -> None:
    tool_actions = _get_tool_actions(tool_name)
    if tool_action not in tool_actions:
        error_message = ("'{0}' is not a valid action for the {1} tool.\nAvailable actions: {2}"
                         .format(tool_action, tool_name, ", ".join(tool_actions)))
        raise ValueError(error_message)


def main() -> None:
    # Begin.
    # Start the performance counter.
    perf_counter_start = perf_counter()
    # Get the version.
    version = get_version(__VERSION_FILE)
    # Parse the GOFFLS arguments.
    usage = "goffls [--version] [--help] <action> [args]"
    description = '''Generic Optimization Framework for Federated Learning Schedules (GOFFLS).\n'''
    epilog = ('''{0}'''.format("\n".join([tool_info for tool_info in _get_tools_info()])))
    ap = ArgumentParser(usage=usage,
                        description=description,
                        epilog=epilog,
                        formatter_class=RawTextHelpFormatter)
    ap.add_argument("-v", "--version",
                    action="version",
                    help="show version number and exit",
                    version="%(prog)s version {0}".format(version))
    ap.add_argument("action",
                    type=str,
                    help="action to perform with GOFFLS tool")
    if "launch_server" in argv:
        ap.add_argument("--implementation",
                        type=str,
                        required=True,
                        help="Server Implementation (no default)")
        ap.add_argument("--id",
                        type=int,
                        required=True,
                        help="Server ID (no default)")
        ap.add_argument("--config-file",
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
        ap.add_argument("--config-file",
                        type=Path,
                        required=True,
                        help="Client Config File (no default)")
    parsed_args = ap.parse_args()
    # Get the user-provided arguments.
    action = str(parsed_args.action)
    # Verify if the user-provided action is valid.
    _verify_if_tool_action_is_valid("GOFFLS", action)
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
