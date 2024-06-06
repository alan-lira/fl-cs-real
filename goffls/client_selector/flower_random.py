from logging import Logger

from flwr.server import ClientManager

from goffls.utils.logger_util import log_message


def select_clients_using_random(client_manager: ClientManager,
                                phase: str,
                                num_available_clients: int,
                                clients_fraction: float,
                                logger: Logger) -> dict:
    # Log a 'selecting clients' message.
    message = "Selecting {0}ing clients using Random...".format(phase)
    log_message(logger, message, "INFO")
    # Initialize the selection dictionary and the list of selected clients.
    selection = {}
    selected_clients = []
    # Determine the number of clients to select.
    num_clients_to_select = max(1, int(num_available_clients * clients_fraction))
    # Select clients via random sampling.
    sampled_clients_proxies = client_manager.sample(num_clients_to_select)
    # Append their corresponding proxies objects into the selected clients list.
    for sampled_client_proxy in sampled_clients_proxies:
        selected_clients.append({"client_proxy": sampled_client_proxy})
    # Update the selection dictionary with the selected clients for the schedule.
    selection.update({"selected_clients": selected_clients})
    # Log a 'number of clients selected' message.
    message = "{0} {1} (out of {2}) {3} selected!".format(len(selected_clients),
                                                          "clients" if len(selected_clients) != 1 else "client",
                                                          num_available_clients,
                                                          "were" if len(selected_clients) != 1 else "was")
    log_message(logger, message, "INFO")
    # Return the selection dictionary.
    return selection
