from logging import Logger

from flwr.server import ClientManager

from goffls.utils.logger_util import log_message


def select_clients_using_random(client_manager: ClientManager,
                                phase: str,
                                num_available_clients: int,
                                min_available_clients: int,
                                clients_fraction: float,
                                logger: Logger) -> list:
    # Log a 'selecting clients' message.
    message = "Selecting {0}ing clients using Random...".format(phase)
    log_message(logger, message, "INFO")
    selected_clients = []
    num_clients_to_select = int(num_available_clients * clients_fraction)
    sampled_clients_proxies = client_manager.sample(num_clients=num_clients_to_select,
                                                    min_num_clients=min_available_clients)
    for sampled_client_proxy in sampled_clients_proxies:
        selected_clients.append({"client_proxy": sampled_client_proxy})
    # Log a 'number of clients selected' message.
    message = "{0} {1} selected!".format(len(selected_clients),
                                         "clients were" if len(selected_clients) != 1 else "client was")
    log_message(logger, message, "INFO")
    return selected_clients
