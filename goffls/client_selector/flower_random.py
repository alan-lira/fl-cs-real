from flwr.server import ClientManager


def select_clients_randomly(client_manager: ClientManager,
                            num_available_clients: int,
                            min_available_clients: int,
                            clients_fraction: float) -> list:
    selected_clients = []
    num_clients_to_select = int(num_available_clients * clients_fraction)
    sampled_clients_proxies = client_manager.sample(num_clients=num_clients_to_select,
                                                    min_num_clients=min_available_clients)
    for sampled_client_proxy in sampled_clients_proxies:
        selected_clients.append({"client_proxy": sampled_client_proxy})
    return selected_clients
