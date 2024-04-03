from flwr.server import ClientManager
from flwr.server.client_proxy import ClientProxy


def select_clients_randomly(client_manager: ClientManager,
                            num_available_clients: int,
                            min_available_clients: int,
                            fraction: float) -> list[ClientProxy]:
    num_clients_to_select = int(num_available_clients * fraction)
    selected_clients = client_manager.sample(num_clients=num_clients_to_select, min_num_clients=min_available_clients)
    return selected_clients
