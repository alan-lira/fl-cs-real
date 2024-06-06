from typing import Union


def select_all_available_clients(available_clients_map: dict,
                                 phase: str) -> list:
    selected_clients = []
    for client_key, client_values in available_clients_map.items():
        client_proxy = client_values["client_proxy"]
        client_capacity = client_values["client_num_{0}ing_examples_available".format(phase)]
        selected_clients.append({"client_proxy": client_proxy,
                                 "client_capacity": client_capacity,
                                 "client_num_tasks_scheduled": 0})
    return selected_clients


def sum_clients_capacities(clients: Union[list, dict],
                           phase: str) -> int:
    clients_capacities_sum = 0
    if isinstance(clients, list):
        clients_capacities_sum = sum([client["client_capacity"] for client in clients])
    elif isinstance(clients, dict):
        clients_capacities_sum = sum([client_values["client_num_{0}ing_examples_available".format(phase)]
                                      for _, client_values in clients.items()])
    return clients_capacities_sum


def schedule_tasks_to_selected_clients(num_tasks_to_schedule: int,
                                       selected_clients: list) -> None:
    # If no clients were selected, the tasks can't be scheduled.
    if not selected_clients:
        return
    # While there are tasks left to schedule...
    while True:
        # Get the number of tasks already scheduled.
        num_tasks_scheduled = sum([selected_clients[sel_index]["client_num_tasks_scheduled"]
                                   for sel_index, _ in enumerate(selected_clients)])
        # Get the number of remaining tasks to schedule.
        remaining_tasks_to_schedule = num_tasks_to_schedule - num_tasks_scheduled
        # Get the clients with remaining capacity.
        clients_with_remaining_capacity = [selected_clients[index] for index, _ in enumerate(selected_clients)
                                           if selected_clients[index]["client_capacity"] -
                                           selected_clients[index]["client_num_tasks_scheduled"] > 0]
        num_clients_with_remaining_capacity = len(clients_with_remaining_capacity)
        # If no more clients left with remaining capacity, end.
        if num_clients_with_remaining_capacity == 0:
            break
        # If the number of remaining tasks to scheduler is lesser than the number of clients with remaining capacity...
        if remaining_tasks_to_schedule < num_clients_with_remaining_capacity:
            for rem_index, _ in enumerate(clients_with_remaining_capacity):
                client_proxy = clients_with_remaining_capacity[rem_index]["client_proxy"]
                client_num_tasks_to_schedule = 1
                for sel_index, _ in enumerate(selected_clients):
                    if selected_clients[sel_index]["client_proxy"] == client_proxy:
                        client_num_tasks_scheduled_before = selected_clients[sel_index]["client_num_tasks_scheduled"]
                        client_num_tasks_scheduled_then \
                            = client_num_tasks_scheduled_before + client_num_tasks_to_schedule
                        selected_clients[sel_index].update({"client_num_tasks_scheduled":
                                                            client_num_tasks_scheduled_then})
                remaining_tasks_to_schedule -= client_num_tasks_to_schedule
                # If no more tasks left to schedule, end.
                if remaining_tasks_to_schedule == 0:
                    break
        else:
            # Otherwise, schedule the remaining tasks as equal as possible to the clients with remaining capacity.
            num_tasks_per_client = remaining_tasks_to_schedule // num_clients_with_remaining_capacity
            for rem_index, _ in enumerate(clients_with_remaining_capacity):
                client_proxy = clients_with_remaining_capacity[rem_index]["client_proxy"]
                client_capacity = clients_with_remaining_capacity[rem_index]["client_capacity"]
                client_num_tasks_scheduled = clients_with_remaining_capacity[rem_index]["client_num_tasks_scheduled"]
                client_remaining_capacity = client_capacity - client_num_tasks_scheduled
                client_num_tasks_to_schedule = min(num_tasks_per_client, client_remaining_capacity)
                for sel_index, _ in enumerate(selected_clients):
                    if selected_clients[sel_index]["client_proxy"] == client_proxy:
                        client_num_tasks_scheduled_before = selected_clients[sel_index]["client_num_tasks_scheduled"]
                        client_num_tasks_scheduled_then \
                            = client_num_tasks_scheduled_before + client_num_tasks_to_schedule
                        selected_clients[sel_index].update({"client_num_tasks_scheduled":
                                                            client_num_tasks_scheduled_then})
        # Get the number of tasks already scheduled.
        num_tasks_scheduled = sum([selected_clients[sel_index]["client_num_tasks_scheduled"]
                                   for sel_index, _ in enumerate(selected_clients)])
        # Get the number of remaining tasks to schedule.
        remaining_tasks_to_schedule = num_tasks_to_schedule - num_tasks_scheduled
        # If no more tasks left to schedule, end.
        if remaining_tasks_to_schedule == 0:
            break


def map_available_participating_clients(comm_rounds: list,
                                        available_clients_map: dict,
                                        individual_metrics_history: dict) -> dict:
    # Initialize the available participating clients map.
    available_participating_clients_map = {}
    # Iterate through the list of communication rounds.
    for comm_round in comm_rounds:
        # Get the communication round's key.
        comm_round_key = "comm_round_{0}".format(comm_round)
        # Verify if there is an entry in the individual metrics history for the communication round.
        if comm_round_key in individual_metrics_history:
            # If so, get the individual metrics entry for the communication round.
            individual_metrics_entry_comm_round = individual_metrics_history[comm_round_key]
            # Iterate through the list of clients who participated on the communication round.
            for participating_client_dict in individual_metrics_entry_comm_round["clients_metrics_dicts"]:
                client_id_str = list(participating_client_dict.keys())[0]
                # If the participating client is available...
                if client_id_str in available_clients_map:
                    client_proxy = available_clients_map[client_id_str]["client_proxy"]
                    client_num_training_examples_available \
                        = available_clients_map[client_id_str]["client_num_training_examples_available"]
                    client_num_testing_examples_available \
                        = available_clients_map[client_id_str]["client_num_testing_examples_available"]
                    client_metrics = participating_client_dict.values()
                    # Verify if the available participating client has been mapped yet...
                    if client_id_str not in available_participating_clients_map:
                        # If not, append his information and his metrics of the current communication round to the map.
                        client_map = {"client_proxy": client_proxy,
                                      "client_num_training_examples_available": client_num_training_examples_available,
                                      "client_num_testing_examples_available": client_num_testing_examples_available,
                                      comm_round_key: client_metrics}
                        available_participating_clients_map.update({client_id_str: client_map})
                    else:
                        # If so, append his metrics of the current communication round to the map.
                        available_participating_clients_map[client_id_str].update({comm_round_key: client_metrics})
    return available_participating_clients_map


def get_metric_mean_value(individual_metrics_history: dict,
                          client_key: str,
                          num_examples_key: str,
                          num_examples_used: int,
                          metric_key: str) -> float:
    metric_mean_value = 0
    metric_values = []
    for comm_round_key, individual_metrics_entry_comm_round in individual_metrics_history.items():
        for participating_client_dict in individual_metrics_entry_comm_round["clients_metrics_dicts"]:
            if client_key in participating_client_dict:
                comm_round_num_examples_used = participating_client_dict[client_key][num_examples_key]
                if comm_round_num_examples_used == num_examples_used:
                    metric_value = participating_client_dict[client_key][metric_key]
                    if metric_value > 0:
                        metric_values.append(metric_value)
    if metric_values:
        metric_mean_value = sum(metric_values) / len(metric_values)
    return metric_mean_value


def calculate_linear_interpolation_or_extrapolation(x1: float,
                                                    x2: float,
                                                    y1: float,
                                                    y2: float,
                                                    x: float) -> float:
    # Calculate the slope m of the line.
    m = (y2 - y1) / (x2 - x1)
    # Calculate the value of y using the line equation.
    y = y1 + m * (x - x1)
    return y
