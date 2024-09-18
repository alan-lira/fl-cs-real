from bisect import bisect_left
from random import sample
from typing import Union


def select_all_available_clients(available_clients_map: dict,
                                 phase: str) -> list:
    selected_clients = []
    for client_key, client_values in available_clients_map.items():
        client_proxy = client_values["client_proxy"]
        client_task_assignment_capacities_phase_key = "client_task_assignment_capacities_{0}".format(phase)
        client_task_assignment_capacities_phase = client_values[client_task_assignment_capacities_phase_key]
        client_max_task_capacity = max(client_values["client_task_assignment_capacities_{0}".format(phase)])
        selected_clients.append({"client_proxy": client_proxy,
                                 client_task_assignment_capacities_phase_key: client_task_assignment_capacities_phase,
                                 "client_max_task_capacity": client_max_task_capacity,
                                 "client_num_tasks_scheduled": 0})
    return selected_clients


def sum_clients_max_task_capacities(clients: Union[list, dict],
                                    phase: str) -> int:
    clients_max_task_capacities_sum = 0
    if isinstance(clients, list):
        clients_max_task_capacities_sum = sum([client["client_max_task_capacity"] for client in clients])
    elif isinstance(clients, dict):
        clients_max_task_capacities_sum = sum([max(client_values["client_task_assignment_capacities_{0}".format(phase)])
                                               for _, client_values in clients.items()])
    return clients_max_task_capacities_sum


def take_closest(values: list,
                 value: int) -> int:
    value_idx = bisect_left(values, value)
    if value_idx == 0:
        return values[0]
    if value_idx == len(values):
        return values[-1]
    prev_idx = values[value_idx - 1]
    next_idx = values[value_idx]
    closest = next_idx if (next_idx - value < value - prev_idx) else prev_idx
    return closest


def schedule_tasks_to_selected_clients(num_tasks_to_schedule: int,
                                       selected_clients: list,
                                       phase: str) -> None:
    # If there are no tasks to schedule or selected clients, end.
    if num_tasks_to_schedule == 0 or not selected_clients:
        return
    # Schedule initially the tasks equally possible.
    tasks_per_client = num_tasks_to_schedule // len(selected_clients)
    # The tasks leftover will be scheduled to the first client.
    tasks_leftover = num_tasks_to_schedule % len(selected_clients)
    for client_idx, _ in enumerate(selected_clients):
        selected_clients[client_idx]["client_num_tasks_scheduled"] += tasks_per_client
    selected_clients[0]["client_num_tasks_scheduled"] += tasks_leftover
    # Validate the initial task assignment.
    for client_idx, _ in enumerate(selected_clients):
        # Get the current number of tasks scheduled for client i.
        client_num_tasks_scheduled = selected_clients[client_idx]["client_num_tasks_scheduled"]
        # Get the task assignment capacities of client i.
        client_task_assignment_capacities_phase_key = "client_task_assignment_capacities_{0}".format(phase)
        client_task_assignment_capacities_phase \
            = selected_clients[client_idx][client_task_assignment_capacities_phase_key]
        # Set a valid task assignment for client i.
        selected_clients[client_idx]["client_num_tasks_scheduled"] \
            = take_closest(client_task_assignment_capacities_phase, client_num_tasks_scheduled)
    # While there are tasks left to schedule...
    while True:
        # Get the current number of tasks assigned.
        num_tasks_scheduled = sum([selected_clients[client_idx]["client_num_tasks_scheduled"]
                                   for client_idx, _ in enumerate(selected_clients)])
        # Get the client indices that have at least one task scheduled.
        clients_indices = [client_idx for client_idx in range(0, len(selected_clients))
                           if selected_clients[client_idx]["client_num_tasks_scheduled"] > 0]
        if num_tasks_scheduled == num_tasks_to_schedule and len(clients_indices) == len(selected_clients):
            # All the tasks have been scheduled, and all clients have tasks assigned to them.
            break
        if num_tasks_scheduled > num_tasks_to_schedule:
            # Get the client indices that have at least one task scheduled.
            clients_indices = [client_idx for client_idx in range(0, len(selected_clients))
                               if selected_clients[client_idx]["client_num_tasks_scheduled"] > 0]
            # Randomly sample a client index.
            client_idx_sampled = sample(clients_indices, 1)[0]
            # Get the index of the current capacity used.
            client_task_assignment_capacities \
                = selected_clients[client_idx_sampled]["client_task_assignment_capacities_{0}".format(phase)]
            client_current_num_tasks_scheduled = selected_clients[client_idx_sampled]["client_num_tasks_scheduled"]
            cap_idx = list(client_task_assignment_capacities).index(client_current_num_tasks_scheduled)
            if cap_idx != 0:
                # Get the previous valid capacity of the client i.
                cap_prev = client_task_assignment_capacities[cap_idx - 1]
                # Set the assignment of client i to its previous valid capacity.
                selected_clients[client_idx_sampled]["client_num_tasks_scheduled"] = cap_prev
        else:
            # Get the client indices that have any number of tasks scheduled.
            clients_indices = [client_idx for client_idx in range(0, len(selected_clients))]
            # Randomly sample a client index.
            client_idx_sampled = sample(clients_indices, 1)[0]
            # Get the index of the current capacity used.
            client_task_assignment_capacities \
                = selected_clients[client_idx_sampled]["client_task_assignment_capacities_{0}".format(phase)]
            client_current_num_tasks_scheduled = selected_clients[client_idx_sampled]["client_num_tasks_scheduled"]
            cap_idx = list(client_task_assignment_capacities).index(client_current_num_tasks_scheduled)
            if cap_idx != len(client_task_assignment_capacities) - 1:
                # Get the next valid capacity of the client i.
                cap_next = client_task_assignment_capacities[cap_idx + 1]
                # Set the assignment of client i to its next valid capacity.
                selected_clients[client_idx_sampled]["client_num_tasks_scheduled"] = cap_next


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
                    client_task_assignment_capacities_train \
                        = available_clients_map[client_id_str]["client_task_assignment_capacities_train"]
                    client_task_assignment_capacities_test \
                        = available_clients_map[client_id_str]["client_task_assignment_capacities_test"]
                    client_metrics = participating_client_dict.values()
                    # Verify if the available participating client has been mapped yet...
                    if client_id_str not in available_participating_clients_map:
                        # If not, append his information and his metrics of the current communication round to the map.
                        client_map = {"client_proxy": client_proxy,
                                      "client_num_training_examples_available": client_num_training_examples_available,
                                      "client_num_testing_examples_available": client_num_testing_examples_available,
                                      "client_task_assignment_capacities_train": client_task_assignment_capacities_train,
                                      "client_task_assignment_capacities_test": client_task_assignment_capacities_test,
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
