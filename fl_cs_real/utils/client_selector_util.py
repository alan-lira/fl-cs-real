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


def get_all_possible_sums(lists: list) -> list:
    # Initialize with the sum of an empty combination (0).
    possible_sums = {0}
    for lst in lists:
        # Create a new set to store the updated sums.
        new_sums = set()
        # For each sum already in possible_sums, add each element of the current list.
        for value in lst:
            for s in possible_sums:
                new_sums.add(s + value)
        # Update possible_sums to include the new sums.
        possible_sums = new_sums
    # Return the sorted list of all possible sums.
    return sorted(possible_sums)


def schedule_minimum_tasks_to_all_clients(selected_clients: list,
                                          phase: str) -> None:
    while True:
        # Get the client indices that have no tasks scheduled.
        clients_indices = [client_idx for client_idx in range(0, len(selected_clients))
                           if selected_clients[client_idx]["client_num_tasks_scheduled"] == 0]
        if not clients_indices:
            # All clients received a minimum number of tasks.
            break
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


def deep_copy_selected_client(selected_client_value: dict | list) -> dict | list:
    if isinstance(selected_client_value, dict):
        # Recursively copy dict items.
        return {k: deep_copy_selected_client(v) for k, v in selected_client_value.items()}
    elif isinstance(selected_client_value, list):
        # Recursively copy list items.
        return [deep_copy_selected_client(i) for i in selected_client_value]
    else:
        # Return the value itself for immutable types (int, str).
        return selected_client_value


def find_different_schedule(selected_clients: list,
                            phase: str,
                            previous_schedules_to_avoid: set) -> list:
    # Deep copy the current list of selected clients.
    selected_clients_copy = [deep_copy_selected_client(selected_client) for selected_client in selected_clients]
    # For each selected client i ...
    for client_i, _ in enumerate(selected_clients_copy):
        # Search for the clients j that can swap their scheduled number of tasks with client i.
        swap_clients_indices = [client_j for client_j, _ in enumerate(selected_clients_copy)
                                if selected_clients_copy[client_i]["client_num_tasks_scheduled"] in
                                selected_clients[client_j]["client_task_assignment_capacities_{0}".format(phase)]
                                and selected_clients_copy[client_j]["client_num_tasks_scheduled"] in
                                selected_clients[client_i]["client_task_assignment_capacities_{0}".format(phase)]
                                and client_i != client_j]
        if swap_clients_indices:
            # Randomly sample a client index.
            client_j_sampled = sample(swap_clients_indices, 1)[0]
            # Get the currently number of tasks scheduled for clients i and j.
            client_i_num_tasks_scheduled = selected_clients_copy[client_i]["client_num_tasks_scheduled"]
            client_j_num_tasks_scheduled = selected_clients_copy[client_j_sampled]["client_num_tasks_scheduled"]
            # Perform the swap of tasks.
            selected_clients_copy[client_i]["client_num_tasks_scheduled"] = client_j_num_tasks_scheduled
            selected_clients_copy[client_j_sampled]["client_num_tasks_scheduled"] = client_i_num_tasks_scheduled
    # Get the generated different schedule.
    different_schedule = [selected_clients_copy[client_idx]["client_num_tasks_scheduled"]
                          for client_idx, _ in enumerate(selected_clients_copy)]
    # Verify if the scheduled is really different from the previous ones...
    if tuple(different_schedule) not in previous_schedules_to_avoid:
        return selected_clients_copy
    return selected_clients


def remove_tasks_from_a_random_client(selected_clients: list,
                                      phase: str,
                                      num_tasks_to_remove: int = 0,
                                      schedule_to_all_clients: bool = False) -> int:
    # Get the client indices that have at least one task scheduled.
    clients_indices = [client_idx for client_idx in range(0, len(selected_clients))
                       if selected_clients[client_idx]["client_num_tasks_scheduled"] > 0]
    if clients_indices:
        # Randomly sample a client index.
        client_idx_sampled = sample(clients_indices, 1)[0]
        # Get the index of the current capacity used.
        client_task_assignment_capacities \
            = selected_clients[client_idx_sampled]["client_task_assignment_capacities_{0}".format(phase)]
        client_previous_num_tasks_scheduled = selected_clients[client_idx_sampled]["client_num_tasks_scheduled"]
        cap_idx = list(client_task_assignment_capacities).index(client_previous_num_tasks_scheduled)
        if num_tasks_to_remove > 0:
            # Calculate the new number of tasks after the removal.
            num_tasks_after_removal = client_previous_num_tasks_scheduled - num_tasks_to_remove
            if (num_tasks_after_removal in client_task_assignment_capacities) and \
               ((not schedule_to_all_clients) or (schedule_to_all_clients and num_tasks_after_removal > 0)):
                # Set the assignment of client i to a previous valid capacity.
                selected_clients[client_idx_sampled]["client_num_tasks_scheduled"] = num_tasks_after_removal
        else:
            if (not schedule_to_all_clients and cap_idx != 0) or (schedule_to_all_clients and cap_idx > 1):
                # Get the previous valid capacity of the client i.
                cap_prev = client_task_assignment_capacities[cap_idx - 1]
                # Set the assignment of client i to its previous valid capacity.
                selected_clients[client_idx_sampled]["client_num_tasks_scheduled"] = cap_prev
        client_current_num_tasks_scheduled = selected_clients[client_idx_sampled]["client_num_tasks_scheduled"]
        num_tasks_removed = client_previous_num_tasks_scheduled - client_current_num_tasks_scheduled
        return num_tasks_removed
    return 0


def add_tasks_to_a_random_client(selected_clients: list,
                                 phase: str,
                                 num_tasks_to_add: int = 0) -> int:
    # Get the client indices that have any number of tasks scheduled.
    clients_indices = [client_idx for client_idx in range(0, len(selected_clients))]
    if clients_indices:
        # Randomly sample a client index.
        client_idx_sampled = sample(clients_indices, 1)[0]
        # Get the index of the current capacity used.
        client_task_assignment_capacities \
            = selected_clients[client_idx_sampled]["client_task_assignment_capacities_{0}".format(phase)]
        client_previous_num_tasks_scheduled = selected_clients[client_idx_sampled]["client_num_tasks_scheduled"]
        cap_idx = list(client_task_assignment_capacities).index(client_previous_num_tasks_scheduled)
        if num_tasks_to_add > 0:
            # Calculate the new number of tasks after the addition.
            num_tasks_after_addition = client_previous_num_tasks_scheduled + num_tasks_to_add
            if num_tasks_after_addition in client_task_assignment_capacities:
                # Set the assignment of client i to a next valid capacity.
                selected_clients[client_idx_sampled]["client_num_tasks_scheduled"] = num_tasks_after_addition
        else:
            if cap_idx != len(client_task_assignment_capacities) - 1:
                # Get the next valid capacity of the client i.
                cap_next = client_task_assignment_capacities[cap_idx + 1]
                # Set the assignment of client i to its next valid capacity.
                selected_clients[client_idx_sampled]["client_num_tasks_scheduled"] = cap_next
        client_current_num_tasks_scheduled = selected_clients[client_idx_sampled]["client_num_tasks_scheduled"]
        num_tasks_added = client_current_num_tasks_scheduled - client_previous_num_tasks_scheduled
        return num_tasks_added
    return 0


def schedule_tasks_to_selected_clients(num_tasks_to_schedule: int,
                                       selected_clients: list,
                                       phase: str,
                                       profiling_round: bool,
                                       previous_schedules_to_avoid: set | None = None,
                                       schedule_to_all_clients: bool = False) -> list:
    # If there are no tasks to schedule or selected clients, end.
    if num_tasks_to_schedule == 0 or not selected_clients:
        return selected_clients
    # Initialize the list of task assignment capacities per client.
    task_assignment_capacities_list = []
    for client_idx, _ in enumerate(selected_clients):
        # Get the task assignment capacities of client i.
        client_task_assignment_capacities_phase_key = "client_task_assignment_capacities_{0}".format(phase)
        client_task_assignment_capacities_phase \
            = selected_clients[client_idx][client_task_assignment_capacities_phase_key]
        # Append the task assignment capacities of client i to the list of task assignment capacities.
        task_assignment_capacities_list.append(client_task_assignment_capacities_phase)
    # Compute all the possible sums of task assignments, considering one assignment per client.
    all_possible_task_assignment_sums = get_all_possible_sums(task_assignment_capacities_list)
    # If the number of tasks to schedule is infeasible...
    if num_tasks_to_schedule not in all_possible_task_assignment_sums:
        # Set a new valid number of tasks to schedule.
        num_tasks_to_schedule = take_closest(all_possible_task_assignment_sums, num_tasks_to_schedule)
    # If is a profile round or all clients must have tasks, initially schedule a minimum number of tasks to all clients.
    if profiling_round or schedule_to_all_clients:
        schedule_minimum_tasks_to_all_clients(selected_clients, phase)
    # While there are tasks left to schedule...
    remove_action = True
    while True:
        # Initialize the variable used to invalidate the current schedule.
        invalidate_current_schedule = False
        # If is a profile round and there are previous schedules to be avoided...
        if profiling_round and previous_schedules_to_avoid:
            # Find a different schedule (clients should ideally have different schedules during profiling).
            selected_clients = find_different_schedule(selected_clients,
                                                       phase,
                                                       previous_schedules_to_avoid)
        # If all clients must receive tasks, verify if this constraint was met.
        if schedule_to_all_clients:
            all_clients_have_tasks = all(selected_clients[client_idx]["client_num_tasks_scheduled"] > 0
                                         for client_idx in range(0, len(selected_clients)))
            if not all_clients_have_tasks:
                invalidate_current_schedule = True
        # Invalidate the current schedule, if needed.
        if invalidate_current_schedule:
            if remove_action:
                # Remove tasks from a random client.
                remove_tasks_from_a_random_client(selected_clients, phase, schedule_to_all_clients=schedule_to_all_clients)
            else:
                # Add tasks to a random client.
                add_tasks_to_a_random_client(selected_clients, phase)
            remove_action = not remove_action
        # Get the current schedule.
        current_schedule = [selected_clients[client_idx]["client_num_tasks_scheduled"]
                            for client_idx, _ in enumerate(selected_clients)]
        # Get the current number of tasks assigned.
        num_tasks_scheduled = sum(current_schedule)
        # Verify if all the tasks have been scheduled...
        if not invalidate_current_schedule and num_tasks_scheduled == num_tasks_to_schedule:
            # If so, filter out the clients with no tasks scheduled, if any.
            selected_clients_filtered = []
            for client_idx in range(0, len(selected_clients)):
                if selected_clients[client_idx]["client_num_tasks_scheduled"] > 0:
                    selected_clients_filtered.append(selected_clients[client_idx])
            return selected_clients_filtered
        if num_tasks_scheduled > num_tasks_to_schedule:
            # Remove tasks from a random client.
            remove_tasks_from_a_random_client(selected_clients, phase, schedule_to_all_clients=schedule_to_all_clients)
        else:
            # Add tasks to a random client.
            add_tasks_to_a_random_client(selected_clients, phase)


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


def calculate_linear_interpolation_or_extrapolation(x1: int | float,
                                                    x2: int | float,
                                                    y1: int | float,
                                                    y2: int | float,
                                                    x: int | float) -> int | float:
    # Calculate the slope m of the line.
    m = (y2 - y1) / (x2 - x1)
    # Calculate the value of y using the line equation.
    y = y1 + m * (x - x1)
    return y
