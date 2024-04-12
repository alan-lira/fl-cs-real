from logging import Logger
from numpy import array, inf

from goffls.task_scheduler.mec import mec
from goffls.utils.logger_util import log_message


def _select_all_available_clients(available_clients_map: dict,
                                  phase: str) -> list:
    selected_clients = []
    for client_key, client_values in available_clients_map.items():
        client_proxy = client_values["client_proxy"]
        client_capacity = 0
        if phase == "train":
            client_capacity = client_values["num_training_examples_available"]
        if phase == "test":
            client_capacity = client_values["num_testing_examples_available"]
        selected_clients.append({"client_proxy": client_proxy,
                                 "client_capacity": client_capacity,
                                 "client_num_tasks_scheduled": 0})
    return selected_clients


def _sum_selected_clients_capacities(selected_clients: list) -> int:
    selected_clients_capacities_sum = sum([client["client_capacity"] for client in selected_clients])
    return selected_clients_capacities_sum


def _schedule_tasks_to_selected_clients(num_tasks_to_schedule: int,
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
        # Schedule the remaining tasks as equal as possible to the clients with remaining capacity.
        num_tasks_per_client = remaining_tasks_to_schedule // len(clients_with_remaining_capacity)
        for rem_index, _ in enumerate(clients_with_remaining_capacity):
            client_proxy = clients_with_remaining_capacity[rem_index]["client_proxy"]
            client_capacity = clients_with_remaining_capacity[rem_index]["client_capacity"]
            client_num_tasks_scheduled = clients_with_remaining_capacity[rem_index]["client_num_tasks_scheduled"]
            client_remaining_capacity = client_capacity - client_num_tasks_scheduled
            client_num_tasks_to_schedule = min(num_tasks_per_client, client_remaining_capacity)
            for sel_index, _ in enumerate(selected_clients):
                if selected_clients[sel_index]["client_proxy"] == client_proxy:
                    client_num_tasks_scheduled_before = selected_clients[sel_index]["client_num_tasks_scheduled"]
                    client_num_tasks_scheduled_then = client_num_tasks_scheduled_before + client_num_tasks_to_schedule
                    selected_clients[sel_index].update({"client_num_tasks_scheduled": client_num_tasks_scheduled_then})
        # Get the number of tasks already scheduled.
        num_tasks_scheduled = sum([selected_clients[sel_index]["client_num_tasks_scheduled"]
                                   for sel_index, _ in enumerate(selected_clients)])
        # Get the number of remaining tasks to schedule.
        remaining_tasks_to_schedule = num_tasks_to_schedule - num_tasks_scheduled
        # If no more tasks left to schedule, end.
        if remaining_tasks_to_schedule == 0:
            break


def _map_available_participating_clients(comm_rounds: list,
                                         available_clients_map: dict,
                                         individual_metrics_history: dict) -> dict:
    # Initialize the available participating clients map.
    available_participating_clients_map = {}
    # Iterate through the list of communication rounds.
    for comm_round in comm_rounds:
        # Get the communication round key.
        comm_round_key = "comm_round_{0}".format(comm_round)
        # Verify if there is an entry in the individual metrics history for the communication round.
        if comm_round_key in individual_metrics_history:
            # If so, get the individual metrics entry for the communication round.
            individual_metrics_entry_comm_round = individual_metrics_history[comm_round_key]
            # Iterate through the list of clients who participated on the communication round.
            for participating_client_dict in individual_metrics_entry_comm_round:
                client_id_str = list(participating_client_dict.keys())[0]
                # If the participating client is available...
                if client_id_str in available_clients_map:
                    client_proxy = available_clients_map[client_id_str]["client_proxy"]
                    num_training_examples_available \
                        = available_clients_map[client_id_str]["num_training_examples_available"]
                    num_testing_examples_available \
                        = available_clients_map[client_id_str]["num_testing_examples_available"]
                    client_metrics = participating_client_dict.values()
                    client_map = {"client_proxy": client_proxy,
                                  "num_training_examples_available": num_training_examples_available,
                                  "num_testing_examples_available": num_testing_examples_available,
                                  comm_round_key: client_metrics}
                    # Append him to the available participating clients map.
                    available_participating_clients_map.update({client_id_str: client_map})
    return available_participating_clients_map


def select_clients_using_mec(comm_round: int,
                             phase: str,
                             num_tasks_to_schedule: int,
                             available_clients_map: dict,
                             individual_metrics_history: dict,
                             history_check_approach: str,
                             enable_complementary_selection: bool,
                             complementary_selection_settings: dict,
                             logger: Logger) -> list:
    # Log a 'selecting clients' message.
    message = "Selecting {0}ing clients using MEC...".format(phase)
    log_message(logger, message, "INFO")
    # Initialize the list of selected clients.
    selected_clients = []
    # Verify if there are any entries in the individual metrics history.
    if not individual_metrics_history:
        # If not, this means it is the first communication round. Therefore, all available clients will be selected.
        selected_all_available_clients = _select_all_available_clients(available_clients_map, phase)
        # Get the maximum number of tasks that can be scheduled to the selected clients.
        selected_clients_capacities_sum = _sum_selected_clients_capacities(selected_all_available_clients)
        # Redefine the number of tasks to schedule, if the selected clients capacities sum is lower.
        num_tasks_to_schedule = min(num_tasks_to_schedule, selected_clients_capacities_sum)
        # Schedule the tasks to the selected clients.
        _schedule_tasks_to_selected_clients(num_tasks_to_schedule, selected_all_available_clients)
        # Append the selected clients into the selected clients list.
        selected_clients.extend(selected_all_available_clients)
    else:
        # Otherwise, the clients will be selected according to their entries in the individual metrics history.
        comm_rounds = []
        if history_check_approach == "ImmediatelyPreviousRound":
            # Consider only available clients who participated on the immediately previous round.
            comm_rounds = [comm_round - 1]
        elif history_check_approach == "AnyPreviousRound":
            # Consider only available clients who participated on any of the previous rounds.
            comm_rounds = list(range(1, comm_round))
        # Load the available participating clients map.
        available_participating_clients_map = _map_available_participating_clients(comm_rounds,
                                                                                   available_clients_map,
                                                                                   individual_metrics_history)
        # Set the number of resources,
        # based on the number of available clients with entries in the individual metrics history.
        num_resources = len(available_participating_clients_map)
        # Initialize the global lists that will be transformed to array.
        client_ids = []
        assignment_capacities = []
        time_costs = []
        energy_costs = []
        # For each available client that has entries in the individual metrics history...
        for client_key, client_values in available_participating_clients_map.items():
            # Initialize his assignment capacities list, based on his number of examples available.
            client_assignment_capacity = 0
            if phase == "train":
                client_assignment_capacity = client_values["num_training_examples_available"]
            if phase == "test":
                client_assignment_capacity = client_values["num_testing_examples_available"]
            assignment_capacities_client = [i for i in range(0, client_assignment_capacity+1)]
            # Initialize his costs lists, based on the number of tasks (examples) to be scheduled.
            time_costs_client = [inf for _ in range(0, num_tasks_to_schedule+1)]
            energy_costs_client = [inf for _ in range(0, num_tasks_to_schedule+1)]
            # Get his individual metrics history entries...
            individual_metrics_history_entries = [value
                                                  for key, value in client_values.items() if "comm_round_" in key][0]
            for individual_metrics_history_entry in individual_metrics_history_entries:
                # Get the number of examples used by him.
                num_examples_key = "num_{0}ing_examples".format(phase)
                num_examples = individual_metrics_history_entry[num_examples_key]
                # Get the time spent by him (if available).
                time_key = "{0}ing_time".format(phase)
                if time_key in individual_metrics_history_entry:
                    # Update his time costs list for this number of examples.
                    time_cost = individual_metrics_history_entry[time_key]
                    time_costs_client[num_examples] = time_cost
                # Get the energy consumed by his CPU (if available).
                energy_cpu_key = "{0}ing_energy_cpu".format(phase)
                energy_cpu_cost = 0
                if energy_cpu_key in individual_metrics_history_entry:
                    energy_cpu_cost = individual_metrics_history_entry[energy_cpu_key]
                # Get the energy consumed by his NVIDIA GPU (if available).
                energy_nvidia_gpu_key = "{0}ing_energy_nvidia_gpu".format(phase)
                energy_nvidia_gpu_cost = 0
                if energy_nvidia_gpu_key in individual_metrics_history_entry:
                    energy_nvidia_gpu_cost = individual_metrics_history_entry[energy_nvidia_gpu_key]
                # Update his energy costs list for this number of examples.
                energy_costs_client[num_examples] = energy_cpu_cost + energy_nvidia_gpu_cost
            # Append his lists into the global lists.
            client_ids.append(client_key)
            assignment_capacities.append(assignment_capacities_client)
            time_costs.append(time_costs_client)
            energy_costs.append(energy_costs_client)
        # Convert the global lists into Numpy arrays.
        assignment_capacities = array(assignment_capacities, dtype=object)
        time_costs = array(time_costs, dtype=object)
        energy_costs = array(energy_costs, dtype=object)
        # Execute the MEC algorithm.
        optimal_schedule, minimal_makespan, minimal_energy_consumption = mec(num_resources,
                                                                             num_tasks_to_schedule,
                                                                             assignment_capacities,
                                                                             time_costs,
                                                                             energy_costs)
        # Log the MEC algorithm's result.
        message = "X*: {0}\nMinimal makespan (Cₘₐₓ): {1}\nMinimal energy consumption (ΣE): {2}" \
                  .format(optimal_schedule, minimal_makespan, minimal_energy_consumption)
        log_message(logger, message, "DEBUG")
        # Get the list of indices of the selected clients.
        selected_clients_indices = [sel_index for sel_index, client_num_tasks_scheduled
                                    in enumerate(optimal_schedule) if client_num_tasks_scheduled > 0]
        # Append their corresponding proxies objects and numbers of tasks scheduled into the selected clients list.
        for sel_index in selected_clients_indices:
            client_id_str = client_ids[sel_index]
            client_proxy = available_participating_clients_map[client_id_str]["client_proxy"]
            client_capacity = 0
            if phase == "train":
                client_capacity = available_participating_clients_map[client_id_str]["num_training_examples_available"]
            if phase == "test":
                client_capacity = available_participating_clients_map[client_id_str]["num_testing_examples_available"]
            client_num_tasks_scheduled = int(optimal_schedule[sel_index])
            selected_clients.append({"client_proxy": client_proxy,
                                     "client_capacity": client_capacity,
                                     "client_num_tasks_scheduled": client_num_tasks_scheduled})
    # Log a 'number of clients selected' message.
    message = "{0} {1} selected!".format(len(selected_clients),
                                         "clients were" if len(selected_clients) != 1 else "client was")
    log_message(logger, message, "INFO")
    return selected_clients
