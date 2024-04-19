from logging import Logger
from numpy import array, inf

from goffls.task_scheduler.ecmtc import ecmtc
from goffls.util.logger_util import log_message


def _select_all_available_clients(available_clients_map: dict,
                                  phase: str) -> list:
    selected_clients = []
    for client_key, client_values in available_clients_map.items():
        client_proxy = client_values["client_proxy"]
        client_capacity = client_values["num_{0}ing_examples_available".format(phase)]
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
                    # Verify if the available participating client has been mapped yet...
                    if client_id_str not in available_participating_clients_map:
                        # If not, append his information and his metrics of the current communication round to the map.
                        client_map = {"client_proxy": client_proxy,
                                      "num_training_examples_available": num_training_examples_available,
                                      "num_testing_examples_available": num_testing_examples_available,
                                      comm_round_key: client_metrics}
                        available_participating_clients_map.update({client_id_str: client_map})
                    else:
                        # If so, just append his metrics of the current communication round to the map.
                        available_participating_clients_map[client_id_str].update({comm_round_key: client_metrics})
    return available_participating_clients_map


def _calculate_linear_interpolation_or_extrapolation(x1: float,
                                                     x2: float,
                                                     y1: float,
                                                     y2: float,
                                                     x: float) -> float:
    # Calculate the slope m of the line.
    m = (y2 - y1) / (x2 - x1)
    # Calculate the value of y using the line equation.
    y = y1 + m * (x - x1)
    return y


def select_clients_using_ecmtc(comm_round: int,
                               phase: str,
                               num_tasks_to_schedule: int,
                               deadline_in_seconds: float,
                               available_clients_map: dict,
                               individual_metrics_history: dict,
                               history_checker: str,
                               assignment_capacities_init_settings: dict,
                               logger: Logger) -> list:
    # Log a 'selecting clients' message.
    message = "Selecting {0}ing clients using ECMTC...".format(phase)
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
        # Otherwise, the available clients will be selected considering their entries in the individual metrics history.
        comm_rounds = []
        if history_checker == "Only_Immediately_Previous_Round":
            # Check the immediately previous round's history only.
            comm_rounds = [comm_round - 1]
        elif history_checker == "All_Previous_Rounds":
            # Check all the previous rounds' history.
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
            # Initialize his costs lists, based on the number of tasks (examples) to be scheduled.
            time_costs_client = [inf for _ in range(0, num_tasks_to_schedule+1)]
            energy_costs_client = [inf for _ in range(0, num_tasks_to_schedule+1)]
            # Get his individual metrics history entries...
            individual_metrics_history_entries = [list(comm_round_metrics)[0]
                                                  for key, comm_round_metrics in client_values.items()
                                                  if "comm_round_" in key]
            for individual_metrics_history_entry in individual_metrics_history_entries:
                # Get the number of examples used by him.
                num_examples_key = "num_{0}ing_examples_used".format(phase)
                num_examples = individual_metrics_history_entry[num_examples_key]
                # Get the time spent by him (if available).
                time_key = "{0}ing_time".format(phase)
                if time_key in individual_metrics_history_entry:
                    # Update his time costs list for this number of examples.
                    time_cost = individual_metrics_history_entry[time_key]
                    time_costs_client[num_examples] = time_cost
                energy_cost = 0
                # Get the energy consumed by his CPU (if available).
                energy_cpu_key = "{0}ing_energy_cpu".format(phase)
                if energy_cpu_key in individual_metrics_history_entry:
                    energy_cpu_cost = individual_metrics_history_entry[energy_cpu_key]
                    if energy_cpu_cost > 0:
                        energy_cost += energy_cpu_cost
                    else:
                        # TODO: Take the mean CPU energy cost for this number of examples, if available.
                        pass
                # Get the energy consumed by his NVIDIA GPU (if available).
                energy_nvidia_gpu_key = "{0}ing_energy_nvidia_gpu".format(phase)
                if energy_nvidia_gpu_key in individual_metrics_history_entry:
                    energy_nvidia_gpu_cost = individual_metrics_history_entry[energy_nvidia_gpu_key]
                    if energy_nvidia_gpu_cost > 0:
                        energy_cost += energy_nvidia_gpu_cost
                    else:
                        # TODO: Take the mean NVIDIA GPU energy cost for this number of examples, if available.
                        pass
                # If no valid energy costs were found, set the energy cost as infinity.
                if energy_cost == 0:
                    energy_cost = inf
                # Update his energy costs list for this number of examples.
                energy_costs_client[num_examples] = energy_cost
            # Initialize his assignment capacities list...
            assignment_capacities_client = None
            assignment_capacities_initializer = assignment_capacities_init_settings["assignment_capacities_initializer"]
            if assignment_capacities_initializer == "Only_Previous_Num_Tasks_Assigned_Set":
                # Based only on his previous round(s) participation, i.e., the set of previously numbers of tasks
                # assigned to him.
                previous_num_tasks_assigned = [list(comm_round_metrics)[0]["num_{0}ing_examples_used".format(phase)]
                                               for key, comm_round_metrics in client_values.items()
                                               if "comm_round_" in key]
                previous_num_tasks_assigned_set = list(set(previous_num_tasks_assigned))
                assignment_capacities_client = previous_num_tasks_assigned_set
            elif assignment_capacities_initializer == "Custom_Range_Set_Union_Previous_Num_Tasks_Assigned_Set":
                # Based on a custom range set (ordered in ascending order), which also includes his previous round(s)
                # participation, i.e., the set of previously numbers of tasks assigned to him.
                lower_bound = assignment_capacities_init_settings["lower_bound"]
                upper_bound = assignment_capacities_init_settings["upper_bound"]
                if upper_bound == "client_capacity":
                    upper_bound = client_values["num_{0}ing_examples_available".format(phase)]
                step = assignment_capacities_init_settings["step"]
                custom_range = list(range(lower_bound, min(upper_bound+1, num_tasks_to_schedule+1), step))
                previous_num_tasks_assigned = [list(comm_round_metrics)[0]["num_{0}ing_examples_used".format(phase)]
                                               for key, comm_round_metrics in client_values.items()
                                               if "comm_round_" in key]
                previous_num_tasks_assigned_set = list(set(previous_num_tasks_assigned))
                custom_range.extend(previous_num_tasks_assigned_set)
                custom_range_set = list(set(custom_range))
                custom_range_set_sorted = sorted(custom_range_set)
                assignment_capacities_client = custom_range_set_sorted
                # Set the costs of zero tasks scheduled, allowing the data point (x=0, y=0) to be used during the
                # estimation of costs for the unknown values belonging to the custom range.
                time_costs_client[0] = 0
                energy_costs_client[0] = 0
                previous_num_tasks_assigned.append(0)
                # Estimates the costs for the unknown values via linear interpolation/extrapolation.
                for assignment_capacity in assignment_capacities_client:
                    if assignment_capacity not in previous_num_tasks_assigned:
                        # Determine x1 and x2, which are two known values of previously numbers of tasks assigned.
                        x1_candidates = [i for i in previous_num_tasks_assigned if i < assignment_capacity]
                        x2_candidates = [i for i in previous_num_tasks_assigned if i > assignment_capacity]
                        if x2_candidates:
                            # Interpolation.
                            x1 = x1_candidates[-1]
                            x2 = x2_candidates[0]
                        else:
                            # Extrapolation.
                            x1 = x1_candidates[-2]
                            x2 = x1_candidates[-1]
                        # Calculate the linear interpolation/extrapolation for the time cost.
                        y1_time = time_costs_client[x1]
                        y2_time = time_costs_client[x2]
                        time_cost_estimation = _calculate_linear_interpolation_or_extrapolation(x1,
                                                                                                x2,
                                                                                                y1_time,
                                                                                                y2_time,
                                                                                                assignment_capacity)
                        # Calculate the linear interpolation/extrapolation for the energy cost.
                        y1_energy = energy_costs_client[x1]
                        y2_energy = energy_costs_client[x2]
                        energy_cost_estimation = _calculate_linear_interpolation_or_extrapolation(x1,
                                                                                                  x2,
                                                                                                  y1_energy,
                                                                                                  y2_energy,
                                                                                                  assignment_capacity)
                        # Update the costs lists with the estimated values.
                        time_costs_client[assignment_capacity] = time_cost_estimation
                        energy_costs_client[assignment_capacity] = energy_cost_estimation
            # Append his lists into the global lists.
            client_ids.append(client_key)
            assignment_capacities.append(assignment_capacities_client)
            time_costs.append(time_costs_client)
            energy_costs.append(energy_costs_client)
        # Convert the global lists into Numpy arrays.
        assignment_capacities = array(assignment_capacities, dtype=object)
        time_costs = array(time_costs, dtype=object)
        energy_costs = array(energy_costs, dtype=object)
        # Execute the ECMTC algorithm.
        optimal_schedule, minimal_energy_consumption, minimal_makespan = ecmtc(num_resources,
                                                                               num_tasks_to_schedule,
                                                                               assignment_capacities,
                                                                               time_costs,
                                                                               energy_costs,
                                                                               deadline_in_seconds)
        # Log the ECMTC algorithm's result.
        message = "X*: {0}\nMinimal makespan (Cₘₐₓ): {1}\nMinimal energy consumption (ΣE): {2}" \
                  .format(optimal_schedule, minimal_makespan, minimal_energy_consumption)
        log_message(logger, message, "DEBUG")
        # Get the list of indices of the selected clients.
        selected_clients_indices = [sel_index for sel_index, client_num_tasks_scheduled
                                    in enumerate(optimal_schedule) if client_num_tasks_scheduled > 0]
        # Append their corresponding proxies objects and numbers of tasks scheduled into the selected clients list.
        for sel_index in selected_clients_indices:
            client_id_str = client_ids[sel_index]
            client_map = available_participating_clients_map[client_id_str]
            client_proxy = client_map["client_proxy"]
            client_capacity = client_map["num_{0}ing_examples_available".format(phase)]
            client_num_tasks_scheduled = int(optimal_schedule[sel_index])
            selected_clients.append({"client_proxy": client_proxy,
                                     "client_capacity": client_capacity,
                                     "client_num_tasks_scheduled": client_num_tasks_scheduled})
    # Log a 'number of clients selected' message.
    message = "{0} {1} selected!".format(len(selected_clients),
                                         "clients were" if len(selected_clients) != 1 else "client was")
    log_message(logger, message, "INFO")
    return selected_clients
