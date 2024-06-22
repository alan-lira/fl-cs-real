from logging import Logger
from numpy import array, inf
from random import sample

from goffls.task_scheduler.ecmtc import ecmtc
from goffls.utils.client_selector_util import calculate_linear_interpolation_or_extrapolation, get_metric_mean_value, \
    map_available_participating_clients, schedule_tasks_to_selected_clients, select_all_available_clients, \
    sum_clients_capacities
from goffls.utils.logger_util import log_message


def select_clients_using_ecmtc(comm_round: int,
                               phase: str,
                               num_tasks_to_schedule: int,
                               deadline_in_seconds: float,
                               available_clients_map: dict,
                               individual_metrics_history: dict,
                               history_checker: str,
                               assignment_capacities_init_settings: dict,
                               complementary_clients_fraction: float,
                               complementary_tasks_fraction: float,
                               logger: Logger) -> dict:
    # Log a 'selecting clients' message.
    message = "Selecting {0}ing clients using ECMTC...".format(phase)
    log_message(logger, message, "INFO")
    # Initialize the selection dictionary and the list of selected clients.
    selection = {}
    selected_clients = []
    # Verify if there are any entries in the individual metrics history.
    if not individual_metrics_history:
        # If not, this means it is the first communication round. Therefore, all available clients will be selected.
        selected_all_available_clients = select_all_available_clients(available_clients_map, phase)
        # Get the maximum number of tasks that can be scheduled to the selected clients.
        selected_clients_capacities_sum = sum_clients_capacities(selected_all_available_clients, phase)
        # Redefine the number of tasks to schedule, if the selected clients capacities sum is lower.
        num_tasks_to_schedule = min(num_tasks_to_schedule, selected_clients_capacities_sum)
        # Schedule the tasks to the selected clients.
        schedule_tasks_to_selected_clients(num_tasks_to_schedule, selected_all_available_clients)
        # Append the selected clients into the selected clients list.
        selected_clients.extend(selected_all_available_clients)
        # Update the selection dictionary with the selected clients for the schedule.
        selection.update({"selected_clients": selected_clients})
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
        available_participating_clients_map = map_available_participating_clients(comm_rounds,
                                                                                  available_clients_map,
                                                                                  individual_metrics_history)
        # Set the number of resources,
        # based on the number of available clients with entries in the individual metrics history.
        num_resources = len(available_participating_clients_map)
        # Get the maximum number of tasks that can be scheduled to the available participating clients.
        available_participating_clients_capacities_sum = sum_clients_capacities(available_participating_clients_map,
                                                                                phase)
        # Set the number of tasks that will be scheduled to the complementary clients (if any).
        num_complementary_tasks_to_schedule = int(num_tasks_to_schedule * complementary_tasks_fraction)
        # If others than the clients selected by the ECMTC algorithm are to be used (i.e., complementary clients)...
        if complementary_clients_fraction != 0 and complementary_tasks_fraction != 0:
            # Set the number of tasks that will be scheduled to the selected clients.
            num_tasks_to_schedule = num_tasks_to_schedule - num_complementary_tasks_to_schedule
        # Redefine the number of tasks to schedule, if the available participating clients capacities sum is lower.
        num_tasks_to_schedule = min(num_tasks_to_schedule, available_participating_clients_capacities_sum)
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
                time_key = "{0}ing_elapsed_time".format(phase)
                if time_key in individual_metrics_history_entry:
                    # Update his time costs list for this number of examples.
                    time_cost = individual_metrics_history_entry[time_key]
                    time_costs_client[num_examples] = time_cost
                # Initialize the energy cost with the zero value, so different energy costs can be summed up.
                energy_cost = 0
                # Get the energy consumed by his CPU (if available).
                energy_cpu_key = "{0}ing_energy_cpu".format(phase)
                if energy_cpu_key in individual_metrics_history_entry:
                    energy_cpu_cost = individual_metrics_history_entry[energy_cpu_key]
                    # If the CPU energy cost is a valid value (higher than 0), take it.
                    if energy_cpu_cost > 0:
                        energy_cost += energy_cpu_cost
                    # Otherwise, take the mean of the CPU energy costs for this number of examples, if available.
                    else:
                        energy_cpu_cost_mean_value = get_metric_mean_value(individual_metrics_history,
                                                                           client_key,
                                                                           num_examples_key,
                                                                           num_examples,
                                                                           energy_cpu_key)
                        energy_cost += energy_cpu_cost_mean_value
                # Get the energy consumed by his NVIDIA GPU (if available).
                energy_nvidia_gpu_key = "{0}ing_energy_nvidia_gpu".format(phase)
                if energy_nvidia_gpu_key in individual_metrics_history_entry:
                    energy_nvidia_gpu_cost = individual_metrics_history_entry[energy_nvidia_gpu_key]
                    # If the NVIDIA GPU energy cost is a valid value (higher than 0), take it.
                    if energy_nvidia_gpu_cost > 0:
                        energy_cost += energy_nvidia_gpu_cost
                    # Otherwise, take the mean of the NVIDIA GPU energy costs for this number of examples, if available.
                    else:
                        energy_nvidia_gpu_cost_mean_value = get_metric_mean_value(individual_metrics_history,
                                                                                  client_key,
                                                                                  num_examples_key,
                                                                                  num_examples,
                                                                                  energy_nvidia_gpu_key)
                        energy_cost += energy_nvidia_gpu_cost_mean_value
                # Update his energy costs list for this number of examples.
                energy_costs_client[num_examples] = energy_cost
            # Initialize his assignment capacities list...
            assignment_capacities_client = None
            assignment_capacities_initializer = assignment_capacities_init_settings["assignment_capacities_initializer"]
            if assignment_capacities_initializer == "Only_Previous_Num_Tasks_Assigned_Set":
                # Based only on his previous round(s) participation, i.e., the set of previous numbers of tasks
                # assigned to him.
                previous_num_tasks_assigned = [list(comm_round_metrics)[0]["num_{0}ing_examples_used".format(phase)]
                                               for key, comm_round_metrics in client_values.items()
                                               if "comm_round_" in key]
                previous_num_tasks_assigned_set = list(set(previous_num_tasks_assigned))
                assignment_capacities_client = previous_num_tasks_assigned_set
            elif assignment_capacities_initializer == "Custom_Range_Set_Union_Previous_Num_Tasks_Assigned_Set":
                # Based on a custom range set (ordered in ascending order), which also includes his previous round(s)
                # participation, i.e., the set of previous numbers of tasks assigned to him.
                lower_bound = assignment_capacities_init_settings["lower_bound"]
                upper_bound = assignment_capacities_init_settings["upper_bound"]
                if upper_bound == "client_capacity":
                    upper_bound = client_values["client_num_{0}ing_examples_available".format(phase)]
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
                        # Determine x1 and x2, which are two known previous numbers of tasks assigned.
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
                        time_cost_estimation = calculate_linear_interpolation_or_extrapolation(x1,
                                                                                               x2,
                                                                                               y1_time,
                                                                                               y2_time,
                                                                                               assignment_capacity)
                        # Calculate the linear interpolation/extrapolation for the energy cost.
                        y1_energy = energy_costs_client[x1]
                        y2_energy = energy_costs_client[x2]
                        energy_cost_estimation = calculate_linear_interpolation_or_extrapolation(x1,
                                                                                                 x2,
                                                                                                 y1_energy,
                                                                                                 y2_energy,
                                                                                                 assignment_capacity)
                        # Update the cost lists with the estimated values.
                        time_costs_client[assignment_capacity] = time_cost_estimation
                        energy_costs_client[assignment_capacity] = energy_cost_estimation
            # Filter his costs lists.
            filtered_time_costs_client = []
            filtered_energy_costs_client = []
            for index in range(0, len(time_costs_client)):
                if time_costs_client[index] != inf:
                    filtered_time_costs_client.append(time_costs_client[index])
                if energy_costs_client[index] != inf:
                    filtered_energy_costs_client.append(energy_costs_client[index])
            # Append his lists into the global lists.
            client_ids.append(client_key)
            assignment_capacities.append(assignment_capacities_client)
            time_costs.append(filtered_time_costs_client)
            energy_costs.append(filtered_energy_costs_client)
        # Convert the global lists into Numpy arrays.
        assignment_capacities = array(assignment_capacities, dtype=object)
        time_costs = array(time_costs, dtype=object)
        energy_costs = array(energy_costs, dtype=object)
        # Execute the ECMTC algorithm.
        ecmtc_schedule, ecmtc_energy_consumption, ecmtc_makespan = ecmtc(num_resources,
                                                                         num_tasks_to_schedule,
                                                                         assignment_capacities,
                                                                         time_costs,
                                                                         energy_costs,
                                                                         deadline_in_seconds)
        # Update the selection dictionary with the expected metrics for the schedule.
        selection.update({"expected_makespan": ecmtc_makespan,
                          "expected_energy_consumption": ecmtc_energy_consumption})
        # Log the ECMTC algorithm's result.
        message = "X*: {0}\nMinimal makespan (Cₘₐₓ): {1}\nMinimal energy consumption (ΣE): {2}" \
                  .format(ecmtc_schedule, ecmtc_makespan, ecmtc_energy_consumption)
        log_message(logger, message, "DEBUG")
        # Get the list of indices from the selected clients.
        selected_clients_indices = [sel_index for sel_index, client_num_tasks_scheduled
                                    in enumerate(ecmtc_schedule) if client_num_tasks_scheduled > 0]
        # Append their corresponding proxies objects and numbers of tasks scheduled into the selected clients list.
        for sel_index in selected_clients_indices:
            client_id_str = client_ids[sel_index]
            client_map = available_participating_clients_map[client_id_str]
            client_proxy = client_map["client_proxy"]
            client_capacity = client_map["client_num_{0}ing_examples_available".format(phase)]
            client_num_tasks_scheduled = int(ecmtc_schedule[sel_index])
            selected_clients.append({"client_proxy": client_proxy,
                                     "client_capacity": client_capacity,
                                     "client_num_tasks_scheduled": client_num_tasks_scheduled})
        # If there are complementary clients to be used other than the ones selected by the ECMTC algorithm...
        if complementary_clients_fraction != 0 and complementary_tasks_fraction != 0:
            # Initialize the list of complementary clients.
            complementary_clients = []
            # Determine the number of complementary clients to select.
            num_complementary_clients_to_select = max(1, int(num_resources * complementary_clients_fraction))
            # Filter (remove) the already selected clients from the available participating clients map.
            available_participating_clients_filtered_map = available_participating_clients_map.copy()
            for sel_index in selected_clients_indices:
                client_id_str = client_ids[sel_index]
                if client_id_str in available_participating_clients_filtered_map:
                    del available_participating_clients_filtered_map[client_id_str]
            # Select clients via random sampling if there are any available participating clients not selected yet.
            if available_participating_clients_filtered_map:
                sampled_clients_keys = sample(sorted(available_participating_clients_filtered_map),
                                              num_complementary_clients_to_select)
                for client_key in sampled_clients_keys:
                    client_map = available_clients_map[client_key]
                    client_proxy = client_map["client_proxy"]
                    client_capacity = client_map["client_num_{0}ing_examples_available".format(phase)]
                    complementary_clients.append({"client_proxy": client_proxy,
                                                  "client_capacity": client_capacity,
                                                  "client_num_tasks_scheduled": 0})
                # Get the maximum number of tasks that can be scheduled to the complementary clients.
                complementary_clients_capacities_sum = sum_clients_capacities(complementary_clients, phase)
                # Redefine the number of tasks to schedule if the complementary clients capacities sum is lower.
                num_complementary_tasks_to_schedule = min(num_complementary_tasks_to_schedule,
                                                          complementary_clients_capacities_sum)
                # Schedule the tasks to the complementary clients.
                schedule_tasks_to_selected_clients(num_complementary_tasks_to_schedule, complementary_clients)
                # Append the complementary clients to the list of selected clients.
                selected_clients.extend(complementary_clients)
        # Update the selection dictionary with the selected clients for the schedule.
        selection.update({"selected_clients": selected_clients})
    # Log a 'number of clients selected' message.
    message = "{0} {1} (out of {2}) {3} selected!".format(len(selected_clients),
                                                          "clients" if len(selected_clients) != 1 else "client",
                                                          len(available_clients_map),
                                                          "were" if len(selected_clients) != 1 else "was")
    log_message(logger, message, "INFO")
    # Return the selection dictionary.
    return selection
