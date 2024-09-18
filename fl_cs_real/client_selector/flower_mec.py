from logging import Logger
from numpy import array, inf

from fl_cs_real.task_scheduler.mec import mec
from fl_cs_real.utils.client_selector_util import calculate_linear_interpolation_or_extrapolation, \
    get_metric_mean_value, map_available_participating_clients, schedule_tasks_to_selected_clients, \
    select_all_available_clients, sum_clients_max_task_capacities
from fl_cs_real.utils.logger_util import log_message


def select_clients_using_mec(comm_round: int,
                             phase: str,
                             num_tasks_to_schedule: int,
                             available_clients_map: dict,
                             individual_metrics_history: dict,
                             history_checker: str,
                             logger: Logger) -> dict:
    # Log a 'selecting clients' message.
    message = "Selecting {0}ing clients for round {1} using MEC...".format(phase, comm_round)
    log_message(logger, message, "INFO")
    # Initialize the selection dictionary and the list of selected clients.
    selection = {}
    selected_clients = []
    # Verify if there are any entries in the individual metrics history.
    if not individual_metrics_history:
        # If not, this means it is the first communication round. Therefore, all available clients will be selected.
        selected_all_available_clients = select_all_available_clients(available_clients_map, phase)
        # Get the maximum number of tasks that can be scheduled to the selected clients.
        selected_clients_capacities_sum = sum_clients_max_task_capacities(selected_all_available_clients, phase)
        # Redefine the number of tasks to schedule, if the selected clients capacities sum is lower.
        num_tasks_to_schedule = min(num_tasks_to_schedule, selected_clients_capacities_sum)
        # Schedule the tasks to the selected clients.
        schedule_tasks_to_selected_clients(num_tasks_to_schedule,
                                           selected_all_available_clients,
                                           phase)
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
        available_participating_clients_capacities_sum = sum_clients_max_task_capacities(available_participating_clients_map,
                                                                                         phase)
        # Redefine the number of tasks to schedule, if the available participating clients capacities sum is lower.
        num_tasks_to_schedule = min(num_tasks_to_schedule, available_participating_clients_capacities_sum)
        # Initialize the global lists that will be transformed to array.
        client_ids = []
        assignment_capacities = []
        time_costs = []
        energy_costs = []
        # For each available client that has entries in the individual metrics history...
        for client_key, client_values in available_participating_clients_map.items():
            # Get his assignment capacities list.
            assignment_capacities_client = client_values["client_task_assignment_capacities_{0}".format(phase)]
            # Initialize his costs lists, based on the number of tasks (examples) to be scheduled.
            time_costs_client = [inf for _ in range(0, num_tasks_to_schedule + 1)]
            energy_costs_client = [inf for _ in range(0, num_tasks_to_schedule + 1)]
            # Set the costs of zero tasks scheduled, allowing the data point (x=0, y=0) to be used during the
            # estimation of his costs for the unused task assignment capacities.
            time_costs_client[0] = 0
            energy_costs_client[0] = 0
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
            # Get his set of previous numbers of tasks assigned.
            previous_num_tasks_assigned = [list(comm_round_metrics)[0]["num_{0}ing_examples_used".format(phase)]
                                           for key, comm_round_metrics in client_values.items()
                                           if "comm_round_" in key]
            previous_num_tasks_assigned.insert(0, 0)
            # Estimate his costs for the unused task assignment capacities.
            for assignment_capacity in assignment_capacities_client:
                if assignment_capacity not in previous_num_tasks_assigned and \
                   assignment_capacity <= num_tasks_to_schedule:
                    # Determine x1 and x2, which are two known previous numbers of tasks assigned.
                    prev_known_assignments = [i for i in previous_num_tasks_assigned if i < assignment_capacity]
                    next_known_assignments = [i for i in previous_num_tasks_assigned if i > assignment_capacity]
                    prev_known_assignment = None
                    if next_known_assignments:
                        # Interpolation.
                        prev_known_assignment = prev_known_assignments[-1]
                        next_known_assignment = next_known_assignments[0]
                    else:
                        # Extrapolation.
                        if len(prev_known_assignments) > 1:
                            prev_known_assignment = prev_known_assignments[-2]
                        next_known_assignment = prev_known_assignments[-1]
                    if prev_known_assignment != next_known_assignment:
                        # Calculate the linear interpolation/extrapolation for the time cost.
                        y1_time = time_costs_client[prev_known_assignment]
                        y2_time = time_costs_client[next_known_assignment]
                        time_cost_estimation = calculate_linear_interpolation_or_extrapolation(prev_known_assignment,
                                                                                               next_known_assignment,
                                                                                               y1_time,
                                                                                               y2_time,
                                                                                               assignment_capacity)
                        # Calculate the linear interpolation/extrapolation for the energy cost.
                        y1_energy = energy_costs_client[prev_known_assignment]
                        y2_energy = energy_costs_client[next_known_assignment]
                        energy_cost_estimation = calculate_linear_interpolation_or_extrapolation(prev_known_assignment,
                                                                                                 next_known_assignment,
                                                                                                 y1_energy,
                                                                                                 y2_energy,
                                                                                                 assignment_capacity)
                        # Update the cost lists with the estimated values.
                        time_costs_client[assignment_capacity] = time_cost_estimation
                        energy_costs_client[assignment_capacity] = energy_cost_estimation
            # Filter his lists.
            filtered_assignment_capacities_client = []
            filtered_time_costs_client = []
            filtered_energy_costs_client = []
            for index in range(0, num_tasks_to_schedule + 1):
                if time_costs_client[index] != inf and energy_costs_client[index] != inf:
                    filtered_assignment_capacities_client.append(index)
                    filtered_time_costs_client.append(time_costs_client[index])
                    filtered_energy_costs_client.append(energy_costs_client[index])
            # Append his lists into the global lists.
            client_ids.append(client_key)
            assignment_capacities.append(filtered_assignment_capacities_client)
            time_costs.append(filtered_time_costs_client)
            energy_costs.append(filtered_energy_costs_client)
        # Convert the global lists into Numpy arrays.
        assignment_capacities = array(assignment_capacities, dtype=object)
        time_costs = array(time_costs, dtype=object)
        energy_costs = array(energy_costs, dtype=object)
        # Execute the MEC algorithm.
        mec_schedule, mec_makespan, mec_energy_consumption = mec(num_resources,
                                                                 num_tasks_to_schedule,
                                                                 assignment_capacities,
                                                                 time_costs,
                                                                 energy_costs)
        # Update the selection dictionary with the expected metrics for the schedule.
        selection.update({"expected_makespan": mec_makespan,
                          "expected_energy_consumption": mec_energy_consumption})
        # Log the MEC algorithm's result.
        message = "X*: {0}\nMinimal makespan (Cₘₐₓ): {1}\nMinimal energy consumption (ΣE): {2}" \
                  .format(mec_schedule, mec_makespan, mec_energy_consumption)
        log_message(logger, message, "DEBUG")
        # Get the list of indices from the selected clients.
        selected_clients_indices = [sel_index for sel_index, client_num_tasks_scheduled
                                    in enumerate(mec_schedule) if client_num_tasks_scheduled > 0]
        # Append their corresponding proxies objects and numbers of tasks scheduled into the selected clients list.
        for sel_index in selected_clients_indices:
            client_id_str = client_ids[sel_index]
            client_map = available_participating_clients_map[client_id_str]
            client_proxy = client_map["client_proxy"]
            client_max_task_capacity = max(client_map["client_task_assignment_capacities_{0}".format(phase)])
            client_num_tasks_scheduled = int(mec_schedule[sel_index])
            i_index = list(assignment_capacities[sel_index]).index(client_num_tasks_scheduled)
            client_expected_duration = time_costs[sel_index][i_index]
            client_expected_energy_consumption = energy_costs[sel_index][i_index]
            selected_clients.append({"client_proxy": client_proxy,
                                     "client_max_task_capacity": client_max_task_capacity,
                                     "client_num_tasks_scheduled": client_num_tasks_scheduled,
                                     "client_expected_duration": client_expected_duration,
                                     "client_expected_energy_consumption": client_expected_energy_consumption})
        # Update the selection dictionary with the selected clients for the schedule.
        selection.update({"selected_clients": selected_clients})
    # Log a 'number of clients selected' message.
    message = "{0} {1}ing {2} (out of {3}) {4} selected for round {5}!" \
              .format(len(selected_clients),
                      phase,
                      "clients" if len(selected_clients) != 1 else "client",
                      len(available_clients_map),
                      "were" if len(selected_clients) != 1 else "was",
                      comm_round)
    log_message(logger, message, "INFO")
    # Return the selection dictionary.
    return selection
