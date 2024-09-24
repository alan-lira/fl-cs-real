from logging import Logger
from numpy import array
from random import sample
from statistics import mean

from fl_cs_real.task_scheduler.mc2mkp_adapted import mc2mkp_adapted
from fl_cs_real.utils.client_selector_util import calculate_linear_interpolation_or_extrapolation, \
    map_available_participating_clients, schedule_tasks_to_selected_clients, \
    select_all_available_clients, sum_clients_max_task_capacities
from fl_cs_real.utils.logger_util import log_message


def select_clients_using_mc2mkp_adapted(comm_round: int,
                                        phase: str,
                                        num_tasks_to_schedule: int,
                                        available_clients_map: dict,
                                        individual_metrics_history: dict,
                                        history_checker: str,
                                        candidate_clients_fraction: float,
                                        logger: Logger) -> dict:
    # Log a 'selecting clients' message.
    message = "Selecting {0}ing clients for round {1} using (MC)²MKP (adapted)...".format(phase, comm_round)
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
        # If only a fraction of the available clients is to be used by the algorithm (subset of candidate clients)...
        if candidate_clients_fraction != 0:
            # Set the number of resources,
            # based on the number of available clients with entries in the individual metrics history.
            num_resources = len(available_participating_clients_map)
            # Determine the number of candidate clients.
            num_candidate_clients = max(1, int(num_resources * candidate_clients_fraction))
            # Filter the subset of candidate clients via random sampling.
            sampled_clients_keys = sample(sorted(available_participating_clients_map), num_candidate_clients)
            available_participating_clients_map = {client_key: available_participating_clients_map[client_key]
                                                   for client_key in sampled_clients_keys}
        # Set the number of resources.
        num_resources = len(available_participating_clients_map)
        # Get the maximum number of tasks that can be scheduled to the available participating clients.
        available_participating_clients_capacities_sum = sum_clients_max_task_capacities(available_participating_clients_map,
                                                                                         phase)
        # Redefine the number of tasks to schedule, if the available participating clients capacities sum is lower.
        num_tasks_to_schedule = min(num_tasks_to_schedule, available_participating_clients_capacities_sum)
        # Initialize the global lists that will be transformed to array.
        client_ids = []
        assignment_capacities = []
        energy_costs = []
        # For each available client that has entries in the individual metrics history...
        for client_key, client_values in available_participating_clients_map.items():
            # Get his assignment capacities list.
            assignment_capacities_client = client_values["client_task_assignment_capacities_{0}".format(phase)]
            # Initialize his costs lists, based on his assignment capacities.
            energy_costs_client = [0] * len(assignment_capacities_client)
            # Set the costs of zero tasks scheduled, if needed, allowing the data point (x=0, y=0) to be used
            # during the estimation of his costs for the unused task assignment capacities.
            if 0 not in assignment_capacities_client:
                energy_costs_client.insert(0, 0)
            # Get his individual metrics history entries...
            individual_metrics_history_entries = [list(comm_round_metrics)[0]
                                                  for key, comm_round_metrics in client_values.items()
                                                  if "comm_round_" in key]
            for individual_metrics_history_entry in individual_metrics_history_entries:
                # Get the number of examples used by him.
                num_examples_key = "num_{0}ing_examples_used".format(phase)
                num_examples = individual_metrics_history_entry[num_examples_key]
                num_examples_idx = assignment_capacities_client.index(num_examples)
                # Initialize the energy cost with the zero value, so energy costs can be summed up.
                energy_cost = 0
                # Get the energy consumed by his CPU (if available).
                energy_cpu_key = "{0}ing_energy_cpu".format(phase)
                if energy_cpu_key in individual_metrics_history_entry:
                    energy_cpu_cost = individual_metrics_history_entry[energy_cpu_key]
                    energy_cost += energy_cpu_cost
                # Get the energy consumed by his NVIDIA GPU (if available).
                energy_nvidia_gpu_key = "{0}ing_energy_nvidia_gpu".format(phase)
                if energy_nvidia_gpu_key in individual_metrics_history_entry:
                    energy_nvidia_gpu_cost = individual_metrics_history_entry[energy_nvidia_gpu_key]
                    energy_cost += energy_nvidia_gpu_cost
                # Update his energy costs list for this number of examples.
                if not isinstance(energy_costs_client[num_examples_idx], list):
                    energy_costs_client[num_examples_idx] = [energy_cost]
                else:
                    energy_costs_client[num_examples_idx].append(energy_cost)
            # Calculate the averages of his historical costs per number of tasks.
            for idx, _ in enumerate(assignment_capacities_client):
                energy_cost_client_idx = energy_costs_client[idx]
                if isinstance(energy_cost_client_idx, list):
                    energy_costs_client[idx] = mean(energy_cost_client_idx)
            # Estimate his costs for the unused task assignment capacities:
            # 1. Energy costs.
            unknown_energy_costs_indices = [idx for idx in range(1, len(energy_costs_client))
                                            if energy_costs_client[idx] == 0]
            known_energy_costs_indices = [idx for idx in range(0, len(energy_costs_client))
                                          if idx not in unknown_energy_costs_indices]
            for _, unknown_idx in enumerate(unknown_energy_costs_indices):
                prev_known_indices = [known_idx for known_idx in known_energy_costs_indices if known_idx < unknown_idx]
                next_known_indices = [known_idx for known_idx in known_energy_costs_indices if known_idx > unknown_idx]
                prev_known_idx = None
                if next_known_indices:
                    # Interpolation.
                    prev_known_idx = prev_known_indices[-1]
                    next_known_idx = next_known_indices[0]
                else:
                    # Extrapolation.
                    if len(prev_known_indices) > 1:
                        prev_known_idx = prev_known_indices[-2]
                    next_known_idx = prev_known_indices[-1]
                if prev_known_idx is not None:
                    # Estimate the unknown energy cost via linear interpolation/extrapolation.
                    prev_known_x_i = assignment_capacities_client[prev_known_idx]
                    next_known_x_i = assignment_capacities_client[next_known_idx]
                    prev_known_energy_cost = energy_costs_client[prev_known_idx]
                    next_known_energy_cost = energy_costs_client[next_known_idx]
                    x_i_to_estimate = assignment_capacities_client[unknown_idx]
                    if prev_known_x_i < next_known_x_i and prev_known_energy_cost < next_known_energy_cost:
                        estimated_energy_cost = calculate_linear_interpolation_or_extrapolation(prev_known_x_i,
                                                                                                next_known_x_i,
                                                                                                prev_known_energy_cost,
                                                                                                next_known_energy_cost,
                                                                                                x_i_to_estimate)
                        energy_costs_client[unknown_idx] = estimated_energy_cost
            # Append his lists into the global lists.
            client_ids.append(client_key)
            assignment_capacities.append(assignment_capacities_client)
            energy_costs.append(energy_costs_client)
        # Convert the global lists into Numpy arrays.
        assignment_capacities = array(assignment_capacities, dtype=object)
        energy_costs = array(energy_costs, dtype=object)
        # Execute the (MC)²MKP adapted algorithm.
        mc2mkp_schedule = mc2mkp_adapted(num_tasks_to_schedule,
                                         num_resources,
                                         energy_costs,
                                         assignment_capacities)
        # Update the selection dictionary with the expected metrics for the schedule.
        mc2mkp_energy_consumption = 0
        for sel_index, client_num_tasks_scheduled in enumerate(list(mc2mkp_schedule)):
            if client_num_tasks_scheduled > 0:
                i_index = list(assignment_capacities[sel_index]).index(client_num_tasks_scheduled)
                energy_cost_i = energy_costs[sel_index][i_index]
                mc2mkp_energy_consumption += energy_cost_i
        selection.update({"expected_energy_consumption": mc2mkp_energy_consumption})
        # Log the (MC)²MKP adapted algorithm's result.
        message = "X*: {0}".format(mc2mkp_schedule)
        log_message(logger, message, "DEBUG")
        # Get the list of indices from the selected clients.
        selected_clients_indices = [sel_index for sel_index, client_num_tasks_scheduled
                                    in enumerate(list(mc2mkp_schedule)) if client_num_tasks_scheduled > 0]
        # Append their corresponding proxies objects and numbers of tasks scheduled into the selected clients list.
        for sel_index in selected_clients_indices:
            client_id_str = client_ids[sel_index]
            client_map = available_participating_clients_map[client_id_str]
            client_proxy = client_map["client_proxy"]
            client_max_task_capacity = max(client_map["client_task_assignment_capacities_{0}".format(phase)])
            client_num_tasks_scheduled = int(mc2mkp_schedule[sel_index])
            i_index = list(assignment_capacities[sel_index]).index(client_num_tasks_scheduled)
            client_expected_energy_consumption = energy_costs[sel_index][i_index]
            selected_clients.append({"client_proxy": client_proxy,
                                     "client_max_task_capacity": client_max_task_capacity,
                                     "client_num_tasks_scheduled": client_num_tasks_scheduled,
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
