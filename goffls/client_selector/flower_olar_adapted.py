from logging import Logger
from numpy import array, inf

from goffls.task_scheduler.olar_adapted import olar_adapted
from goffls.utils.client_selector_util import calculate_linear_interpolation_or_extrapolation, \
    map_available_participating_clients, schedule_tasks_to_selected_clients, select_all_available_clients, \
    sum_clients_capacities
from goffls.utils.logger_util import log_message


def select_clients_using_olar_adapted(comm_round: int,
                                      phase: str,
                                      num_tasks_to_schedule: int,
                                      available_clients_map: dict,
                                      individual_metrics_history: dict,
                                      history_checker: str,
                                      assignment_capacities_init_settings: dict,
                                      logger: Logger) -> dict:
    # Log a 'selecting clients' message.
    message = "Selecting {0}ing clients for round {1} using OLAR (adapted)...".format(phase, comm_round)
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
        # Redefine the number of tasks to schedule, if the available participating clients capacities sum is lower.
        num_tasks_to_schedule = min(num_tasks_to_schedule, available_participating_clients_capacities_sum)
        # Initialize the global lists that will be transformed to array.
        client_ids = []
        assignment_capacities = []
        time_costs = []
        # For each available client that has entries in the individual metrics history...
        for client_key, client_values in available_participating_clients_map.items():
            # Initialize his costs lists, based on the number of tasks (examples) to be scheduled.
            time_costs_client = [inf for _ in range(0, num_tasks_to_schedule+1)]
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
                        # Update the cost lists with the estimated values.
                        time_costs_client[assignment_capacity] = time_cost_estimation
            # Filter his costs lists.
            filtered_time_costs_client = []
            for index in range(0, len(time_costs_client)):
                if time_costs_client[index] != inf:
                    filtered_time_costs_client.append(time_costs_client[index])
            # Append his lists into the global lists.
            client_ids.append(client_key)
            assignment_capacities.append(assignment_capacities_client)
            time_costs.append(filtered_time_costs_client)
        # Convert the global lists into Numpy arrays.
        assignment_capacities = array(assignment_capacities, dtype=object)
        time_costs = array(time_costs, dtype=object)
        # Execute the OLAR adapted algorithm.
        olar_schedule = olar_adapted(num_tasks_to_schedule,
                                     num_resources,
                                     time_costs,
                                     assignment_capacities)
        # Update the selection dictionary with the expected metrics for the schedule.
        olar_makespan = 0
        for sel_index, client_num_tasks_scheduled in enumerate(list(olar_schedule)):
            if client_num_tasks_scheduled > 0:
                i_index = list(assignment_capacities[sel_index]).index(client_num_tasks_scheduled)
                time_cost_i = time_costs[sel_index][i_index]
                if time_cost_i > olar_makespan:
                    olar_makespan = time_cost_i
        selection.update({"expected_makespan": olar_makespan})
        # Log the OLAR adapted algorithm's result.
        message = "X*: {0}".format(olar_schedule)
        log_message(logger, message, "DEBUG")
        # Get the list of indices from the selected clients.
        selected_clients_indices = [sel_index for sel_index, client_num_tasks_scheduled
                                    in enumerate(list(olar_schedule)) if client_num_tasks_scheduled > 0]
        # Append their corresponding proxies objects and numbers of tasks scheduled into the selected clients list.
        for sel_index in selected_clients_indices:
            client_id_str = client_ids[sel_index]
            client_map = available_participating_clients_map[client_id_str]
            client_proxy = client_map["client_proxy"]
            client_capacity = client_map["client_num_{0}ing_examples_available".format(phase)]
            client_num_tasks_scheduled = int(olar_schedule[sel_index])
            i_index = list(assignment_capacities[sel_index]).index(client_num_tasks_scheduled)
            client_expected_duration = time_costs[sel_index][i_index]
            selected_clients.append({"client_proxy": client_proxy,
                                     "client_capacity": client_capacity,
                                     "client_num_tasks_scheduled": client_num_tasks_scheduled,
                                     "client_expected_duration": client_expected_duration})
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
