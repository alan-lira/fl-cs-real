from logging import Logger
from random import sample

from goffls.utils.client_selector_util import schedule_tasks_to_selected_clients, sum_clients_capacities
from goffls.utils.logger_util import log_message


def select_clients_using_random(phase: str,
                                num_tasks_to_schedule: int,
                                clients_fraction: float,
                                available_clients_map: dict,
                                logger: Logger) -> dict:
    # Log a 'selecting clients' message.
    message = "Selecting {0}ing clients using Random...".format(phase)
    log_message(logger, message, "INFO")
    # Initialize the selection dictionary and the list of selected clients.
    selection = {}
    selected_clients = []
    # Get the number of available clients.
    num_available_clients = len(available_clients_map)
    # Determine the number of clients to select.
    num_clients_to_select = max(1, int(num_available_clients * clients_fraction))
    # Select clients via random sampling.
    sampled_clients_keys = sample(sorted(available_clients_map), num_clients_to_select)
    for client_key in sampled_clients_keys:
        client_map = available_clients_map[client_key]
        client_proxy = client_map["client_proxy"]
        client_capacity = client_map["client_num_{0}ing_examples_available".format(phase)]
        selected_clients.append({"client_proxy": client_proxy,
                                 "client_capacity": client_capacity,
                                 "client_num_tasks_scheduled": 0})
    # Get the maximum number of tasks that can be scheduled to the selected clients.
    selected_clients_capacities_sum = sum_clients_capacities(selected_clients, phase)
    # Redefine the number of tasks to schedule, if the selected clients capacities sum is lower.
    num_tasks_to_schedule = min(num_tasks_to_schedule, selected_clients_capacities_sum)
    # Schedule the tasks to the selected clients.
    schedule_tasks_to_selected_clients(num_tasks_to_schedule, selected_clients)
    # Update the selection dictionary with the selected clients for the schedule.
    selection.update({"selected_clients": selected_clients})
    # Log a 'number of clients selected' message.
    message = "{0} {1} (out of {2}) {3} selected!".format(len(selected_clients),
                                                          "clients" if len(selected_clients) != 1 else "client",
                                                          num_available_clients,
                                                          "were" if len(selected_clients) != 1 else "was")
    log_message(logger, message, "INFO")
    # Return the selection dictionary.
    return selection
