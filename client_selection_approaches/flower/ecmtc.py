from numpy import array, inf
from random import sample

from flwr.common import GetPropertiesIns

from schedulers.ecmtc import ecmtc


def select_clients_using_ecmtc(comm_round: int,
                               phase: str,
                               num_tasks: int,
                               deadline: float,
                               allow_complementary_clients_random_selection: bool,
                               complementary_tasks_fraction: float,
                               complementary_clients_fraction: float,
                               available_clients_map: dict,
                               individual_metrics_history: dict) -> list:
    selected_clients = []
    # Verify if there are any entries in the individual metrics history.
    if not individual_metrics_history:
        # If not, this means it is the first communication round. Therefore, all available clients will be selected.
        # Schedule the tasks as equally possible for the initial available clients.
        # If there are leftovers, they will be added to the client at index 0.
        mean_tasks = num_tasks // len(available_clients_map)
        leftover = num_tasks % len(available_clients_map)
        initial_schedule = []
        for _ in range(0, len(available_clients_map)):
            initial_schedule.append(mean_tasks)
        initial_schedule[0] += leftover
        for selected_client_index, _ in enumerate(available_clients_map.items()):
            selected_client_proxy = list(available_clients_map.items())[selected_client_index][1]
            selected_client_num_tasks = initial_schedule[selected_client_index]
            selected_clients.append({"client_proxy": selected_client_proxy,
                                     "client_num_tasks": selected_client_num_tasks})
    else:
        # Otherwise, the clients will be selected according to their entries in the individual metrics history.
        previous_comm_round_key = "comm_round_{0}".format(comm_round-1)
        # Verify if there is an entry in the individual metrics history for the previous communication round.
        if previous_comm_round_key in individual_metrics_history:
            # Get the individual metrics entry of the previous communication round.
            all_individual_metrics_previous_comm_round = individual_metrics_history[previous_comm_round_key]
            # Get the available clients that participated on the previous communication round.
            previous_comm_round_available_clients_map = {}
            previous_comm_round_individual_metrics = {}
            previous_comm_round_available_clients_capacities = {}
            for individual_metrics_previous_comm_round in all_individual_metrics_previous_comm_round:
                client_id_str = list(individual_metrics_previous_comm_round.keys())[0]
                client_metrics = individual_metrics_previous_comm_round[client_id_str]
                client_proxy = available_clients_map[client_id_str]
                num_examples_available_property = "num_{0}ing_examples_available".format(phase)
                gpi = GetPropertiesIns({num_examples_available_property: "?"})
                client_prompted = client_proxy.get_properties(gpi, timeout=9999)
                client_assignment_capacity = client_prompted.properties[num_examples_available_property]
                previous_comm_round_available_clients_map.update({client_id_str: client_proxy})
                previous_comm_round_individual_metrics.update({client_id_str: client_metrics})
                previous_comm_round_available_clients_capacities.update({client_id_str: client_assignment_capacity})
            # If the random selection of complementary clients is allowed...
            if allow_complementary_clients_random_selection:
                # Get the available clients that did not participate on the previous communication round.
                complementary_available_clients_map = {}
                complementary_available_clients_capacities = {}
                complementary_available_clients_ids_str \
                    = available_clients_map.keys() - previous_comm_round_available_clients_map
                # If there are available clients that did not participate on the previous communication round...
                if len(complementary_available_clients_ids_str) > 0:
                    for complementary_available_client_id_str in complementary_available_clients_ids_str:
                        client_proxy = available_clients_map[complementary_available_client_id_str]
                        num_examples_available_property = "num_{0}ing_examples_available".format(phase)
                        gpi = GetPropertiesIns({num_examples_available_property: "?"})
                        client_prompted = client_proxy.get_properties(gpi, timeout=9999)
                        client_assignment_capacity = client_prompted.properties[num_examples_available_property]
                        complementary_available_clients_map.update({complementary_available_client_id_str:
                                                                    client_proxy})
                        complementary_available_clients_capacities.update({complementary_available_client_id_str:
                                                                           client_assignment_capacity})
                    # Set the number of available clients that will be randomly selected.
                    complementary_available_clients_num_selected \
                        = int(len(complementary_available_clients_ids_str) * complementary_clients_fraction)
                    # Set the number of tasks that will be scheduled for the randomly selected clients.
                    complementary_available_clients_num_tasks = int(num_tasks * complementary_tasks_fraction)
                    # Schedule the tasks as equally possible for the randomly selected clients.
                    # If there are leftovers, they will be added to the client at index 0.
                    complementary_available_clients_mean_tasks \
                        = complementary_available_clients_num_tasks // len(complementary_available_clients_ids_str)
                    complementary_available_clients_leftover \
                        = complementary_available_clients_num_tasks % len(complementary_available_clients_ids_str)
                    complementary_available_clients_schedule = []
                    for _ in range(0, len(complementary_available_clients_ids_str)):
                        complementary_available_clients_schedule.append(complementary_available_clients_mean_tasks)
                    complementary_available_clients_schedule[0] += complementary_available_clients_leftover
                    # Randomly select a fraction of available clients that did not participate on the previous
                    # communication round.
                    sampled_complementary_available_clients_ids = sample(complementary_available_clients_ids_str,
                                                                         complementary_available_clients_num_selected)
                    for selected_client_index, _ in enumerate(sampled_complementary_available_clients_ids):
                        selected_client_id_str = sampled_complementary_available_clients_ids[selected_client_index]
                        selected_client_proxy = available_clients_map[selected_client_id_str]
                        selected_client_num_tasks = complementary_available_clients_schedule[selected_client_index]
                        selected_clients.append({"client_proxy": selected_client_proxy,
                                                 "client_num_tasks": selected_client_num_tasks})
            # Set the number of resources.
            # (Based on the number of available clients that participated on the previous communication round).
            num_resources = len(previous_comm_round_available_clients_map)
            # Initialize the global lists that will be transformed to array and be used by the ECMTC algorithm.
            client_ids = []
            assignment_capacities = []
            time_costs = []
            energy_costs = []
            # For each available client that participated on the previous communication round...
            for client_id_str, client_proxy in previous_comm_round_available_clients_map.items():
                # Initialize his assignment capacities list (based on his number of examples available).
                client_assignment_capacity = previous_comm_round_available_clients_capacities[client_id_str]
                assignment_capacities_client = [i for i in range(0, client_assignment_capacity+1)]
                # Initialize his costs lists (based on the number of examples required).
                time_costs_client = [inf for _ in range(0, num_tasks+1)]
                energy_costs_client = [inf for _ in range(0, num_tasks+1)]
                # Get his metrics from the previous communication round.
                client_metrics = previous_comm_round_individual_metrics[client_id_str]
                # Get the number of examples used by him.
                num_examples_key = "num_{0}ing_examples".format(phase)
                num_examples = client_metrics[num_examples_key]
                # Get the time spent by him (if available).
                time_key = "{0}ing_time".format(phase)
                if time_key in client_metrics:
                    time_cost = client_metrics[time_key]
                    time_costs_client[num_examples] = time_cost
                # Get the energy spent by him (if available).
                energy_key = "{0}ing_energy_cpu".format(phase)
                if energy_key in client_metrics:
                    energy_cost = client_metrics[energy_key]
                    energy_costs_client[num_examples] = energy_cost
                # Append his lists into the global lists.
                client_ids.append(client_id_str)
                assignment_capacities.append(assignment_capacities_client)
                time_costs.append(time_costs_client)
                energy_costs.append(energy_costs_client)
            # Convert the global lists into Numpy arrays.
            assignment_capacities = array(assignment_capacities, dtype=object)
            time_costs = array(time_costs, dtype=object)
            energy_costs = array(energy_costs, dtype=object)
            # Execute the ECMTC algorithm.
            optimal_schedule, minimal_energy_consumption, minimal_makespan = ecmtc(num_resources,
                                                                                   num_tasks,
                                                                                   assignment_capacities,
                                                                                   time_costs,
                                                                                   energy_costs,
                                                                                   deadline)
            # Debugging prints.
            print("X*: {0}".format(optimal_schedule))
            print("Minimal makespan (Cₘₐₓ): {0}".format(minimal_makespan))
            print("Minimal energy consumption (ΣE): {0}".format(minimal_energy_consumption))
            # Get the list of indices of the selected clients.
            selected_clients_indices = [client_index for client_index, client_num_tasks in enumerate(optimal_schedule)
                                        if client_num_tasks > 0]
            # Append their corresponding proxies objects and numbers of tasks scheduled into the selected clients list.
            for selected_client_index in selected_clients_indices:
                selected_client_id_str = client_ids[selected_client_index]
                selected_client_proxy = available_clients_map[selected_client_id_str]
                selected_client_num_tasks = int(optimal_schedule[selected_client_index])
                selected_clients.append({"client_proxy": selected_client_proxy,
                                         "client_num_tasks": selected_client_num_tasks})
        else:
            # No entries in the individual metrics history were found for the previous communication round.
            pass
    return selected_clients
