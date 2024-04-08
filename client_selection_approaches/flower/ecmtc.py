from numpy import array, inf
from random import sample

from flwr.common import GetPropertiesIns
from flwr.server.client_proxy import ClientProxy

from schedulers.ecmtc import ecmtc


def _get_client_assignment_capacity(phase: str,
                                    client_proxy: ClientProxy) -> int:
    num_examples_available_property = "num_{0}ing_examples_available".format(phase)
    gpi = GetPropertiesIns({num_examples_available_property: "?"})
    client_prompted = client_proxy.get_properties(gpi, timeout=9999)
    client_assignment_capacity = client_prompted.properties[num_examples_available_property]
    return client_assignment_capacity


def _load_available_participating_clients_dicts_from_previous_round(comm_round: int,
                                                                    phase: str,
                                                                    available_clients_map: dict,
                                                                    individual_metrics_history: dict) -> tuple:
    # Initialize the available clients dictionaries.
    available_participating_clients_history_map = {}
    available_participating_clients_individual_metrics = {}
    available_participating_clients_capacities = {}
    # Get the previous communication round key.
    previous_comm_round_key = "comm_round_{0}".format(comm_round - 1)
    # Verify if there is an entry in the individual metrics history for the previous communication round.
    if previous_comm_round_key in individual_metrics_history:
        # If so, get the individual metrics entry of the previous communication round.
        all_individual_metrics_previous_comm_round = individual_metrics_history[previous_comm_round_key]
        # Get the available clients that participated in the previous communication round.
        for individual_metrics_previous_comm_round in all_individual_metrics_previous_comm_round:
            client_id_str = list(individual_metrics_previous_comm_round.keys())[0]
            client_metrics = individual_metrics_previous_comm_round[client_id_str]
            client_proxy = available_clients_map[client_id_str]
            client_assignment_capacity = _get_client_assignment_capacity(phase, client_proxy)
            available_participating_clients_history_map.update({client_id_str: client_proxy})
            client_individual_metrics = {client_id_str: client_metrics}
            if previous_comm_round_key not in available_participating_clients_individual_metrics:
                round_individual_metrics = {previous_comm_round_key:
                                            {"individual_metrics": [client_individual_metrics]}}
                available_participating_clients_individual_metrics.update(round_individual_metrics)
            else:
                available_participating_clients_individual_metrics[previous_comm_round_key]["individual_metrics"] \
                    .append(client_individual_metrics)
            available_participating_clients_capacities.update({client_id_str: client_assignment_capacity})
    return available_participating_clients_history_map, available_participating_clients_individual_metrics, \
        available_participating_clients_capacities


def _load_available_absent_clients_dicts_from_previous_round(comm_round: int,
                                                             phase: str,
                                                             available_clients_map: dict,
                                                             individual_metrics_history: dict) -> tuple:
    # Initialize the available clients dictionaries.
    available_absent_clients_map = {}
    available_absent_clients_capacities = {}
    # Get the available clients that participated in the previous communication round.
    available_participating_clients_history_map, _, _ \
        = _load_available_participating_clients_dicts_from_previous_round(comm_round,
                                                                          phase,
                                                                          available_clients_map,
                                                                          individual_metrics_history)
    # Get the available clients that did not participate in the previous communication round.
    available_absent_clients_ids_str = available_clients_map.keys() - available_participating_clients_history_map
    # If there are available clients that did not participate in the previous communication round...
    if len(available_absent_clients_ids_str) > 0:
        for available_absent_client_id_str in available_absent_clients_ids_str:
            client_proxy = available_clients_map[available_absent_client_id_str]
            client_assignment_capacity = _get_client_assignment_capacity(phase, client_proxy)
            available_absent_clients_map.update({available_absent_client_id_str: client_proxy})
            available_absent_clients_capacities.update({available_absent_client_id_str: client_assignment_capacity})
    return available_absent_clients_map, available_absent_clients_capacities


def _determine_number_of_complementary_clients_and_tasks(phase: str,
                                                         num_tasks: int,
                                                         available_absent_clients_map: dict,
                                                         complementary_selection_settings: dict) -> tuple:
    num_complementary_clients = 0
    num_complementary_tasks = 0
    complementary_selection_approach = complementary_selection_settings["approach"]
    if complementary_selection_approach == "RandomAbsent":
        tasks_fraction = None
        clients_fraction = None
        if phase == "train":
            tasks_fraction = complementary_selection_settings["fit_tasks_fraction"]
            clients_fraction = complementary_selection_settings["fit_clients_fraction"]
        elif phase == "test":
            tasks_fraction = complementary_selection_settings["evaluate_tasks_fraction"]
            clients_fraction = complementary_selection_settings["evaluate_clients_fraction"]
        num_complementary_clients = int(len(available_absent_clients_map) * clients_fraction)
        num_complementary_tasks = int(num_tasks * tasks_fraction)
    return num_complementary_clients, num_complementary_tasks


def _determine_complementary_tasks_schedule(num_complementary_clients: int,
                                            num_complementary_tasks: int) -> list:
    complementary_tasks_schedule = []
    # Schedule the tasks as equally possible.
    complementary_available_clients_mean_tasks = num_complementary_tasks // num_complementary_clients
    for _ in range(0, num_complementary_clients):
        complementary_tasks_schedule.append(complementary_available_clients_mean_tasks)
    # If there are leftovers, they will be added to the client at index 0.
    complementary_available_clients_leftover = num_complementary_tasks % num_complementary_clients
    complementary_tasks_schedule[0] += complementary_available_clients_leftover
    return complementary_tasks_schedule


def _randomly_select_complementary_clients(available_absent_clients_map: dict,
                                           num_complementary_clients: int,
                                           complementary_tasks_schedule: list) -> list:
    randomly_selected_complementary_clients = []
    available_absent_clients_ids_str = [available_absent_client_id_str
                                        for available_absent_client_id_str, _ in available_absent_clients_map.items()]
    selected_complementary_clients_ids_str = sample(available_absent_clients_ids_str, num_complementary_clients)
    for selected_complementary_client_index, _ in enumerate(selected_complementary_clients_ids_str):
        selected_complementary_clients_id_str \
            = selected_complementary_clients_ids_str[selected_complementary_client_index]
        selected_complementary_client_proxy = available_absent_clients_map[selected_complementary_clients_id_str]
        selected_complementary_client_num_tasks = complementary_tasks_schedule[selected_complementary_client_index]
        randomly_selected_complementary_clients.append({"client_proxy": selected_complementary_client_proxy,
                                                        "client_num_tasks": selected_complementary_client_num_tasks})
    return randomly_selected_complementary_clients


def select_clients_using_ecmtc(comm_round: int,
                               phase: str,
                               num_tasks: int,
                               deadline: float,
                               available_clients_map: dict,
                               individual_metrics_history: dict,
                               history_check_approach: str,
                               enable_complementary_selection: bool,
                               complementary_selection_settings: dict) -> list:
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
        available_participating_clients_history_map = {}
        available_participating_clients_individual_metrics = {}
        available_participating_clients_capacities = {}
        available_absent_clients_map = {}
        available_absent_clients_capacities = {}
        # If the history check approach is to consider participating clients only from the previous round...
        if history_check_approach == "PreviousRoundOnly":
            # Load the available participating clients dicts from the previous round.
            available_participating_clients_history_map, available_participating_clients_individual_metrics, \
                available_participating_clients_capacities \
                = _load_available_participating_clients_dicts_from_previous_round(comm_round,
                                                                                  phase,
                                                                                  available_clients_map,
                                                                                  individual_metrics_history)
        # If the selection of complementary clients is enabled...
        if enable_complementary_selection:
            complementary_selection_approach = complementary_selection_settings["approach"]
            complementary_selection_history_check_approach = complementary_selection_settings["history_check_approach"]
            # If the complementary clients selection approach is to randomly select absent clients...
            if complementary_selection_approach == "RandomAbsent":
                # If the history check approach is to consider absent clients only from the previous round...
                if complementary_selection_history_check_approach == "PreviousRoundOnly":
                    # Load the available absent clients dicts from the previous round.
                    available_absent_clients_map, available_absent_clients_capacities \
                        = _load_available_absent_clients_dicts_from_previous_round(comm_round,
                                                                                   phase,
                                                                                   available_clients_map,
                                                                                   individual_metrics_history)
                # Determine the number of complementary clients to select and the number of complementary tasks
                # to be scheduled to them.
                num_complementary_clients, num_complementary_tasks \
                    = _determine_number_of_complementary_clients_and_tasks(phase,
                                                                           num_tasks,
                                                                           available_absent_clients_map,
                                                                           complementary_selection_settings)
                # Determine the complementary tasks schedule.
                complementary_tasks_schedule = _determine_complementary_tasks_schedule(num_complementary_clients,
                                                                                       num_complementary_tasks)
                # Randomly select a fraction of complementary clients.
                randomly_selected_complementary_clients \
                    = _randomly_select_complementary_clients(available_absent_clients_map,
                                                             num_complementary_clients,
                                                             complementary_tasks_schedule)
                # Append the randomly selected complementary clients into the selected clients list.
                selected_clients.extend(randomly_selected_complementary_clients)
        # Set the number of resources,
        # based on the number of available clients with entries in the individual metrics history.
        num_resources = len(available_participating_clients_history_map)
        # Initialize the global lists that will be transformed to array.
        client_ids = []
        assignment_capacities = []
        time_costs = []
        energy_costs = []
        # For each available client that has entries in the individual metrics history...
        for client_id_str, client_proxy in available_participating_clients_history_map.items():
            # Initialize his assignment capacities list, based on his number of examples available.
            client_assignment_capacity = available_participating_clients_capacities[client_id_str]
            assignment_capacities_client = [i for i in range(0, client_assignment_capacity+1)]
            # Initialize his costs lists, based on the number of tasks (examples) required.
            time_costs_client = [inf for _ in range(0, num_tasks+1)]
            energy_costs_client = [inf for _ in range(0, num_tasks+1)]
            # Get his entries in the individual metrics history...
            for comm_round_key, _ in available_participating_clients_individual_metrics.items():
                comm_round_dict = available_participating_clients_individual_metrics[comm_round_key]
                individual_metrics = comm_round_dict["individual_metrics"]
                for client_individual_metrics in individual_metrics:
                    if client_id_str in client_individual_metrics:
                        # Get the client metrics.
                        client_metrics = client_individual_metrics[client_id_str]
                        # Get the number of examples used by him.
                        num_examples_key = "num_{0}ing_examples".format(phase)
                        num_examples = client_metrics[num_examples_key]
                        # Get the time spent by him (if available).
                        time_key = "{0}ing_time".format(phase)
                        if time_key in client_metrics:
                            # Update his time costs list for this number of examples.
                            time_cost = client_metrics[time_key]
                            time_costs_client[num_examples] = time_cost
                        # Get the energy spent by him (if available).
                        energy_key = "{0}ing_energy_cpu".format(phase)
                        if energy_key in client_metrics:
                            # Update his energy costs list for this number of examples.
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
    return selected_clients
