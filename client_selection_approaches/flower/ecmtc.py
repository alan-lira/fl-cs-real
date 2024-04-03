from flwr.common import GetPropertiesIns
from numpy import array, inf

from schedulers.ecmtc import ecmtc
from flwr.server.client_proxy import ClientProxy


def select_clients_using_ecmtc(comm_round: int,
                               phase: str,
                               num_tasks: int,
                               deadline: float,
                               available_clients_map: dict,
                               individual_metrics_history: dict) -> list[ClientProxy]:
    selected_clients = []
    # Verify if it's the first communication round.
    if comm_round == 1:
        # If so, all available clients will be selected, since there are no individual metrics history yet.
        selected_clients = [client_proxy for _, client_proxy in available_clients_map.items()]
    else:
        # Otherwise, the clients will be selected according to their individual metrics history.
        # Get the individual metrics history of the previous communication round.
        previous_comm_round_key = "comm_round_{0}".format(comm_round-1)
        individual_metrics_previous_comm_round = individual_metrics_history[previous_comm_round_key][0]
        # Get the number of resources (based on the participating clients of the previous communication round).
        num_resources = len(individual_metrics_previous_comm_round)
        # Initialize the global lists.
        client_ids = []
        assignment_capacities = []
        time_costs = []
        energy_costs = []
        for client_id_str, client_metrics in individual_metrics_previous_comm_round.items():
            # Get the assignment capacity of the current client (number of examples available).
            client_proxy = available_clients_map[client_id_str]
            gpi = GetPropertiesIns({"num_{0}ing_examples_available".format(phase): "?"})
            client_prompted_properties = client_proxy.get_properties(gpi, timeout=9999)
            assignment_capacity = client_prompted_properties.properties["num_{0}ing_examples_available".format(phase)]
            # Initialize his assignment capacities list (based on his number of examples available).
            assignment_capacities_client = [i for i in range(0, assignment_capacity+1)]
            # Initialize his costs lists (based on the number of examples required).
            time_costs_client = [inf for _ in range(0, num_tasks+1)]
            energy_costs_client = [inf for _ in range(0, num_tasks+1)]
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
        print(optimal_schedule, minimal_makespan, minimal_energy_consumption)
        # Get the list of indices of the selected clients.
        selected_clients_index = [index for index, value in enumerate(optimal_schedule) if value > 0]
        # Append their corresponding proxy object into the list of selected clients.
        for selected_client_index in selected_clients_index:
            selected_client_id_str = client_ids[selected_client_index]
            selected_client_proxy = available_clients_map[selected_client_id_str]
            selected_clients.append(selected_client_proxy)
    return selected_clients
