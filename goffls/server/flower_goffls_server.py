from copy import deepcopy
from dateutil import parser
from logging import Logger
from time import perf_counter
from typing import Dict, List, Optional, Tuple, Union

from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, GetPropertiesIns, Metrics, NDArrays, Parameters, \
    parameters_to_ndarrays, Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.strategy import Strategy

from goffls.client_selector.flower_ecmtc import select_clients_using_ecmtc
from goffls.client_selector.flower_elastic_adapted import select_clients_using_elastic_adapted
from goffls.client_selector.flower_mec import select_clients_using_mec
from goffls.client_selector.flower_olar_adapted import select_clients_using_olar_adapted
from goffls.client_selector.flower_random import select_clients_using_random
from goffls.metrics_aggregator.flower_weighted_average import aggregate_loss_by_weighted_average, \
    aggregate_metrics_by_weighted_average
from goffls.model_aggregator.flower_weighted_average import aggregate_parameters_by_weighted_average
from goffls.utils.logger_util import log_message


class FlowerGOFFLSServer(Strategy):

    def __init__(self,
                 *,
                 id_: int,
                 fl_settings: dict,
                 client_selection_settings: dict,
                 model_aggregation_settings: dict,
                 metrics_aggregation_settings: dict,
                 fit_config: dict,
                 evaluate_config: dict,
                 initial_parameters: Optional[NDArrays],
                 logger: Logger) -> None:
        # Initialize the attributes.
        super().__init__()
        self._server_id = id_
        self._fl_settings = fl_settings
        self._client_selection_settings = client_selection_settings
        self._model_aggregation_settings = model_aggregation_settings
        self._metrics_aggregation_settings = metrics_aggregation_settings
        self._fit_config = fit_config
        self._evaluate_config = evaluate_config
        self._initial_parameters = initial_parameters
        self._logger = logger
        self._available_clients_map = {}
        self._selected_fit_clients_history = {}
        self._selected_evaluate_clients_history = {}
        self._individual_fit_metrics_history = {}
        self._aggregated_fit_metrics_history = {}
        self._individual_evaluate_metrics_history = {}
        self._aggregated_evaluate_metrics_history = {}

    def _set_attribute(self,
                       attribute_name: str,
                       attribute_value: any) -> None:
        setattr(self, attribute_name, attribute_value)

    def get_attribute(self,
                      attribute_name: str) -> any:
        return getattr(self, attribute_name)

    def initialize_parameters(self,
                              client_manager: ClientManager) -> Optional[Parameters]:
        """Initialize the model parameters.
           \nImplementation of the abstract method from the Strategy class."""
        # Get the initial parameters.
        initial_parameters = self.get_attribute("_initial_parameters")
        # Discard it from the memory.
        self._set_attribute("_initial_parameters", None)
        # Return the initial parameters.
        return initial_parameters

    def _update_fit_config(self,
                           comm_round: int) -> Optional[dict]:
        """Updates the training configuration (fit_config) that will be sent to clients.
        \nCalled by Flower prior to each training phase."""
        # Get the necessary attributes.
        server_id = self.get_attribute("_server_id")
        logger = self.get_attribute("_logger")
        # Get the training configuration.
        fit_config = self.get_attribute("_fit_config")
        # Update its current communication round.
        fit_config.update({"comm_round": comm_round})
        # Apply the training configuration changes.
        self._set_attribute("_fit_config", fit_config)
        # Replace None values to 'None' (necessary workaround on Flower).
        fit_config = {k: ("None" if v is None else v) for k, v in fit_config.items()}
        # Log the current training configuration (fit_config).
        message = "[Server {0} | Round {1}] Current fit_config: {2}".format(server_id, comm_round, fit_config)
        log_message(logger, message, "DEBUG")
        # Log the current communication round.
        message = "[Server {0} | Round {1}] Starting the training phase...".format(server_id, comm_round)
        log_message(logger, message, "INFO")
        # Return the training configuration (fit_config).
        return fit_config

    @staticmethod
    def _map_available_clients(client_manager: ClientManager) -> dict:
        available_clients = client_manager.all()
        available_clients_map = {}
        for _, client_proxy in available_clients.items():
            client_id_property = "client_id"
            client_hostname_property = "client_hostname"
            client_num_cpus_property = "client_num_cpus"
            client_cpu_cores_list_property = "client_cpu_cores_list"
            client_num_training_examples_available_property = "client_num_training_examples_available"
            client_num_testing_examples_available_property = "client_num_testing_examples_available"
            gpi = GetPropertiesIns({client_id_property: "?",
                                    client_hostname_property: "?",
                                    client_num_cpus_property: "?",
                                    client_cpu_cores_list_property: "?",
                                    client_num_training_examples_available_property: "?",
                                    client_num_testing_examples_available_property: "?"})
            client_prompted = client_proxy.get_properties(gpi, timeout=9999)
            client_id = client_prompted.properties[client_id_property]
            client_hostname = client_prompted.properties[client_hostname_property]
            client_num_cpus = client_prompted.properties[client_num_cpus_property]
            client_cpu_cores_list = client_prompted.properties[client_cpu_cores_list_property]
            client_num_training_examples_available = \
                client_prompted.properties[client_num_training_examples_available_property]
            client_num_testing_examples_available = \
                client_prompted.properties[client_num_testing_examples_available_property]
            client_id_str = "client_{0}".format(client_id)
            client_map = {"client_proxy": client_proxy,
                          "client_hostname": client_hostname,
                          "client_num_cpus": client_num_cpus,
                          "client_cpu_cores_list": client_cpu_cores_list,
                          "client_num_training_examples_available": client_num_training_examples_available,
                          "client_num_testing_examples_available": client_num_testing_examples_available}
            available_clients_map.update({client_id_str: client_map})
        return available_clients_map

    def _update_selected_fit_clients_history(self,
                                             comm_round: int,
                                             num_fit_tasks: int,
                                             available_fit_clients_map: dict,
                                             selection_duration: float,
                                             selected_fit_clients: list) -> None:
        selected_fit_clients_history = self.get_attribute("_selected_fit_clients_history")
        comm_round_key = "comm_round_{0}".format(comm_round)
        if comm_round_key not in selected_fit_clients_history:
            client_selection_settings = self.get_attribute("_client_selection_settings")
            client_selector = client_selection_settings["client_selector"]
            available_fit_clients_ids = list(available_fit_clients_map.keys())
            num_available_fit_clients = len(available_fit_clients_ids)
            num_selected_fit_clients = len(selected_fit_clients)
            available_fit_clients_proxies = [client_values["client_proxy"]
                                             for client_values in list(available_fit_clients_map.values())]
            selected_fit_clients_ids = []
            for client in selected_fit_clients:
                client_proxy = client["client_proxy"]
                client_index = available_fit_clients_proxies.index(client_proxy)
                client_id_str = available_fit_clients_ids[client_index]
                selected_fit_clients_ids.append(client_id_str)
            comm_round_values = {"client_selector": client_selector,
                                 "selection_duration": selection_duration,
                                 "num_tasks": num_fit_tasks,
                                 "num_available_clients": num_available_fit_clients,
                                 "num_selected_clients": num_selected_fit_clients,
                                 "selected_clients": selected_fit_clients_ids}
            comm_round_selected_fit_clients = {comm_round_key: comm_round_values}
            selected_fit_clients_history.update(comm_round_selected_fit_clients)
            self._set_attribute("_selected_fit_clients_history", selected_fit_clients_history)

    def configure_fit(self,
                      server_round: int,
                      parameters: Parameters,
                      client_manager: ClientManager) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training.
           \nImplementation of the abstract method from the Strategy class."""
        # Get the necessary attributes.
        fl_settings = self.get_attribute("_fl_settings")
        enable_training = fl_settings["enable_training"]
        num_fit_tasks = fl_settings["num_fit_tasks"]
        client_selection_settings = self.get_attribute("_client_selection_settings")
        client_selector = client_selection_settings["client_selector"]
        individual_fit_metrics_history = self.get_attribute("_individual_fit_metrics_history")
        logger = self.get_attribute("_logger")
        # Wait for the initial clients to connect before starting the first round.
        if server_round == 1:
            wait_for_initial_clients = fl_settings["wait_for_initial_clients"]
            num_clients_to_wait = wait_for_initial_clients["num_clients_to_wait"]
            waiting_timeout_in_seconds = wait_for_initial_clients["waiting_timeout_in_seconds"]
            client_manager.wait_for(num_clients_to_wait, waiting_timeout_in_seconds)
        # Do not configure federated training if it is not enabled.
        if not enable_training:
            return []
        # Set the base training configuration (fit_config).
        fit_config = self._update_fit_config(server_round)
        # Initialize the list of the selected clients for training (selected_fit_clients).
        selected_fit_clients = []
        # Map the available fit clients.
        available_fit_clients_map = self._map_available_clients(client_manager)
        # Set the available clients map.
        self._set_attribute("_available_clients_map", available_fit_clients_map)
        # Get the number of available fit clients.
        num_available_fit_clients = len(available_fit_clients_map)
        # Set the phase value.
        phase = "train"
        # Start the clients selection duration timer.
        selection_duration_start = perf_counter()
        if client_selector == "Random":
            # Select clients using the Random algorithm.
            fit_clients_fraction = client_selection_settings["fit_clients_fraction"]
            selected_fit_clients = select_clients_using_random(client_manager,
                                                               phase,
                                                               num_available_fit_clients,
                                                               fit_clients_fraction,
                                                               logger)
        elif client_selector == "MEC":
            # Select clients using the MEC algorithm.
            history_checker = client_selection_settings["history_checker"]
            assignment_capacities_init_settings = client_selection_settings["assignment_capacities_init_settings"]
            selected_fit_clients = select_clients_using_mec(server_round,
                                                            phase,
                                                            num_fit_tasks,
                                                            available_fit_clients_map,
                                                            individual_fit_metrics_history,
                                                            history_checker,
                                                            assignment_capacities_init_settings,
                                                            logger)
        elif client_selector == "ECMTC":
            # Select clients using the ECMTC algorithm.
            history_checker = client_selection_settings["history_checker"]
            assignment_capacities_init_settings = client_selection_settings["assignment_capacities_init_settings"]
            fit_deadline_in_seconds = client_selection_settings["fit_deadline_in_seconds"]
            selected_fit_clients = select_clients_using_ecmtc(server_round,
                                                              phase,
                                                              num_fit_tasks,
                                                              fit_deadline_in_seconds,
                                                              available_fit_clients_map,
                                                              individual_fit_metrics_history,
                                                              history_checker,
                                                              assignment_capacities_init_settings,
                                                              logger)
        elif client_selector == "OLAR":
            # Select clients using the OLAR adapted algorithm.
            history_checker = client_selection_settings["history_checker"]
            assignment_capacities_init_settings = client_selection_settings["assignment_capacities_init_settings"]
            selected_fit_clients = select_clients_using_olar_adapted(server_round,
                                                                     phase,
                                                                     num_fit_tasks,
                                                                     available_fit_clients_map,
                                                                     individual_fit_metrics_history,
                                                                     history_checker,
                                                                     assignment_capacities_init_settings,
                                                                     logger)
        elif client_selector == "ELASTIC":
            # Select clients using the ELASTIC adapted algorithm.
            history_checker = client_selection_settings["history_checker"]
            assignment_capacities_init_settings = client_selection_settings["assignment_capacities_init_settings"]
            fit_deadline_in_seconds = client_selection_settings["fit_deadline_in_seconds"]
            alpha = client_selection_settings["alpha"]
            selected_fit_clients = select_clients_using_elastic_adapted(server_round,
                                                                        phase,
                                                                        num_fit_tasks,
                                                                        fit_deadline_in_seconds,
                                                                        alpha,
                                                                        available_fit_clients_map,
                                                                        individual_fit_metrics_history,
                                                                        history_checker,
                                                                        assignment_capacities_init_settings,
                                                                        logger)
        # Get the clients selection duration.
        selection_duration = perf_counter() - selection_duration_start
        # Update the history of selected clients for training (selected_fit_clients).
        self._update_selected_fit_clients_history(server_round,
                                                  num_fit_tasks,
                                                  available_fit_clients_map,
                                                  selection_duration,
                                                  selected_fit_clients)
        # Set the list of (fit_client_proxy, fit_client_instructions) pairs.
        fit_pairs = []
        for selected_fit_client in selected_fit_clients:
            selected_fit_client_proxy = selected_fit_client["client_proxy"]
            selected_fit_client_config = deepcopy(fit_config)
            if "client_num_tasks_scheduled" in selected_fit_client:
                num_training_examples_to_use = selected_fit_client["client_num_tasks_scheduled"]
                selected_fit_client_config.update({"num_training_examples_to_use": num_training_examples_to_use})
            selected_fit_client_instructions = FitIns(parameters, selected_fit_client_config)
            fit_pairs.append((selected_fit_client_proxy, selected_fit_client_instructions))
        # Return the list of (fit_client_proxy, fit_client_instructions) pairs.
        return fit_pairs

    @staticmethod
    def _calculate_energy_timestamp_metrics_of_client_hostname(metrics: list[tuple[int, Metrics]],
                                                               phase: str,
                                                               hostname: str) -> None:
        clients_key = "{0}ing_clients".format(phase)
        energy_cpu_key = "{0}ing_energy_cpu".format(phase)
        energy_cpu_timestamp = energy_cpu_key + "_"
        energy_cpu_timestamps = {}
        for metric_tuple in metrics:
            client_metrics = metric_tuple[1]
            client_hostname = client_metrics["client_hostname"]
            if client_hostname == hostname:
                client_id = client_metrics["client_id"]
                client_num_cpus = client_metrics["client_num_cpus"]
                client_energy_cpu_timestamps = []
                client_metrics_to_delete = []
                for client_metric_key, client_metric_value in client_metrics.items():
                    if energy_cpu_timestamp in client_metric_key:
                        client_timestamp = client_metric_key.split(energy_cpu_timestamp, 1)[1]
                        client_energy_cpu_timestamps.append({client_timestamp: client_metric_value})
                        client_metrics_to_delete.append(client_metric_key)
                for client_metric_to_delete in client_metrics_to_delete:
                    del client_metrics[client_metric_to_delete]
                client_energy_cpu_timestamps = sorted(client_energy_cpu_timestamps, key=lambda x: list(x.keys()))
                for client_energy_cpu_timestamp in client_energy_cpu_timestamps:
                    for client_timestamp, client_metric_value in client_energy_cpu_timestamp.items():
                        if client_timestamp not in energy_cpu_timestamps:
                            energy_cpu_timestamp_dict = {energy_cpu_key: [client_metric_value],
                                                         clients_key: [client_id],
                                                         "total_cpus": client_num_cpus}
                            energy_cpu_timestamps.update({client_timestamp: energy_cpu_timestamp_dict})
                        else:
                            energy_cpu_timestamps[client_timestamp][energy_cpu_key].append(client_metric_value)
                            energy_cpu_timestamps[client_timestamp][clients_key].append(client_id)
                            total_cpus = energy_cpu_timestamps[client_timestamp]["total_cpus"] + client_num_cpus
                            energy_cpu_timestamps[client_timestamp]["total_cpus"] = total_cpus
        energy_cpu_timestamps = dict(sorted(energy_cpu_timestamps.items()))
        for metric_tuple in metrics:
            client_metrics = metric_tuple[1]
            client_hostname = client_metrics["client_hostname"]
            if client_hostname == hostname:
                client_id = client_metrics["client_id"]
                client_num_cpus = client_metrics["client_num_cpus"]
                client_start_timestamp = client_metrics["{0}ing_start_timestamp".format(phase)]
                client_end_timestamp = client_metrics["{0}ing_end_timestamp".format(phase)]
                client_energy_cpu = 0
                for energy_cpu_timestamp, energy_cpu_timestamp_dict in energy_cpu_timestamps.items():
                    if energy_cpu_key in energy_cpu_timestamp_dict:
                        clients = energy_cpu_timestamp_dict[clients_key]
                        if client_id in clients:
                            client_index = clients.index(client_id)
                            energy_cpu = energy_cpu_timestamp_dict[energy_cpu_key][client_index]
                            total_cpus = energy_cpu_timestamp_dict["total_cpus"]
                            microseconds_diff_factor = 1
                            if str(client_start_timestamp).startswith(energy_cpu_timestamp):
                                client_start_timestamp = parser.parse(client_start_timestamp).replace(tzinfo=None)
                                client_start_timestamp_microseconds = client_start_timestamp.microsecond
                                if client_start_timestamp_microseconds > 0:
                                    microseconds_diff_factor = (1000000 - client_start_timestamp_microseconds) / 1000000
                            if str(client_end_timestamp).startswith(energy_cpu_timestamp):
                                client_end_timestamp = parser.parse(client_end_timestamp).replace(tzinfo=None)
                                client_end_timestamp_microseconds = client_end_timestamp.microsecond
                                if client_end_timestamp_microseconds:
                                    microseconds_diff_factor \
                                        = 1 - ((1000000 - client_end_timestamp_microseconds) / 1000000)
                            client_energy_cpu_t = (client_num_cpus * energy_cpu) / total_cpus
                            client_energy_cpu_t *= microseconds_diff_factor
                            client_energy_cpu += client_energy_cpu_t
                        client_metrics.update({energy_cpu_key: client_energy_cpu})

    def _calculate_energy_timestamp_metrics(self,
                                            metrics: list[tuple[int, Metrics]],
                                            phase: str) -> None:
        hostnames = []
        for metric_tuple in metrics:
            client_metrics = metric_tuple[1]
            client_hostname = client_metrics["client_hostname"]
            hostnames.append(client_hostname)
        hostnames = list(set(hostnames))
        for hostname in hostnames:
            self._calculate_energy_timestamp_metrics_of_client_hostname(metrics, phase, hostname)

    def _update_individual_fit_metrics_history(self,
                                               comm_round: int,
                                               fit_metrics: list[tuple[int, Metrics]]) -> None:
        individual_fit_metrics_history = self.get_attribute("_individual_fit_metrics_history")
        comm_round_key = "comm_round_{0}".format(comm_round)
        if comm_round_key not in individual_fit_metrics_history:
            client_selection_settings = self.get_attribute("_client_selection_settings")
            client_selector = client_selection_settings["client_selector"]
            selected_fit_clients_history = self.get_attribute("_selected_fit_clients_history")
            num_tasks = selected_fit_clients_history[comm_round_key]["num_tasks"]
            num_available_clients = selected_fit_clients_history[comm_round_key]["num_available_clients"]
            fit_clients_metrics = []
            for metric_tuple in fit_metrics:
                client_metrics = metric_tuple[1]
                client_id = client_metrics["client_id"]
                client_id_str = "client_{0}".format(client_id)
                client_metrics_copy = client_metrics.copy()
                client_metrics_copy.pop("client_id")
                client_metrics_copy["hostname"] = client_metrics_copy.pop("client_hostname")
                client_metrics_copy["num_cpus"] = client_metrics_copy.pop("client_num_cpus")
                client_metrics_copy["cpu_cores_list"] = client_metrics_copy.pop("client_cpu_cores_list")
                fit_clients_metrics.append({client_id_str: client_metrics_copy})
            comm_round_values = {"client_selector": client_selector,
                                 "num_tasks": num_tasks,
                                 "num_available_clients": num_available_clients,
                                 "clients_metrics_dicts": fit_clients_metrics}
            comm_round_individual_fit_metrics = {comm_round_key: comm_round_values}
            individual_fit_metrics_history.update(comm_round_individual_fit_metrics)
            self._set_attribute("_individual_fit_metrics_history", individual_fit_metrics_history)

    @staticmethod
    def _remove_undesired_metrics(metrics_tuples: list[tuple[int, Metrics]],
                                  undesired_metrics: list) -> list[tuple[int, Metrics]]:
        for metric_tuple in metrics_tuples:
            client_metrics = metric_tuple[1]
            for undesired_metric in undesired_metrics:
                if undesired_metric in client_metrics:
                    del client_metrics[undesired_metric]
        return metrics_tuples

    def _update_aggregated_fit_metrics_history(self,
                                               comm_round: int,
                                               metrics_aggregator: str,
                                               aggregated_fit_metrics: dict) -> None:
        aggregated_fit_metrics_history = self.get_attribute("_aggregated_fit_metrics_history")
        comm_round_key = "comm_round_{0}".format(comm_round)
        if comm_round_key not in aggregated_fit_metrics_history:
            client_selection_settings = self.get_attribute("_client_selection_settings")
            client_selector = client_selection_settings["client_selector"]
            selected_fit_clients_history = self.get_attribute("_selected_fit_clients_history")
            num_tasks = selected_fit_clients_history[comm_round_key]["num_tasks"]
            num_available_clients = selected_fit_clients_history[comm_round_key]["num_available_clients"]
            comm_round_values = {"client_selector": client_selector,
                                 "metrics_aggregator": metrics_aggregator,
                                 "num_tasks": num_tasks,
                                 "num_available_clients": num_available_clients,
                                 "aggregated_metrics": aggregated_fit_metrics}
            comm_round_aggregated_fit_metrics = {comm_round_key: comm_round_values}
            aggregated_fit_metrics_history.update(comm_round_aggregated_fit_metrics)
            self._set_attribute("_aggregated_fit_metrics_history", aggregated_fit_metrics_history)

    def _aggregate_fit_metrics(self,
                               comm_round: int,
                               fit_metrics: list[tuple[int, Metrics]]) -> Optional[Metrics]:
        """Aggregates the training metrics (fit_metrics).
        \nCalled by Flower after each training phase."""
        # Get the necessary attributes.
        metrics_aggregation_settings = self.get_attribute("_metrics_aggregation_settings")
        metrics_aggregator = metrics_aggregation_settings["metrics_aggregator"]
        server_id = self.get_attribute("_server_id")
        logger = self.get_attribute("_logger")
        available_clients_map = self.get_attribute("_available_clients_map")
        # Update (from the available clients map) the hostname and number of CPUs for each participating client.
        for metric_tuple in fit_metrics:
            client_metrics = metric_tuple[1]
            client_id = client_metrics["client_id"]
            client_id_str = "client_{0}".format(client_id)
            client_hostname = available_clients_map[client_id_str]["client_hostname"]
            client_num_cpus = available_clients_map[client_id_str]["client_num_cpus"]
            client_cpu_cores_list = available_clients_map[client_id_str]["client_cpu_cores_list"]
            client_metrics.update({"client_hostname": client_hostname,
                                   "client_num_cpus": client_num_cpus,
                                   "client_cpu_cores_list": client_cpu_cores_list})
        # Set the phase value.
        phase = "train"
        # Calculate the energy timestamp metrics.
        self._calculate_energy_timestamp_metrics(fit_metrics, phase)
        # Update the individual training metrics history.
        self._update_individual_fit_metrics_history(comm_round, fit_metrics)
        # Remove the undesired metrics, if any.
        undesired_metrics = ["client_id", "client_hostname", "client_num_cpus", "client_cpu_cores_list",
                             "training_start_timestamp", "training_end_timestamp"]
        fit_metrics = self._remove_undesired_metrics(fit_metrics, undesired_metrics)
        # Initialize the aggregated training metrics dictionary (aggregated_fit_metrics).
        aggregated_fit_metrics = {}
        # Aggregate the training metrics according to the user-defined aggregator.
        if metrics_aggregator == "Weighted_Average":
            aggregated_fit_metrics = aggregate_metrics_by_weighted_average(fit_metrics)
        # Update the aggregated training metrics history.
        self._update_aggregated_fit_metrics_history(comm_round, metrics_aggregator, aggregated_fit_metrics)
        # Get the number of participating clients.
        num_participating_clients = len(fit_metrics)
        num_participating_clients_str = "".join([str(num_participating_clients),
                                                 " Clients" if num_participating_clients > 1 else " Client"])
        # Log the aggregated training metrics.
        message = "[Server {0} | Round {1}] Aggregated training metrics ({2} of {3}): {4}" \
                  .format(server_id, comm_round, metrics_aggregator, num_participating_clients_str,
                          aggregated_fit_metrics)
        log_message(logger, message, "DEBUG")
        # Return the aggregated training metrics (aggregated_fit_metrics).
        return aggregated_fit_metrics

    def aggregate_fit(self,
                      server_round: int,
                      results: List[Tuple[ClientProxy, FitRes]],
                      failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]) \
            -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate the model parameters and the training metrics based on the fit_results.
           \nImplementation of the abstract method from the Strategy class."""
        # Get the necessary attributes.
        fl_settings = self.get_attribute("_fl_settings")
        accept_clients_failures = fl_settings["accept_clients_failures"]
        model_aggregation_settings = self.get_attribute("_model_aggregation_settings")
        model_aggregator = model_aggregation_settings["model_aggregator"]
        # Do not aggregate if there are no results or if there are clients' failures and failures are not accepted.
        if not results or (failures and not accept_clients_failures):
            return None, {}
        # Initialize the aggregated model parameters.
        aggregated_model_parameters = None
        if model_aggregator == "FedAvg":
            # Aggregate the model parameters by weighted average.
            inplace_aggregation = model_aggregation_settings["inplace_aggregation"]
            aggregated_model_parameters = aggregate_parameters_by_weighted_average(results, inplace_aggregation)
        # Aggregate the training metrics.
        fit_metrics = [(result.num_examples, result.metrics) for _, result in results]
        aggregated_fit_metrics = self._aggregate_fit_metrics(server_round, fit_metrics)
        # Return the aggregated model parameters and aggregated training metrics.
        return aggregated_model_parameters, aggregated_fit_metrics

    def _update_evaluate_config(self,
                                comm_round: int) -> Optional[dict]:
        """Updates the testing configuration (evaluate_config) that will be sent to clients.
        \nCalled by Flower prior to each testing phase."""
        # Get the necessary attributes.
        server_id = self.get_attribute("_server_id")
        logger = self.get_attribute("_logger")
        # Get the testing configuration.
        evaluate_config = self.get_attribute("_evaluate_config")
        # Update its current communication round.
        evaluate_config.update({"comm_round": comm_round})
        # Apply the testing configuration changes.
        self._set_attribute("_evaluate_config", evaluate_config)
        # Replace None values to 'None' (necessary workaround on Flower).
        evaluate_config = {k: ("None" if v is None else v) for k, v in evaluate_config.items()}
        # Log the current testing configuration (evaluate_config).
        message = "[Server {0} | Round {1}] Current evaluate_config: {2}".format(server_id, comm_round, evaluate_config)
        log_message(logger, message, "DEBUG")
        # Log the current communication round.
        message = "[Server {0} | Round {1}] Starting the testing phase...".format(server_id, comm_round)
        log_message(logger, message, "INFO")
        # Return the testing configuration (evaluate_config).
        return evaluate_config

    def _update_selected_evaluate_clients_history(self,
                                                  comm_round: int,
                                                  num_evaluate_tasks: int,
                                                  available_evaluate_clients_map: dict,
                                                  selection_duration: float,
                                                  selected_evaluate_clients: list) -> None:
        selected_evaluate_clients_history = self.get_attribute("_selected_evaluate_clients_history")
        comm_round_key = "comm_round_{0}".format(comm_round)
        if comm_round_key not in selected_evaluate_clients_history:
            client_selection_settings = self.get_attribute("_client_selection_settings")
            client_selector = client_selection_settings["client_selector"]
            available_evaluate_clients_ids = list(available_evaluate_clients_map.keys())
            num_available_evaluate_clients = len(available_evaluate_clients_ids)
            num_selected_evaluate_clients = len(selected_evaluate_clients)
            available_evaluate_clients_proxies = [client_values["client_proxy"]
                                                  for client_values in list(available_evaluate_clients_map.values())]
            selected_evaluate_clients_ids = []
            for client in selected_evaluate_clients:
                client_proxy = client["client_proxy"]
                client_index = available_evaluate_clients_proxies.index(client_proxy)
                client_id_str = available_evaluate_clients_ids[client_index]
                selected_evaluate_clients_ids.append(client_id_str)
            comm_round_values = {"client_selector": client_selector,
                                 "selection_duration": selection_duration,
                                 "num_tasks": num_evaluate_tasks,
                                 "num_available_clients": num_available_evaluate_clients,
                                 "num_selected_clients": num_selected_evaluate_clients,
                                 "selected_clients": selected_evaluate_clients_ids}
            comm_round_selected_evaluate_clients = {comm_round_key: comm_round_values}
            selected_evaluate_clients_history.update(comm_round_selected_evaluate_clients)
            self._set_attribute("_selected_evaluate_clients_history", selected_evaluate_clients_history)

    def configure_evaluate(self,
                           server_round: int,
                           parameters: Parameters,
                           client_manager: ClientManager) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of testing.
           \nImplementation of the abstract method from the Strategy class."""
        # Get the necessary attributes.
        fl_settings = self.get_attribute("_fl_settings")
        enable_testing = fl_settings["enable_testing"]
        num_evaluate_tasks = fl_settings["num_evaluate_tasks"]
        client_selection_settings = self.get_attribute("_client_selection_settings")
        client_selector = client_selection_settings["client_selector"]
        individual_evaluate_metrics_history = self.get_attribute("_individual_evaluate_metrics_history")
        logger = self.get_attribute("_logger")
        # Do not configure federated testing if it is not enabled.
        if not enable_testing:
            return []
        # Set the base testing configuration (evaluate_config).
        evaluate_config = self._update_evaluate_config(server_round)
        # Initialize the list of the selected clients for testing (selected_evaluate_clients).
        selected_evaluate_clients = []
        # Map the available evaluate clients.
        available_evaluate_clients_map = self._map_available_clients(client_manager)
        # Set the available clients map.
        self._set_attribute("_available_clients_map", available_evaluate_clients_map)
        # Get the number of available evaluate clients.
        num_available_evaluate_clients = len(available_evaluate_clients_map)
        # Set the phase value.
        phase = "test"
        # Start the clients selection duration timer.
        selection_duration_start = perf_counter()
        if client_selector == "Random":
            # Select clients for testing randomly.
            evaluate_clients_fraction = client_selection_settings["evaluate_clients_fraction"]
            selected_evaluate_clients = select_clients_using_random(client_manager,
                                                                    phase,
                                                                    num_available_evaluate_clients,
                                                                    evaluate_clients_fraction,
                                                                    logger)
        elif client_selector == "MEC":
            # Select clients using the MEC algorithm.
            history_checker = client_selection_settings["history_checker"]
            assignment_capacities_init_settings = client_selection_settings["assignment_capacities_init_settings"]
            selected_evaluate_clients = select_clients_using_mec(server_round,
                                                                 phase,
                                                                 num_evaluate_tasks,
                                                                 available_evaluate_clients_map,
                                                                 individual_evaluate_metrics_history,
                                                                 history_checker,
                                                                 assignment_capacities_init_settings,
                                                                 logger)
        elif client_selector == "ECMTC":
            # Select clients using the ECMTC algorithm.
            history_checker = client_selection_settings["history_checker"]
            assignment_capacities_init_settings = client_selection_settings["assignment_capacities_init_settings"]
            evaluate_deadline_in_seconds = client_selection_settings["evaluate_deadline_in_seconds"]
            selected_evaluate_clients = select_clients_using_ecmtc(server_round,
                                                                   phase,
                                                                   num_evaluate_tasks,
                                                                   evaluate_deadline_in_seconds,
                                                                   available_evaluate_clients_map,
                                                                   individual_evaluate_metrics_history,
                                                                   history_checker,
                                                                   assignment_capacities_init_settings,
                                                                   logger)
        elif client_selector == "OLAR":
            # Select clients using the OLAR adapted algorithm.
            history_checker = client_selection_settings["history_checker"]
            assignment_capacities_init_settings = client_selection_settings["assignment_capacities_init_settings"]
            selected_evaluate_clients = select_clients_using_olar_adapted(server_round,
                                                                          phase,
                                                                          num_evaluate_tasks,
                                                                          available_evaluate_clients_map,
                                                                          individual_evaluate_metrics_history,
                                                                          history_checker,
                                                                          assignment_capacities_init_settings,
                                                                          logger)
        elif client_selector == "ELASTIC":
            # Select clients using the ELASTIC adapted algorithm.
            history_checker = client_selection_settings["history_checker"]
            assignment_capacities_init_settings = client_selection_settings["assignment_capacities_init_settings"]
            evaluate_deadline_in_seconds = client_selection_settings["evaluate_deadline_in_seconds"]
            alpha = client_selection_settings["alpha"]
            selected_evaluate_clients = select_clients_using_elastic_adapted(server_round,
                                                                             phase,
                                                                             num_evaluate_tasks,
                                                                             evaluate_deadline_in_seconds,
                                                                             alpha,
                                                                             available_evaluate_clients_map,
                                                                             individual_evaluate_metrics_history,
                                                                             history_checker,
                                                                             assignment_capacities_init_settings,
                                                                             logger)
        # Get the clients selection duration.
        selection_duration = perf_counter() - selection_duration_start
        # Update the history of selected clients for testing (selected_evaluate_clients).
        self._update_selected_evaluate_clients_history(server_round,
                                                       num_evaluate_tasks,
                                                       available_evaluate_clients_map,
                                                       selection_duration,
                                                       selected_evaluate_clients)
        # Set the list of (evaluate_client_proxy, evaluate_client_instructions) pairs.
        evaluate_pairs = []
        for selected_evaluate_client in selected_evaluate_clients:
            selected_evaluate_client_proxy = selected_evaluate_client["client_proxy"]
            selected_evaluate_client_config = deepcopy(evaluate_config)
            if "client_num_tasks_scheduled" in selected_evaluate_client:
                num_testing_examples_to_use = selected_evaluate_client["client_num_tasks_scheduled"]
                selected_evaluate_client_config.update({"num_testing_examples_to_use": num_testing_examples_to_use})
            selected_evaluate_client_instructions = EvaluateIns(parameters, selected_evaluate_client_config)
            evaluate_pairs.append((selected_evaluate_client_proxy, selected_evaluate_client_instructions))
        # Return the list of (evaluate_client_proxy, evaluate_client_instructions) pairs.
        return evaluate_pairs

    def _update_individual_evaluate_metrics_history(self,
                                                    comm_round: int,
                                                    evaluate_metrics: list[tuple[int, Metrics]]) -> None:
        individual_evaluate_metrics_history = self.get_attribute("_individual_evaluate_metrics_history")
        comm_round_key = "comm_round_{0}".format(comm_round)
        if comm_round_key not in individual_evaluate_metrics_history:
            client_selection_settings = self.get_attribute("_client_selection_settings")
            client_selector = client_selection_settings["client_selector"]
            selected_evaluate_clients_history = self.get_attribute("_selected_evaluate_clients_history")
            num_tasks = selected_evaluate_clients_history[comm_round_key]["num_tasks"]
            num_available_clients = selected_evaluate_clients_history[comm_round_key]["num_available_clients"]
            evaluate_clients_metrics = []
            for metric_tuple in evaluate_metrics:
                client_metrics = metric_tuple[1]
                client_id = client_metrics["client_id"]
                client_id_str = "client_{0}".format(client_id)
                client_metrics_copy = client_metrics.copy()
                client_metrics_copy.pop("client_id")
                client_metrics_copy["hostname"] = client_metrics_copy.pop("client_hostname")
                client_metrics_copy["num_cpus"] = client_metrics_copy.pop("client_num_cpus")
                client_metrics_copy["cpu_cores_list"] = client_metrics_copy.pop("client_cpu_cores_list")
                evaluate_clients_metrics.append({client_id_str: client_metrics_copy})
            comm_round_values = {"client_selector": client_selector,
                                 "num_tasks": num_tasks,
                                 "num_available_clients": num_available_clients,
                                 "clients_metrics_dicts": evaluate_clients_metrics}
            comm_round_individual_evaluate_metrics = {comm_round_key: comm_round_values}
            individual_evaluate_metrics_history.update(comm_round_individual_evaluate_metrics)
            self._set_attribute("_individual_evaluate_metrics_history", individual_evaluate_metrics_history)

    def _update_aggregated_evaluate_metrics_history(self,
                                                    comm_round: int,
                                                    metrics_aggregator: str,
                                                    aggregated_evaluate_metrics: dict) -> None:
        aggregated_evaluate_metrics_history = self.get_attribute("_aggregated_evaluate_metrics_history")
        comm_round_key = "comm_round_{0}".format(comm_round)
        if comm_round_key not in aggregated_evaluate_metrics_history:
            client_selection_settings = self.get_attribute("_client_selection_settings")
            client_selector = client_selection_settings["client_selector"]
            selected_evaluate_clients_history = self.get_attribute("_selected_evaluate_clients_history")
            num_tasks = selected_evaluate_clients_history[comm_round_key]["num_tasks"]
            num_available_clients = selected_evaluate_clients_history[comm_round_key]["num_available_clients"]
            comm_round_values = {"client_selector": client_selector,
                                 "metrics_aggregator": metrics_aggregator,
                                 "num_tasks": num_tasks,
                                 "num_available_clients": num_available_clients,
                                 "aggregated_metrics": aggregated_evaluate_metrics}
            comm_round_aggregated_evaluate_metrics = {comm_round_key: comm_round_values}
            aggregated_evaluate_metrics_history.update(comm_round_aggregated_evaluate_metrics)
            self._set_attribute("_aggregated_evaluate_metrics_history", aggregated_evaluate_metrics_history)

    def _aggregate_evaluate_metrics(self,
                                    comm_round: int,
                                    evaluate_metrics: list[tuple[int, Metrics]]) -> Optional[Metrics]:
        """Aggregates the testing metrics (evaluate_metrics).
        \nCalled by Flower after each testing phase."""
        # Get the necessary attributes.
        metrics_aggregation_settings = self.get_attribute("_metrics_aggregation_settings")
        metrics_aggregator = metrics_aggregation_settings["metrics_aggregator"]
        server_id = self.get_attribute("_server_id")
        logger = self.get_attribute("_logger")
        available_clients_map = self.get_attribute("_available_clients_map")
        # Update (from the available clients map) the hostname and number of CPUs for each participating client.
        for metric_tuple in evaluate_metrics:
            client_metrics = metric_tuple[1]
            client_id = client_metrics["client_id"]
            client_id_str = "client_{0}".format(client_id)
            client_hostname = available_clients_map[client_id_str]["client_hostname"]
            client_num_cpus = available_clients_map[client_id_str]["client_num_cpus"]
            client_cpu_cores_list = available_clients_map[client_id_str]["client_cpu_cores_list"]
            client_metrics.update({"client_hostname": client_hostname,
                                   "client_num_cpus": client_num_cpus,
                                   "client_cpu_cores_list": client_cpu_cores_list})
        # Set the phase value.
        phase = "test"
        # Calculate the energy timestamp metrics.
        self._calculate_energy_timestamp_metrics(evaluate_metrics, phase)
        # Update the individual testing metrics history.
        self._update_individual_evaluate_metrics_history(comm_round, evaluate_metrics)
        # Remove the undesired metrics, if any.
        undesired_metrics = ["client_id", "client_hostname", "client_num_cpus", "client_cpu_cores_list",
                             "testing_start_timestamp", "testing_end_timestamp"]
        evaluate_metrics = self._remove_undesired_metrics(evaluate_metrics, undesired_metrics)
        # Initialize the aggregated testing metrics dictionary (aggregated_evaluate_metrics).
        aggregated_evaluate_metrics = {}
        # Aggregate the testing metrics according to the user-defined aggregator.
        if metrics_aggregator == "Weighted_Average":
            aggregated_evaluate_metrics = aggregate_metrics_by_weighted_average(evaluate_metrics)
        # Update the aggregated testing metrics history.
        self._update_aggregated_evaluate_metrics_history(comm_round, metrics_aggregator, aggregated_evaluate_metrics)
        # Get the number of participating clients.
        num_participating_clients = len(evaluate_metrics)
        num_participating_clients_str = "".join([str(num_participating_clients),
                                                 " Clients" if num_participating_clients > 1 else " Client"])
        # Log the aggregated testing metrics.
        message = "[Server {0} | Round {1}] Aggregated testing metrics ({2} of {3}): {4}" \
                  .format(server_id, comm_round, metrics_aggregator, num_participating_clients_str,
                          aggregated_evaluate_metrics)
        log_message(logger, message, "DEBUG")
        # Return the aggregated testing metrics (aggregated_evaluate_metrics).
        return aggregated_evaluate_metrics

    def aggregate_evaluate(self,
                           server_round: int,
                           results: List[Tuple[ClientProxy, EvaluateRes]],
                           failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]]) \
            -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate the testing metrics based on the evaluate_results.
           \nImplementation of the abstract method from the Strategy class."""
        # Get the necessary attributes.
        fl_settings = self.get_attribute("_fl_settings")
        accept_clients_failures = fl_settings["accept_clients_failures"]
        model_aggregation_settings = self.get_attribute("_model_aggregation_settings")
        model_aggregator = model_aggregation_settings["model_aggregator"]
        # Do not aggregate if there are no results or if there are clients' failures and failures are not accepted.
        if not results or (failures and not accept_clients_failures):
            return None, {}
        # Initialize the aggregated loss value.
        aggregated_loss = 0
        if model_aggregator == "FedAvg":
            # Aggregate the loss by weighted average.
            aggregated_loss = aggregate_loss_by_weighted_average(results)
        # Aggregate the testing metrics.
        evaluate_metrics = [(result.num_examples, result.metrics) for _, result in results]
        aggregated_evaluate_metrics = self._aggregate_evaluate_metrics(server_round, evaluate_metrics)
        # Return the aggregated loss and aggregated testing metrics.
        return aggregated_loss, aggregated_evaluate_metrics

    @staticmethod
    def _evaluate_centrally_on_server(comm_round: int,
                                      model_parameters: NDArrays,
                                      evaluate_config: dict) -> Optional[Metrics]:
        """Evaluates, in a centralized fashion (server-sided), the updated model parameters.
        \nCalled by Flower after each training phase.
        \nRequires a local testing dataset on the server.
        \nThe evaluation outcome will be stored in the 'losses_centralized' and 'metrics_centralized' dictionaries."""
        return None

    def evaluate(self,
                 server_round: int,
                 parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate in a centralized fashion the updated model parameters.
           \nImplementation of the abstract method from the Strategy class."""
        # Convert the model parameters to NumPy ndarrays.
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        # Evaluate the model parameters centrally on the server.
        eval_res = self._evaluate_centrally_on_server(server_round, parameters_ndarrays, {})
        # Verify if the outcome is empty.
        if eval_res is None:
            return None
        # If not, get the centralized loss and centralized testing metrics.
        centralized_loss, centralized_evaluate_metrics = eval_res
        # Return the centralized loss and centralized testing metrics.
        return centralized_loss, centralized_evaluate_metrics
