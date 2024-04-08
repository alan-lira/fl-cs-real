from copy import deepcopy
from logging import Logger
from time import perf_counter
from typing import Dict, List, Optional, Tuple, Union

from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, GetPropertiesIns, Metrics, NDArrays, Parameters, \
    parameters_to_ndarrays, Scalar
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.strategy import Strategy

from client_selection_approaches.flower.ecmtc import select_clients_using_ecmtc
from client_selection_approaches.flower.mec import select_clients_using_mec
from client_selection_approaches.flower.random import select_clients_randomly
from metrics_aggregation_approaches.flower.weighted_average import aggregate_loss_by_weighted_average, \
    aggregate_metrics_by_weighted_average
from model_aggregation_approaches.flower.weighted_average import aggregate_parameters_by_weighted_average
from utils.logger_util import log_message


class FlowerGOFFLSServer(Strategy):

    def __init__(self,
                 *,
                 server_id: int,
                 enable_training: bool,
                 enable_testing: bool,
                 accept_clients_failures: bool,
                 client_selection_settings: dict,
                 model_aggregation_settings: dict,
                 metrics_aggregation_settings: dict,
                 fit_config: dict,
                 evaluate_config: dict,
                 initial_parameters: Optional[NDArrays],
                 logger: Logger) -> None:
        super().__init__()
        self._server_id = server_id
        self._enable_training = enable_training
        self._enable_testing = enable_testing
        self._accept_clients_failures = accept_clients_failures
        self._client_selection_settings = client_selection_settings
        self._model_aggregation_settings = model_aggregation_settings
        self._metrics_aggregation_settings = metrics_aggregation_settings
        self._fit_config = fit_config
        self._evaluate_config = evaluate_config
        self._initial_parameters = initial_parameters
        self._logger = logger
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
           \nImplementation of the abstract method of the Strategy class."""
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
    def _map_available_clients(available_clients: dict) -> dict:
        available_clients_map = {}
        for _, client_proxy in available_clients.items():
            client_id_property = "client_id"
            gpi = GetPropertiesIns({client_id_property: "?"})
            client_prompted = client_proxy.get_properties(gpi, timeout=9999)
            client_id = client_prompted.properties[client_id_property]
            available_clients_map.update({"client_{0}".format(client_id): client_proxy})
        return available_clients_map

    def _update_selected_fit_clients_history(self,
                                             comm_round: int,
                                             available_fit_clients_map: dict,
                                             selection_duration: float,
                                             selected_fit_clients: list) -> None:
        selected_fit_clients_history = self.get_attribute("_selected_fit_clients_history")
        available_fit_clients_map_keys = list(available_fit_clients_map.keys())
        available_fit_clients_map_values = list(available_fit_clients_map.values())
        for selected_fit_client in selected_fit_clients:
            client_proxy = selected_fit_client["client_proxy"]
            client_index = available_fit_clients_map_values.index(client_proxy)
            client_id_str = available_fit_clients_map_keys[client_index]
            comm_round_key = "comm_round_{0}".format(comm_round)
            if comm_round_key not in selected_fit_clients_history:
                comm_round_selected_fit_metrics_dict = {comm_round_key:
                                                        {"selection_duration": selection_duration,
                                                         "selected_fit_clients": [client_id_str]}}
                selected_fit_clients_history.update(comm_round_selected_fit_metrics_dict)
            else:
                selected_fit_clients_history[comm_round_key]["selected_fit_clients"].append(client_id_str)
        self._set_attribute("_selected_fit_clients_history", selected_fit_clients_history)

    def configure_fit(self,
                      server_round: int,
                      parameters: Parameters,
                      client_manager: ClientManager) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training.
           \nImplementation of the abstract method of the Strategy class."""
        # Get the necessary attributes.
        enable_training = self.get_attribute("_enable_training")
        client_selection_settings = self.get_attribute("_client_selection_settings")
        client_selection_approach = client_selection_settings["approach"]
        individual_fit_metrics_history = self.get_attribute("_individual_fit_metrics_history")
        # Do not configure federated training if it is not enabled.
        if not enable_training:
            return []
        # Set the base training configuration (fit_config).
        fit_config = self._update_fit_config(server_round)
        # Initialize the list of the selected clients for training (selected_fit_clients).
        selected_fit_clients = []
        # Get the available fit clients.
        available_fit_clients = client_manager.all()
        # Get the number of available fit clients.
        num_available_fit_clients = len(available_fit_clients)
        # Map the available fit clients.
        available_fit_clients_map = self._map_available_clients(available_fit_clients)
        # Start the clients selection duration timer.
        selection_duration_start = perf_counter()
        if client_selection_approach == "Random":
            # Select clients for training randomly.
            min_available_clients = client_selection_settings["min_available_clients"]
            fit_clients_fraction = client_selection_settings["fit_clients_fraction"]
            selected_fit_clients = select_clients_randomly(client_manager,
                                                           num_available_fit_clients,
                                                           min_available_clients,
                                                           fit_clients_fraction)
        elif client_selection_approach == "MEC":
            # Select clients using the MEC algorithm.
            phase = "train"
            num_fit_tasks = client_selection_settings["num_fit_tasks"]
            history_check_approach = client_selection_settings["history_check_approach"]
            enable_complementary_selection = client_selection_settings["enable_complementary_selection"]
            complementary_selection_settings = client_selection_settings["complementary_selection_settings"]
            selected_fit_clients = select_clients_using_mec(server_round,
                                                            phase,
                                                            num_fit_tasks,
                                                            available_fit_clients_map,
                                                            individual_fit_metrics_history,
                                                            history_check_approach,
                                                            enable_complementary_selection,
                                                            complementary_selection_settings)
        elif client_selection_approach == "ECMTC":
            # Select clients using the ECMTC algorithm.
            phase = "train"
            num_fit_tasks = client_selection_settings["num_fit_tasks"]
            fit_deadline = client_selection_settings["fit_deadline"]
            history_check_approach = client_selection_settings["history_check_approach"]
            enable_complementary_selection = client_selection_settings["enable_complementary_selection"]
            complementary_selection_settings = client_selection_settings["complementary_selection_settings"]
            selected_fit_clients = select_clients_using_ecmtc(server_round,
                                                              phase,
                                                              num_fit_tasks,
                                                              fit_deadline,
                                                              available_fit_clients_map,
                                                              individual_fit_metrics_history,
                                                              history_check_approach,
                                                              enable_complementary_selection,
                                                              complementary_selection_settings)
        # Get the clients selection duration.
        selection_duration = perf_counter() - selection_duration_start
        # Update the history of selected clients for training (selected_fit_clients).
        self._update_selected_fit_clients_history(server_round,
                                                  available_fit_clients_map,
                                                  selection_duration,
                                                  selected_fit_clients)
        # Set the list of (fit_client_proxy, fit_client_instructions) pairs.
        fit_pairs = []
        for selected_fit_client in selected_fit_clients:
            selected_fit_client_proxy = selected_fit_client["client_proxy"]
            selected_fit_client_config = deepcopy(fit_config)
            if "client_num_tasks" in selected_fit_client:
                num_training_examples = selected_fit_client["client_num_tasks"]
                selected_fit_client_config.update({"num_training_examples": num_training_examples})
            selected_fit_client_instructions = FitIns(parameters, selected_fit_client_config)
            fit_pairs.append((selected_fit_client_proxy, selected_fit_client_instructions))
        # Return the list of (fit_client_proxy, fit_client_instructions) pairs.
        return fit_pairs

    def _update_individual_fit_metrics_history(self,
                                               comm_round: int,
                                               fit_metrics: list[tuple[int, Metrics]]) -> None:
        individual_fit_metrics_history = self.get_attribute("_individual_fit_metrics_history")
        for metric_tuple in fit_metrics:
            client_metrics = metric_tuple[1]
            client_id = client_metrics["client_id"]
            del client_metrics["client_id"]
            client_metrics_dict = {"client_{0}".format(client_id): client_metrics}
            comm_round_key = "comm_round_{0}".format(comm_round)
            if comm_round_key not in individual_fit_metrics_history:
                comm_round_individual_fit_metrics_dict = {comm_round_key: [client_metrics_dict]}
                individual_fit_metrics_history.update(comm_round_individual_fit_metrics_dict)
            else:
                individual_fit_metrics_history[comm_round_key].append(client_metrics_dict)
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
                                               aggregated_fit_metrics: dict) -> None:
        aggregated_fit_metrics_history = self.get_attribute("_aggregated_fit_metrics_history")
        comm_round_key = "comm_round_{0}".format(comm_round)
        if comm_round_key not in aggregated_fit_metrics_history:
            comm_round_aggregated_fit_metrics_dict = {comm_round_key: aggregated_fit_metrics}
            aggregated_fit_metrics_history.update(comm_round_aggregated_fit_metrics_dict)
        self._set_attribute("_aggregated_fit_metrics_history", aggregated_fit_metrics_history)

    def _aggregate_fit_metrics(self,
                               comm_round: int,
                               fit_metrics: list[tuple[int, Metrics]]) -> Optional[Metrics]:
        """Aggregates the training metrics (fit_metrics).
        \nCalled by Flower after each training phase."""
        # Get the necessary attributes.
        metrics_aggregation_settings = self.get_attribute("_metrics_aggregation_settings")
        metrics_aggregation_approach = metrics_aggregation_settings["approach"]
        server_id = self.get_attribute("_server_id")
        logger = self.get_attribute("_logger")
        # Update the individual training metrics history.
        self._update_individual_fit_metrics_history(comm_round, fit_metrics)
        # Remove the undesired metrics, if any.
        undesired_metrics = ["client_id"]
        fit_metrics = self._remove_undesired_metrics(fit_metrics, undesired_metrics)
        # Initialize the aggregated training metrics dictionary (aggregated_fit_metrics).
        aggregated_fit_metrics = {}
        # Aggregate the training metrics according to the user-defined approach.
        if metrics_aggregation_approach == "WeightedAverage":
            aggregated_fit_metrics = aggregate_metrics_by_weighted_average(fit_metrics)
        # Update the aggregated training metrics history.
        self._update_aggregated_fit_metrics_history(comm_round, aggregated_fit_metrics)
        # Get the number of participating clients.
        num_participating_clients = len(fit_metrics)
        num_participating_clients_str = "".join([str(num_participating_clients),
                                                 " Clients" if num_participating_clients > 1 else " Client"])
        # Log the aggregated training metrics.
        message = "[Server {0} | Round {1}] Aggregated training metrics ({2} of {3}): {4}" \
                  .format(server_id, comm_round, metrics_aggregation_approach, num_participating_clients_str,
                          aggregated_fit_metrics)
        log_message(logger, message, "DEBUG")
        # Return the aggregated training metrics (aggregated_fit_metrics).
        return aggregated_fit_metrics

    def aggregate_fit(self,
                      server_round: int,
                      results: List[Tuple[ClientProxy, FitRes]],
                      failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate the model parameters and the training metrics based on the fit_results.
           \nImplementation of the abstract method of the Strategy class."""
        # Get the necessary attributes.
        accept_clients_failures = self.get_attribute("_accept_clients_failures")
        model_aggregation_settings = self.get_attribute("_model_aggregation_settings")
        model_aggregation_approach = model_aggregation_settings["approach"]
        # Do not aggregate if there are no results or if there are clients failures and failures are not accepted.
        if not results or (failures and not accept_clients_failures):
            return None, {}
        # Initialize the aggregated model parameters.
        aggregated_model_parameters = None
        if model_aggregation_approach == "FedAvg":
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
                                                  available_evaluate_clients_map: dict,
                                                  selection_duration: float,
                                                  selected_evaluate_clients: list) -> None:
        selected_evaluate_clients_history = self.get_attribute("_selected_evaluate_clients_history")
        available_evaluate_clients_map_keys = list(available_evaluate_clients_map.keys())
        available_evaluate_clients_map_values = list(available_evaluate_clients_map.values())
        for selected_evaluate_client in selected_evaluate_clients:
            client_proxy = selected_evaluate_client["client_proxy"]
            client_index = available_evaluate_clients_map_values.index(client_proxy)
            client_id_str = available_evaluate_clients_map_keys[client_index]
            comm_round_key = "comm_round_{0}".format(comm_round)
            if comm_round_key not in selected_evaluate_clients_history:
                comm_round_selected_evaluate_metrics_dict = {comm_round_key:
                                                             {"selection_duration": selection_duration,
                                                              "selected_evaluate_clients": [client_id_str]}}
                selected_evaluate_clients_history.update(comm_round_selected_evaluate_metrics_dict)
            else:
                selected_evaluate_clients_history[comm_round_key]["selected_evaluate_clients"].append(client_id_str)
        self._set_attribute("_selected_evaluate_clients_history", selected_evaluate_clients_history)

    def configure_evaluate(self,
                           server_round: int,
                           parameters: Parameters,
                           client_manager: ClientManager) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of testing.
           \nImplementation of the abstract method of the Strategy class."""
        # Get the necessary attributes.
        enable_testing = self.get_attribute("_enable_testing")
        client_selection_settings = self.get_attribute("_client_selection_settings")
        client_selection_approach = client_selection_settings["approach"]
        individual_evaluate_metrics_history = self.get_attribute("_individual_evaluate_metrics_history")
        # Do not configure federated testing if it is not enabled.
        if not enable_testing:
            return []
        # Set the base testing configuration (evaluate_config).
        evaluate_config = self._update_evaluate_config(server_round)
        # Initialize the list of the selected clients for testing (selected_evaluate_clients).
        selected_evaluate_clients = []
        # Get the available evaluate clients.
        available_evaluate_clients = client_manager.all()
        # Get the number of available evaluate clients.
        num_available_evaluate_clients = len(available_evaluate_clients)
        # Map the available evaluate clients.
        available_evaluate_clients_map = self._map_available_clients(available_evaluate_clients)
        # Start the clients selection duration timer.
        selection_duration_start = perf_counter()
        if client_selection_approach == "Random":
            # Select clients for testing randomly.
            min_available_clients = client_selection_settings["min_available_clients"]
            evaluate_clients_fraction = client_selection_settings["evaluate_clients_fraction"]
            selected_evaluate_clients = select_clients_randomly(client_manager,
                                                                num_available_evaluate_clients,
                                                                min_available_clients,
                                                                evaluate_clients_fraction)
        elif client_selection_approach == "MEC":
            # Select clients using the MEC algorithm.
            phase = "test"
            num_evaluate_tasks = client_selection_settings["num_evaluate_tasks"]
            history_check_approach = client_selection_settings["history_check_approach"]
            enable_complementary_selection = client_selection_settings["enable_complementary_selection"]
            complementary_selection_settings = client_selection_settings["complementary_selection_settings"]
            selected_evaluate_clients = select_clients_using_mec(server_round,
                                                                 phase,
                                                                 num_evaluate_tasks,
                                                                 available_evaluate_clients_map,
                                                                 individual_evaluate_metrics_history,
                                                                 history_check_approach,
                                                                 enable_complementary_selection,
                                                                 complementary_selection_settings)
        elif client_selection_approach == "ECMTC":
            # Select clients using the ECMTC algorithm.
            phase = "test"
            num_evaluate_tasks = client_selection_settings["num_evaluate_tasks"]
            evaluate_deadline = client_selection_settings["evaluate_deadline"]
            history_check_approach = client_selection_settings["history_check_approach"]
            enable_complementary_selection = client_selection_settings["enable_complementary_selection"]
            complementary_selection_settings = client_selection_settings["complementary_selection_settings"]
            selected_evaluate_clients = select_clients_using_ecmtc(server_round,
                                                                   phase,
                                                                   num_evaluate_tasks,
                                                                   evaluate_deadline,
                                                                   available_evaluate_clients_map,
                                                                   individual_evaluate_metrics_history,
                                                                   history_check_approach,
                                                                   enable_complementary_selection,
                                                                   complementary_selection_settings)
        # Get the clients selection duration.
        selection_duration = perf_counter() - selection_duration_start
        # Update the history of selected clients for testing (selected_evaluate_clients).
        self._update_selected_evaluate_clients_history(server_round,
                                                       available_evaluate_clients_map,
                                                       selection_duration,
                                                       selected_evaluate_clients)
        # Set the list of (evaluate_client_proxy, evaluate_client_instructions) pairs.
        evaluate_pairs = []
        for selected_evaluate_client in selected_evaluate_clients:
            selected_evaluate_client_proxy = selected_evaluate_client["client_proxy"]
            selected_evaluate_client_config = deepcopy(evaluate_config)
            if "client_num_tasks" in selected_evaluate_client:
                num_testing_examples = selected_evaluate_client["client_num_tasks"]
                selected_evaluate_client_config.update({"num_testing_examples": num_testing_examples})
            selected_evaluate_client_instructions = EvaluateIns(parameters, selected_evaluate_client_config)
            evaluate_pairs.append((selected_evaluate_client_proxy, selected_evaluate_client_instructions))
        # Return the list of (evaluate_client_proxy, evaluate_client_instructions) pairs.
        return evaluate_pairs

    def _update_individual_evaluate_metrics_history(self,
                                                    comm_round: int,
                                                    evaluate_metrics: list[tuple[int, Metrics]]) -> None:
        individual_evaluate_metrics_history = self.get_attribute("_individual_evaluate_metrics_history")
        for metric_tuple in evaluate_metrics:
            client_metrics = metric_tuple[1]
            client_id = client_metrics["client_id"]
            del client_metrics["client_id"]
            client_metrics_dict = {"client_{0}".format(client_id): client_metrics}
            comm_round_key = "comm_round_{0}".format(comm_round)
            if comm_round_key not in individual_evaluate_metrics_history:
                comm_round_individual_evaluate_metrics_dict = {comm_round_key: [client_metrics_dict]}
                individual_evaluate_metrics_history.update(comm_round_individual_evaluate_metrics_dict)
            else:
                individual_evaluate_metrics_history[comm_round_key].append(client_metrics_dict)
        self._set_attribute("_individual_evaluate_metrics_history", individual_evaluate_metrics_history)

    def _update_aggregated_evaluate_metrics_history(self,
                                                    comm_round: int,
                                                    aggregated_evaluate_metrics: dict) -> None:
        aggregated_evaluate_metrics_history = self.get_attribute("_aggregated_evaluate_metrics_history")
        comm_round_key = "comm_round_{0}".format(comm_round)
        if comm_round_key not in aggregated_evaluate_metrics_history:
            comm_round_aggregated_evaluate_metrics_dict = {comm_round_key: aggregated_evaluate_metrics}
            aggregated_evaluate_metrics_history.update(comm_round_aggregated_evaluate_metrics_dict)
        self._set_attribute("_aggregated_evaluate_metrics_history", aggregated_evaluate_metrics_history)

    def _aggregate_evaluate_metrics(self,
                                    comm_round: int,
                                    evaluate_metrics: list[tuple[int, Metrics]]) -> Optional[Metrics]:
        """Aggregates the testing metrics (evaluate_metrics).
        \nCalled by Flower after each testing phase."""
        # Get the necessary attributes.
        metrics_aggregation_settings = self.get_attribute("_metrics_aggregation_settings")
        metrics_aggregation_approach = metrics_aggregation_settings["approach"]
        server_id = self.get_attribute("_server_id")
        logger = self.get_attribute("_logger")
        # Update the individual testing metrics history.
        self._update_individual_evaluate_metrics_history(comm_round, evaluate_metrics)
        # Remove the undesired metrics, if any.
        undesired_metrics = ["client_id"]
        evaluate_metrics = self._remove_undesired_metrics(evaluate_metrics, undesired_metrics)
        # Initialize the aggregated testing metrics dictionary (aggregated_evaluate_metrics).
        aggregated_evaluate_metrics = {}
        # Aggregate the testing metrics according to the user-defined approach.
        if metrics_aggregation_approach == "WeightedAverage":
            aggregated_evaluate_metrics = aggregate_metrics_by_weighted_average(evaluate_metrics)
        # Update the aggregated testing metrics history.
        self._update_aggregated_evaluate_metrics_history(comm_round, aggregated_evaluate_metrics)
        # Get the number of participating clients.
        num_participating_clients = len(evaluate_metrics)
        num_participating_clients_str = "".join([str(num_participating_clients),
                                                 " Clients" if num_participating_clients > 1 else " Client"])
        # Log the aggregated testing metrics.
        message = "[Server {0} | Round {1}] Aggregated testing metrics ({2} of {3}): {4}" \
                  .format(server_id, comm_round, metrics_aggregation_approach, num_participating_clients_str,
                          aggregated_evaluate_metrics)
        log_message(logger, message, "DEBUG")
        # Return the aggregated testing metrics (aggregated_evaluate_metrics).
        return aggregated_evaluate_metrics

    def aggregate_evaluate(self,
                           server_round: int,
                           results: List[Tuple[ClientProxy, EvaluateRes]],
                           failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]]) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate the testing metrics based on the evaluate_results.
           \nImplementation of the abstract method of the Strategy class."""
        # Get the necessary attributes.
        accept_clients_failures = self.get_attribute("_accept_clients_failures")
        model_aggregation_settings = self.get_attribute("_model_aggregation_settings")
        model_aggregation_approach = model_aggregation_settings["approach"]
        # Do not aggregate if there are no results or if there are clients failures and failures are not accepted.
        if not results or (failures and not accept_clients_failures):
            return None, {}
        # Initialize the aggregated loss value.
        aggregated_loss = 0
        if model_aggregation_approach == "FedAvg":
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
        \nThe outcome will be stored in the 'losses_centralized' and 'metrics_centralized' dictionaries."""
        return None

    def evaluate(self,
                 server_round: int,
                 parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate in a centralized fashion the updated model parameters.
           \nImplementation of the abstract method of the Strategy class."""
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
