from pathlib import Path
from typing import Optional

from flwr.common import NDArrays
from flwr.server import Server, ServerConfig, SimpleClientManager, start_server
from flwr.server.strategy import Strategy

from goffls.server.flower_goffls_server import FlowerGOFFLSServer
from goffls.utils.config_parser_util import parse_config_section
from goffls.utils.logger_util import load_logger, log_message


class FlowerServerLauncher:
    def __init__(self,
                 id_: int,
                 config_file: Path) -> None:
        # Initialize the attributes.
        self._server_id = id_
        self._config_file = config_file
        self._logging_settings = None
        self._fl_settings = None
        self._server_strategy_settings = None
        self._metrics_aggregation_settings = None
        self._ssl_settings = None
        self._grpc_settings = None
        self._fit_config_settings = None
        self._evaluate_config_settings = None
        self._output_settings = None
        self._logger = None
        self._server_strategy = None
        # Parse the settings.
        self._parse_settings()
        # Set the logger.
        self._set_logger()

    def _set_attribute(self,
                       attribute_name: str,
                       attribute_value: any) -> None:
        setattr(self, attribute_name, attribute_value)

    def get_attribute(self,
                      attribute_name: str) -> any:
        return getattr(self, attribute_name)

    def _parse_settings(self) -> None:
        # Get the necessary attributes.
        config_file = self.get_attribute("_config_file")
        # Parse and set the logging settings.
        logging_section = "Logging Settings"
        logging_settings = parse_config_section(config_file, logging_section)
        self._set_attribute("_logging_settings", logging_settings)
        # Parse and set the fl settings.
        fl_section = "FL Settings"
        fl_settings = parse_config_section(config_file, fl_section)
        self._set_attribute("_fl_settings", fl_settings)
        # Parse and set the server strategy settings.
        server_strategy_section = "Server Strategy Settings"
        server_strategy_settings = parse_config_section(config_file, server_strategy_section)
        server_strategy = server_strategy_settings["strategy"]
        server_strategy_implementation_section = "{0} Server Strategy Settings".format(server_strategy)
        server_strategy_implementation_settings = parse_config_section(config_file,
                                                                       server_strategy_implementation_section)
        client_selection_settings = {}
        client_selector_for_training = server_strategy_implementation_settings["client_selector_for_training"]
        client_selection_for_training_section = "{0} Client Selection Settings".format(client_selector_for_training)
        client_selection_for_training_settings = parse_config_section(config_file,
                                                                      client_selection_for_training_section)
        if "assignment_capacities_initializer" in client_selection_for_training_settings:
            assignment_capacities_initializer \
                = client_selection_for_training_settings["assignment_capacities_initializer"]
            assignment_capacities_init_settings = {}
            try:
                assignment_capacities_init_section = "{0} Settings".format(assignment_capacities_initializer)
                assignment_capacities_init_settings = parse_config_section(config_file,
                                                                           assignment_capacities_init_section)
            except KeyError:
                pass
            assignment_capacities_init_settings.update({"assignment_capacities_initializer":
                                                        assignment_capacities_initializer})
            client_selection_for_training_settings.update({"assignment_capacities_init_settings":
                                                          assignment_capacities_init_settings})
            client_selection_for_training_settings.pop("assignment_capacities_initializer")
        client_selection_for_training_settings.update({"client_selector_for_training":
                                                      client_selector_for_training})
        client_selection_settings.update({"client_selection_for_training_settings":
                                          client_selection_for_training_settings})
        client_selector_for_testing = server_strategy_implementation_settings["client_selector_for_testing"]
        client_selection_for_testing_section = "{0} Client Selection Settings".format(client_selector_for_testing)
        client_selection_for_testing_settings = parse_config_section(config_file,
                                                                     client_selection_for_testing_section)
        if "assignment_capacities_initializer" in client_selection_for_testing_settings:
            assignment_capacities_initializer \
                = client_selection_for_testing_settings["assignment_capacities_initializer"]
            assignment_capacities_init_settings = {}
            try:
                assignment_capacities_init_section = "{0} Settings".format(assignment_capacities_initializer)
                assignment_capacities_init_settings = parse_config_section(config_file,
                                                                           assignment_capacities_init_section)
            except KeyError:
                pass
            assignment_capacities_init_settings.update({"assignment_capacities_initializer":
                                                        assignment_capacities_initializer})
            client_selection_for_testing_settings.update({"assignment_capacities_init_settings":
                                                          assignment_capacities_init_settings})
            client_selection_for_testing_settings.pop("assignment_capacities_initializer")
        client_selection_for_testing_settings.update({"client_selector_for_testing":
                                                     client_selector_for_testing})
        client_selection_settings.update({"client_selection_for_testing_settings":
                                         client_selection_for_testing_settings})
        model_aggregator = server_strategy_implementation_settings["model_aggregator"]
        model_aggregator_section = "{0} Model Aggregation Settings".format(model_aggregator)
        model_aggregator_settings = parse_config_section(config_file, model_aggregator_section)
        model_aggregator_settings.update({"model_aggregator": model_aggregator})
        server_strategy_settings = {}
        server_strategy_settings.update({"strategy": server_strategy})
        server_strategy_settings.update({"client_selection": client_selection_settings})
        server_strategy_settings.update({"model_aggregation": model_aggregator_settings})
        self._set_attribute("_server_strategy_settings", server_strategy_settings)
        # Parse and set the metrics aggregation settings.
        metrics_aggregation_section = "Metrics Aggregation Settings"
        metrics_aggregation_settings = parse_config_section(config_file, metrics_aggregation_section)
        self._set_attribute("_metrics_aggregation_settings", metrics_aggregation_settings)
        # Parse and set the ssl settings.
        ssl_section = "SSL Settings"
        ssl_settings = parse_config_section(config_file, ssl_section)
        self._set_attribute("_ssl_settings", ssl_settings)
        # Parse and set the grpc settings.
        grpc_section = "gRPC Settings"
        grpc_settings = parse_config_section(config_file, grpc_section)
        self._set_attribute("_grpc_settings", grpc_settings)
        # Parse and set the fit_config settings.
        fit_config_section = "Fit_Config Settings"
        fit_config_settings = parse_config_section(config_file, fit_config_section)
        self._set_attribute("_fit_config_settings", fit_config_settings)
        # Parse and set the evaluate_config settings.
        evaluate_config_section = "Evaluate_Config Settings"
        evaluate_config_settings = parse_config_section(config_file, evaluate_config_section)
        self._set_attribute("_evaluate_config_settings", evaluate_config_settings)
        # Parse and set the output settings.
        output_section = "Output Settings"
        output_settings = parse_config_section(config_file, output_section)
        self._set_attribute("_output_settings", output_settings)

    def _set_logger(self) -> None:
        # Get the necessary attributes.
        logging_settings = self.get_attribute("_logging_settings")
        server_id = self.get_attribute("_server_id")
        # Append the server's id to the output file name.
        file_name = Path(logging_settings["file_name"]).absolute()
        file_name = str(file_name.parent.joinpath(file_name.stem + "_{0}".format(server_id) + file_name.suffix))
        logging_settings["file_name"] = file_name
        # Set the logger name.
        logger_name = type(self).__name__ + "_Logger"
        # Load and set the logger.
        logger = load_logger(logging_settings, logger_name)
        self._set_attribute("_logger", logger)

    def _get_ssl_certificates(self) -> Optional[tuple[bytes]]:
        # Get the necessary attributes.
        ssl_settings = self.get_attribute("_ssl_settings")
        enable_ssl = ssl_settings["enable_ssl"]
        ca_certificate_file = ssl_settings["ca_certificate_file"]
        server_certificate_file = ssl_settings["server_certificate_file"]
        server_rsa_private_key_file = ssl_settings["server_rsa_private_key_file"]
        # Initialize the SSL certificates tuple.
        ssl_certificates = None
        # If SSL secure connection is enabled...
        if enable_ssl:
            # Read the SSL certificates bytes.
            ca_certificate_bytes = ca_certificate_file.read_bytes()
            server_certificate_bytes = server_certificate_file.read_bytes()
            server_rsa_private_key_bytes = server_rsa_private_key_file.read_bytes()
            # Mount the SSL certificates tuple.
            ssl_certificates = (ca_certificate_bytes, server_certificate_bytes, server_rsa_private_key_bytes)
        # Return the SSL certificates tuple.
        return ssl_certificates

    def _get_flower_server_address(self) -> str:
        # Get the necessary attributes.
        grpc_settings = self.get_attribute("_grpc_settings")
        listen_ip_address = grpc_settings["listen_ip_address"]
        listen_port = str(grpc_settings["listen_port"])
        # Mount the flower server address.
        flower_server_address = listen_ip_address + ":" + listen_port
        # Return the flower server address.
        return flower_server_address

    def _get_max_message_length_in_bytes(self) -> int:
        # Get the necessary attributes.
        grpc_settings = self.get_attribute("_grpc_settings")
        max_message_length_in_bytes = grpc_settings["max_message_length_in_bytes"]
        # Return the maximum message length in bytes.
        return max_message_length_in_bytes

    def _load_fit_config(self) -> dict:
        # Get the necessary attributes.
        fit_config_settings = self.get_attribute("_fit_config_settings")
        server_id = self.get_attribute("_server_id")
        logger = self.get_attribute("_logger")
        # Load the initial training configuration (fit_config).
        fit_config = {"comm_round": 0}
        fit_config.update(fit_config_settings)
        # Log the initial training configuration (fit_config).
        message = "[Server {0}] Base fit_config: {1}".format(server_id, fit_config)
        log_message(logger, message, "DEBUG")
        # Return the initial training configuration (fit_config).
        return fit_config

    def _load_evaluate_config(self) -> dict:
        # Get the necessary attributes.
        evaluate_config_settings = self.get_attribute("_evaluate_config_settings")
        server_id = self.get_attribute("_server_id")
        logger = self.get_attribute("_logger")
        # Load the initial testing configuration (evaluate_config).
        evaluate_config = {"comm_round": 0}
        evaluate_config.update(evaluate_config_settings)
        self._set_attribute("_evaluate_config", evaluate_config)
        # Log the initial testing configuration (evaluate_config).
        message = "[Server {0}] Base evaluate_config: {1}".format(server_id, evaluate_config)
        log_message(logger, message, "DEBUG")
        # Return the initial testing configuration (evaluate_config).
        return evaluate_config

    def _load_initial_parameters(self) -> Optional[NDArrays]:
        """Server-side parameter initialization. A powerful mechanism which can be used, for instance:
        \n - To resume the training from a previously saved checkpoint;
        \n - To implement hybrid approaches, such as to fine-tune a pre-trained model using federated learning.
        \n If no parameters are set, the server will randomly select one client and ask its parameters."""
        # Get the necessary attributes.
        server_id = self.get_attribute("_server_id")
        logger = self.get_attribute("_logger")
        # Load the initial model parameters.
        initial_parameters = None
        # Log the initial model parameters.
        message = "[Server {0}] Initial model parameters: {1}".format(server_id, initial_parameters)
        log_message(logger, message, "DEBUG")
        # Return the initial model parameters.
        return initial_parameters

    @staticmethod
    def _instantiate_simple_client_manager() -> SimpleClientManager:
        # Instantiate a simple client manager.
        simple_client_manager = SimpleClientManager()
        # Return the simple client manager.
        return simple_client_manager

    def _instantiate_server_strategy(self,
                                     fit_config: dict,
                                     evaluate_config: dict,
                                     initial_parameters: Optional[NDArrays]) -> Strategy:
        # Get the necessary attributes.
        server_id = self.get_attribute("_server_id")
        logger = self.get_attribute("_logger")
        fl_settings = self.get_attribute("_fl_settings")
        server_strategy_settings = self.get_attribute("_server_strategy_settings")
        strategy = server_strategy_settings["strategy"]
        client_selection_settings = server_strategy_settings["client_selection"]
        model_aggregation_settings = server_strategy_settings["model_aggregation"]
        metrics_aggregation_settings = self.get_attribute("_metrics_aggregation_settings")
        # Initialize the server strategy.
        server_strategy = None
        if strategy == "GOFFLS":
            # Instantiate the GOFFLS (Generic Optimization Framework for Federated Learning Schedules) server strategy.
            server_strategy = FlowerGOFFLSServer(id_=server_id,
                                                 fl_settings=fl_settings,
                                                 client_selection_settings=client_selection_settings,
                                                 model_aggregation_settings=model_aggregation_settings,
                                                 metrics_aggregation_settings=metrics_aggregation_settings,
                                                 fit_config=fit_config,
                                                 evaluate_config=evaluate_config,
                                                 initial_parameters=initial_parameters,
                                                 logger=logger)
        # Set the server strategy.
        self._set_attribute("_server_strategy", server_strategy)
        # Return the server strategy.
        return server_strategy

    @staticmethod
    def _instantiate_flower_server(simple_client_manager: SimpleClientManager,
                                   server_strategy: Strategy) -> Server:
        # Instantiate the flower server.
        flower_server = Server(client_manager=simple_client_manager,
                               strategy=server_strategy)
        # Return the flower server.
        return flower_server

    @staticmethod
    def _instantiate_flower_server_config(num_rounds: int,
                                          round_timeout: int) -> ServerConfig:
        # Instantiate the flower server config.
        flower_server_config = ServerConfig(num_rounds=num_rounds,
                                            round_timeout=round_timeout)
        # Return the flower server config.
        return flower_server_config

    @staticmethod
    def _start_flower_server(server_address: str,
                             server: Optional[Server],
                             config: Optional[ServerConfig],
                             grpc_max_message_length: int,
                             certificates: Optional[tuple[bytes, bytes, bytes]]) -> None:
        # Start the flower server.
        start_server(server_address=server_address,
                     server=server,
                     config=config,
                     grpc_max_message_length=grpc_max_message_length,
                     certificates=certificates)

    def _generate_selected_fit_clients_history_output_file(self) -> None:
        # Get the necessary attributes.
        server_strategy = self.get_attribute("_server_strategy")
        selected_fit_clients_history = server_strategy.get_attribute("_selected_fit_clients_history")
        output_settings = self.get_attribute("_output_settings")
        remove_output_files = output_settings["remove_output_files"]
        selected_fit_clients_history_file = Path(output_settings["selected_fit_clients_history_file"]).absolute()
        # Remove the history output file (if it exists and if removing is enabled).
        if remove_output_files:
            selected_fit_clients_history_file.unlink(missing_ok=True)
        # Create the parents directories of the output file (if not exist yet).
        selected_fit_clients_history_file.parent.mkdir(exist_ok=True, parents=True)
        # Write the header line to the output file (if not exist yet).
        if not selected_fit_clients_history_file.exists():
            with open(file=selected_fit_clients_history_file, mode="a", encoding="utf-8") as file:
                header_line = "{0},{1},{2},{3},{4},{5},{6}\n" \
                              .format("comm_round",
                                      "client_selector",
                                      "selection_duration",
                                      "num_tasks",
                                      "num_available_clients",
                                      "num_selected_clients",
                                      "selected_clients")
                file.write(header_line)
        # Order the selected_fit_clients_history dictionary in ascending order of communication round.
        sorted_comm_round_keys = sorted(list(selected_fit_clients_history.keys()), key=lambda x: (len(x), x))
        selected_fit_clients_history = {comm_round_key: selected_fit_clients_history[comm_round_key]
                                        for comm_round_key in sorted_comm_round_keys}
        # Write the history data lines to the output file.
        with open(file=selected_fit_clients_history_file, mode="a", encoding="utf-8") as file:
            for comm_round_key, comm_round_values in selected_fit_clients_history.items():
                client_selector = comm_round_values["client_selector"]
                selection_duration = comm_round_values["selection_duration"]
                num_tasks = comm_round_values["num_tasks"]
                num_available_clients = comm_round_values["num_available_clients"]
                num_selected_clients = comm_round_values["num_selected_clients"]
                selected_clients = comm_round_values["selected_clients"]
                data_line = "{0},{1},{2},{3},{4},{5},{6}\n" \
                            .format(comm_round_key,
                                    client_selector,
                                    selection_duration,
                                    num_tasks,
                                    num_available_clients,
                                    num_selected_clients,
                                    "|".join(selected_clients))
                file.write(data_line)

    def _generate_individual_fit_metrics_history_output_file(self) -> None:
        # Get the necessary attributes.
        server_strategy = self.get_attribute("_server_strategy")
        selected_fit_clients_history = server_strategy.get_attribute("_selected_fit_clients_history")
        individual_fit_metrics_history = server_strategy.get_attribute("_individual_fit_metrics_history")
        output_settings = self.get_attribute("_output_settings")
        remove_output_files = output_settings["remove_output_files"]
        individual_fit_metrics_history_file = Path(output_settings["individual_fit_metrics_history_file"]).absolute()
        # Remove the history output file (if it exists and if removing is enabled).
        if remove_output_files:
            individual_fit_metrics_history_file.unlink(missing_ok=True)
        # Create the parents directories of the output file (if not exist yet).
        individual_fit_metrics_history_file.parent.mkdir(exist_ok=True, parents=True)
        # Get the ordered set of fit metrics names.
        fit_metrics_names = []
        for _, comm_round_values in individual_fit_metrics_history.items():
            clients_metrics_dicts = comm_round_values["clients_metrics_dicts"]
            for client_metrics_dict in clients_metrics_dicts:
                client_metrics = list(client_metrics_dict.values())[0]
                fit_metrics_names.extend(client_metrics.keys())
        fit_metrics_names = sorted(set(fit_metrics_names))
        # Write the header line to the output file (if not exist yet).
        if not individual_fit_metrics_history_file.is_file():
            with open(file=individual_fit_metrics_history_file, mode="a", encoding="utf-8") as file:
                header_line = "{0},{1},{2},{3},{4},{5},{6},{7},{8}\n" \
                              .format("comm_round",
                                      "client_selector",
                                      "num_tasks",
                                      "num_available_clients",
                                      "client_id",
                                      "client_expected_duration",
                                      "client_expected_energy_consumption",
                                      "client_expected_accuracy",
                                      ",".join(fit_metrics_names))
                file.write(header_line)
        # Order the individual_fit_metrics_history dictionary in ascending order of communication round.
        sorted_comm_round_keys = sorted(list(individual_fit_metrics_history.keys()), key=lambda x: (len(x), x))
        individual_fit_metrics_history = {comm_round_key: individual_fit_metrics_history[comm_round_key]
                                          for comm_round_key in sorted_comm_round_keys}
        # Write the history data lines to the output file.
        with open(file=individual_fit_metrics_history_file, mode="a", encoding="utf-8") as file:
            for comm_round_key, comm_round_values in individual_fit_metrics_history.items():
                client_selector = comm_round_values["client_selector"]
                num_tasks = comm_round_values["num_tasks"]
                num_available_clients = comm_round_values["num_available_clients"]
                clients_metrics_dicts = comm_round_values["clients_metrics_dicts"]
                clients_metrics_dicts = sorted(clients_metrics_dicts, key=lambda x: list(x.keys()))
                selected_clients = list(selected_fit_clients_history[comm_round_key]["selected_clients"])
                expected_durations = list(selected_fit_clients_history[comm_round_key]["expected_durations"])
                expected_energy_consumptions \
                    = list(selected_fit_clients_history[comm_round_key]["expected_energy_consumptions"])
                expected_accuracies = list(selected_fit_clients_history[comm_round_key]["expected_accuracies"])
                for client_metrics_dict in clients_metrics_dicts:
                    client_id_str = list(client_metrics_dict.keys())[0]
                    client_id_str_index = selected_clients.index(client_id_str)
                    client_expected_duration = expected_durations[client_id_str_index]
                    client_expected_energy_consumption = expected_energy_consumptions[client_id_str_index]
                    client_expected_accuracy = expected_accuracies[client_id_str_index]
                    client_metrics = list(client_metrics_dict.values())[0]
                    fit_metrics_values = []
                    for fit_metric_name in fit_metrics_names:
                        fit_metric_value = "N/A"
                        if fit_metric_name in client_metrics:
                            fit_metric_value = str(client_metrics[fit_metric_name])
                        fit_metrics_values.append(fit_metric_value)
                    data_line = "{0},{1},{2},{3},{4},{5},{6},{7},{8}\n" \
                                .format(comm_round_key,
                                        client_selector,
                                        num_tasks,
                                        num_available_clients,
                                        client_id_str,
                                        client_expected_duration,
                                        client_expected_energy_consumption,
                                        client_expected_accuracy,
                                        ",".join(fit_metrics_values))
                    file.write(data_line)

    def _generate_aggregated_fit_metrics_history_output_file(self) -> None:
        # Get the necessary attributes.
        server_strategy = self.get_attribute("_server_strategy")
        aggregated_fit_metrics_history = server_strategy.get_attribute("_aggregated_fit_metrics_history")
        output_settings = self.get_attribute("_output_settings")
        remove_output_files = output_settings["remove_output_files"]
        aggregated_fit_metrics_history_file = Path(output_settings["aggregated_fit_metrics_history_file"]).absolute()
        # Remove the history output file (if it exists and if removing is enabled).
        if remove_output_files:
            aggregated_fit_metrics_history_file.unlink(missing_ok=True)
        # Create the parents directories of the output file (if not exist yet).
        aggregated_fit_metrics_history_file.parent.mkdir(exist_ok=True, parents=True)
        # Get the ordered set of fit metrics names.
        fit_metrics_names = []
        for _, comm_round_values in aggregated_fit_metrics_history.items():
            aggregated_metrics = comm_round_values["aggregated_metrics"]
            fit_metrics_names.extend(aggregated_metrics.keys())
        fit_metrics_names = sorted(set(fit_metrics_names))
        # Write the header line to the output file (if not exist yet).
        if not aggregated_fit_metrics_history_file.is_file():
            with open(file=aggregated_fit_metrics_history_file, mode="a", encoding="utf-8") as file:
                header_line = "{0},{1},{2},{3},{4},{5}\n" \
                              .format("comm_round",
                                      "client_selector",
                                      "metrics_aggregator",
                                      "num_tasks",
                                      "num_available_clients",
                                      ",".join(fit_metrics_names))
                file.write(header_line)
        # Order the aggregated_fit_metrics_history dictionary in ascending order of communication round.
        sorted_comm_round_keys = sorted(list(aggregated_fit_metrics_history.keys()), key=lambda x: (len(x), x))
        aggregated_fit_metrics_history = {comm_round_key: aggregated_fit_metrics_history[comm_round_key]
                                          for comm_round_key in sorted_comm_round_keys}
        # Write the history data lines to the output file.
        with open(file=aggregated_fit_metrics_history_file, mode="a", encoding="utf-8") as file:
            for comm_round, comm_round_values in aggregated_fit_metrics_history.items():
                client_selector = comm_round_values["client_selector"]
                metrics_aggregator = comm_round_values["metrics_aggregator"]
                num_tasks = comm_round_values["num_tasks"]
                num_available_clients = comm_round_values["num_available_clients"]
                aggregated_metrics = comm_round_values["aggregated_metrics"]
                fit_metrics_values = []
                for fit_metric_name in fit_metrics_names:
                    fit_metric_value = "N/A"
                    if fit_metric_name in aggregated_metrics:
                        fit_metric_value = str(aggregated_metrics[fit_metric_name])
                    fit_metrics_values.append(fit_metric_value)
                data_line = "{0},{1},{2},{3},{4},{5}\n" \
                            .format(comm_round,
                                    client_selector,
                                    metrics_aggregator,
                                    num_tasks,
                                    num_available_clients,
                                    ",".join(fit_metrics_values))
                file.write(data_line)

    def _generate_fit_selection_performance_history_output_file(self) -> None:
        # Get the necessary attributes.
        server_strategy = self.get_attribute("_server_strategy")
        fit_selection_performance_history = server_strategy.get_attribute("_fit_selection_performance_history")
        output_settings = self.get_attribute("_output_settings")
        remove_output_files = output_settings["remove_output_files"]
        fit_selection_performance_history_file \
            = Path(output_settings["fit_selection_performance_history_file"]).absolute()
        # Remove the history output file (if it exists and if removing is enabled).
        if remove_output_files:
            fit_selection_performance_history_file.unlink(missing_ok=True)
        # Create the parents directories of the output file (if not exist yet).
        fit_selection_performance_history_file.parent.mkdir(exist_ok=True, parents=True)
        # Write the header line to the output file (if not exist yet).
        if not fit_selection_performance_history_file.exists():
            with open(file=fit_selection_performance_history_file, mode="a", encoding="utf-8") as file:
                header_line = "{0},{1},{2},{3},{4},{5},{6},{7},{8}\n" \
                              .format("comm_round",
                                      "client_selector",
                                      "num_tasks",
                                      "expected_makespan",
                                      "actual_makespan",
                                      "expected_energy_consumption",
                                      "actual_energy_consumption",
                                      "expected_accuracy",
                                      "actual_accuracy")
                file.write(header_line)
        # Order the fit_selection_performance_history dictionary in ascending order of communication round.
        sorted_comm_round_keys = sorted(list(fit_selection_performance_history.keys()), key=lambda x: (len(x), x))
        fit_selection_performance_history = {comm_round_key: fit_selection_performance_history[comm_round_key]
                                             for comm_round_key in sorted_comm_round_keys}
        # Write the history data lines to the output file.
        with open(file=fit_selection_performance_history_file, mode="a", encoding="utf-8") as file:
            for comm_round_key, comm_round_values in fit_selection_performance_history.items():
                client_selector = comm_round_values["client_selector"]
                num_tasks = comm_round_values["num_tasks"]
                expected_makespan = comm_round_values["expected_makespan"]
                actual_makespan = comm_round_values["actual_makespan"]
                expected_energy_consumption = comm_round_values["expected_energy_consumption"]
                actual_energy_consumption = comm_round_values["actual_energy_consumption"]
                expected_accuracy = comm_round_values["expected_accuracy"]
                actual_accuracy = comm_round_values["actual_accuracy"]
                data_line = "{0},{1},{2},{3},{4},{5},{6},{7},{8}\n" \
                            .format(comm_round_key,
                                    client_selector,
                                    num_tasks,
                                    expected_makespan,
                                    actual_makespan,
                                    expected_energy_consumption,
                                    actual_energy_consumption,
                                    expected_accuracy,
                                    actual_accuracy)
                file.write(data_line)

    def _generate_selected_evaluate_clients_history_output_file(self) -> None:
        # Get the necessary attributes.
        server_strategy_settings = self.get_attribute("_server_strategy_settings")
        client_selection_settings = server_strategy_settings["client_selection"]
        client_selection_for_training_settings = client_selection_settings["client_selection_for_training_settings"]
        client_selector_for_training = client_selection_for_training_settings["client_selector_for_training"]
        server_strategy = self.get_attribute("_server_strategy")
        selected_evaluate_clients_history = server_strategy.get_attribute("_selected_evaluate_clients_history")
        output_settings = self.get_attribute("_output_settings")
        remove_output_files = output_settings["remove_output_files"]
        selected_evaluate_clients_history_file = \
            Path(output_settings["selected_evaluate_clients_history_file"]).absolute()
        # Remove the history output file (if it exists and if removing is enabled).
        if remove_output_files:
            selected_evaluate_clients_history_file.unlink(missing_ok=True)
        # Create the parents directories of the output file (if not exist yet).
        selected_evaluate_clients_history_file.parent.mkdir(exist_ok=True, parents=True)
        # Write the header line to the output file (if not exist yet).
        if not selected_evaluate_clients_history_file.is_file():
            with open(file=selected_evaluate_clients_history_file, mode="a", encoding="utf-8") as file:
                header_line = "{0},{1},{2},{3},{4},{5},{6}\n" \
                              .format("comm_round",
                                      "client_selector",
                                      "selection_duration",
                                      "num_tasks",
                                      "num_available_clients",
                                      "num_selected_clients",
                                      "selected_clients")
                file.write(header_line)
        # Order the selected_evaluate_clients_history dictionary in ascending order of communication round.
        sorted_comm_round_keys = sorted(list(selected_evaluate_clients_history.keys()), key=lambda x: (len(x), x))
        selected_evaluate_clients_history = {comm_round_key: selected_evaluate_clients_history[comm_round_key]
                                             for comm_round_key in sorted_comm_round_keys}
        # Write the history data lines to the output file.
        with open(file=selected_evaluate_clients_history_file, mode="a", encoding="utf-8") as file:
            for comm_round_key, comm_round_values in selected_evaluate_clients_history.items():
                client_selector = "{0} ({1})".format(comm_round_values["client_selector"], client_selector_for_training)
                selection_duration = comm_round_values["selection_duration"]
                num_tasks = comm_round_values["num_tasks"]
                num_available_clients = comm_round_values["num_available_clients"]
                num_selected_clients = comm_round_values["num_selected_clients"]
                selected_clients = comm_round_values["selected_clients"]
                data_line = "{0},{1},{2},{3},{4},{5},{6}\n" \
                            .format(comm_round_key,
                                    client_selector,
                                    selection_duration,
                                    num_tasks,
                                    num_available_clients,
                                    num_selected_clients,
                                    "|".join(selected_clients))
                file.write(data_line)

    def _generate_individual_evaluate_metrics_history_output_file(self) -> None:
        # Get the necessary attributes.
        server_strategy = self.get_attribute("_server_strategy")
        selected_evaluate_clients_history = server_strategy.get_attribute("_selected_evaluate_clients_history")
        individual_evaluate_metrics_history = server_strategy.get_attribute("_individual_evaluate_metrics_history")
        output_settings = self.get_attribute("_output_settings")
        remove_output_files = output_settings["remove_output_files"]
        individual_evaluate_metrics_history_file \
            = Path(output_settings["individual_evaluate_metrics_history_file"]).absolute()
        # Remove the history output file (if it exists and if removing is enabled).
        if remove_output_files:
            individual_evaluate_metrics_history_file.unlink(missing_ok=True)
        # Create the parents directories of the output file (if not exist yet).
        individual_evaluate_metrics_history_file.parent.mkdir(exist_ok=True, parents=True)
        # Get the ordered set of evaluate metrics names.
        evaluate_metrics_names = []
        for _, comm_round_values in individual_evaluate_metrics_history.items():
            clients_metrics_dicts = comm_round_values["clients_metrics_dicts"]
            for client_metrics_dict in clients_metrics_dicts:
                client_metrics = list(client_metrics_dict.values())[0]
                evaluate_metrics_names.extend(client_metrics.keys())
        evaluate_metrics_names = sorted(set(evaluate_metrics_names))
        # Write the header line to the output file (if not exist yet).
        if not individual_evaluate_metrics_history_file.is_file():
            with open(file=individual_evaluate_metrics_history_file, mode="a", encoding="utf-8") as file:
                header_line = "{0},{1},{2},{3},{4},{5},{6},{7},{8}\n" \
                              .format("comm_round",
                                      "client_selector",
                                      "num_tasks",
                                      "num_available_clients",
                                      "client_id",
                                      "client_expected_duration",
                                      "client_expected_energy_consumption",
                                      "client_expected_accuracy",
                                      ",".join(evaluate_metrics_names))
                file.write(header_line)
        # Order the individual_evaluate_metrics_history dictionary in ascending order of communication round.
        sorted_comm_round_keys = sorted(list(individual_evaluate_metrics_history.keys()), key=lambda x: (len(x), x))
        individual_evaluate_metrics_history = {comm_round_key: individual_evaluate_metrics_history[comm_round_key]
                                               for comm_round_key in sorted_comm_round_keys}
        # Write the history data lines to the output file.
        with open(file=individual_evaluate_metrics_history_file, mode="a", encoding="utf-8") as file:
            for comm_round_key, comm_round_values in individual_evaluate_metrics_history.items():
                client_selector = comm_round_values["client_selector"]
                num_tasks = comm_round_values["num_tasks"]
                num_available_clients = comm_round_values["num_available_clients"]
                clients_metrics_dicts = comm_round_values["clients_metrics_dicts"]
                clients_metrics_dicts = sorted(clients_metrics_dicts, key=lambda x: list(x.keys()))
                selected_clients = list(selected_evaluate_clients_history[comm_round_key]["selected_clients"])
                expected_durations = list(selected_evaluate_clients_history[comm_round_key]["expected_durations"])
                expected_energy_consumptions \
                    = list(selected_evaluate_clients_history[comm_round_key]["expected_energy_consumptions"])
                expected_accuracies = list(selected_evaluate_clients_history[comm_round_key]["expected_accuracies"])
                for client_metrics_dict in clients_metrics_dicts:
                    client_id_str = list(client_metrics_dict.keys())[0]
                    client_id_str_index = selected_clients.index(client_id_str)
                    client_expected_duration = expected_durations[client_id_str_index]
                    client_expected_energy_consumption = expected_energy_consumptions[client_id_str_index]
                    client_expected_accuracy = expected_accuracies[client_id_str_index]
                    client_metrics = list(client_metrics_dict.values())[0]
                    evaluate_metrics_values = []
                    for evaluate_metric_name in evaluate_metrics_names:
                        evaluate_metric_value = "N/A"
                        if evaluate_metric_name in client_metrics:
                            evaluate_metric_value = str(client_metrics[evaluate_metric_name])
                        evaluate_metrics_values.append(evaluate_metric_value)
                    data_line = "{0},{1},{2},{3},{4},{5},{6},{7},{8}\n" \
                                .format(comm_round_key,
                                        client_selector,
                                        num_tasks,
                                        num_available_clients,
                                        client_id_str,
                                        client_expected_duration,
                                        client_expected_energy_consumption,
                                        client_expected_accuracy,
                                        ",".join(evaluate_metrics_values))
                    file.write(data_line)

    def _generate_aggregated_evaluate_metrics_history_output_file(self) -> None:
        # Get the necessary attributes.
        server_strategy_settings = self.get_attribute("_server_strategy_settings")
        client_selection_settings = server_strategy_settings["client_selection"]
        client_selection_for_training_settings = client_selection_settings["client_selection_for_training_settings"]
        client_selector_for_training = client_selection_for_training_settings["client_selector_for_training"]
        server_strategy = self.get_attribute("_server_strategy")
        aggregated_evaluate_metrics_history = server_strategy.get_attribute("_aggregated_evaluate_metrics_history")
        output_settings = self.get_attribute("_output_settings")
        remove_output_files = output_settings["remove_output_files"]
        aggregated_evaluate_metrics_history_file = \
            Path(output_settings["aggregated_evaluate_metrics_history_file"]).absolute()
        # Remove the history output file (if it exists and if removing is enabled).
        if remove_output_files:
            aggregated_evaluate_metrics_history_file.unlink(missing_ok=True)
        # Create the parents directories of the output file (if not exist yet).
        aggregated_evaluate_metrics_history_file.parent.mkdir(exist_ok=True, parents=True)
        # Get the ordered set of evaluate metrics names.
        evaluate_metrics_names = []
        for _, comm_round_values in aggregated_evaluate_metrics_history.items():
            aggregated_metrics = comm_round_values["aggregated_metrics"]
            evaluate_metrics_names.extend(aggregated_metrics.keys())
        evaluate_metrics_names = sorted(set(evaluate_metrics_names))
        # Write the header line to the output file (if not exist yet).
        if not aggregated_evaluate_metrics_history_file.is_file():
            with open(file=aggregated_evaluate_metrics_history_file, mode="a", encoding="utf-8") as file:
                header_line = "{0},{1},{2},{3},{4},{5}\n" \
                              .format("comm_round",
                                      "client_selector",
                                      "metrics_aggregator",
                                      "num_tasks",
                                      "num_available_clients",
                                      ",".join(evaluate_metrics_names))
                file.write(header_line)
        # Order the aggregated_evaluate_metrics_history dictionary in ascending order of communication round.
        sorted_comm_round_keys = sorted(list(aggregated_evaluate_metrics_history.keys()), key=lambda x: (len(x), x))
        aggregated_evaluate_metrics_history = {comm_round_key: aggregated_evaluate_metrics_history[comm_round_key]
                                               for comm_round_key in sorted_comm_round_keys}
        # Write the history data lines to the output file.
        with open(file=aggregated_evaluate_metrics_history_file, mode="a", encoding="utf-8") as file:
            for comm_round, comm_round_values in aggregated_evaluate_metrics_history.items():
                client_selector = "{0} ({1})".format(comm_round_values["client_selector"], client_selector_for_training)
                metrics_aggregator = comm_round_values["metrics_aggregator"]
                num_tasks = comm_round_values["num_tasks"]
                num_available_clients = comm_round_values["num_available_clients"]
                aggregated_metrics = comm_round_values["aggregated_metrics"]
                evaluate_metrics_values = []
                for evaluate_metric_name in evaluate_metrics_names:
                    evaluate_metric_value = "N/A"
                    if evaluate_metric_name in aggregated_metrics:
                        evaluate_metric_value = str(aggregated_metrics[evaluate_metric_name])
                    evaluate_metrics_values.append(evaluate_metric_value)
                data_line = "{0},{1},{2},{3},{4},{5}\n" \
                            .format(comm_round,
                                    client_selector,
                                    metrics_aggregator,
                                    num_tasks,
                                    num_available_clients,
                                    ",".join(evaluate_metrics_values))
                file.write(data_line)

    def _generate_evaluate_selection_performance_history_output_file(self) -> None:
        # Get the necessary attributes.
        server_strategy_settings = self.get_attribute("_server_strategy_settings")
        client_selection_settings = server_strategy_settings["client_selection"]
        client_selection_for_training_settings = client_selection_settings["client_selection_for_training_settings"]
        client_selector_for_training = client_selection_for_training_settings["client_selector_for_training"]
        server_strategy = self.get_attribute("_server_strategy")
        evaluate_selection_performance_history \
            = server_strategy.get_attribute("_evaluate_selection_performance_history")
        output_settings = self.get_attribute("_output_settings")
        remove_output_files = output_settings["remove_output_files"]
        evaluate_selection_performance_history_file \
            = Path(output_settings["evaluate_selection_performance_history_file"]).absolute()
        # Remove the history output file (if it exists and if removing is enabled).
        if remove_output_files:
            evaluate_selection_performance_history_file.unlink(missing_ok=True)
        # Create the parents directories of the output file (if not exist yet).
        evaluate_selection_performance_history_file.parent.mkdir(exist_ok=True, parents=True)
        # Write the header line to the output file (if not exist yet).
        if not evaluate_selection_performance_history_file.exists():
            with open(file=evaluate_selection_performance_history_file, mode="a", encoding="utf-8") as file:
                header_line = "{0},{1},{2},{3},{4},{5},{6},{7},{8}\n" \
                              .format("comm_round",
                                      "client_selector",
                                      "num_tasks",
                                      "expected_makespan",
                                      "actual_makespan",
                                      "expected_energy_consumption",
                                      "actual_energy_consumption",
                                      "expected_accuracy",
                                      "actual_accuracy")
                file.write(header_line)
        # Order the evaluate_selection_performance_history dictionary in ascending order of communication round.
        sorted_comm_round_keys = sorted(list(evaluate_selection_performance_history.keys()), key=lambda x: (len(x), x))
        evaluate_selection_performance_history = {comm_round_key: evaluate_selection_performance_history[comm_round_key]
                                                  for comm_round_key in sorted_comm_round_keys}
        # Write the history data lines to the output file.
        with open(file=evaluate_selection_performance_history_file, mode="a", encoding="utf-8") as file:
            for comm_round_key, comm_round_values in evaluate_selection_performance_history.items():
                client_selector = "{0} ({1})".format(comm_round_values["client_selector"], client_selector_for_training)
                num_tasks = comm_round_values["num_tasks"]
                expected_makespan = comm_round_values["expected_makespan"]
                actual_makespan = comm_round_values["actual_makespan"]
                expected_energy_consumption = comm_round_values["expected_energy_consumption"]
                actual_energy_consumption = comm_round_values["actual_energy_consumption"]
                expected_accuracy = comm_round_values["expected_accuracy"]
                actual_accuracy = comm_round_values["actual_accuracy"]
                data_line = "{0},{1},{2},{3},{4},{5},{6},{7},{8}\n" \
                            .format(comm_round_key,
                                    client_selector,
                                    num_tasks,
                                    expected_makespan,
                                    actual_makespan,
                                    expected_energy_consumption,
                                    actual_energy_consumption,
                                    expected_accuracy,
                                    actual_accuracy)
                file.write(data_line)

    def launch_server(self) -> None:
        # Get the necessary attributes.
        fl_settings = self.get_attribute("_fl_settings")
        num_rounds = fl_settings["num_rounds"]
        round_timeout_in_seconds = fl_settings["round_timeout_in_seconds"]
        enable_training = fl_settings["enable_training"]
        enable_testing = fl_settings["enable_testing"]
        # Get the Secure Socket Layer (SSL) certificates (SSL-enabled secure connection).
        ssl_certificates = self._get_ssl_certificates()
        # Get the flower server address (to-listen IP address and port).
        flower_server_address = self._get_flower_server_address()
        # Get the maximum message length in bytes.
        max_message_length_in_bytes = self._get_max_message_length_in_bytes()
        # Load the initial fit config.
        fit_config = self._load_fit_config()
        # Load the initial evaluate config.
        evaluate_config = self._load_evaluate_config()
        # Load the initial parameters.
        initial_parameters = self._load_initial_parameters()
        # Instantiate the simple client manager.
        simple_client_manager = self._instantiate_simple_client_manager()
        # Instantiate the server strategy.
        server_strategy = self._instantiate_server_strategy(fit_config, evaluate_config, initial_parameters)
        # Instantiate the flower server.
        flower_server = self._instantiate_flower_server(simple_client_manager, server_strategy)
        # Instantiate the flower server config.
        flower_server_config = self._instantiate_flower_server_config(num_rounds, round_timeout_in_seconds)
        # Start the flower server.
        self._start_flower_server(flower_server_address,
                                  flower_server,
                                  flower_server_config,
                                  max_message_length_in_bytes,
                                  ssl_certificates)
        # Generate the output files for the training step if it was enabled.
        if enable_training:
            # Generate the output file for the selected fit clients' history.
            self._generate_selected_fit_clients_history_output_file()
            # Generate the output file for the individual fit metrics history.
            self._generate_individual_fit_metrics_history_output_file()
            # Generate the output file for the aggregated fit metrics history.
            self._generate_aggregated_fit_metrics_history_output_file()
            # Generate the output file for the fit selection performance history.
            self._generate_fit_selection_performance_history_output_file()
        # Generate the output files for the testing step if it was enabled.
        if enable_testing:
            # Generate the output file for the selected evaluate clients' history.
            self._generate_selected_evaluate_clients_history_output_file()
            # Generate the output file for the individual evaluate metrics history.
            self._generate_individual_evaluate_metrics_history_output_file()
            # Generate the output file for the aggregated evaluate metrics history.
            self._generate_aggregated_evaluate_metrics_history_output_file()
            # Generate the output file for the evaluate selection performance history.
            self._generate_evaluate_selection_performance_history_output_file()
        # End.
        exit(0)
