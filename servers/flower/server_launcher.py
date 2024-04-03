from pathlib import Path
from typing import Optional

from flwr.common import NDArrays
from flwr.server import Server, ServerConfig, SimpleClientManager, start_server
from flwr.server.strategy import Strategy

from servers.flower.goffls_server import FlowerGOFFLSServer
from utils.configparser_util import parse_config_section
from utils.logger_util import load_logger, log_message


class FlowerServerLauncher:
    def __init__(self,
                 server_id: int,
                 server_config_file: Path) -> None:
        self._server_id = server_id
        self._server_config_file = server_config_file
        self._logging_settings = None
        self._fl_settings = None
        self._server_strategy_settings = None
        self._metrics_aggregation_settings = None
        self._ssl_settings = None
        self._grpc_settings = None
        self._fit_config_settings = None
        self._evaluate_config_settings = None
        self._logger = None
        self._server_strategy = None

    def _set_attribute(self,
                       attribute_name: str,
                       attribute_value: any) -> None:
        setattr(self, attribute_name, attribute_value)

    def get_attribute(self,
                      attribute_name: str) -> any:
        return getattr(self, attribute_name)

    def _parse_settings(self) -> None:
        # Get the necessary attributes.
        config_file = self.get_attribute("_server_config_file")
        # Parse and set the logging settings.
        logging_settings = parse_config_section(config_file, "Logging Settings")
        self._set_attribute("_logging_settings", logging_settings)
        # Parse and set the fl settings.
        fl_settings = parse_config_section(config_file, "FL Settings")
        self._set_attribute("_fl_settings", fl_settings)
        # Parse and set the server strategy settings.
        server_strategy_settings = parse_config_section(config_file, "Server Strategy Settings")
        server_strategy = server_strategy_settings["strategy"]
        server_strategy_implementation_settings \
            = parse_config_section(config_file, "{0} Server Strategy Settings".format(server_strategy))
        client_selection_approach = server_strategy_implementation_settings["client_selection_approach"]
        client_selection_approach_settings \
            = parse_config_section(config_file, "{0} Client Selection Settings".format(client_selection_approach))
        client_selection_approach_settings.update({"approach": client_selection_approach})
        model_aggregation_approach = server_strategy_implementation_settings["model_aggregation_approach"]
        model_aggregation_approach_settings \
            = parse_config_section(config_file, "{0} Model Aggregation Settings".format(model_aggregation_approach))
        model_aggregation_approach_settings.update({"approach": model_aggregation_approach})
        server_strategy_settings = {}
        server_strategy_settings.update({"strategy": server_strategy})
        server_strategy_settings.update({"client_selection": client_selection_approach_settings})
        server_strategy_settings.update({"model_aggregation": model_aggregation_approach_settings})
        self._set_attribute("_server_strategy_settings", server_strategy_settings)
        # Parse and set the metrics aggregation settings.
        metrics_aggregation_settings = parse_config_section(config_file, "Metrics Aggregation Settings")
        self._set_attribute("_metrics_aggregation_settings", metrics_aggregation_settings)
        # Parse and set the ssl settings.
        ssl_settings = parse_config_section(config_file, "SSL Settings")
        self._set_attribute("_ssl_settings", ssl_settings)
        # Parse and set the grpc settings.
        grpc_settings = parse_config_section(config_file, "gRPC Settings")
        self._set_attribute("_grpc_settings", grpc_settings)
        # Parse and set the fit_config settings.
        fit_config_settings = parse_config_section(config_file, "Fit_Config Settings")
        self._set_attribute("_fit_config_settings", fit_config_settings)
        # Parse and set the evaluate_config settings.
        evaluate_config_settings = parse_config_section(config_file, "Evaluate_Config Settings")
        self._set_attribute("_evaluate_config_settings", evaluate_config_settings)

    def _set_logger(self) -> None:
        # Get the necessary attributes.
        logging_settings = self.get_attribute("_logging_settings")
        server_id = self.get_attribute("_server_id")
        # Append the server's id to the output file name.
        file_name = Path(logging_settings["file_name"])
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
        """Server-side parameter initialization. A powerful mechanism which can be used, for example:
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
        enable_training = fl_settings["enable_training"]
        enable_testing = fl_settings["enable_testing"]
        accept_clients_failures = fl_settings["accept_clients_failures"]
        server_strategy_settings = self.get_attribute("_server_strategy_settings")
        strategy = server_strategy_settings["strategy"]
        client_selection_settings = server_strategy_settings["client_selection"]
        model_aggregation_settings = server_strategy_settings["model_aggregation"]
        metrics_aggregation_settings = self.get_attribute("_metrics_aggregation_settings")
        # Initialize the server strategy.
        server_strategy = None
        if strategy == "GOFFLS":
            # Instantiate the GOFFLS (Generic Optimization Framework for Federated Learning Schedules) server strategy.
            server_strategy = FlowerGOFFLSServer(server_id=server_id,
                                                 enable_training=enable_training,
                                                 enable_testing=enable_testing,
                                                 accept_clients_failures=accept_clients_failures,
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

    def launch_server(self) -> None:
        # Parse the settings.
        self._parse_settings()
        # Set the logger.
        self._set_logger()
        # Get the Secure Socket Layer (SSL) certificates (SSL-enabled secure connection).
        ssl_certificates = self._get_ssl_certificates()
        # Get the flower server address (listen IP address and port).
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
        fl_settings = self.get_attribute("_fl_settings")
        num_rounds = fl_settings["num_rounds"]
        round_timeout_in_seconds = fl_settings["round_timeout_in_seconds"]
        flower_server_config = self._instantiate_flower_server_config(num_rounds, round_timeout_in_seconds)
        # Start the flower server.
        self._start_flower_server(flower_server_address,
                                  flower_server,
                                  flower_server_config,
                                  max_message_length_in_bytes,
                                  ssl_certificates)
        # Get the server strategy.
        server_strategy = self.get_attribute("_server_strategy")
        selected_fit_clients_history = server_strategy.get_attribute("_selected_fit_clients_history")
        selected_evaluate_clients_history = server_strategy.get_attribute("_selected_evaluate_clients_history")
        individual_fit_metrics_history = server_strategy.get_attribute("_individual_fit_metrics_history")
        aggregated_fit_metrics_history = server_strategy.get_attribute("_aggregated_fit_metrics_history")
        individual_evaluate_metrics_history = server_strategy.get_attribute("_individual_evaluate_metrics_history")
        aggregated_evaluate_metrics_history = server_strategy.get_attribute("_aggregated_evaluate_metrics_history")
        print("Selected training clients history:\n{0}".format(selected_fit_clients_history))
        print("Selected testing clients history:\n{0}".format(selected_evaluate_clients_history))
        print("Individual training metrics history:\n{0}".format(individual_fit_metrics_history))
        print("Aggregated training metrics history:\n{0}".format(aggregated_fit_metrics_history))
        print("Individual testing metrics history:\n{0}".format(individual_evaluate_metrics_history))
        print("Aggregated testing metrics history:\n{0}".format(aggregated_evaluate_metrics_history))
        # End.
        exit(0)
