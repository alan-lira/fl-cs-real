from keras.applications import MobileNetV2
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import SGD
from logging import Logger
from numpy import empty
from pathlib import Path
from PIL import Image
from time import perf_counter
from typing import Optional

from flwr.client import Client, start_client
from flwr.common import NDArray

from goffls.client.flower_numpy_client import FlowerNumpyClient
from goffls.energy_monitor.powerjoular_energy_monitor import PowerJoularEnergyMonitor
from goffls.energy_monitor.pyjoules_energy_monitor import PyJoulesEnergyMonitor
from goffls.utils.config_parser_util import parse_config_section
from goffls.utils.logger_util import load_logger, log_message


class FlowerClientLauncher:
    def __init__(self,
                 id_: int,
                 config_file: Path) -> None:
        # Initialize the attributes.
        self._client_id = id_
        self._config_file = config_file
        self._logging_settings = None
        self._daemon_settings = None
        self._affinity_settings = None
        self._ssl_settings = None
        self._grpc_settings = None
        self._dataset_settings = None
        self._energy_monitoring_settings = None
        self._model_settings = None
        self._logger = None
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
        # Parse and set the daemon settings.
        daemon_section = "Daemon Settings"
        daemon_settings = parse_config_section(config_file, daemon_section)
        self._set_attribute("_daemon_settings", daemon_settings)
        # Parse and set the affinity settings.
        affinity_section = "Affinity Settings"
        affinity_settings = parse_config_section(config_file, affinity_section)
        self._set_attribute("_affinity_settings", affinity_settings)
        # Parse and set the ssl settings.
        ssl_section = "SSL Settings"
        ssl_settings = parse_config_section(config_file, ssl_section)
        self._set_attribute("_ssl_settings", ssl_settings)
        # Parse and set the grpc settings.
        grpc_section = "gRPC Settings"
        grpc_settings = parse_config_section(config_file, grpc_section)
        self._set_attribute("_grpc_settings", grpc_settings)
        # Parse and set the dataset settings.
        dataset_section = "Dataset Settings"
        dataset_settings = parse_config_section(config_file, dataset_section)
        self._set_attribute("_dataset_settings", dataset_settings)
        # Parse and set the energy monitoring settings.
        energy_monitoring_section = "Energy Monitoring Settings"
        energy_monitoring_settings = parse_config_section(config_file,
                                                          energy_monitoring_section)
        energy_monitor_name = energy_monitoring_settings["energy_monitor"]
        energy_monitor_section = "{0} Monitor Settings".format(energy_monitor_name)
        energy_monitor_settings = parse_config_section(config_file, energy_monitor_section)
        energy_monitoring_settings.update({energy_monitor_name: energy_monitor_settings})
        self._set_attribute("_energy_monitoring_settings", energy_monitoring_settings)
        # Parse and set the model settings.
        model_section = "Model Settings"
        model_settings = parse_config_section(config_file, model_section)
        model_provider = model_settings["provider"]
        model_provider_section = "{0} Model Settings".format(model_provider)
        model_provider_settings = parse_config_section(config_file, model_provider_section)
        model_name = model_provider_settings["name"]
        model_provider_specific_section = "{0} {1} Settings".format(model_provider, model_name)
        model_provider_specific_settings = parse_config_section(config_file, model_provider_specific_section)
        optimizer = model_provider_settings["optimizer"]
        optimizer_section = "{0} {1} Settings".format(model_provider, optimizer)
        optimizer_settings = parse_config_section(config_file, optimizer_section)
        loss = model_provider_settings["loss"]
        loss_section = "{0} {1} Settings".format(model_provider, loss)
        loss_settings = parse_config_section(config_file, loss_section)
        model_settings.update({model_provider: model_provider_settings,
                               model_name: model_provider_specific_settings,
                               optimizer: optimizer_settings,
                               loss: loss_settings})
        self._set_attribute("_model_settings", model_settings)

    def _set_logger(self) -> None:
        # Get the necessary attributes.
        logging_settings = self.get_attribute("_logging_settings")
        client_id = self.get_attribute("_client_id")
        # Append the client's id to the output file name.
        file_name = Path(logging_settings["file_name"]).absolute()
        file_name = str(file_name.parent.joinpath(file_name.stem + "_{0}".format(client_id) + file_name.suffix))
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
        # Initialize the SSL certificates tuple.
        ssl_certificates = None
        # If SSL secure connection is enabled...
        if enable_ssl:
            # Read the SSL certificates bytes.
            ca_certificate_bytes = ca_certificate_file.read_bytes()
            # Mount the SSL certificates tuple.
            ssl_certificates = ca_certificate_bytes
        # Return the SSL certificates tuple.
        return ssl_certificates

    def _get_flower_server_address(self) -> str:
        # Get the necessary attributes.
        grpc_settings = self.get_attribute("_grpc_settings")
        server_ip_address = grpc_settings["server_ip_address"]
        server_port = str(grpc_settings["server_port"])
        # Mount the flower server address.
        flower_server_address = server_ip_address + ":" + server_port
        # Return the flower server address.
        return flower_server_address

    def _get_max_message_length_in_bytes(self) -> int:
        # Get the necessary attributes.
        grpc_settings = self.get_attribute("_grpc_settings")
        max_message_length_in_bytes = grpc_settings["max_message_length_in_bytes"]
        # Return the maximum message length in bytes.
        return max_message_length_in_bytes

    @staticmethod
    def _derive_num_images(y_phase_labels_file: Path) -> int:
        return sum(1 for _ in open(file=y_phase_labels_file, mode="r"))

    @staticmethod
    def _derive_images_attributes(x_phase_folder: Path,
                                  y_phase_labels_file: Path) -> tuple:
        first_line = next(open(file=y_phase_labels_file, mode="r"))
        split_line = first_line.rstrip().split(", ")
        image_file = x_phase_folder.joinpath(split_line[0])
        im = Image.open(fp=image_file)
        width, height = im.size
        depth = len(im.getbands())
        return width, height, depth

    def _load_x_y_for_multi_class_image_classification(self,
                                                       phase: str) -> tuple:
        # Get the necessary attributes.
        dataset_settings = self.get_attribute("_dataset_settings")
        root_folder = Path(dataset_settings["root_folder"]).absolute()
        x_phase_folder = root_folder.joinpath("x_{0}".format(phase))
        y_phase_folder = root_folder.joinpath("y_{0}".format(phase))
        y_phase_labels_file = y_phase_folder.joinpath("labels.txt")
        number_of_examples = self._derive_num_images(y_phase_labels_file)
        width, height, depth = self._derive_images_attributes(x_phase_folder, y_phase_labels_file)
        derived_x_shape = (number_of_examples, height, width, depth)
        derived_y_shape = (number_of_examples, 1)
        x_phase = empty(shape=derived_x_shape, dtype="uint8")
        y_phase = empty(shape=derived_y_shape, dtype="uint8")
        with open(file=y_phase_labels_file, mode="r") as labels_file:
            index = 0
            lines = [next(labels_file) for _ in range(number_of_examples)]
            for line in lines:
                split_line = line.rstrip().split(", ")
                image_file = x_phase_folder.joinpath(split_line[0])
                x_phase[index] = Image.open(fp=image_file)
                label = split_line[1]
                y_phase[index] = label
                index += 1
        return x_phase, y_phase

    def _load_dataset(self) -> tuple:
        # Start the dataset loading duration timer.
        dataset_loading_duration_start = perf_counter()
        # Get the necessary attributes.
        dataset_settings = self.get_attribute("_dataset_settings")
        storage_location = dataset_settings["storage_location"]
        category = dataset_settings["category"]
        client_id = self.get_attribute("_client_id")
        logger = self.get_attribute("_logger")
        # Update the dataset root folder with the partition number that is associated to the client id.
        dataset_settings["root_folder"] = dataset_settings["root_folder"] + "partition_{0}".format(client_id)
        self._set_attribute("_dataset_settings", dataset_settings)
        root_folder = dataset_settings["root_folder"]
        # Log a 'loading the dataset' message.
        message = "[Client {0}] Loading the '{1}' dataset ({2} storage)..." \
                  .format(client_id, root_folder, storage_location)
        log_message(logger, message, "INFO")
        # Initialize x_train, y_train, x_test, and y_test.
        x_train = y_train = x_test = y_test = None
        if category == "multi_class_image_classification":
            if storage_location == "Local":
                # Load x_train and y_train.
                x_train, y_train = self._load_x_y_for_multi_class_image_classification("train")
                # Load x_test and y_test.
                x_test, y_test = self._load_x_y_for_multi_class_image_classification("test")
        # Get the dataset load duration.
        dataset_loading_duration = perf_counter() - dataset_loading_duration_start
        # Log the dataset loading duration.
        message = "[Client {0}] The dataset loading took {1} seconds.".format(client_id, dataset_loading_duration)
        log_message(logger, message, "INFO")
        # Return the loaded dataset (x_train, y_train, x_test, and y_test).
        return x_train, y_train, x_test, y_test

    def _load_energy_monitor(self) -> any:
        # Get the necessary attributes.
        energy_monitoring_settings = self.get_attribute("_energy_monitoring_settings")
        enable_energy_monitoring = energy_monitoring_settings["enable_energy_monitoring"]
        energy_monitor_name = energy_monitoring_settings["energy_monitor"]
        energy_monitor_settings = energy_monitoring_settings[energy_monitor_name]
        # Initialize the energy monitor.
        energy_monitor = None
        # If energy monitoring is enabled...
        if enable_energy_monitoring:
            if energy_monitor_name == "pyJoules":
                monitoring_domains = energy_monitor_settings["monitoring_domains"]
                unit = energy_monitor_settings["unit"]
                energy_monitor = PyJoulesEnergyMonitor(monitoring_domains, unit)
            elif energy_monitor_name == "PowerJoular":
                env_variables = energy_monitor_settings["env_variables"]
                monitoring_domains = energy_monitor_settings["monitoring_domains"]
                unit = energy_monitor_settings["unit"]
                process_monitoring = energy_monitor_settings["process_monitoring"]
                report_consumptions_per_second = energy_monitor_settings["report_consumptions_per_second"]
                energy_monitor = PowerJoularEnergyMonitor(env_variables,
                                                          monitoring_domains,
                                                          unit,
                                                          process_monitoring,
                                                          report_consumptions_per_second)
        # Return the energy monitor.
        return energy_monitor

    def _instantiate_optimizer(self) -> any:
        # Get the necessary attributes.
        model_settings = self.get_attribute("_model_settings")
        model_provider = model_settings["provider"]
        model_provider_settings = model_settings[model_provider]
        optimizer_name = model_provider_settings["optimizer"]
        optimizer_settings = model_settings[optimizer_name]
        # Initialize the optimizer.
        optimizer = None
        if model_provider == "Keras":
            if optimizer_name == "SGD":
                # Instantiate the Kera's SGD optimizer (Stochastic Gradient Descent).
                optimizer = SGD(learning_rate=optimizer_settings["learning_rate"],
                                momentum=optimizer_settings["momentum"],
                                nesterov=optimizer_settings["nesterov"],
                                name=optimizer_settings["name"])
        # Return the optimizer.
        return optimizer

    def _instantiate_loss_function(self) -> any:
        # Get the necessary attributes.
        model_settings = self.get_attribute("_model_settings")
        model_provider = model_settings["provider"]
        model_provider_settings = model_settings[model_provider]
        loss_name = model_provider_settings["loss"]
        loss_settings = model_settings[loss_name]
        # Initialize the loss.
        loss = None
        if model_provider == "Keras":
            if loss_name == "SparseCategoricalCrossentropy":
                # Instantiate the Kera's SparseCategoricalCrossentropy loss function.
                loss = SparseCategoricalCrossentropy(from_logits=loss_settings["from_logits"],
                                                     ignore_class=loss_settings["ignore_class"],
                                                     reduction=loss_settings["reduction"],
                                                     name=loss_settings["name"])
        # Return the loss function.
        return loss

    def _instantiate_and_compile_model(self,
                                       optimizer: any,
                                       loss_function: any) -> any:
        # Get the necessary attributes.
        model_settings = self.get_attribute("_model_settings")
        model_provider = model_settings["provider"]
        model_provider_settings = model_settings[model_provider]
        model_name = model_provider_settings["name"]
        model_provider_specific_settings = model_settings[model_name]
        # Initialize the model.
        model = None
        if model_provider == "Keras":
            if model_name == "MobileNetV2":
                # Instantiate the Kera's MobileNetV2 model (Image Classification Architecture).
                model = MobileNetV2(input_shape=model_provider_specific_settings["input_shape"],
                                    alpha=model_provider_specific_settings["alpha"],
                                    include_top=model_provider_specific_settings["include_top"],
                                    weights=model_provider_specific_settings["weights"],
                                    input_tensor=model_provider_specific_settings["input_tensor"],
                                    pooling=model_provider_specific_settings["pooling"],
                                    classes=model_provider_specific_settings["classes"],
                                    classifier_activation=model_provider_specific_settings["classifier_activation"])
            # Compile the Kera's model.
            loss_weights = model_provider_settings["loss_weights"]
            metrics = model_provider_settings["metrics"]
            weighted_metrics = model_provider_settings["weighted_metrics"]
            run_eagerly = model_provider_settings["run_eagerly"]
            steps_per_execution = model_provider_settings["steps_per_execution"]
            jit_compile = model_provider_settings["jit_compile"]
            auto_scale_loss = model_provider_settings["auto_scale_loss"]
            model.compile(optimizer=optimizer,
                          loss=loss_function,
                          loss_weights=loss_weights,
                          metrics=metrics,
                          weighted_metrics=weighted_metrics,
                          run_eagerly=run_eagerly,
                          steps_per_execution=steps_per_execution,
                          jit_compile=jit_compile,
                          auto_scale_loss=auto_scale_loss)
        # Return the model.
        return model

    @staticmethod
    def _instantiate_flower_client(id_: int,
                                   model: any,
                                   x_train: NDArray,
                                   y_train: NDArray,
                                   x_test: NDArray,
                                   y_test: NDArray,
                                   energy_monitor: any,
                                   daemon_settings: dict,
                                   affinity_settings: dict,
                                   logger: Logger) -> Client:
        # Instantiate the flower client.
        flower_client = FlowerNumpyClient(id_=id_,
                                          model=model,
                                          x_train=x_train,
                                          y_train=y_train,
                                          x_test=x_test,
                                          y_test=y_test,
                                          energy_monitor=energy_monitor,
                                          daemon_settings=daemon_settings,
                                          affinity_settings=affinity_settings,
                                          logger=logger)
        flower_client = flower_client.to_client()
        # Return the flower server.
        return flower_client

    @staticmethod
    def _start_flower_client(server_address: str,
                             client: Client,
                             grpc_max_message_length: int,
                             root_certificates: Optional[tuple[bytes, bytes, bytes]]) -> None:
        # Start the flower client.
        start_client(server_address=server_address,
                     client=client,
                     grpc_max_message_length=grpc_max_message_length,
                     root_certificates=root_certificates)

    def launch_client(self) -> None:
        # Get the necessary attributes.
        client_id = self.get_attribute("_client_id")
        logger = self.get_attribute("_logger")
        daemon_settings = self.get_attribute("_daemon_settings")
        affinity_settings = self.get_attribute("_affinity_settings")
        # Get the Secure Socket Layer (SSL) certificates (SSL-enabled secure connection).
        ssl_certificates = self._get_ssl_certificates()
        # Get the flower server address (IP address and port).
        flower_server_address = self._get_flower_server_address()
        # Get the maximum message length in bytes.
        max_message_length_in_bytes = self._get_max_message_length_in_bytes()
        # Load the dataset (x_train, y_train, x_test, and y_test).
        x_train, y_train, x_test, y_test = self._load_dataset()
        # Load the energy monitor.
        energy_monitor = self._load_energy_monitor()
        # Instantiate the optimizer.
        optimizer = self._instantiate_optimizer()
        # Instantiate the loss function.
        loss_function = self._instantiate_loss_function()
        # Instantiate and compile the model.
        model = self._instantiate_and_compile_model(optimizer, loss_function)
        # Instantiate the flower client.
        flower_client = self._instantiate_flower_client(client_id, model, x_train, y_train, x_test, y_test,
                                                        energy_monitor, daemon_settings, affinity_settings, logger)
        # Start the flower client.
        self._start_flower_client(flower_server_address,
                                  flower_client,
                                  max_message_length_in_bytes,
                                  ssl_certificates)
        # End.
        exit(0)
