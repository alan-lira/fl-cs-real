from grpc._channel import _MultiThreadedRendezvous
from keras import layers, Sequential
from keras.applications import DenseNet121, EfficientNetB0, EfficientNetV2L, MobileNetV2, ResNet50, VGG16
from keras.applications.densenet import preprocess_input as densenet121_preprocess_input
from keras.applications.efficientnet import preprocess_input as efficientnet_preprocess_input
from keras.applications.efficientnet_v2  import preprocess_input as efficientnet_v2_preprocess_input
from keras.applications.mobilenet_v2 import preprocess_input as mobilenet_v2_preprocess_input
from keras.applications.resnet import preprocess_input as resnet50_preprocess_input
from keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from keras.losses import Loss, SparseCategoricalCrossentropy
from keras.metrics import Metric, SparseCategoricalAccuracy
from keras.optimizers import Adam, Optimizer, SGD
from logging import Logger
from numpy import array, int64, ndarray
from os import environ
from pathlib import Path
from random import uniform
from time import perf_counter, sleep
from traceback import format_exc
from typing import Optional

from flwr.client import Client, start_client
from flwr.common import NDArray
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, IidPartitioner, PathologicalPartitioner

from fl_cs_real.client.flower_numpy_client import FlowerNumpyClient
from fl_cs_real.energy_monitor.powerjoular_energy_monitor import PowerJoularEnergyMonitor
from fl_cs_real.energy_monitor.pyjoules_energy_monitor import PyJoulesEnergyMonitor
from fl_cs_real.utils.config_parser_util import parse_config_section
from fl_cs_real.utils.logger_util import load_logger, log_message
from fl_cs_real.utils.multiclass_image_dataset_loader_util import load_x_y_for_multiclass_image_dataset

environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # Make TensorFlow log less verbose.

fds = None  # Cache FederatedDataset.


class FlowerClientLauncher:
    def __init__(self,
                 id_: int,
                 config_file: Path,
                 personalized_settings: dict = None) -> None:
        # Initialize the attributes.
        self._client_id = id_
        self._config_file = config_file
        self._logging_settings = None
        self._daemon_settings = None
        self._affinity_settings = None
        self._learning_rate_schedule_settings = None
        self._callbacks_settings = None
        self._ssl_settings = None
        self._grpc_settings = None
        self._dataset_settings = None
        self._local_dataset_settings = None
        self._federated_dataset_settings = None
        self._task_assignment_capacities_settings = None
        self._energy_monitoring_settings = None
        self._device_emulation_settings = None
        self._model_settings = None
        self._simulation_resources_settings = None
        self._root_output_folder = None
        # Parse the settings.
        self._parse_settings()
        # Update the settings if the personalized_settings dictionary was provided.
        if isinstance(personalized_settings, dict):
            for setting_key, config_pairs_dict in personalized_settings.items():
                if hasattr(self, setting_key):
                    setting = self.get_attribute(setting_key)
                    if setting:
                        for k, v in config_pairs_dict.items():
                            if k in setting:
                                setting[k] = v
                            else:
                                setting.update({k: v})
                        self._set_attribute(setting_key, setting)
                    else:
                        self._set_attribute(setting_key, config_pairs_dict)
                else:
                    self._set_attribute(setting_key, config_pairs_dict)
        # Load the logger.
        self._logger = self._load_logger()
        # Load the dataset.
        self._x_train, self._y_train, self._x_test, self._y_test = self._load_dataset()
        # Get the task assignment capacities.
        self._train_task_capacities, self._test_task_capacities = self._get_task_assignment_capacities()
        # Load the energy monitor.
        self._energy_monitor = self._load_energy_monitor()
        # Instantiate and compile the model.
        self._model, self._metrics_names = self._load_model()
        # Pre-process the dataset.
        self._pre_process_dataset()
        # Instantiate the client.
        self._client = self._instantiate_client()

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
        # Parse and set the local dataset settings.
        local_dataset_section = "Local Dataset Settings"
        local_dataset_settings = parse_config_section(config_file, local_dataset_section)
        self._set_attribute("_local_dataset_settings", local_dataset_settings)
        # Parse and set the federated dataset settings.
        federated_dataset_section = "FederatedDataset Settings"
        federated_dataset_settings = parse_config_section(config_file, federated_dataset_section)
        dataset_partitioner = federated_dataset_settings["dataset_partitioner"]
        match dataset_partitioner:
            case "DirichletPartitioner":
                dirichlet_partitioner_section = "DirichletPartitioner Settings"
                dirichlet_partitioner_settings = parse_config_section(config_file, dirichlet_partitioner_section)
                federated_dataset_settings.update(dirichlet_partitioner_settings)
            case "PathologicalPartitioner":
                pathological_partitioner_section = "PathologicalPartitioner Settings"
                pathological_partitioner_settings = parse_config_section(config_file, pathological_partitioner_section)
                federated_dataset_settings.update(pathological_partitioner_settings)
        self._set_attribute("_federated_dataset_settings", federated_dataset_settings)
        # Parse and set the task assignment capacities settings.
        task_assignment_capacities_section = "Task Assignment Capacities Settings"
        task_assignment_capacities_settings = parse_config_section(config_file, task_assignment_capacities_section)
        self._set_attribute("_task_assignment_capacities_settings", task_assignment_capacities_settings)
        # Parse and set the energy monitoring settings.
        energy_monitoring_section = "Energy Monitoring Settings"
        energy_monitoring_settings = parse_config_section(config_file,
                                                          energy_monitoring_section)
        energy_monitor_name = energy_monitoring_settings["energy_monitor"]
        energy_monitor_section = "{0} Monitor Settings".format(energy_monitor_name)
        energy_monitor_settings = parse_config_section(config_file, energy_monitor_section)
        energy_monitoring_settings.update({energy_monitor_name: energy_monitor_settings})
        self._set_attribute("_energy_monitoring_settings", energy_monitoring_settings)
        # Parse and set the device emulation settings.
        device_emulation_section = "Device Emulation Settings"
        device_emulation_settings = parse_config_section(config_file, device_emulation_section)
        self._set_attribute("_device_emulation_settings", device_emulation_settings)
        # Parse and set the model settings.
        model_section = "Model Settings"
        model_settings = parse_config_section(config_file, model_section)
        model_provider = model_settings["provider"]
        model_provider_section = "{0} Model Settings".format(model_provider)
        model_provider_settings = parse_config_section(config_file, model_provider_section)
        model_name = model_provider_settings["model_name"]
        model_provider_specific_section = "{0} {1} Settings".format(model_provider, model_name)
        model_provider_specific_settings = parse_config_section(config_file, model_provider_specific_section)
        optimizer = model_provider_settings["optimizer_name"]
        optimizer_section = "{0} {1} Settings".format(model_provider, optimizer)
        optimizer_settings = parse_config_section(config_file, optimizer_section)
        loss = model_provider_settings["loss_name"]
        loss_section = "{0} {1} Settings".format(model_provider, loss)
        loss_settings = parse_config_section(config_file, loss_section)
        model_settings.update({model_provider: model_provider_settings,
                               model_name: model_provider_specific_settings,
                               optimizer: optimizer_settings,
                               loss: loss_settings})
        self._set_attribute("_model_settings", model_settings)

    def _load_logger(self) -> Logger:
        # Get the necessary attributes.
        logging_settings = self.get_attribute("_logging_settings")
        client_id = self.get_attribute("_client_id")
        # Append the client's id to the output file name.
        file_name = Path(logging_settings["file_name"]).absolute()
        file_name = str(file_name.parent.joinpath(file_name.stem + "_{0}".format(client_id) + file_name.suffix))
        logging_settings["file_name"] = file_name
        # Set the logger name.
        logger_name = type(self).__name__ + "_Logger"
        # Load the logger.
        logger = load_logger(logging_settings, logger_name)
        # Return the logger.
        return logger

    def _load_local_dataset(self) -> tuple:
        # Start the dataset loading duration timer.
        dataset_loading_duration_start = perf_counter()
        # Get the necessary attributes.
        local_dataset_settings = self.get_attribute("_local_dataset_settings")
        dataset_root_folder = Path(local_dataset_settings["dataset_root_folder"])
        dataset_type = local_dataset_settings["dataset_type"]
        client_id = self.get_attribute("_client_id")
        logger = self.get_attribute("_logger")
        # Log a 'loading the dataset' message.
        message = "[Client {0}] Loading the '{1}' dataset (local storage)..." \
                  .format(client_id, dataset_root_folder)
        log_message(logger, message, "INFO")
        # Initialize x_train, y_train, x_test, and y_test.
        x_train = y_train = x_test = y_test = None
        match dataset_type:
            case "multi_class_image_classification":
                # Load x_train and y_train.
                x_train, y_train = load_x_y_for_multiclass_image_dataset(dataset_root_folder, "train")
                # Load x_test and y_test.
                x_test, y_test = load_x_y_for_multiclass_image_dataset(dataset_root_folder, "test")
        # Get the dataset load duration.
        dataset_loading_duration = perf_counter() - dataset_loading_duration_start
        # Log the dataset loading duration.
        message = "[Client {0}] The dataset loading took {1} seconds.".format(client_id, dataset_loading_duration)
        log_message(logger, message, "INFO")
        # Return the loaded dataset (x_train, y_train, x_test, and y_test).
        return x_train, y_train, x_test, y_test

    def _load_federated_dataset(self) -> tuple:
        # Get the necessary attributes.
        federated_dataset_settings = self.get_attribute("_federated_dataset_settings")
        dataset = federated_dataset_settings["dataset"]
        subset = federated_dataset_settings["subset"]
        dataset_partitioner = federated_dataset_settings["dataset_partitioner"]
        client_id = self.get_attribute("_client_id")
        num_partitions = federated_dataset_settings["num_partitions"]
        # Set the necessary keys.
        x_field_key = ""
        y_field_key = ""
        train_partitioner_key = ""
        test_partitioner_key = ""
        train_split_key = ""
        test_split_key = ""
        match dataset:
            case "uoft-cs/cifar10":
                x_field_key = "img"
                y_field_key = "label"
                train_partitioner_key = "train"
                test_partitioner_key = "test"
                train_split_key = "train"
                test_split_key = "test"
            case "uoft-cs/cifar100":
                x_field_key = "img"
                y_field_key = "coarse_label" # TODO: Improve the definition of the y_field key.
                train_partitioner_key = "train"
                test_partitioner_key = "test"
                train_split_key = "train"
                test_split_key = "test"
            case "ylecun/mnist":
                x_field_key = "image"
                y_field_key = "label"
                train_partitioner_key = "train"
                test_partitioner_key = "test"
                train_split_key = "train"
                test_split_key = "test"
            case "zalando-datasets/fashion_mnist":
                x_field_key = "image"
                y_field_key = "label"
                train_partitioner_key = "train"
                test_partitioner_key = "test"
                train_split_key = "train"
                test_split_key = "test"
            case "zh-plus/tiny-imagenet":
                x_field_key = "image"
                y_field_key = "label"
                train_partitioner_key = "train"
                test_partitioner_key = "valid"
                train_split_key = "train"
                test_split_key = "valid"
            case "benjamin-paine/imagenet-1k":
                x_field_key = "image"
                y_field_key = "label"
                train_partitioner_key = "train"
                test_partitioner_key = "validation"
                train_split_key = "train"
                test_split_key = "validation"
            case "ufldl-stanford/svhn":
                x_field_key = "image"
                y_field_key = "label"
                train_partitioner_key = "train"
                test_partitioner_key = "test"
                train_split_key = "train"
                test_split_key = "test"
            case "flwrlabs/cinic10":
                x_field_key = "image"
                y_field_key = "label"
                train_partitioner_key = "train"
                test_partitioner_key = "test"
                train_split_key = "train"
                test_split_key = "test"
        global fds # Initialize (download) 'FederatedDataset' only once.
        if fds is None:
            partitioners = {}
            training_dataset_partitioner = None
            test_dataset_partitioner = None
            match dataset_partitioner:
                case "IidPartitioner":
                    # Set the training dataset partitioner.
                    training_dataset_partitioner = IidPartitioner(num_partitions=num_partitions)
                    # Set the test dataset partitioner.
                    test_dataset_partitioner = IidPartitioner(num_partitions=num_partitions)
                case "DirichletPartitioner":
                    # Get the training dataset settings.
                    training_dataset_min_partition_size = federated_dataset_settings["training_dataset_min_partition_size"]
                    training_dataset_alpha = federated_dataset_settings["training_dataset_alpha"]
                    training_dataset_partition_by = federated_dataset_settings["training_dataset_partition_by"]
                    training_dataset_self_balancing = federated_dataset_settings["training_dataset_self_balancing"]
                    training_dataset_shuffle = federated_dataset_settings["training_dataset_shuffle"]
                    training_dataset_seed = federated_dataset_settings["training_dataset_seed"]
                    # Get the test dataset settings.
                    test_dataset_min_partition_size = federated_dataset_settings["test_dataset_min_partition_size"]
                    test_dataset_alpha = federated_dataset_settings["test_dataset_alpha"]
                    test_dataset_partition_by = federated_dataset_settings["test_dataset_partition_by"]
                    test_dataset_self_balancing = federated_dataset_settings["test_dataset_self_balancing"]
                    test_dataset_shuffle = federated_dataset_settings["test_dataset_shuffle"]
                    test_dataset_seed = federated_dataset_settings["test_dataset_seed"]
                    # Set the training dataset partitioner.
                    training_dataset_partitioner = DirichletPartitioner(num_partitions=num_partitions,
                                                                        partition_by=training_dataset_partition_by,
                                                                        alpha=training_dataset_alpha,
                                                                        min_partition_size=training_dataset_min_partition_size,
                                                                        self_balancing=training_dataset_self_balancing,
                                                                        shuffle=training_dataset_shuffle,
                                                                        seed=training_dataset_seed)
                    # Set the test dataset partitioner.
                    test_dataset_partitioner = DirichletPartitioner(num_partitions=num_partitions,
                                                                    partition_by=test_dataset_partition_by,
                                                                    alpha=test_dataset_alpha,
                                                                    min_partition_size=test_dataset_min_partition_size,
                                                                    self_balancing=test_dataset_self_balancing,
                                                                    shuffle=test_dataset_shuffle,
                                                                    seed=test_dataset_seed)
                case "PathologicalPartitioner":
                    # Get the training dataset settings.
                    training_dataset_partition_by = federated_dataset_settings["training_dataset_partition_by"]
                    training_num_classes_per_partition = federated_dataset_settings["training_num_classes_per_partition"]
                    training_class_assignment_mode = federated_dataset_settings["training_class_assignment_mode"]
                    training_dataset_shuffle = federated_dataset_settings["training_dataset_shuffle"]
                    training_dataset_seed = federated_dataset_settings["training_dataset_seed"]
                    # Get the test dataset settings.
                    test_dataset_partition_by = federated_dataset_settings["test_dataset_partition_by"]
                    test_num_classes_per_partition = federated_dataset_settings["test_num_classes_per_partition"]
                    test_class_assignment_mode = federated_dataset_settings["test_class_assignment_mode"]
                    test_dataset_shuffle = federated_dataset_settings["test_dataset_shuffle"]
                    test_dataset_seed = federated_dataset_settings["test_dataset_seed"]
                    # Set the training dataset partitioner.
                    training_dataset_partitioner = PathologicalPartitioner(num_partitions=num_partitions,
                                                                           partition_by=training_dataset_partition_by,
                                                                           num_classes_per_partition=training_num_classes_per_partition,
                                                                           class_assignment_mode=training_class_assignment_mode,
                                                                           shuffle=training_dataset_shuffle,
                                                                           seed=training_dataset_seed)
                    # Set the test dataset partitioner.
                    test_dataset_partitioner = PathologicalPartitioner(num_partitions=num_partitions,
                                                                       partition_by=test_dataset_partition_by,
                                                                       num_classes_per_partition=test_num_classes_per_partition,
                                                                       class_assignment_mode=test_class_assignment_mode,
                                                                       shuffle=test_dataset_shuffle,
                                                                       seed=test_dataset_seed)
            # Update the dictionary of partitioners.
            partitioners.update({train_partitioner_key: training_dataset_partitioner,
                                 test_partitioner_key: test_dataset_partitioner})
            # Instantiate the FederatedDataset object.
            fds = FederatedDataset(dataset=dataset,
                                   subset=subset,
                                   partitioners=partitioners)
        # Get the client's partitions (based on its id).
        partition_train = fds.load_partition(client_id, train_split_key)
        partition_train.set_format("numpy")
        partition_test = fds.load_partition(client_id, test_split_key)
        partition_test.set_format("numpy")
        # Load x_train and y_train.
        x_train, y_train = partition_train[x_field_key], partition_train[y_field_key]
        # Load x_test and y_test.
        x_test, y_test = partition_test[x_field_key], partition_test[y_field_key]
        # Return the loaded dataset (x_train, y_train, x_test, and y_test).
        return x_train, y_train, x_test, y_test

    def _load_dataset(self) -> tuple:
        # Get the necessary attributes.
        dataset_settings = self.get_attribute("_dataset_settings")
        loading_approach = dataset_settings["loading_approach"]
        # Initialize x_train, y_train, x_test, and y_test.
        x_train = y_train = x_test = y_test = None
        match loading_approach:
            case "Local":
                x_train, y_train, x_test, y_test = self._load_local_dataset()
            case "FederatedDataset":
                x_train, y_train, x_test, y_test = self._load_federated_dataset()
        # Return the loaded dataset (x_train, y_train, x_test, and y_test).
        return x_train, y_train, x_test, y_test

    def _get_task_assignment_capacities(self) -> tuple:
        # Get the necessary attributes.
        x_train = self.get_attribute("_x_train")
        x_test = self.get_attribute("_x_test")
        task_assignment_capacities_settings = self.get_attribute("_task_assignment_capacities_settings")
        task_assignment_capacities_train = task_assignment_capacities_settings["task_assignment_capacities_train"]
        task_assignment_capacities_test = task_assignment_capacities_settings["task_assignment_capacities_test"]
        if not task_assignment_capacities_train:
            lower_bound = task_assignment_capacities_settings["lower_bound"]
            upper_bound = task_assignment_capacities_settings["upper_bound"]
            if upper_bound == "client_capacity":
                upper_bound = len(x_train)
            task_assignment_capacities_train = [lower_bound, upper_bound]
            step = task_assignment_capacities_settings["step"]
            task_assignment_capacities_train.extend(list(range(lower_bound, upper_bound + 1, step)))
        if not task_assignment_capacities_test:
            lower_bound = task_assignment_capacities_settings["lower_bound"]
            upper_bound = task_assignment_capacities_settings["upper_bound"]
            if upper_bound == "client_capacity":
                upper_bound = len(x_test)
            task_assignment_capacities_test = [lower_bound, upper_bound]
            step = task_assignment_capacities_settings["step"]
            task_assignment_capacities_test.extend(list(range(lower_bound, upper_bound + 1, step)))
        task_assignment_capacities_train = sorted(list(set(task_assignment_capacities_train)))
        task_assignment_capacities_train_extension = list(range(task_assignment_capacities_train[-2] + 1,
                                                                task_assignment_capacities_train[-1]))
        task_assignment_capacities_train.extend(task_assignment_capacities_train_extension)
        task_assignment_capacities_train = sorted(list(set(task_assignment_capacities_train)))
        task_assignment_capacities_test = sorted(list(set(task_assignment_capacities_test)))
        task_assignment_capacities_test_extension = list(range(task_assignment_capacities_test[-2] + 1,
                                                               task_assignment_capacities_test[-1]))
        task_assignment_capacities_test.extend(task_assignment_capacities_test_extension)
        task_assignment_capacities_test = sorted(list(set(task_assignment_capacities_test)))
        return task_assignment_capacities_train, task_assignment_capacities_test

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
            match energy_monitor_name:
                case "pyJoules":
                    monitoring_domains = energy_monitor_settings["monitoring_domains"]
                    unit = energy_monitor_settings["unit"]
                    energy_monitor = PyJoulesEnergyMonitor(monitoring_domains, unit)
                case "PowerJoular":
                    monitoring_domains = energy_monitor_settings["monitoring_domains"]
                    unit = energy_monitor_settings["unit"]
                    process_monitoring = energy_monitor_settings["process_monitoring"]
                    unique_monitor = energy_monitor_settings["unique_monitor"]
                    report_consumptions_per_timestamp = energy_monitor_settings["report_consumptions_per_timestamp"]
                    remove_energy_consumptions_files = energy_monitor_settings["remove_energy_consumptions_files"]
                    energy_consumptions_file = energy_monitor_settings["energy_consumptions_file"]
                    energy_monitor = PowerJoularEnergyMonitor(monitoring_domains,
                                                              unit,
                                                              process_monitoring,
                                                              unique_monitor,
                                                              report_consumptions_per_timestamp,
                                                              remove_energy_consumptions_files,
                                                              energy_consumptions_file)
        self._set_attribute("_energy_monitor", energy_monitor)
        # Return the energy monitor.
        return energy_monitor

    def _load_optimizer(self) -> Optimizer:
        # Get the necessary attributes.
        model_settings = self.get_attribute("_model_settings")
        model_provider = model_settings["provider"]
        model_provider_settings = model_settings[model_provider]
        optimizer_name = model_provider_settings["optimizer_name"]
        optimizer_settings = model_settings[optimizer_name]
        # Initialize the optimizer.
        optimizer = None
        if model_provider == "Keras":
            match optimizer_name:
                case "Adam":
                    # Instantiate the Kera's Adam optimizer.
                    optimizer = Adam(learning_rate=optimizer_settings["learning_rate"],
                                     beta_1=optimizer_settings["beta_1"],
                                     beta_2=optimizer_settings["beta_2"],
                                     epsilon=optimizer_settings["epsilon"],
                                     amsgrad=optimizer_settings["amsgrad"],
                                     weight_decay=optimizer_settings["weight_decay"],
                                     clipnorm=optimizer_settings["clipnorm"],
                                     clipvalue=optimizer_settings["clipvalue"],
                                     global_clipnorm=optimizer_settings["global_clipnorm"],
                                     use_ema=optimizer_settings["use_ema"],
                                     ema_momentum=optimizer_settings["ema_momentum"],
                                     ema_overwrite_frequency=optimizer_settings["ema_overwrite_frequency"],
                                     loss_scale_factor=optimizer_settings["loss_scale_factor"],
                                     gradient_accumulation_steps=optimizer_settings["gradient_accumulation_steps"])
                case "SGD":
                    # Instantiate the Kera's SGD optimizer (Stochastic Gradient Descent).
                    optimizer = SGD(learning_rate=optimizer_settings["learning_rate"],
                                    momentum=optimizer_settings["momentum"],
                                    nesterov=optimizer_settings["nesterov"],
                                    name=optimizer_settings["optimizer_name"])
        # Return the optimizer.
        return optimizer

    def _load_loss_function(self) -> Loss:
        # Get the necessary attributes.
        model_settings = self.get_attribute("_model_settings")
        model_provider = model_settings["provider"]
        model_provider_settings = model_settings[model_provider]
        loss_name = model_provider_settings["loss_name"]
        loss_settings = model_settings[loss_name]
        # Initialize the loss.
        loss = None
        if model_provider == "Keras":
            match loss_name:
                case "SparseCategoricalCrossentropy":
                    # Instantiate the Kera's SparseCategoricalCrossentropy loss function.
                    loss = SparseCategoricalCrossentropy(from_logits=loss_settings["from_logits"],
                                                         ignore_class=loss_settings["ignore_class"],
                                                         reduction=loss_settings["reduction"],
                                                         name=loss_settings["loss_name"])
        # Return the loss function.
        return loss

    def _load_metrics(self) -> list[Metric]:
        # Get the necessary attributes.
        model_settings = self.get_attribute("_model_settings")
        model_provider = model_settings["provider"]
        model_provider_settings = model_settings[model_provider]
        metrics = model_provider_settings["metrics"]
        for index, metric in enumerate(metrics):
            if model_provider == "Keras":
                match metric:
                    case "sparse_categorical_accuracy":
                        # Instantiate the Kera's SparseCategoricalAccuracy metric.
                        metrics[index] = SparseCategoricalAccuracy()
        # Return the list of metrics.
        return metrics

    def _load_model(self) -> tuple:
        # Get the necessary attributes.
        model_settings = self.get_attribute("_model_settings")
        model_provider = model_settings["provider"]
        model_provider_settings = model_settings[model_provider]
        model_name = model_provider_settings["model_name"]
        model_provider_specific_settings = model_settings[model_name]
        # Load the optimizer.
        optimizer = self._load_optimizer()
        # Load the loss function.
        loss_function = self._load_loss_function()
        # Load the list of metrics.
        metrics = self._load_metrics()
        # Initialize the model and its metrics names.
        model = None
        metrics_names = None
        if model_provider == "Keras":
            match model_name:
                case "Custom_CNN_CIFAR-10":
                    # Define a custom CNN for the CIFAR-10 dataset (FedCS Paper):
                    #  Specifically, our model consisted of six 3 × 3 convolution layers (32, 32, 64, 64,
                    #  128, 128 channels, each of which was activated by ReLU and batch normalized,
                    #  and every two of which were followed by 2 × 2 max pooling)
                    #  followed by three fully-connected layers (382 and 192 units with ReLU activation and
                    #  another 10 units activated by soft-max).
                    model = Sequential()
                    # Input Layer.
                    model.add(layers.Input(shape=(32, 32, 3)))
                    # First Block: Conv(32) -> BatchNorm -> ReLU.
                    model.add(layers.Conv2D(32, (3, 3), padding="same"))
                    model.add(layers.BatchNormalization())
                    model.add(layers.ReLU())
                    # Second Block: Conv(32) -> BatchNorm -> ReLU.
                    model.add(layers.Conv2D(32, (3, 3), padding="same"))
                    model.add(layers.BatchNormalization())
                    model.add(layers.ReLU())
                    # 2 x 2 Max Pooling.
                    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
                    # Third Block: Conv(64) -> BatchNorm -> ReLU.
                    model.add(layers.Conv2D(64, (3, 3), padding="same"))
                    model.add(layers.BatchNormalization())
                    model.add(layers.ReLU())
                    # Fourth Block: Conv(64) -> BatchNorm -> ReLU.
                    model.add(layers.Conv2D(64, (3, 3), padding="same"))
                    model.add(layers.BatchNormalization())
                    model.add(layers.ReLU())
                    # 2 x 2 Max Pooling.
                    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
                    # Fifth Block: Conv(128) -> BatchNorm -> ReLU.
                    model.add(layers.Conv2D(128, (3, 3), padding="same"))
                    model.add(layers.BatchNormalization())
                    model.add(layers.ReLU())
                    # Sixth Block: Conv(128) -> BatchNorm -> ReLU.
                    model.add(layers.Conv2D(128, (3, 3), padding="same"))
                    model.add(layers.BatchNormalization())
                    model.add(layers.ReLU())
                    # 2 x 2 Max Pooling.
                    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
                    # Flatten the output from the convolutional layers.
                    model.add(layers.Flatten())
                    # First Fully Connected Layer: 382 units, ReLU activation.
                    model.add(layers.Dense(382, activation="relu"))
                    # Second Fully Connected Layer: 192 units, ReLU activation.
                    model.add(layers.Dense(192, activation="relu"))
                    # Third Fully Connected Layer: 10 units (output layer), softmax activation for multi-class classification.
                    model.add(layers.Dense(10, activation="softmax"))
                case "Custom_CNN_CIFAR-100_Fine_Labels":
                    # Define a custom CNN for the CIFAR-100 dataset (100 fine labels).
                    model = Sequential()
                    # Input Layer.
                    model.add(layers.Input(shape=(32, 32, 3)))
                    # First Convolutional Block.
                    model.add(layers.Conv2D(32, (3, 3), activation="relu"))
                    model.add(layers.MaxPooling2D((2, 2)))
                    # Second Convolutional Block.
                    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
                    model.add(layers.MaxPooling2D((2, 2)))
                    # Third Convolutional Block.
                    model.add(layers.Conv2D(128, (3, 3), activation="relu"))
                    model.add(layers.MaxPooling2D((2, 2)))
                    # Flatten the output for the fully connected layers.
                    model.add(layers.Flatten())
                    # Fully Connected Layers.
                    model.add(layers.Dense(128, activation="relu"))
                    # Dropout layer to reduce overfitting.
                    model.add(layers.Dropout(0.5))
                    # Output layer for 100 classes (softmax activation for multi-class classification).
                    model.add(layers.Dense(100, activation="softmax"))
                case "Custom_CNN_CIFAR-100_Coarse_Labels":
                    # Define a custom CNN for the CIFAR-100 dataset (20 coarse labels).
                    model = Sequential()
                    # Input Layer.
                    model.add(layers.Input(shape=(32, 32, 3)))
                    # First Convolutional Block.
                    model.add(layers.Conv2D(32, (3, 3), activation="relu"))
                    model.add(layers.MaxPooling2D((2, 2)))
                    # Second Convolutional Block.
                    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
                    model.add(layers.MaxPooling2D((2, 2)))
                    # Third Convolutional Block.
                    model.add(layers.Conv2D(128, (3, 3), activation="relu"))
                    model.add(layers.MaxPooling2D((2, 2)))
                    # Flatten the output for the fully connected layers.
                    model.add(layers.Flatten())
                    # Fully Connected Layers.
                    model.add(layers.Dense(128, activation="relu"))
                    # Dropout layer to reduce overfitting.
                    model.add(layers.Dropout(0.5))
                    # Output layer for 20 classes (softmax activation for multi-class classification).
                    model.add(layers.Dense(20, activation="softmax"))
                case "Custom_CNN_FashionMNIST":
                    # Define a custom CNN for the FashionMNIST dataset.
                    model = Sequential()
                    # Input Layer.
                    model.add(layers.Input(shape=(28, 28, 1)))
                    # First Convolutional Block.
                    model.add(layers.Conv2D(32, (3, 3), activation="relu"))
                    model.add(layers.MaxPooling2D((2, 2)))
                    # Second Convolutional Block.
                    model.add(layers.Conv2D(128, (3, 3), activation="relu"))
                    model.add(layers.MaxPooling2D((2, 2)))
                    # Flatten the output for the fully connected layers.
                    model.add(layers.Flatten())
                    # Dropout layer to reduce overfitting.
                    model.add(layers.Dropout(0.5))
                    # Output layer for 10 classes (softmax activation for multi-class classification).
                    model.add(layers.Dense(10, activation="softmax"))
                case "Custom_CNN_TinyImageNet":
                    # Define a custom CNN for the Tiny ImageNet dataset.
                    model = Sequential()
                    # Input Layer.
                    model.add(layers.Input(shape=(64, 64, 3)))
                    # First Convolutional Block.
                    model.add(layers.Conv2D(32, (3, 3), activation="relu"))
                    model.add(layers.MaxPooling2D((2, 2)))
                    # Second Convolutional Block.
                    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
                    model.add(layers.MaxPooling2D((2, 2)))
                    # Third Convolutional Block.
                    model.add(layers.Conv2D(128, (3, 3), activation="relu"))
                    model.add(layers.MaxPooling2D((2, 2)))
                    # Flatten the output for the fully connected layers.
                    model.add(layers.Flatten())
                    # Fully Connected Layers.
                    model.add(layers.Dense(512, activation="relu"))
                    # Dropout layer to reduce overfitting.
                    model.add(layers.Dropout(0.5))
                    # Output layer for 200 classes (softmax activation for multi-class classification).
                    model.add(layers.Dense(200, activation="softmax"))
                case "Custom_CNN_SVHN":
                    # Define a custom CNN for the SVHN dataset.
                    model = Sequential()
                    # Input Layer.
                    model.add(layers.Input(shape=(32, 32, 3)))
                    # First Convolutional Block.
                    model.add(layers.Conv2D(32, (3, 3), activation="relu"))
                    model.add(layers.MaxPooling2D((2, 2)))
                    # Second Convolutional Block.
                    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
                    model.add(layers.MaxPooling2D((2, 2)))
                    # Flatten the output for the fully connected layers.
                    model.add(layers.Flatten())
                    # Fully Connected Layers.
                    model.add(layers.Dense(128, activation="relu"))
                    # Dropout layer to reduce overfitting.
                    model.add(layers.Dropout(0.5))
                    # Output layer for 10 classes (softmax activation for multi-class classification).
                    model.add(layers.Dense(10, activation="softmax"))
                case "Custom_CNN_CINIC-10":
                    # Define a custom CNN for the CINIC-10 dataset.
                    model = Sequential()
                    # Input Layer.
                    model.add(layers.Input(shape=(32, 32, 3)))
                    # First Convolutional Block.
                    model.add(layers.Conv2D(32, (3, 3), activation="relu"))
                    model.add(layers.MaxPooling2D((2, 2)))
                    # Second Convolutional Block.
                    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
                    model.add(layers.MaxPooling2D((2, 2)))
                    # Third Convolutional Block.
                    model.add(layers.Conv2D(128, (3, 3), activation="relu"))
                    model.add(layers.MaxPooling2D((2, 2)))
                    # Flatten the output for the fully connected layers.
                    model.add(layers.Flatten())
                    # Fully Connected Layers.
                    model.add(layers.Dense(128, activation="relu"))
                    # Dropout layer to reduce overfitting.
                    model.add(layers.Dropout(0.5))
                    # Output layer for 10 classes (softmax activation for multi-class classification).
                    model.add(layers.Dense(10, activation="softmax"))
                case "EfficientNetB0":
                    # Instantiate the Kera's EfficientNetB0 model.
                    model = EfficientNetB0(input_shape=model_provider_specific_settings["input_shape"],
                                           include_top=model_provider_specific_settings["include_top"],
                                           weights=model_provider_specific_settings["weights"],
                                           input_tensor=model_provider_specific_settings["input_tensor"],
                                           pooling=model_provider_specific_settings["pooling"],
                                           classes=model_provider_specific_settings["classes"],
                                           classifier_activation=model_provider_specific_settings["classifier_activation"])
                case "EfficientNetV2L":
                    # Instantiate the Kera's EfficientNetV2L model.
                    model = EfficientNetV2L(input_shape=model_provider_specific_settings["input_shape"],
                                            include_top=model_provider_specific_settings["include_top"],
                                            weights=model_provider_specific_settings["weights"],
                                            input_tensor=model_provider_specific_settings["input_tensor"],
                                            pooling=model_provider_specific_settings["pooling"],
                                            classes=model_provider_specific_settings["classes"],
                                            classifier_activation=model_provider_specific_settings["classifier_activation"])
                case "MobileNetV2":
                    # Instantiate the Kera's MobileNetV2 model.
                    model = MobileNetV2(input_shape=model_provider_specific_settings["input_shape"],
                                        alpha=model_provider_specific_settings["alpha"],
                                        include_top=model_provider_specific_settings["include_top"],
                                        weights=model_provider_specific_settings["weights"],
                                        input_tensor=model_provider_specific_settings["input_tensor"],
                                        pooling=model_provider_specific_settings["pooling"],
                                        classes=model_provider_specific_settings["classes"],
                                        classifier_activation=model_provider_specific_settings["classifier_activation"])
                case "VGG16":
                    # Instantiate the Kera's VGG16 model.
                    model = VGG16(input_shape=model_provider_specific_settings["input_shape"],
                                  include_top=model_provider_specific_settings["include_top"],
                                  weights=model_provider_specific_settings["weights"],
                                  input_tensor=model_provider_specific_settings["input_tensor"],
                                  pooling=model_provider_specific_settings["pooling"],
                                  classes=model_provider_specific_settings["classes"],
                                  classifier_activation=model_provider_specific_settings["classifier_activation"])
                case "ResNet50":
                    # Instantiate the Kera's ResNet50 model.
                    model = ResNet50(input_shape=model_provider_specific_settings["input_shape"],
                                     include_top=model_provider_specific_settings["include_top"],
                                     weights=model_provider_specific_settings["weights"],
                                     input_tensor=model_provider_specific_settings["input_tensor"],
                                     pooling=model_provider_specific_settings["pooling"],
                                     classes=model_provider_specific_settings["classes"],
                                     classifier_activation=model_provider_specific_settings["classifier_activation"])
                case "DenseNet121":
                    # Instantiate the Kera's DenseNet121 model.
                    model = DenseNet121(input_shape=model_provider_specific_settings["input_shape"],
                                        include_top=model_provider_specific_settings["include_top"],
                                        weights=model_provider_specific_settings["weights"],
                                        input_tensor=model_provider_specific_settings["input_tensor"],
                                        pooling=model_provider_specific_settings["pooling"],
                                        classes=model_provider_specific_settings["classes"],
                                        classifier_activation=model_provider_specific_settings["classifier_activation"])
            # Compile the Kera's model.
            loss_weights = model_provider_settings["loss_weights"]
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
            # Get the model's metrics names.
            metrics_names = [metric.name for metric in vars(model)["_compile_metrics"]._user_metrics]
        # Return the model and its list of metrics names.
        return model, metrics_names

    @staticmethod
    def reshape_images(x: NDArray,
                       new_shape: tuple) -> NDArray:
        x_reshaped = []
        for index in range(0, len(x)):
            xi_copy = x[index].copy()
            xi_copy.resize(new_shape)
            x_reshaped.append(xi_copy)
        x_reshaped = array(x_reshaped)
        return x_reshaped

    def _pre_process_dataset(self) -> None:
        # Get the necessary attributes.
        federated_dataset_settings = self.get_attribute("_federated_dataset_settings")
        dataset = federated_dataset_settings["dataset"]
        x_train = self.get_attribute("_x_train")
        x_test = self.get_attribute("_x_test")
        model_settings = self.get_attribute("_model_settings")
        model_provider = model_settings["provider"]
        model_provider_settings = model_settings[model_provider]
        model_name = model_provider_settings["model_name"]
        # Reshape the data instances, if needed.
        match dataset:
            case "flwrlabs/cinic10":
                new_shape = (32, 32, 3)
                x_train = self.reshape_images(x_train, new_shape)
                x_test = self.reshape_images(x_test, new_shape)
            case "zh-plus/tiny-imagenet":
                new_shape = (64, 64, 3)
                x_train = self.reshape_images(x_train, new_shape)
                x_test = self.reshape_images(x_test, new_shape)
        # Set the list of custom CNNs.
        custom_cnns = ["Custom_CNN_CIFAR-10", "Custom_CNN_CIFAR-100_Fine_Labels", "Custom_CNN_CIFAR-100_Coarse_Labels",
                       "Custom_CNN_FashionMNIST", "Custom_CNN_SVHN", "Custom_CNN_CINIC-10", "Custom_CNN_TinyImageNet"]
        # Pre-process the dataset.
        if model_provider == "Keras":
            match model_name:
                case w if w in custom_cnns:
                    x_train = x_train / 255.0
                    x_test = x_test / 255.0
                case "EfficientNetB0":
                    x_train = efficientnet_preprocess_input(x=x_train)
                    x_test = efficientnet_preprocess_input(x=x_test)
                case "EfficientNetV2L":
                    x_train = efficientnet_v2_preprocess_input(x=x_train)
                    x_test = efficientnet_v2_preprocess_input(x=x_test)
                case "MobileNetV2":
                    x_train = mobilenet_v2_preprocess_input(x=x_train)
                    x_test = mobilenet_v2_preprocess_input(x=x_test)
                case "VGG16":
                    x_train = vgg16_preprocess_input(x=x_train)
                    x_test = vgg16_preprocess_input(x=x_test)
                case "ResNet50":
                    x_train = resnet50_preprocess_input(x=x_train)
                    x_test = resnet50_preprocess_input(x=x_test)
                case "DenseNet121":
                    x_train = densenet121_preprocess_input(x=x_train)
                    x_test = densenet121_preprocess_input(x=x_test)
        # Set the pre-processed dataset (x_train, x_test).
        self._set_attribute("_x_train", x_train)
        self._set_attribute("_x_test", x_test)

    def _instantiate_client(self) -> Client:
        # Get the necessary attributes.
        client_id = self.get_attribute("_client_id")
        logger = self.get_attribute("_logger")
        daemon_settings = self.get_attribute("_daemon_settings")
        affinity_settings = self.get_attribute("_affinity_settings")
        device_emulation_settings = self.get_attribute("_device_emulation_settings")
        simulation_resources_settings = self.get_attribute("_simulation_resources_settings")
        root_output_folder = self.get_attribute("_root_output_folder")
        model = self.get_attribute("_model")
        metrics_names = self.get_attribute("_metrics_names")
        x_train = self.get_attribute("_x_train")
        y_train = self.get_attribute("_y_train")
        x_test = self.get_attribute("_x_test")
        y_test = self.get_attribute("_y_test")
        train_task_capacities = self.get_attribute("_train_task_capacities")
        test_task_capacities = self.get_attribute("_test_task_capacities")
        energy_monitor = self.get_attribute("_energy_monitor")
        # Verify if the energy consumptions monitor to be used is PowerJoular
        # and if only one monitoring process is allowed to run in the system.
        if isinstance(energy_monitor, PowerJoularEnergyMonitor) and energy_monitor.get_attribute("_unique_monitor"):
            # Get the unique PowerJoular attributes.
            powerjoular_unique_attributes = vars(energy_monitor)
            powerjoular_unique_attributes.update({"_energy_monitor": "PowerJoular_Unique"})
            powerjoular_unique_attributes = list(powerjoular_unique_attributes.items())
            energy_monitor = powerjoular_unique_attributes
        # Instantiate the flower client.
        client = FlowerNumpyClient(id_=client_id,
                                   model=model,
                                   metrics_names=metrics_names,
                                   x_train=x_train,
                                   y_train=y_train,
                                   x_test=x_test,
                                   y_test=y_test,
                                   task_assignment_capacities_train=train_task_capacities,
                                   task_assignment_capacities_test=test_task_capacities,
                                   energy_monitor=energy_monitor,
                                   daemon_settings=daemon_settings,
                                   affinity_settings=affinity_settings,
                                   device_emulation_settings=device_emulation_settings,
                                   simulation_resources_settings = simulation_resources_settings,
                                   root_output_folder=root_output_folder,
                                   logger=logger)
        client = client.to_client()
        # Return the flower server.
        return client

    def _load_ssl_certificates(self) -> Optional[tuple[bytes]]:
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

    def _get_server_address(self) -> str:
        # Get the necessary attributes.
        grpc_settings = self.get_attribute("_grpc_settings")
        server_ip_address = grpc_settings["server_ip_address"]
        server_port = str(grpc_settings["server_port"])
        # Return the server address.
        return server_ip_address + ":" + server_port

    def _get_max_message_length_in_bytes(self) -> int:
        # Get the necessary attributes.
        grpc_settings = self.get_attribute("_grpc_settings")
        max_message_length_in_bytes = grpc_settings["max_message_length_in_bytes"]
        # Return the maximum message length in bytes.
        return max_message_length_in_bytes

    def _get_connection_retries_settings(self) -> tuple:
        # Get the necessary attributes.
        grpc_settings = self.get_attribute("_grpc_settings")
        max_connection_retries = grpc_settings["max_connection_retries"]
        max_backoff_in_seconds = grpc_settings["max_backoff_in_seconds"]
        # Return the maximum connection retries and maximum backoff in seconds.
        return max_connection_retries, max_backoff_in_seconds

    @staticmethod
    def _start_flower_client(client_id: int,
                             server_address: str,
                             client: Client,
                             grpc_max_message_length: int,
                             root_certificates: Optional[tuple[bytes, bytes, bytes]],
                             max_connection_retries: int,
                             max_backoff_in_seconds: float,
                             logger: Logger) -> None:
        # Start the flower client.
        current_try = 1
        while True:
            try:
                start_client(server_address=server_address,
                             client=client,
                             grpc_max_message_length=grpc_max_message_length,
                             root_certificates=root_certificates)
                break
            except _MultiThreadedRendezvous:
                traceback_exception_str = format_exc()
                if current_try == max_connection_retries:
                    raise traceback_exception_str
                random_second_fraction = round(uniform(0, 1), 2)
                wait_time = min(((2 ** current_try) + random_second_fraction), max_backoff_in_seconds)
                if "grpc_status:14" in traceback_exception_str:
                    message = ("[Client {0}] Could not connect to the Server ({1})! "
                               "Retrying in {2} seconds (retries left: {3})...") \
                              .format(client_id, server_address, wait_time, max_connection_retries - current_try)
                    log_message(logger, message, "INFO")
                sleep(wait_time)
                current_try += 1

    def launch_client(self) -> None:
        # Get the necessary attributes.
        client_id = self.get_attribute("_client_id")
        logger = self.get_attribute("_logger")
        client = self.get_attribute("_client")
        energy_monitor = self.get_attribute("_energy_monitor")
        # Get the flower server address (IP address and port).
        server_address = self._get_server_address()
        # Get the maximum message length in bytes.
        max_message_length_in_bytes = self._get_max_message_length_in_bytes()
        # Get the Secure Socket Layer (SSL) certificates (SSL-enabled secure connection).
        ssl_certificates = self._load_ssl_certificates()
        # Get the settings for connection retries to the server.
        max_connection_retries, max_backoff_in_seconds = self._get_connection_retries_settings()
        # Start the flower client.
        if isinstance(energy_monitor, PowerJoularEnergyMonitor) and energy_monitor.get_attribute("_unique_monitor"):
            # Start the unique PowerJoular monitoring process.
            energy_monitor.start()
            self._start_flower_client(client_id,
                                      server_address,
                                      client,
                                      max_message_length_in_bytes,
                                      ssl_certificates,
                                      max_connection_retries,
                                      max_backoff_in_seconds,
                                      logger)
            # Stop the unique PowerJoular monitoring process.
            energy_monitor.stop()
        else:
            # Start the flower client.
            self._start_flower_client(client_id,
                                      server_address,
                                      client,
                                      max_message_length_in_bytes,
                                      ssl_certificates,
                                      max_connection_retries,
                                      max_backoff_in_seconds,
                                      logger)
        # End.
        exit(0)
