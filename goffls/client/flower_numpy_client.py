from datetime import datetime
from keras.callbacks import Callback
from keras.saving import load_model, save_model
from logging import Logger
from multiprocessing import Process, Queue, set_start_method
from numpy.random import randint
from os import getpid
from pathlib import Path
from psutil import cpu_count
from socket import gethostname
from time import perf_counter, process_time

from flwr.client import NumPyClient
from flwr.common import NDArray, NDArrays
from goffls.energy_monitor.powerjoular_energy_monitor import PowerJoularEnergyMonitor
from goffls.energy_monitor.pyjoules_energy_monitor import PyJoulesEnergyMonitor
from goffls.utils.logger_util import log_message
from goffls.utils.platform_util import get_system


class TrainingMeasurementsCallback(Callback):

    def __init__(self,
                 energy_monitor: any) -> None:
        super().__init__()
        # Initialize the attributes.
        self._energy_monitor = energy_monitor
        self._energy_monitor_tag = "training_energy"
        self._training_start_timestamp = None
        self._training_elapsed_time_start = None
        self._training_elapsed_time = None
        self._training_cpu_time_start = None
        self._training_cpu_time = None
        self._training_energy_consumptions = {}

    def _set_attribute(self,
                       attribute_name: str,
                       attribute_value: any) -> None:
        setattr(self, attribute_name, attribute_value)

    def get_attribute(self,
                      attribute_name: str) -> any:
        return getattr(self, attribute_name)

    def on_train_begin(self,
                       logs=None) -> None:
        # Get the energy consumption monitor.
        energy_monitor = self.get_attribute("_energy_monitor")
        energy_monitor_tag = self.get_attribute("_energy_monitor_tag")
        # If there is an energy consumption monitor...
        if energy_monitor:
            # Start the model training energy consumption monitoring.
            if isinstance(energy_monitor, PyJoulesEnergyMonitor):
                energy_monitor.start(energy_monitor_tag)
            elif isinstance(energy_monitor, PowerJoularEnergyMonitor):
                model_training_pid = getpid()
                energy_monitor.start(model_training_pid)
        # Set the model training start.
        training_elapsed_time_start = perf_counter()
        training_cpu_time_start = process_time()
        training_start_timestamp = datetime.now()
        self._set_attribute("_training_elapsed_time_start", training_elapsed_time_start)
        self._set_attribute("_training_cpu_time_start", training_cpu_time_start)
        self._set_attribute("_training_start_timestamp", training_start_timestamp)

    def on_train_end(self,
                     logs=None) -> None:
        # Get the model training start.
        training_elapsed_time_start = self.get_attribute("_training_elapsed_time_start")
        training_cpu_time_start = self.get_attribute("_training_cpu_time_start")
        training_start_timestamp = self.get_attribute("_training_start_timestamp")
        # Set the model training duration (elapsed and CPU times).
        training_elapsed_time = perf_counter() - training_elapsed_time_start
        training_cpu_time = process_time() - training_cpu_time_start
        self._set_attribute("_training_elapsed_time", training_elapsed_time)
        self._set_attribute("_training_cpu_time", training_cpu_time)
        # Get the model training end timestamp.
        training_end_timestamp = datetime.now()
        # Get the energy consumption monitor.
        energy_monitor = self.get_attribute("_energy_monitor")
        energy_monitor_tag = self.get_attribute("_energy_monitor_tag")
        # If there is an energy consumption monitor...
        if energy_monitor:
            # Initialize the model training energy consumptions.
            training_energy_consumptions = {}
            # Stop the model training energy consumption monitoring and get the energy consumptions measurements.
            if isinstance(energy_monitor, PyJoulesEnergyMonitor):
                energy_monitor.stop()
                training_energy_consumptions = energy_monitor.get_energy_consumptions(energy_monitor_tag)
            elif isinstance(energy_monitor, PowerJoularEnergyMonitor):
                energy_monitor.stop()
                training_energy_consumptions = energy_monitor.get_energy_consumptions(energy_monitor_tag,
                                                                                      training_start_timestamp,
                                                                                      training_end_timestamp)
            # Set the model training energy consumptions.
            self._set_attribute("_training_energy_consumptions", training_energy_consumptions)


class TestingMeasurementsCallback(Callback):

    def __init__(self,
                 energy_monitor: any) -> None:
        super().__init__()
        # Initialize the attributes.
        self._energy_monitor = energy_monitor
        self._energy_monitor_tag = "testing_energy"
        self._testing_start_timestamp = None
        self._testing_elapsed_time_start = None
        self._testing_elapsed_time = None
        self._testing_cpu_time_start = None
        self._testing_cpu_time = None
        self._testing_energy_consumptions = {}

    def _set_attribute(self,
                       attribute_name: str,
                       attribute_value: any) -> None:
        setattr(self, attribute_name, attribute_value)

    def get_attribute(self,
                      attribute_name: str) -> any:
        return getattr(self, attribute_name)

    def on_test_begin(self,
                      logs=None) -> None:
        # Get the energy consumption monitor.
        energy_monitor = self.get_attribute("_energy_monitor")
        energy_monitor_tag = self.get_attribute("_energy_monitor_tag")
        # If there is an energy consumption monitor...
        if energy_monitor:
            # Start the model testing energy consumption monitoring.
            if isinstance(energy_monitor, PyJoulesEnergyMonitor):
                energy_monitor.start(energy_monitor_tag)
            elif isinstance(energy_monitor, PowerJoularEnergyMonitor):
                model_testing_pid = getpid()
                energy_monitor.start(model_testing_pid)
        # Set the model testing start.
        testing_elapsed_time_start = perf_counter()
        testing_cpu_time_start = process_time()
        testing_start_timestamp = datetime.now()
        self._set_attribute("_testing_elapsed_time_start", testing_elapsed_time_start)
        self._set_attribute("_testing_cpu_time_start", testing_cpu_time_start)
        self._set_attribute("_testing_start_timestamp", testing_start_timestamp)

    def on_test_end(self,
                    logs=None) -> None:
        # Get the model testing start.
        testing_elapsed_time_start = self.get_attribute("_testing_elapsed_time_start")
        testing_cpu_time_start = self.get_attribute("_testing_cpu_time_start")
        testing_start_timestamp = self.get_attribute("_testing_start_timestamp")
        # Set the model testing duration (elapsed and CPU times).
        testing_elapsed_time = perf_counter() - testing_elapsed_time_start
        testing_cpu_time = process_time() - testing_cpu_time_start
        self._set_attribute("_testing_elapsed_time", testing_elapsed_time)
        self._set_attribute("_testing_cpu_time", testing_cpu_time)
        # Get the model testing end timestamp.
        testing_end_timestamp = datetime.now()
        # Get the energy consumption monitor.
        energy_monitor = self.get_attribute("_energy_monitor")
        energy_monitor_tag = self.get_attribute("_energy_monitor_tag")
        # If there is an energy consumption monitor...
        if energy_monitor:
            # Initialize the model testing energy consumptions.
            testing_energy_consumptions = {}
            # Stop the model testing energy consumption monitoring and get the energy consumptions measurements.
            if isinstance(energy_monitor, PyJoulesEnergyMonitor):
                energy_monitor.stop()
                testing_energy_consumptions = energy_monitor.get_energy_consumptions(energy_monitor_tag)
            elif isinstance(energy_monitor, PowerJoularEnergyMonitor):
                energy_monitor.stop()
                testing_energy_consumptions = energy_monitor.get_energy_consumptions(energy_monitor_tag,
                                                                                     testing_start_timestamp,
                                                                                     testing_end_timestamp)
            # Set the model testing energy consumptions.
            self._set_attribute("_testing_energy_consumptions", testing_energy_consumptions)


class FlowerNumpyClient(NumPyClient):

    def __init__(self,
                 id_: int,
                 model: any,
                 x_train: NDArray,
                 y_train: NDArray,
                 x_test: NDArray,
                 y_test: NDArray,
                 energy_monitor: any,
                 daemon_settings: dict,
                 affinity_settings: dict,
                 logger: Logger) -> None:
        # Initialize the attributes.
        self._client_id = id_
        self._model = model
        self._x_train = x_train
        self._y_train = y_train
        self._x_test = x_test
        self._y_test = y_test
        self._energy_monitor = energy_monitor
        self._daemon_settings = daemon_settings
        self._affinity_settings = affinity_settings
        self._logger = logger
        self._model_file = None
        self._training_measurements_callback = TrainingMeasurementsCallback(energy_monitor)
        self._testing_measurements_callback = TestingMeasurementsCallback(energy_monitor)
        self._hostname = gethostname()
        self._num_cpus = cpu_count(logical=True)
        # Set the starting method of daemon processes.
        self._set_starting_method_of_daemon_processes()
        # Set the list of CPU cores to be used by the client (Linux only).
        self._set_list_of_cpu_cores()

    def _set_attribute(self,
                       attribute_name: str,
                       attribute_value: any) -> None:
        setattr(self, attribute_name, attribute_value)

    def get_attribute(self,
                      attribute_name: str) -> any:
        return getattr(self, attribute_name)

    def _save_model(self,
                    model: any) -> None:
        # Get the necessary attributes.
        client_id = self.get_attribute("_client_id")
        daemon_settings = self.get_attribute("_daemon_settings")
        enable_daemon_mode = daemon_settings["enable_daemon_mode"]
        if enable_daemon_mode:
            # Set the local model file path.
            model_file = Path("output/models/flower_client_{0}.keras".format(client_id)).absolute()
            model_file.parent.mkdir(exist_ok=True, parents=True)
            self._set_attribute("_model_file", model_file)
            # Dump the local model to file.
            save_model(model=model, filepath=model_file, overwrite=True)
            self._set_attribute("_model", None)
        else:
            # Set the local model.
            self._set_attribute("_model", model)

    def _set_starting_method_of_daemon_processes(self) -> None:
        # Get the necessary attributes.
        daemon_settings = self.get_attribute("_daemon_settings")
        enable_daemon_mode = daemon_settings["enable_daemon_mode"]
        start_method = daemon_settings["start_method"]
        model = self.get_attribute("_model")
        # If the daemon mode is enabled...
        if enable_daemon_mode:
            # Set the starting method of daemon processes.
            set_start_method(start_method)
            # Dump the local model to file.
            self._save_model(model)

    def _get_affinity_list(self) -> list:
        # Get the necessary attributes.
        client_id = self.get_attribute("_client_id")
        num_cpus = self.get_attribute("_num_cpus")
        affinity_settings = self.get_attribute("_affinity_settings")
        affinity_method = affinity_settings["affinity_method"]
        affinity_list = []
        cpus_ids = list(range(0, num_cpus))
        if affinity_method == "One_CPU_Core_Only":
            if client_id in cpus_ids:
                cpu_index = client_id
            else:
                cpu_index = client_id % num_cpus
            cpu_to_allocate = [cpus_ids[cpu_index]]
            affinity_list.extend(cpu_to_allocate)
        elif affinity_method == "CPU_Cores_List":
            cpu_cores_list = affinity_settings["cpu_cores_list"]
            if not cpu_cores_list:
                cpu_cores_list = cpus_ids
            affinity_list.extend(cpu_cores_list)
        return affinity_list

    def _set_list_of_cpu_cores(self) -> None:
        # If the operating system is Linux...
        if get_system() == "Linux":
            # Import the necessary functions.
            from os import sched_setaffinity
            # Get the necessary attributes.
            client_id = self.get_attribute("_client_id")
            logger = self.get_attribute("_logger")
            # Set the client process affinity (the list of CPU cores IDs to be used).
            affinity_list = self._get_affinity_list()
            sched_setaffinity(0, affinity_list)
            # Set the number of CPU cores to be used by the client.
            num_cpus = len(affinity_list)
            self._set_attribute("_num_cpus", num_cpus)
            # Log a 'eligible CPU cores' message.
            message = "[Client {0}] {1} CPU cores will be used (list of IDs): {2}" \
                      .format(client_id,
                              num_cpus,
                              ",".join([str(cpu_core_id) for cpu_core_id in affinity_list]))
            log_message(logger, message, "INFO")

    def _load_model(self) -> any:
        # Get the necessary attributes.
        model = self.get_attribute("_model")
        model_file = self.get_attribute("_model_file")
        if not model:
            # Load the local model from file.
            model = load_model(filepath=model_file, compile=True, safe_mode=True)
        return model

    def get_properties(self,
                       config: dict) -> dict:
        """ Implementation of the abstract method from the NumPyClient class."""
        if "client_id" in config:
            client_id = self.get_attribute("_client_id")
            config.update({"client_id": client_id})
        if "client_hostname" in config:
            client_hostname = self.get_attribute("_hostname")
            config.update({"client_hostname": client_hostname})
        if "client_num_cpus" in config:
            client_num_cpus = self.get_attribute("_num_cpus")
            config.update({"client_num_cpus": client_num_cpus})
        if "client_num_training_examples_available" in config:
            client_num_training_examples_available = len(self.get_attribute("_x_train"))
            config.update({"client_num_training_examples_available": client_num_training_examples_available})
        if "client_num_testing_examples_available" in config:
            client_num_testing_examples_available = len(self.get_attribute("_x_test"))
            config.update({"client_num_testing_examples_available": client_num_testing_examples_available})
        # Return the properties requested by the server.
        return config

    def get_parameters(self,
                       config: dict) -> NDArrays:
        """ Implementation of the abstract method from the NumPyClient class."""
        model = self._load_model()
        local_model_parameters = model.get_weights()
        # Return the current parameters (weights) of the local model requested by the server.
        return local_model_parameters

    @staticmethod
    def _get_slice_indices(num_examples_available: int,
                           num_examples_to_use: int) -> list:
        if num_examples_to_use == num_examples_available:
            slice_indices = list(range(0, num_examples_available))
        else:
            lower_index = 0
            higher_index = 0
            while (higher_index - lower_index) != (num_examples_to_use - 1):
                lower_index = randint(low=0, high=num_examples_available)
                higher_index = randint(low=lower_index, high=num_examples_available)
            slice_indices = list(range(lower_index, higher_index + 1))
        return slice_indices

    def _train_model(self,
                     global_parameters: NDArrays,
                     x_train_sliced: NDArray,
                     y_train_sliced: NDArray,
                     fit_config: dict,
                     fit_queue: Queue) -> None:
        # Get the necessary attributes.
        training_measurements_callback = self.get_attribute("_training_measurements_callback")
        # Load the local model.
        model = self._load_model()
        # Update the parameters (weights) of the local model with those received from the server (global parameters).
        model.set_weights(global_parameters)
        # Train the local model using the local training dataset slice.
        history = model.fit(x=x_train_sliced,
                            y=y_train_sliced,
                            shuffle=fit_config["shuffle"],
                            batch_size=fit_config["batch_size"],
                            initial_epoch=fit_config["initial_epoch"],
                            epochs=fit_config["epochs"],
                            steps_per_epoch=fit_config["steps_per_epoch"],
                            validation_split=fit_config["validation_split"],
                            validation_batch_size=fit_config["validation_batch_size"],
                            verbose=fit_config["verbose"],
                            callbacks=[training_measurements_callback]).history
        # Save the local model with the parameters (weights) obtained from the training.
        self._save_model(model)
        # Put the model training result into the fit_queue.
        training_elapsed_time = training_measurements_callback.get_attribute("_training_elapsed_time")
        training_cpu_time = training_measurements_callback.get_attribute("_training_cpu_time")
        training_energy_consumptions = training_measurements_callback.get_attribute("_training_energy_consumptions")
        model_training_result = {"history": history,
                                 "training_elapsed_time": training_elapsed_time,
                                 "training_cpu_time": training_cpu_time,
                                 "training_energy_consumptions": training_energy_consumptions}
        fit_queue.put({"model_training_result": model_training_result})

    def fit(self,
            global_parameters: NDArrays,
            fit_config: dict) -> tuple[NDArrays, int, dict]:
        """ Implementation of the abstract method from the NumPyClient class."""
        # Get the necessary attributes.
        client_id = self.get_attribute("_client_id")
        x_train = self.get_attribute("_x_train")
        y_train = self.get_attribute("_y_train")
        daemon_settings = self.get_attribute("_daemon_settings")
        enable_daemon_mode = daemon_settings["enable_daemon_mode"]
        logger = self.get_attribute("_logger")
        # Initialize the training metrics dictionary.
        training_metrics = {}
        # Get the current communication round.
        comm_round = fit_config["comm_round"]
        # Get the number of training examples available.
        num_training_examples_available = len(x_train)
        # Get the number of training examples to use.
        if "num_training_examples_to_use" in fit_config:
            num_training_examples_to_use = fit_config["num_training_examples_to_use"]
        else:
            num_training_examples_to_use = len(x_train)
        # Get the indices that will be used to slice the training dataset.
        slice_indices = self._get_slice_indices(num_training_examples_available, num_training_examples_to_use)
        # Slice the training dataset.
        x_train_sliced = x_train[slice_indices]
        y_train_sliced = y_train[slice_indices]
        # Replace 'None' values to None (necessary workaround on Flower).
        fit_config = {k: (None if v == "None" else v) for k, v in fit_config.items()}
        # Log the training configuration (fit_config) received from the server.
        message = "[Client {0} | Round {1}] Received fit_config: {2}".format(client_id, comm_round, fit_config)
        log_message(logger, message, "DEBUG")
        # Log a 'training the model' message.
        message = "[Client {0} | Round {1}] Training the model (daemon mode: {2})..." \
                  .format(client_id, comm_round, str(enable_daemon_mode).lower())
        log_message(logger, message, "INFO")
        # Unset the logger.
        self._set_attribute("_logger", None)
        # Initialize the model training queue (fit_queue).
        fit_queue = Queue()
        if enable_daemon_mode:
            # Launch the model training process.
            target = self._train_model
            args = (global_parameters, x_train_sliced, y_train_sliced, fit_config, fit_queue)
            model_training_process = Process(target=target, args=args)
            model_training_process.start()
            # Wait for the model training process completion.
            model_training_process.join()
        else:
            # Execute the model training task.
            self._train_model(global_parameters, x_train_sliced, y_train_sliced, fit_config, fit_queue)
        # Get the model training result.
        fit_queue_element = fit_queue.get()
        model_training_result = fit_queue_element["model_training_result"]
        history = model_training_result["history"]
        training_elapsed_time = model_training_result["training_elapsed_time"]
        training_cpu_time = model_training_result["training_cpu_time"]
        training_energy_consumptions = model_training_result["training_energy_consumptions"]
        # Add the model training duration to the training metrics.
        training_metrics.update({"training_elapsed_time": training_elapsed_time,
                                 "training_cpu_time": training_cpu_time})
        # Add the model training energy consumptions to the training metrics.
        training_metrics = training_metrics | training_energy_consumptions
        # Get the parameters (weights) of the local model obtained from the training.
        model = self._load_model()
        local_model_parameters = model.get_weights()
        # Get the training metrics names.
        training_metrics_names = history.keys()
        # Store the training metrics of the last epoch.
        for training_metric_name in training_metrics_names:
            training_metrics.update({training_metric_name: history[training_metric_name][-1]})
        # Add the number of training examples used to the training metrics.
        training_metrics.update({"num_training_examples_used": num_training_examples_to_use})
        # Set the logger.
        self._set_attribute("_logger", logger)
        # Log the model training duration.
        message = "[Client {0} | Round {1}] The model training elapsed time: {2} seconds (CPU time: {3} seconds)." \
                  .format(client_id, comm_round, training_elapsed_time, training_cpu_time)
        log_message(logger, message, "INFO")
        # Send to the server the local model parameters (weights), number of training examples, and training metrics.
        return local_model_parameters, num_training_examples_to_use, training_metrics

    def _test_model(self,
                    global_parameters: NDArrays,
                    x_test_sliced: NDArray,
                    y_test_sliced: NDArray,
                    evaluate_config: dict,
                    evaluate_queue: Queue) -> None:
        # Get the necessary attributes.
        testing_measurements_callback = self.get_attribute("_testing_measurements_callback")
        # Load the local model.
        model = self._load_model()
        # Update the parameters (weights) of the local model with those received from the server (global parameters).
        model.set_weights(global_parameters)
        # Test the local model using the local testing dataset slice.
        history = model.evaluate(x=x_test_sliced,
                                 y=y_test_sliced,
                                 batch_size=evaluate_config["batch_size"],
                                 steps=evaluate_config["steps"],
                                 verbose=evaluate_config["verbose"],
                                 callbacks=[testing_measurements_callback])
        # Save the local model with the parameters (weights) received from the server (global parameters).
        self._save_model(model)
        # Put the model testing result into the evaluate_queue.
        testing_elapsed_time = testing_measurements_callback.get_attribute("_testing_elapsed_time")
        testing_cpu_time = testing_measurements_callback.get_attribute("_testing_cpu_time")
        testing_energy_consumptions = testing_measurements_callback.get_attribute("_testing_energy_consumptions")
        model_testing_result = {"history": history,
                                "testing_elapsed_time": testing_elapsed_time,
                                "testing_cpu_time": testing_cpu_time,
                                "testing_energy_consumptions": testing_energy_consumptions}
        evaluate_queue.put({"model_testing_result": model_testing_result})

    def evaluate(self,
                 global_parameters: NDArrays,
                 evaluate_config: dict) -> tuple[float, int, dict]:
        """ Implementation of the abstract method from the NumPyClient class."""
        # Get the necessary attributes.
        client_id = self.get_attribute("_client_id")
        x_test = self.get_attribute("_x_test")
        y_test = self.get_attribute("_y_test")
        daemon_settings = self.get_attribute("_daemon_settings")
        enable_daemon_mode = daemon_settings["enable_daemon_mode"]
        logger = self.get_attribute("_logger")
        # Initialize the testing metrics dictionary.
        testing_metrics = {}
        # Get the current communication round.
        comm_round = evaluate_config["comm_round"]
        # Get the number of testing examples available.
        num_testing_examples_available = len(x_test)
        # Get the number of testing examples to use.
        if "num_testing_examples_to_use" in evaluate_config:
            num_testing_examples_to_use = evaluate_config["num_testing_examples_to_use"]
        else:
            num_testing_examples_to_use = len(x_test)
        # Get the indices that will be used to slice the testing dataset.
        slice_indices = self._get_slice_indices(num_testing_examples_available, num_testing_examples_to_use)
        # Slice the testing dataset.
        x_test_sliced = x_test[slice_indices]
        y_test_sliced = y_test[slice_indices]
        # Replace 'None' values to None (necessary workaround on Flower).
        evaluate_config = {k: (None if v == "None" else v) for k, v in evaluate_config.items()}
        # Log the testing configuration (evaluate_config) received from the server.
        message = "[Client {0} | Round {1}] Received evaluate_config: {2}" \
                  .format(client_id, comm_round, evaluate_config)
        log_message(logger, message, "DEBUG")
        # Log a 'testing the model' message.
        message = "[Client {0} | Round {1}] Testing the model (daemon mode: {2})..." \
                  .format(client_id, comm_round, str(enable_daemon_mode).lower())
        log_message(logger, message, "INFO")
        # Unset the logger.
        self._set_attribute("_logger", None)
        # Initialize the model testing queue (evaluate_queue).
        evaluate_queue = Queue()
        if enable_daemon_mode:
            # Launch the model testing process.
            target = self._test_model
            args = (global_parameters, x_test_sliced, y_test_sliced, evaluate_config, evaluate_queue)
            model_testing_process = Process(target=target, args=args)
            model_testing_process.start()
            # Wait for the model testing process completion.
            model_testing_process.join()
        else:
            # Execute the model testing task.
            self._test_model(global_parameters, x_test_sliced, y_test_sliced, evaluate_config, evaluate_queue)
        # Get the model testing result.
        evaluate_queue_element = evaluate_queue.get()
        model_testing_result = evaluate_queue_element["model_testing_result"]
        history = model_testing_result["history"]
        testing_elapsed_time = model_testing_result["testing_elapsed_time"]
        testing_cpu_time = model_testing_result["testing_cpu_time"]
        testing_energy_consumptions = model_testing_result["testing_energy_consumptions"]
        # Add the model testing duration to the testing metrics.
        testing_metrics.update({"testing_elapsed_time": testing_elapsed_time,
                                "testing_cpu_time": testing_cpu_time})
        # Add the model testing energy consumptions to the testing metrics.
        testing_metrics = testing_metrics | testing_energy_consumptions
        # Get the testing metrics names.
        model = self._load_model()
        testing_metrics_names = model.metrics_names
        # Store the testing metrics.
        for index, testing_metric_name in enumerate(testing_metrics_names):
            testing_metrics.update({testing_metric_name: history[index]})
        # Add the number of testing examples used to the testing metrics.
        testing_metrics.update({"num_testing_examples_used": num_testing_examples_to_use})
        # Get the loss value.
        loss = testing_metrics["loss"]
        # Set the logger.
        self._set_attribute("_logger", logger)
        # Log the model testing duration.
        message = "[Client {0} | Round {1}] The model testing elapsed time: {2} seconds (CPU time: {3} seconds)." \
                  .format(client_id, comm_round, testing_elapsed_time, testing_cpu_time)
        log_message(logger, message, "INFO")
        # Send to the server the loss, number of testing examples, and testing metrics.
        return loss, num_testing_examples_to_use, testing_metrics
