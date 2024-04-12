from logging import Logger
from numpy.random import randint
from pyJoules.energy_meter import EnergyMeter
from time import perf_counter

from flwr.client import NumPyClient
from flwr.common import NDArray, NDArrays

from goffls.util.logger_util import log_message


class FlowerNumpyClient(NumPyClient):

    def __init__(self,
                 client_id: int,
                 model: any,
                 x_train: NDArray,
                 y_train: NDArray,
                 x_test: NDArray,
                 y_test: NDArray,
                 energy_monitor: any,
                 logger: Logger) -> None:
        self._client_id = client_id
        self._model = model
        self._x_train = x_train
        self._y_train = y_train
        self._x_test = x_test
        self._y_test = y_test
        self._energy_monitor = energy_monitor
        self._logger = logger

    def _set_attribute(self,
                       attribute_name: str,
                       attribute_value: any) -> None:
        setattr(self, attribute_name, attribute_value)

    def get_attribute(self,
                      attribute_name: str) -> any:
        return getattr(self, attribute_name)

    def get_properties(self,
                       config: dict) -> dict:
        if "client_id" in config:
            client_id = self.get_attribute("_client_id")
            config.update({"client_id": client_id})
        if "num_training_examples_available" in config:
            num_training_examples_available = len(self.get_attribute("_x_train"))
            config.update({"num_training_examples_available": num_training_examples_available})
        if "num_testing_examples_available" in config:
            num_testing_examples_available = len(self.get_attribute("_x_test"))
            config.update({"num_testing_examples_available": num_testing_examples_available})
        # Return the properties requested by the server.
        return config

    def get_parameters(self,
                       config: dict) -> NDArrays:
        model = self.get_attribute("_model")
        local_model_parameters = model.get_weights()
        # Return the current parameters (weights) of the local model requested by the server.
        return local_model_parameters

    def _update_model_parameters(self,
                                 model_parameters: NDArrays) -> None:
        model = self.get_attribute("_model")
        model.set_weights(model_parameters)
        self._set_attribute("_model", model)

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

    @staticmethod
    def _get_pyjoules_energy_consumption_measurements(last_trace: any,
                                                      tag: str) -> dict:
        energy_consumption_measurements = {}
        if last_trace["tag"] == tag:
            energy_dict = last_trace["energy"]
            if "package_0" in energy_dict:
                # Get the CPU energy consumption, returned as Micro-Joules (μJ), then convert to Joules (J).
                energy_cpu = energy_dict["package_0"] / (1 * pow(10, 6))
                # Add the CPU energy consumption to the dictionary of measurements.
                energy_consumption_measurements.update({tag + "_cpu": energy_cpu})
            if "core_0" in energy_dict:
                # Get the CPU Cores energy consumption, returned as Micro-Joules (μJ), then convert to Joules (J).
                energy_cpu_cores = energy_dict["core_0"] / (1 * pow(10, 6))
                # Add the CPU Cores energy consumption to the dictionary of measurements.
                energy_consumption_measurements.update({tag + "_cpu_cores": energy_cpu_cores})
            if "uncore_0" in energy_dict:
                # Get the Integrated GPU energy consumption, returned as Micro-Joules (μJ), then convert to Joules (J).
                energy_integrated_gpu = energy_dict["uncore_0"] / (1 * pow(10, 6))
                # Add the Integrated GPU energy consumption to the dictionary of measurements.
                energy_consumption_measurements.update({tag + "_integrated_gpu": energy_integrated_gpu})
            if "nvidia_gpu_0" in energy_dict:
                # Get the NVIDIA GPU energy consumption, returned as Milli-Joules (mJ), then convert to Joules (J).
                energy_nvidia_gpu = energy_dict["nvidia_gpu_0"] / (1 * pow(10, 3))
                # Add the NVIDIA GPU energy consumption to the dictionary of measurements.
                energy_consumption_measurements.update({tag + "_nvidia_gpu": energy_nvidia_gpu})
            if "dram_0" in energy_dict:
                # Get the RAM energy consumption, returned as Micro-Joules (μJ), then convert to Joules (J).
                energy_ram = energy_dict["dram_0"] / (1 * pow(10, 6))
                # Add the RAM energy consumption to the dictionary of measurements.
                energy_consumption_measurements.update({tag + "_ram": energy_ram})
        return energy_consumption_measurements

    def fit(self,
            global_model_parameters: NDArrays,
            fit_config: dict) -> tuple[NDArrays, int, dict]:
        # Start the model training duration timer.
        model_training_duration_start = perf_counter()
        # Start the model training energy consumption monitor, if any.
        energy_monitor = self.get_attribute("_energy_monitor")
        if energy_monitor:
            if isinstance(energy_monitor, EnergyMeter):
                energy_monitor.start(tag="training_energy")
        # Update the parameters (weights) of the local model with those received from the server (global parameters).
        self._update_model_parameters(global_model_parameters)
        # Get the necessary attributes.
        client_id = self.get_attribute("_client_id")
        model = self.get_attribute("_model")
        x_train = self.get_attribute("_x_train")
        y_train = self.get_attribute("_y_train")
        logger = self.get_attribute("_logger")
        # Initialize the training metrics dictionary.
        training_metrics = {}
        # Get the current communication round.
        comm_round = fit_config["comm_round"]
        # Get the number of training examples available.
        num_training_examples_available = len(x_train)
        # Get the number of training examples to use.
        if "num_training_examples" in fit_config:
            num_training_examples_to_use = fit_config["num_training_examples"]
        else:
            num_training_examples_to_use = len(x_train)
        # Get the indices that will be used to slice the training dataset.
        slice_indices = self._get_slice_indices(num_training_examples_available, num_training_examples_to_use)
        # Slice the training dataset.
        x_train_sliced = x_train[slice_indices]
        y_train_sliced = y_train[slice_indices]
        # Add the number of training examples to the training metrics.
        training_metrics.update({"num_training_examples": num_training_examples_to_use})
        # Replace 'None' values to None (necessary workaround on Flower).
        fit_config = {k: (None if v == "None" else v) for k, v in fit_config.items()}
        # Log the training configuration (fit_config) received from the server.
        message = "[Client {0} | Round {1}] Received fit_config: {2}".format(client_id, comm_round, fit_config)
        log_message(logger, message, "DEBUG")
        # Log a 'training the model' message.
        message = "[Client {0} | Round {1}] Training the model...".format(client_id, comm_round)
        log_message(logger, message, "INFO")
        # Train the local model with updated global parameters (weights) using the local training dataset.
        history = model.fit(x=x_train_sliced,
                            y=y_train_sliced,
                            shuffle=fit_config["shuffle"],
                            batch_size=fit_config["batch_size"],
                            initial_epoch=fit_config["initial_epoch"],
                            epochs=fit_config["epochs"],
                            steps_per_epoch=fit_config["steps_per_epoch"],
                            validation_split=fit_config["validation_split"],
                            validation_batch_size=fit_config["validation_batch_size"]).history
        # Get the parameters (weights) of the local model obtained from the training.
        local_model_parameters = model.get_weights()
        # Update the parameters (weights) of the local model with those obtained from the training (local parameters).
        self._update_model_parameters(local_model_parameters)
        # Get the training metrics names.
        training_metrics_names = history.keys()
        # Store the training metrics of the last epoch.
        for training_metric_name in training_metrics_names:
            training_metrics.update({training_metric_name: history[training_metric_name][-1]})
        # Stop the model training energy consumption monitor, if any.
        if energy_monitor:
            if isinstance(energy_monitor, EnergyMeter):
                energy_monitor.stop()
                # Get the pyJoules energy consumption measurements.
                last_trace = vars(energy_monitor.get_trace()[0])
                tag = "training_energy"
                energy_consumption_measurements = self._get_pyjoules_energy_consumption_measurements(last_trace, tag)
                # Add the model training energy consumption measurements to the training metrics.
                training_metrics = training_metrics | energy_consumption_measurements
        # Get the model training duration.
        model_training_duration = perf_counter() - model_training_duration_start
        # Add the model training duration to the training metrics.
        training_metrics.update({"training_time": model_training_duration})
        # Log the model training duration.
        message = "[Client {0} | Round {1}] The model training took {2} seconds." \
                  .format(client_id, comm_round, model_training_duration)
        log_message(logger, message, "INFO")
        # Send to the server the local model parameters (weights), number of training examples, and training metrics.
        return local_model_parameters, num_training_examples_to_use, training_metrics

    def evaluate(self,
                 global_model_parameters: NDArrays,
                 evaluate_config: dict) -> tuple[float, int, dict]:
        # Start the model testing duration timer.
        model_testing_duration_start = perf_counter()
        # Start the model testing energy consumption monitor, if any.
        energy_monitor = self.get_attribute("_energy_monitor")
        if energy_monitor:
            if isinstance(energy_monitor, EnergyMeter):
                energy_monitor.start(tag="testing_energy")
        # Update the parameters (weights) of the local model with those received from the server (global parameters).
        self._update_model_parameters(global_model_parameters)
        # Get the necessary attributes.
        client_id = self.get_attribute("_client_id")
        model = self.get_attribute("_model")
        x_test = self.get_attribute("_x_test")
        y_test = self.get_attribute("_y_test")
        logger = self.get_attribute("_logger")
        # Initialize the testing metrics dictionary.
        testing_metrics = {}
        # Get the current communication round.
        comm_round = evaluate_config["comm_round"]
        # Get the number of testing examples available.
        num_testing_examples_available = len(x_test)
        # Get the number of testing examples to use.
        if "num_testing_examples" in evaluate_config:
            num_testing_examples_to_use = evaluate_config["num_testing_examples"]
        else:
            num_testing_examples_to_use = len(x_test)
        # Get the indices that will be used to slice the testing dataset.
        slice_indices = self._get_slice_indices(num_testing_examples_available, num_testing_examples_to_use)
        # Slice the testing dataset.
        x_test_sliced = x_test[slice_indices]
        y_test_sliced = y_test[slice_indices]
        # Add the number of testing examples to the testing metrics.
        testing_metrics.update({"num_testing_examples": num_testing_examples_to_use})
        # Replace 'None' values to None (necessary workaround on Flower).
        evaluate_config = {k: (None if v == "None" else v) for k, v in evaluate_config.items()}
        # Log the testing configuration (evaluate_config) received from the server.
        message = "[Client {0} | Round {1}] Received evaluate_config: {2}" \
                  .format(client_id, comm_round, evaluate_config)
        log_message(logger, message, "DEBUG")
        # Log a 'testing the model' message.
        message = "[Client {0} | Round {1}] Testing the model...".format(client_id, comm_round)
        log_message(logger, message, "INFO")
        # Test the local model with updated global parameters (weights) using the local testing dataset.
        history = model.evaluate(x=x_test_sliced,
                                 y=y_test_sliced,
                                 batch_size=evaluate_config["batch_size"],
                                 steps=evaluate_config["steps"])
        # Get the testing metrics names.
        testing_metrics_names = model.metrics_names
        # Store the testing metrics.
        for index, testing_metric_name in enumerate(testing_metrics_names):
            testing_metrics.update({testing_metric_name: history[index]})
        # Stop the model testing energy consumption monitor, if any.
        if energy_monitor:
            if isinstance(energy_monitor, EnergyMeter):
                energy_monitor.stop()
                # Get the pyJoules energy consumption measurements.
                last_trace = vars(energy_monitor.get_trace()[0])
                tag = "testing_energy"
                energy_consumption_measurements = self._get_pyjoules_energy_consumption_measurements(last_trace, tag)
                # Add the model testing energy consumption measurements to the testing metrics.
                testing_metrics = testing_metrics | energy_consumption_measurements
        # Get the model testing duration.
        model_testing_duration = perf_counter() - model_testing_duration_start
        # Add the model testing duration to the testing metrics.
        testing_metrics.update({"testing_time": model_testing_duration})
        # Get the loss value.
        loss = testing_metrics["loss"]
        # Log the model testing duration.
        message = "[Client {0} | Round {1}] The model testing took {2} seconds." \
                  .format(client_id, comm_round, model_testing_duration)
        log_message(logger, message, "INFO")
        # Send to the server the loss, number of testing examples, and testing metrics.
        return loss, num_testing_examples_to_use, testing_metrics
