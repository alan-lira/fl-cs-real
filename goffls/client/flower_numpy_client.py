from keras.saving import load_model, save_model
from logging import Logger
from multiprocessing import Process, Queue, set_start_method
from numpy.random import randint
from pathlib import Path
from time import perf_counter

from flwr.client import NumPyClient
from flwr.common import NDArray, NDArrays
from goffls.energy_monitor.powerjoular_energy_monitor import PowerJoularEnergyMonitor
from goffls.energy_monitor.pyjoules_energy_monitor import PyJoulesEnergyMonitor
from goffls.util.logger_util import log_message


class FlowerNumpyClient(NumPyClient):

    def __init__(self,
                 id_: int,
                 model: any,
                 x_train: NDArray,
                 y_train: NDArray,
                 x_test: NDArray,
                 y_test: NDArray,
                 energy_monitor: any,
                 daemon_mode: bool,
                 daemon_start_method: str,
                 logger: Logger) -> None:
        # Initialize the attributes.
        self._client_id = id_
        self._model = model
        self._x_train = x_train
        self._y_train = y_train
        self._x_test = x_test
        self._y_test = y_test
        self._energy_monitor = energy_monitor
        self._daemon_mode = daemon_mode
        self._logger = logger
        self._model_file = None
        # If the daemon mode is enabled...
        if daemon_mode:
            # Set the starting method of daemon processes.
            set_start_method(daemon_start_method)
            # Dump the local model to file.
            self._save_model(model)

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
        daemon_mode = self.get_attribute("_daemon_mode")
        if daemon_mode:
            # Set the local model file path.
            model_file = Path("output/models/flower_client_{0}.keras".format(client_id))
            model_file.parent.mkdir(exist_ok=True, parents=True)
            self._set_attribute("_model_file", model_file)
            # Dump the local model to file.
            save_model(model=model, filepath=model_file, overwrite=True)
            self._set_attribute("_model", None)
        else:
            # Set the local model.
            self._set_attribute("_model", model)

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
        """ Implementation of the abstract method of the NumPyClient class."""
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
        """ Implementation of the abstract method of the NumPyClient class."""
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
        energy_monitor = self.get_attribute("_energy_monitor")
        daemon_mode = self.get_attribute("_daemon_mode")
        # Get the model training process id from the fit queue.
        model_training_pid = None
        if daemon_mode:
            fit_queue_element = fit_queue.get()
            model_training_pid = fit_queue_element["model_training_pid"]
        # Start the model training energy consumption monitoring.
        if energy_monitor:
            tag = "training_energy"
            if isinstance(energy_monitor, PyJoulesEnergyMonitor):
                energy_monitor.start(tag)
            elif isinstance(energy_monitor, PowerJoularEnergyMonitor):
                energy_monitor.start(model_training_pid)
        # Start the model training duration timer.
        duration_start = perf_counter()
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
                            validation_batch_size=fit_config["validation_batch_size"]).history
        # Save the local model with the parameters (weights) obtained from the training.
        self._save_model(model)
        # Get the model training duration.
        duration = perf_counter() - duration_start
        # Stop the model training energy consumption monitoring.
        energy_consumptions = {}
        if energy_monitor:
            tag = "training_energy"
            if isinstance(energy_monitor, PyJoulesEnergyMonitor):
                energy_monitor.stop()
                # Get the energy consumptions.
                energy_consumptions = energy_monitor.get_energy_consumptions(tag)
            elif isinstance(energy_monitor, PowerJoularEnergyMonitor):
                if daemon_mode:
                    energy_monitor.stop()
                    # Get the energy consumptions.
                    energy_consumptions = energy_monitor.get_energy_consumptions(tag)
        # Put the model training result into the fit queue.
        model_training_result = {"history": history,
                                 "duration": duration,
                                 "energy_consumptions": energy_consumptions}
        fit_queue.put({"model_training_result": model_training_result})

    def fit(self,
            global_parameters: NDArrays,
            fit_config: dict) -> tuple[NDArrays, int, dict]:
        """ Implementation of the abstract method of the NumPyClient class."""
        # Get the necessary attributes.
        client_id = self.get_attribute("_client_id")
        x_train = self.get_attribute("_x_train")
        y_train = self.get_attribute("_y_train")
        daemon_mode = self.get_attribute("_daemon_mode")
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
                  .format(client_id, comm_round, str(daemon_mode).lower())
        log_message(logger, message, "INFO")
        # Unset the logger.
        self._set_attribute("_logger", None)
        # Initialize the model training queue.
        fit_queue = Queue()
        if daemon_mode:
            # Launch the model training process.
            target = self._train_model
            args = (global_parameters, x_train_sliced, y_train_sliced, fit_config, fit_queue)
            model_training_process = Process(target=target, args=args)
            model_training_process.start()
            # Get the model training process id.
            model_training_pid = model_training_process.pid
            # Put the model training process id into the fit queue.
            fit_queue.put({"model_training_pid": model_training_pid})
            # Wait for the model training process completion.
            model_training_process.join()
        else:
            # Execute the model training task.
            self._train_model(global_parameters, x_train_sliced, y_train_sliced, fit_config, fit_queue)
        # Get the model training result.
        fit_queue_element = fit_queue.get()
        model_training_result = fit_queue_element["model_training_result"]
        history = model_training_result["history"]
        duration = model_training_result["duration"]
        energy_consumptions = model_training_result["energy_consumptions"]
        # Add the model training duration to the training metrics.
        training_metrics.update({"training_time": duration})
        # Add the model training energy consumptions to the training metrics.
        training_metrics = training_metrics | energy_consumptions
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
        message = "[Client {0} | Round {1}] The model training took {2} seconds." \
                  .format(client_id, comm_round, duration)
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
        energy_monitor = self.get_attribute("_energy_monitor")
        daemon_mode = self.get_attribute("_daemon_mode")
        # Get the model testing process id from the evaluate queue.
        model_testing_pid = None
        if daemon_mode:
            evaluate_queue_element = evaluate_queue.get()
            model_testing_pid = evaluate_queue_element["model_testing_pid"]
        # Start the model testing energy consumption monitoring.
        if energy_monitor:
            tag = "testing_energy"
            if isinstance(energy_monitor, PyJoulesEnergyMonitor):
                energy_monitor.start(tag)
            elif isinstance(energy_monitor, PowerJoularEnergyMonitor):
                energy_monitor.start(model_testing_pid)
        # Start the model testing duration timer.
        duration_start = perf_counter()
        # Load the local model.
        model = self._load_model()
        # Update the parameters (weights) of the local model with those received from the server (global parameters).
        model.set_weights(global_parameters)
        # Test the local model using the local testing dataset slice.
        history = model.evaluate(x=x_test_sliced,
                                 y=y_test_sliced,
                                 batch_size=evaluate_config["batch_size"],
                                 steps=evaluate_config["steps"])
        # Save the local model with the parameters (weights) received from the server (global parameters).
        self._save_model(model)
        # Get the model testing duration.
        duration = perf_counter() - duration_start
        # Stop the model testing energy consumption monitoring.
        energy_consumptions = {}
        if energy_monitor:
            tag = "testing_energy"
            if isinstance(energy_monitor, PyJoulesEnergyMonitor):
                energy_monitor.stop()
                # Get the energy consumptions.
                energy_consumptions = energy_monitor.get_energy_consumptions(tag)
            elif isinstance(energy_monitor, PowerJoularEnergyMonitor):
                if daemon_mode:
                    energy_monitor.stop()
                    # Get the energy consumptions.
                    energy_consumptions = energy_monitor.get_energy_consumptions(tag)
        # Put the model testing result into the evaluate queue.
        model_testing_result = {"history": history,
                                "duration": duration,
                                "energy_consumptions": energy_consumptions}
        evaluate_queue.put({"model_testing_result": model_testing_result})

    def evaluate(self,
                 global_parameters: NDArrays,
                 evaluate_config: dict) -> tuple[float, int, dict]:
        """ Implementation of the abstract method of the NumPyClient class."""
        # Get the necessary attributes.
        client_id = self.get_attribute("_client_id")
        x_test = self.get_attribute("_x_test")
        y_test = self.get_attribute("_y_test")
        daemon_mode = self.get_attribute("_daemon_mode")
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
                  .format(client_id, comm_round, str(daemon_mode).lower())
        log_message(logger, message, "INFO")
        # Unset the logger.
        self._set_attribute("_logger", None)
        # Initialize the model testing queue.
        evaluate_queue = Queue()
        if daemon_mode:
            # Launch the model testing process.
            target = self._test_model
            args = (global_parameters, x_test_sliced, y_test_sliced, evaluate_config, evaluate_queue)
            model_testing_process = Process(target=target, args=args)
            model_testing_process.start()
            # Get the model testing process id.
            model_testing_pid = model_testing_process.pid
            # Put the model testing process id into the evaluate queue.
            evaluate_queue.put({"model_testing_pid": model_testing_pid})
            # Wait for the model testing process completion.
            model_testing_process.join()
        else:
            # Execute the model testing task.
            self._test_model(global_parameters, x_test_sliced, y_test_sliced, evaluate_config, evaluate_queue)
        # Get the model testing result.
        evaluate_queue_element = evaluate_queue.get()
        model_testing_result = evaluate_queue_element["model_testing_result"]
        history = model_testing_result["history"]
        duration = model_testing_result["duration"]
        energy_consumptions = model_testing_result["energy_consumptions"]
        # Add the model testing duration to the testing metrics.
        testing_metrics.update({"testing_time": duration})
        # Add the model testing energy consumptions to the testing metrics.
        testing_metrics = testing_metrics | energy_consumptions
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
        message = "[Client {0} | Round {1}] The model testing took {2} seconds." \
                  .format(client_id, comm_round, duration)
        log_message(logger, message, "INFO")
        # Send to the server the loss, number of testing examples, and testing metrics.
        return loss, num_testing_examples_to_use, testing_metrics
