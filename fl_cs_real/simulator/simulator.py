from pathlib import Path
from random import choices, seed
from time import perf_counter

from flwr.client import Client, ClientApp
from flwr.common import Context
from flwr.server import ServerApp, ServerAppComponents
from flwr.simulation import run_simulation
from fl_cs_real.client_launcher.flower_client_launcher import FlowerClientLauncher
from fl_cs_real.edge_devices.edge_devices import generate_edge_devices
from fl_cs_real.server_launcher.flower_server_launcher import FlowerServerLauncher
from fl_cs_real.utils.config_parser_util import parse_config_section, get_all_section_names


class Simulator:

    def __init__(self,
                 config_file: Path) -> None:
        # Initialize the attributes.
        self._config_file = config_file
        self._current_simulation = {}
        self._current_simulation_devices = []

    def _set_attribute(self,
                       attribute_name: str,
                       attribute_value: any) -> None:
        setattr(self, attribute_name, attribute_value)

    def get_attribute(self,
                      attribute_name: str) -> any:
        return getattr(self, attribute_name)

    @staticmethod
    def _personalize_flower_client(client_id: int,
                                   base_client_config_file: Path,
                                   simulation_dict: dict,
                                   current_simulation_devices: list) -> dict:
        # Initialize the personalized_settings dictionary.
        personalized_settings = {}
        # Get the simulation name and settings.
        simulation_name = next(iter(simulation_dict))
        simulation_settings = simulation_dict[simulation_name]
        # Set the device emulation settings.
        device_name = current_simulation_devices[client_id][0]
        device_emulation_settings = current_simulation_devices[client_id][1]
        personalized_settings.update({"_device_emulation_settings": device_emulation_settings})
        # Set the simulation resources settings.
        simulation_num_cpus = simulation_settings["backend_config"]["client_resources"]["num_cpus"]
        simulation_num_gpus = simulation_settings["backend_config"]["client_resources"]["num_gpus"]
        simulation_resources_settings = {"simulation_num_cpus": simulation_num_cpus,
                                         "simulation_num_gpus": simulation_num_gpus}
        personalized_settings.update({"_simulation_resources_settings": simulation_resources_settings})
        # Get the simulation output folder.
        simulation_output_folder = simulation_settings["simulation_output_folder"]
        # Set the root output folder.
        personalized_settings.update({"_root_output_folder": simulation_output_folder})
        # Set the simulation output folder as the parent folder for the logging file.
        logging_settings = parse_config_section(base_client_config_file, "Logging Settings")
        logging_settings["file_name"] = simulation_output_folder + "/" + logging_settings["file_name"]
        personalized_settings.update({"_logging_settings": logging_settings})
        # Set the number of partitions based on the number of supernodes (same value for all clients).
        num_partitions = simulation_settings["num_supernodes"]
        personalized_settings.update({"_federated_dataset_settings": {"num_partitions": num_partitions}})
        # Return the personalized_settings dictionary.
        return personalized_settings

    def _client_fn(self,
                   context: Context) -> Client:
        # Get the necessary attributes.
        partition_id = context.node_config["partition-id"]
        num_partitions = context.node_config["num-partitions"]
        current_simulation = self.get_attribute("_current_simulation")
        current_simulation_devices = self.get_attribute("_current_simulation_devices")
        # Get the simulation name and settings.
        simulation_name = next(iter(current_simulation))
        simulation_settings = current_simulation[simulation_name]
        # Get the base client config file.
        base_client_config_file = Path(simulation_settings["base_client_config_file"])
        # Load the personalized_settings dictionary for the client.
        personalized_settings = self._personalize_flower_client(partition_id,
                                                                base_client_config_file,
                                                                current_simulation,
                                                                current_simulation_devices)
        # Instantiate the flower client launcher.
        fcl = FlowerClientLauncher(partition_id, base_client_config_file, personalized_settings)
        # Get the flower client.
        flower_client = fcl.get_attribute("_client")
        # Return the flower client.
        return flower_client

    @staticmethod
    def _personalize_flower_server(base_server_config_file: Path,
                                   simulation_dict: dict) -> dict:
        # Initialize the personalized_settings dictionary.
        personalized_settings = {}
        # Get the simulation name and settings.
        simulation_name = next(iter(simulation_dict))
        simulation_settings = simulation_dict[simulation_name]
        # Get the simulation output folder.
        simulation_output_folder = simulation_settings["simulation_output_folder"]
        # Set the root output folder.
        personalized_settings.update({"_root_output_folder": simulation_output_folder})
        # Set the simulation output folder as the parent folder for the logging file.
        logging_settings = parse_config_section(base_server_config_file, "Logging Settings")
        logging_settings["file_name"] = simulation_output_folder + "/" + logging_settings["file_name"]
        personalized_settings.update({"_logging_settings": logging_settings})
        # Set the simulation output folder as the parent folder for all output files.
        output_settings = parse_config_section(base_server_config_file, "Output Settings")
        output_file_keys_to_update = []
        for k, v in output_settings.items():
            if isinstance(v, str) and ".csv" in v:
                output_file_keys_to_update.append(k)
        for k in output_file_keys_to_update:
            output_settings[k] = simulation_output_folder + "/" + output_settings[k]
        personalized_settings.update({"_output_settings": output_settings})
        # Set the number of clients to wait based on the number of supernodes.
        num_clients_to_wait = simulation_settings["num_supernodes"]
        fl_settings = parse_config_section(base_server_config_file, "FL Settings")
        fl_settings["wait_for_initial_clients"]["num_clients_to_wait"] = num_clients_to_wait
        personalized_settings.update({"_fl_settings": fl_settings})
        # Return the personalized_settings dictionary.
        return personalized_settings

    def _server_fn(self,
                   context: Context) -> ServerAppComponents:
        # Get the necessary attributes.
        current_simulation = self.get_attribute("_current_simulation")
        # Get the simulation name and settings.
        simulation_name = next(iter(current_simulation))
        simulation_settings = current_simulation[simulation_name]
        # Get the server id.
        server_id = simulation_settings["server_id"]
        # Get the base server config file.
        base_server_config_file = Path(simulation_settings["base_server_config_file"])
        # Load the personalized_settings dictionary for the server.
        personalized_settings = self._personalize_flower_server(base_server_config_file, current_simulation)
        # Instantiate the flower server launcher.
        fsl = FlowerServerLauncher(server_id, base_server_config_file, personalized_settings)
        # Get the server strategy.
        server_strategy = fsl.get_attribute("_server_strategy")
        # Get the server config.
        server_config = fsl.get_attribute("_server_config")
        # Return the ServerAppComponents.
        return ServerAppComponents(strategy=server_strategy, config=server_config)

    def simulate(self) -> None:
        # Set the list of simulations to execute.
        simulations_to_execute = []
        config_file = self.get_attribute("_config_file")
        section_names = get_all_section_names(config_file)
        for section_name in section_names:
            simulation_settings = parse_config_section(config_file, section_name)
            simulation_name = section_name.split(" Settings")[0]
            simulations_to_execute.append({simulation_name: simulation_settings})
        # Iterate through the list of simulations to execute.
        for simulation_to_execute in simulations_to_execute:
            # Update the current simulation.
            self._set_attribute("_current_simulation", simulation_to_execute)
            # Get the simulation name and settings.
            simulation_name = next(iter(simulation_to_execute))
            simulation_settings = simulation_to_execute[simulation_name]
            # Check if the devices performance emulation is enabled.
            emulate_devices_performance = simulation_settings["emulate_devices_performance"]
            if emulate_devices_performance:
                # Set the seed (base value used by the pseudo-random functions) to allow replicable analysis.
                if "seed" in simulation_settings:
                    seed(simulation_settings["seed"])
                # Initialize the dictionary of emulated devices.
                emulated_devices = {}
                # Get the device types to emulate.
                device_types_to_emulate = simulation_settings["device_types_to_emulate"]
                for device_type_to_emulate in device_types_to_emulate:
                    match device_type_to_emulate:
                        case "edge":
                            # Emulate edge devices performance.
                            edge_devices = generate_edge_devices()
                            emulated_devices = emulated_devices | edge_devices
                        case "edge_4_cores":
                            # Emulate edge devices performance.
                            edge_devices = generate_edge_devices()
                            # Filter out the edge devices whose number of CPU cores is not equal to 4.
                            edge_devices = {k: v for k, v in edge_devices.items() if v["num_cpu_cores"] == 4}
                            emulated_devices = emulated_devices | edge_devices
                # Check if the random sampling of emulated devices is enabled.
                random_sampling_emulated_devices = simulation_settings["random_sampling_emulated_devices"]
                if random_sampling_emulated_devices:
                    # Check if it is needed to filter the emulated devices according to the clients' number of CPUs.
                    filter_devices_according_to_client_resources = simulation_settings["filter_devices_according_to_client_resources"]
                    if filter_devices_according_to_client_resources:
                        # Filter the candidate devices according to the number of CPUs to be used by the clients.
                        simulation_num_cpus = simulation_settings["backend_config"]["client_resources"]["num_cpus"]
                        emulated_devices = {k: v for k, v in emulated_devices.items()
                                            if v["num_cpu_cores"] == simulation_num_cpus}
                    # Get the number of clients.
                    num_supernodes = simulation_settings["num_supernodes"]
                    # Select randomly the performance for each client (allow repetition in the sampling).
                    current_simulation_devices = choices(population=list(emulated_devices.items()), k=num_supernodes)
                    # Update the current simulation devices.
                    self._set_attribute("_current_simulation_devices", current_simulation_devices)
                else:
                    manual_emulated_devices_list = simulation_settings["manual_emulated_devices_list"] # TODO
            # Create the ClientApp passing the client generation function.
            client_app = ClientApp(client_fn=self._client_fn)
            # Create the ServerApp passing the server generation function.
            server_app = ServerApp(server_fn=self._server_fn)
            # Print the start of the simulation.
            print("\nStarting the simulation '{0}'...".format(simulation_name))
            # Run the simulation.
            start = perf_counter()
            run_simulation(server_app=server_app,
                           client_app=client_app,
                           num_supernodes=simulation_settings["num_supernodes"],
                           backend_config=simulation_settings["backend_config"])
            end = perf_counter()
            # Print the elapsed time for the simulation.
            elapsed_time_seconds = round((end - start), 2)
            print("\nElapsed time of '{0}': {1} seconds".format(simulation_name, elapsed_time_seconds))
