from os import getenv, system
from pathlib import Path
from psutil import pid_exists
from random import randint
from subprocess import Popen, PIPE


class PowerJoularEnergyMonitor:

    def __init__(self,
                 env_variables: dict,
                 monitoring_domains: list,
                 unit: str) -> None:
        # Initialize the attributes.
        self._env_variables = env_variables
        self._monitoring_domains = monitoring_domains
        self._unit = unit
        self._energy_consumptions_temp_file = None
        self._to_monitor_pid = None
        self._powerjoular_monitoring_pid = None

    def _set_attribute(self,
                       attribute_name: str,
                       attribute_value: any) -> None:
        setattr(self, attribute_name, attribute_value)

    def get_attribute(self,
                      attribute_name: str) -> any:
        return getattr(self, attribute_name)

    @staticmethod
    def _get_available_monitoring_domains() -> list:
        available_monitoring_domains = ["Total", "CPU", "NVIDIA_GPU"]
        return available_monitoring_domains

    def start(self,
              to_monitor_pid: int) -> None:
        # Get the necessary attributes.
        env_variables = self.get_attribute("_env_variables")
        # Load the password from the environment variable.
        pw = getenv(env_variables["pw"])
        # Set the energy consumptions temporary file.
        energy_consumptions_temp_file = Path("energy_consumptions_temp_" + str(randint(1, 9999999)))
        self._set_attribute("_energy_consumptions_temp_file", energy_consumptions_temp_file)
        # Define the PowerJoular monitoring command.
        monitoring_command = "sudo -S powerjoular -p {0} -f {1}" \
                             .format(to_monitor_pid, energy_consumptions_temp_file).split()
        # Start the PowerJoular monitoring process.
        powerjoular_monitoring_process = Popen(monitoring_command,
                                               stdin=PIPE,
                                               stderr=PIPE,
                                               universal_newlines=True,
                                               shell=False)
        # Communicate the password.
        powerjoular_monitoring_process.communicate("{0}\n".format(pw))
        # Get the PowerJoular monitoring process id.
        powerjoular_monitoring_pid = powerjoular_monitoring_process.pid
        # Set the to-monitor process id.
        self._set_attribute("_to_monitor_pid", to_monitor_pid)
        # Set the PowerJoular monitoring process id.
        self._set_attribute("_powerjoular_monitoring_pid", powerjoular_monitoring_pid)

    def stop(self) -> None:
        # Get the necessary attributes.
        powerjoular_monitoring_pid = self.get_attribute("_powerjoular_monitoring_pid")
        # If the PowerJoular monitoring process is still running...
        if powerjoular_monitoring_pid and pid_exists(powerjoular_monitoring_pid):
            # Set the PowerJoular killing command.
            killing_command = "sudo pkill -9 -P " + str(powerjoular_monitoring_pid)
            # Kill the PowerJoular monitoring process.
            system(killing_command)
            # Unset the PowerJoular monitoring process id.
            self._set_attribute("_powerjoular_monitoring_pid", None)

    def get_energy_consumptions(self,
                                tag: str) -> dict:
        # Initialize the energy consumptions dictionary.
        energy_consumptions = {}
        # Get the necessary attributes.
        monitoring_domains = self.get_attribute("_monitoring_domains")
        energy_consumptions_temp_file = self.get_attribute("_energy_consumptions_temp_file")
        to_monitor_pid = self.get_attribute("_to_monitor_pid")
        # If the energy consumptions temporary file exists...
        if energy_consumptions_temp_file.is_file():
            # Initialize the energy consumptions measurements variables.
            total_energy_measurements = []
            cpu_energy_measurements = []
            nvidia_gpu_energy_measurements = []
            with open(energy_consumptions_temp_file, mode="r") as file:
                next(file)  # Skip the header line.
                # Iterate through the timestamp lines.
                for line in file:
                    line_split = line.strip().split(",")
                    # Get the Total energy consumption for the current timestamp, returned as Joules (J).
                    total_energy = float(line_split[2])
                    # Get the CPU energy consumption for the current timestamp, returned as Joules (J).
                    energy_cpu = float(line_split[3])
                    # Get the NVIDIA GPU energy consumption for the current timestamp, returned as Joules (J).
                    energy_nvidia_gpu = float(line_split[4])
                    total_energy_measurements.append(total_energy)
                    cpu_energy_measurements.append(energy_cpu)
                    nvidia_gpu_energy_measurements.append(energy_nvidia_gpu)
            # Remove the energy consumptions temporary file.
            energy_consumptions_temp_file.unlink(missing_ok=True)
            # Remove the auto-generated energy consumptions .csv file.
            energy_consumptions_csv_file = Path(str(energy_consumptions_temp_file) + "-{0}.csv".format(to_monitor_pid))
            energy_consumptions_csv_file.unlink(missing_ok=True)
            # Unset the energy consumptions temporary file.
            self._set_attribute("_energy_consumptions_temp_file", None)
            # Unset the to-monitor process id.
            self._set_attribute("_to_monitor_pid", None)
            # Iterate through the list of monitoring domains.
            for monitoring_domain in monitoring_domains:
                if monitoring_domain == "Total":
                    # Add the Total energy consumptions sum to the energy consumptions dictionary.
                    energy_consumptions.update({tag + "_total": sum(total_energy_measurements)})
                elif monitoring_domain == "CPU":
                    # Add the CPU energy consumptions sum to the energy consumptions dictionary.
                    energy_consumptions.update({tag + "_cpu": sum(cpu_energy_measurements)})
                elif monitoring_domain == "NVIDIA_GPU":
                    # Add the NVIDIA GPU energy consumptions sum to the energy consumptions dictionary.
                    energy_consumptions.update({tag + "_nvidia_gpu": sum(nvidia_gpu_energy_measurements)})
        # Return the energy consumptions dictionary.
        return energy_consumptions