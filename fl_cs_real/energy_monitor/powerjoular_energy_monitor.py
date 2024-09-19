from datetime import datetime
from dateutil import parser
from os import getenv
from pathlib import Path
from psutil import AccessDenied, NoSuchProcess, process_iter, ZombieProcess
from random import randint
from subprocess import Popen, PIPE, TimeoutExpired
from time import sleep
from typing import Optional


class PowerJoularEnergyMonitor:

    def __init__(self,
                 monitoring_domains: list,
                 unit: str,
                 process_monitoring: bool,
                 unique_monitor: bool,
                 report_consumptions_per_timestamp: bool,
                 remove_energy_consumptions_files: bool,
                 energy_consumptions_file: Path) -> None:
        # Initialize the attributes.
        self._monitoring_domains = monitoring_domains
        self._unit = unit
        self._process_monitoring = process_monitoring
        self._unique_monitor = unique_monitor
        self._report_consumptions_per_timestamp = report_consumptions_per_timestamp
        self._remove_energy_consumptions_files = remove_energy_consumptions_files
        self._energy_consumptions_file = energy_consumptions_file
        self._to_monitor_pid = None
        self._powerjoular_monitoring_process = None

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

    @staticmethod
    def _powerjoular_process_is_already_running() -> bool:
        for process in process_iter():
            try:
                if "powerjoular".casefold() in process.name().casefold():
                    return True
            except (NoSuchProcess, AccessDenied, ZombieProcess):
                pass
        return False

    def start(self,
              to_monitor_pid: Optional[int] = None) -> None:
        # Get the necessary attributes.
        process_monitoring = self.get_attribute("_process_monitoring")
        unique_monitor = self.get_attribute("_unique_monitor")
        energy_consumptions_file = Path(self.get_attribute("_energy_consumptions_file")).absolute()
        powerjoular_process_is_already_running = self._powerjoular_process_is_already_running()
        if not unique_monitor:
            # If more than one monitoring process is running in the system,
            # a different name of the energy consumptions file should be used for each process.
            energy_consumptions_file_prefix = str(energy_consumptions_file)
            energy_consumptions_file = Path(energy_consumptions_file_prefix + "_" + str(randint(1, 9999999)))
        self._set_attribute("_energy_consumptions_file", energy_consumptions_file)
        # Verify if more than one monitoring process is allowed to run in the system.
        # If not, verify if the single monitoring process has not been launched yet.
        if not unique_monitor or (unique_monitor and not powerjoular_process_is_already_running):
            # Load the password from the environment variable.
            pw = getenv("POWERJOULAR_PW")
            # Set the to-monitor process id.
            self._set_attribute("_to_monitor_pid", to_monitor_pid)
            # Define the PowerJoular monitoring command.
            if process_monitoring and isinstance(to_monitor_pid, int):
                monitoring_command = "sudo -S powerjoular -p {0} -f {1}".format(to_monitor_pid,
                                                                                energy_consumptions_file).split()
            else:
                monitoring_command = "sudo -S powerjoular -f {0}".format(energy_consumptions_file).split()
            # Start the PowerJoular monitoring process.
            powerjoular_monitoring_process = Popen(args=monitoring_command,
                                                   stdin=PIPE,
                                                   stdout=None,
                                                   stderr=None,
                                                   universal_newlines=True,
                                                   shell=False)
            # Communicate the password, but do not wait for the process completion.
            try:
                communicate_input = "{0}\n".format(pw)
                communicate_timeout = 0.001
                _, _ = powerjoular_monitoring_process.communicate(input=communicate_input, timeout=communicate_timeout)
            except TimeoutExpired:
                pass
            # Set the PowerJoular monitoring process.
            self._set_attribute("_powerjoular_monitoring_process", powerjoular_monitoring_process)

    def stop(self) -> None:
        # Get the necessary attributes.
        powerjoular_monitoring_process = self.get_attribute("_powerjoular_monitoring_process")
        if powerjoular_monitoring_process:
            # Kill the PowerJoular monitoring process.
            powerjoular_monitoring_process.kill()
            _, _ = powerjoular_monitoring_process.communicate()

    @staticmethod
    def _get_line_energy_consumptions(line: str) -> dict:
        line_split = line.strip().split(",")
        # Get the timestamp.
        timestamp = str(line_split[0])
        # Get the Total energy consumption, returned as Joules (J).
        total_energy = float(line_split[2])
        # Get the CPU energy consumption, returned as Joules (J).
        energy_cpu = float(line_split[3])
        # Get the NVIDIA GPU energy consumption, returned as Joules (J).
        energy_nvidia_gpu = float(line_split[4])
        # Set the timestamp consumptions dict.
        timestamp_consumptions = {"total_energy": total_energy,
                                  "energy_cpu": energy_cpu,
                                  "energy_nvidia_gpu": energy_nvidia_gpu}
        line_energy_consumptions = {timestamp: timestamp_consumptions}
        # Return the line_energy_consumptions dict.
        return line_energy_consumptions

    def _remove_energy_consumption_files(self) -> None:
        # Get the necessary attributes.
        to_monitor_pid = self.get_attribute("_to_monitor_pid")
        energy_consumptions_file = Path(self.get_attribute("_energy_consumptions_file")).absolute()
        # Remove the energy consumptions file.
        energy_consumptions_file.unlink(missing_ok=True)
        # Remove the auto-generated energy consumptions .csv file, if any.
        if isinstance(to_monitor_pid, int):
            energy_consumptions_csv_file = Path(str(energy_consumptions_file)
                                                + "-{0}.csv".format(to_monitor_pid)).absolute()
            energy_consumptions_csv_file.unlink(missing_ok=True)

    def get_energy_consumptions(self,
                                monitoring_tag: str,
                                start_timestamp: datetime,
                                end_timestamp: datetime) -> dict:
        # Get the necessary attributes.
        monitoring_domains = self.get_attribute("_monitoring_domains")
        unique_monitor = self.get_attribute("_unique_monitor")
        report_consumptions_per_timestamp = self.get_attribute("_report_consumptions_per_timestamp")
        remove_energy_consumptions_files = self.get_attribute("_remove_energy_consumptions_files")
        energy_consumptions_file = Path(self.get_attribute("_energy_consumptions_file")).absolute()
        # Initialize the energy consumptions dictionary.
        energy_consumptions = {}
        # If the energy consumptions file exists...
        if energy_consumptions_file.is_file():
            # Wait for one second before reading the energy consumptions file.
            sleep(1)
            # Initialize the timestamps_consumptions dict.
            timestamps_consumptions = {}
            # Read the energy consumptions file.
            with open(energy_consumptions_file, mode="r") as file:
                # Skip the header line.
                next(file)
                # Iterate through the timestamp lines.
                for line in file:
                    # Get the energy consumptions for the current timestamp line.
                    line_energy_consumptions = self._get_line_energy_consumptions(line)
                    # Update the timestamps_consumptions dict.
                    if not unique_monitor:
                        timestamps_consumptions.update(line_energy_consumptions)
                    else:
                        timestamp = next(iter(line_energy_consumptions))
                        timestamp = parser.parse(timestamp).replace(tzinfo=None)
                        if start_timestamp.replace(microsecond=0) <= timestamp <= end_timestamp:
                            timestamps_consumptions.update(line_energy_consumptions)
                        if timestamp > end_timestamp:
                            break
            # Iterate through the energy consumption timestamps.
            for timestamp, timestamp_consumptions in timestamps_consumptions.items():
                # Iterate through the list of monitoring domains for the current timestamp.
                for monitoring_domain in monitoring_domains:
                    if monitoring_domain == "Total":
                        if report_consumptions_per_timestamp:
                            # Add the Total energy consumptions to the energy_consumptions dict.
                            energy_consumption_key = monitoring_tag + "_total_{0}".format(timestamp)
                            energy_consumption_value = timestamp_consumptions["total_energy"]
                        else:
                            # Add the Total energy consumptions sums to the energy_consumptions dict.
                            energy_consumption_key = monitoring_tag + "_total"
                            energy_consumption_value = sum([timestamp_consumptions["total_energy"]
                                                            for _, timestamp_consumptions in
                                                            timestamps_consumptions.items()])
                        energy_consumptions.update({energy_consumption_key: energy_consumption_value})
                    elif monitoring_domain == "CPU":
                        if report_consumptions_per_timestamp:
                            # Add the CPU energy consumptions to the energy_consumptions dict.
                            energy_consumption_key = monitoring_tag + "_cpu_{0}".format(timestamp)
                            energy_consumption_value = timestamp_consumptions["energy_cpu"]
                        else:
                            # Add the CPU energy consumptions sums to the energy_consumptions dict.
                            energy_consumption_key = monitoring_tag + "_cpu"
                            energy_consumption_value = sum([timestamp_consumptions["energy_cpu"]
                                                            for _, timestamp_consumptions in
                                                            timestamps_consumptions.items()])
                        energy_consumptions.update({energy_consumption_key: energy_consumption_value})
                    elif monitoring_domain == "NVIDIA_GPU":
                        if report_consumptions_per_timestamp:
                            # Add the NVIDIA GPU energy consumptions to the energy_consumptions dict.
                            energy_consumption_key = monitoring_tag + "_nvidia_gpu_{0}".format(timestamp)
                            energy_consumption_value = timestamp_consumptions["energy_nvidia_gpu"]
                        else:
                            # Add the NVIDIA GPU energy consumptions sums to the energy_consumptions dict.
                            energy_consumption_key = monitoring_tag + "_nvidia_gpu"
                            energy_consumption_value = sum([timestamp_consumptions["energy_nvidia_gpu"]
                                                            for _, timestamp_consumptions in
                                                            timestamps_consumptions.items()])
                        energy_consumptions.update({energy_consumption_key: energy_consumption_value})
        # Verify if the energy consumptions files are set to be removed.
        if remove_energy_consumptions_files:
            # If so, remove the energy consumption files.
            self._remove_energy_consumption_files()
        # Return the energy consumptions dictionary.
        return energy_consumptions
