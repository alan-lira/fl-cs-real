from pyJoules.device import DeviceFactory
from pyJoules.device.nvidia_device import NvidiaGPUDomain
from pyJoules.device.rapl_device import RaplCoreDomain, RaplDramDomain, RaplPackageDomain, RaplUncoreDomain
from pyJoules.energy_meter import EnergyMeter
from pyJoules.exception import NoSuchDeviceError, NoSuchDomainError


class PyJoulesEnergyMonitor:

    def __init__(self,
                 monitoring_domains: list,
                 unit: str) -> None:
        # Initialize the attributes.
        self._monitoring_domains = monitoring_domains
        self._unit = unit
        self._energy_monitor = None
        # Load the energy monitor.
        self._load_energy_monitor()

    def _set_attribute(self,
                       attribute_name: str,
                       attribute_value: any) -> None:
        setattr(self, attribute_name, attribute_value)

    def get_attribute(self,
                      attribute_name: str) -> any:
        return getattr(self, attribute_name)

    @staticmethod
    def _get_available_monitoring_domains() -> list:
        available_monitoring_domains = ["CPU", "CPU_Cores", "Integrated_GPU", "NVIDIA_GPU", "RAM"]
        return available_monitoring_domains

    def _load_energy_monitor(self) -> any:
        # Get the necessary attributes.
        monitoring_domains = self.get_attribute("_monitoring_domains")
        # Instantiate the pyJoules monitor (EnergyMeter), if the hardware supports monitoring.
        any_monitorable_devices = len(DeviceFactory.create_devices()) > 0
        if any_monitorable_devices:
            devices_to_monitor = []
            # Iterate through the list of monitoring domains.
            for monitoring_domain in monitoring_domains:
                if monitoring_domain == "CPU":
                    try:
                        rapl_package_device = DeviceFactory.create_devices([RaplPackageDomain(0)])
                        devices_to_monitor.extend(rapl_package_device)
                    except (NoSuchDeviceError, NoSuchDomainError):
                        pass
                elif monitoring_domain == "CPU_Cores":
                    try:
                        rapl_core_device = DeviceFactory.create_devices([RaplCoreDomain(0)])
                        devices_to_monitor.extend(rapl_core_device)
                    except (NoSuchDeviceError, NoSuchDomainError):
                        pass
                elif monitoring_domain == "Integrated_GPU":
                    try:
                        rapl_uncore_device = DeviceFactory.create_devices([RaplUncoreDomain(0)])
                        devices_to_monitor.extend(rapl_uncore_device)
                    except (NoSuchDeviceError, NoSuchDomainError):
                        pass
                elif monitoring_domain == "NVIDIA_GPU":
                    try:
                        nvidia_gpu_device = DeviceFactory.create_devices([NvidiaGPUDomain(0)])
                        devices_to_monitor.extend(nvidia_gpu_device)
                    except (NoSuchDeviceError, NoSuchDomainError):
                        pass
                elif monitoring_domain == "RAM":
                    try:
                        rapl_dram_device = DeviceFactory.create_devices([RaplDramDomain(0)])
                        devices_to_monitor.extend(rapl_dram_device)
                    except (NoSuchDeviceError, NoSuchDomainError):
                        pass
            # Load the energy monitor.
            energy_monitor = EnergyMeter(devices_to_monitor)
            # Set the energy monitor.
            self._set_attribute("_energy_monitor", energy_monitor)

    def start(self,
              tag: str) -> None:
        # Get the necessary attributes.
        energy_monitor = self.get_attribute("_energy_monitor")
        if energy_monitor:
            # Start the energy consumption monitoring.
            energy_monitor.start(tag=tag)

    def stop(self) -> None:
        # Get the necessary attributes.
        energy_monitor = self.get_attribute("_energy_monitor")
        if energy_monitor:
            # Stop the energy consumption monitoring.
            energy_monitor.stop()

    def get_energy_consumptions(self,
                                tag: str) -> dict:
        # Initialize the energy consumptions dictionary.
        energy_consumptions = {}
        # Get the necessary attributes.
        energy_monitor = self.get_attribute("_energy_monitor")
        unit = self.get_attribute("_unit")
        if energy_monitor:
            last_trace = vars(energy_monitor.get_trace()[0])
            if last_trace["tag"] == tag:
                energy_dict = last_trace["energy"]
                if "package_0" in energy_dict:
                    # Get the CPU energy consumption, returned as Micro-Joules (μJ).
                    energy_cpu = energy_dict["package_0"]
                    if unit == "Joules":
                        # Convert the CPU energy consumption to Joules (J).
                        energy_cpu /= (1 * pow(10, 6))
                    # Add the CPU energy consumption to the energy consumptions dictionary.
                    energy_consumptions.update({tag + "_cpu": energy_cpu})
                if "core_0" in energy_dict:
                    # Get the CPU Cores energy consumption, returned as Micro-Joules (μJ).
                    energy_cpu_cores = energy_dict["core_0"]
                    if unit == "Joules":
                        # Convert the CPU Cores energy consumption to Joules (J).
                        energy_cpu_cores /= (1 * pow(10, 6))
                    # Add the CPU Cores energy consumption to the energy consumptions dictionary.
                    energy_consumptions.update({tag + "_cpu_cores": energy_cpu_cores})
                if "uncore_0" in energy_dict:
                    # Get the Integrated GPU energy consumption, returned as Micro-Joules (μJ).
                    energy_integrated_gpu = energy_dict["uncore_0"]
                    if unit == "Joules":
                        # Convert the Integrated GPU energy consumption to Joules (J).
                        energy_integrated_gpu /= (1 * pow(10, 6))
                    # Add the Integrated GPU energy consumption to the energy consumptions dictionary.
                    energy_consumptions.update({tag + "_integrated_gpu": energy_integrated_gpu})
                if "nvidia_gpu_0" in energy_dict:
                    # Get the NVIDIA GPU energy consumption, returned as Milli-Joules (mJ).
                    energy_nvidia_gpu = energy_dict["nvidia_gpu_0"]
                    if unit == "Joules":
                        # Convert the NVIDIA GPU energy consumption to Joules (J).
                        energy_nvidia_gpu /= (1 * pow(10, 3))
                    # Add the NVIDIA GPU energy consumption to the energy consumptions dictionary.
                    energy_consumptions.update({tag + "_nvidia_gpu": energy_nvidia_gpu})
                if "dram_0" in energy_dict:
                    # Get the RAM energy consumption, returned as Micro-Joules (μJ).
                    energy_ram = energy_dict["dram_0"]
                    if unit == "Joules":
                        # Convert the RAM energy consumption to Joules (J).
                        energy_ram /= (1 * pow(10, 6))
                    # Add the RAM energy consumption to the energy consumptions dictionary.
                    energy_consumptions.update({tag + "_ram": energy_ram})
        # Return the energy consumptions dictionary.
        return energy_consumptions
