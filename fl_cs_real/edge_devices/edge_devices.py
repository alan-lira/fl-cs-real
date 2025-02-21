from pathlib import Path


def calculate_battery_stored_energy(battery_voltage_in_volts: float,
                                    battery_capacity_in_milliampere_hours: float,
                                    battery_stored_energy_unit: str) -> float:
    battery_stored_energy = 0
    match battery_stored_energy_unit:
        case u if u in ["watt-hours", "Wh"]:
            battery_stored_energy = (battery_voltage_in_volts * battery_capacity_in_milliampere_hours) / 1000
        case u if u in ["joules", "J"]:
            battery_stored_energy = (battery_voltage_in_volts * battery_capacity_in_milliampere_hours * 3600) / 1000
    return battery_stored_energy


def calculate_battery_duration(battery_stored_energy_in_joules: float,
                               average_power_consumption_in_watts: float,
                               battery_duration_unit: str) -> float:
    battery_duration = 0
    match battery_duration_unit:
        case u if u in ["seconds", "s"]:
            battery_duration = battery_stored_energy_in_joules / average_power_consumption_in_watts
    return battery_duration


def emulate_raspberry_pi_performance(device_name: str) -> dict:
    # Raspberry Pi (Broadcom BCM2837/BCM2711)
    # CPU Model: Broadcom BCM2837 (Raspberry Pi 3) / BCM2711 (Raspberry Pi 4)
    # Number of Cores: 4 cores (Quad-core ARM Cortex-A53 (Pi 3) / Cortex-A72 (Pi 4))
    # Frequency: 1.2 GHz (Pi 3) / 1.5 GHz (Pi 4)
    # Energy Consumption: ~2.5W (Pi 3) / 3.5W (Pi 4) under moderate load
    # Battery Specs (Li-ion battery):
    #   Capacity: 5,000 mAh
    #   Voltage: 5V
    num_cpu_cores = 0
    cpu_frequency_in_gigahertz = 0
    average_power_consumption_in_watts = 0
    battery_capacity_in_milliampere_hours = 0
    battery_voltage_in_volts = 0
    match device_name:
        case "Raspberry Pi 3":
            num_cpu_cores = 4
            cpu_frequency_in_gigahertz = 1.2
            average_power_consumption_in_watts = 2.5
            battery_capacity_in_milliampere_hours = 5000
            battery_voltage_in_volts = 5
        case "Raspberry Pi 4":
            num_cpu_cores = 4
            cpu_frequency_in_gigahertz = 1.5
            average_power_consumption_in_watts = 3.5
            battery_capacity_in_milliampere_hours = 5000
            battery_voltage_in_volts = 5
    battery_stored_energy_in_joules = calculate_battery_stored_energy(battery_voltage_in_volts,
                                                                      battery_capacity_in_milliampere_hours,
                                                                      "joules")
    battery_duration_in_seconds = calculate_battery_duration(battery_stored_energy_in_joules,
                                                             average_power_consumption_in_watts,
                                                             "seconds")
    performance_dict = {"device_name": device_name,
                        "num_cpu_cores": num_cpu_cores,
                        "cpu_frequency_in_gigahertz": cpu_frequency_in_gigahertz,
                        "average_power_consumption_in_watts": average_power_consumption_in_watts,
                        "battery_capacity_in_milliampere_hours": battery_capacity_in_milliampere_hours,
                        "battery_voltage_in_volts": battery_voltage_in_volts,
                        "battery_stored_energy_in_joules": battery_stored_energy_in_joules,
                        "battery_duration_in_seconds": battery_duration_in_seconds}
    return performance_dict


def emulate_nvidia_jetson_performance(device_name: str) -> dict:
    # NVIDIA Jetson (TX1, TX2, Nano, Xavier)
    # CPU Model: ARM Cortex-A57 (TX1, TX2, Nano), ARM Cortex-A72 (Xavier)
    # Number of Cores: 2 cores (Nano) 4 cores (TX1/TX2) or 6 cores (Xavier)
    # Frequency: 1.2 GHz (TX1), 2.0 GHz (TX2), 1.43 GHz (Nano), 2.2 GHz (Xavier)
    # Energy Consumption: ~10-15W (TX1/TX2/Nano) / ~30W (Xavier)
    # Battery Specs (Li-ion battery):
    #   Capacity: 20,000 mAh (TX1/TX2/Xavier) and 5,000 mAh (Nano)
    #   Voltage: 12V (TX1/TX2/Xavier) and 3.7V (Nano)
    num_cpu_cores = 0
    cpu_frequency_in_gigahertz = 0
    average_power_consumption_in_watts = 0
    battery_capacity_in_milliampere_hours = 0
    battery_voltage_in_volts = 0
    match device_name:
        case model if model in ["NVIDIA Jetson TX1", "NVIDIA Jetson TX2"]:
            num_cpu_cores = 4
            if model == "NVIDIA Jetson TX1":
                cpu_frequency_in_gigahertz = 1.2
                average_power_consumption_in_watts = 10
            elif model == "NVIDIA Jetson TX2":
                cpu_frequency_in_gigahertz = 2.0
                average_power_consumption_in_watts = 15
            battery_capacity_in_milliampere_hours = 20000
            battery_voltage_in_volts = 12
        case model if model in ["NVIDIA Jetson Nano"]:
            num_cpu_cores = 2
            cpu_frequency_in_gigahertz = 1.43
            average_power_consumption_in_watts = 10
            battery_capacity_in_milliampere_hours = 5000
            battery_voltage_in_volts = 3.7
        case model if model in ["NVIDIA Jetson Xavier"]:
            num_cpu_cores = 6
            cpu_frequency_in_gigahertz = 2.2
            average_power_consumption_in_watts = 30
            battery_capacity_in_milliampere_hours = 20000
            battery_voltage_in_volts = 12
    battery_stored_energy_in_joules = calculate_battery_stored_energy(battery_voltage_in_volts,
                                                                      battery_capacity_in_milliampere_hours,
                                                                      "joules")
    battery_duration_in_seconds = calculate_battery_duration(battery_stored_energy_in_joules,
                                                             average_power_consumption_in_watts,
                                                             "seconds")
    performance_dict = {"device_name": device_name,
                        "num_cpu_cores": num_cpu_cores,
                        "cpu_frequency_in_gigahertz": cpu_frequency_in_gigahertz,
                        "average_power_consumption_in_watts": average_power_consumption_in_watts,
                        "battery_capacity_in_milliampere_hours": battery_capacity_in_milliampere_hours,
                        "battery_voltage_in_volts": battery_voltage_in_volts,
                        "battery_stored_energy_in_joules": battery_stored_energy_in_joules,
                        "battery_duration_in_seconds": battery_duration_in_seconds}
    return performance_dict


def emulate_intel_atom_performance(device_name: str) -> dict:
    # Intel Atom (e.g., x5-Z8350, x7-E3950)
    # CPU Model: Intel Atom x5-Z8350, x7-E3950
    # Number of Cores: 2 cores (x5-Z8350) and 4 cores (x5-Z8350/x7-E3950)
    # Frequency: 1.0 GHz (x5-Z8350 2 Cores) 1.44 GHz (x5-Z8350 4 Cores), 2.00 GHz (x7-E3950)
    # Energy Consumption: ~3W (x5-Z8350 2 Cores), ~4-6W (x5-Z8350 4 Cores), ~10W (x7-E3950)
    # Battery Specs (Li-ion battery):
    #   Capacity: 2,500 mAh (x5-Z8350 2 Cores) 5,000 mAh (x5-Z8350 4 Cores/x7-E3950)
    #   Voltage: 3.7V (x5-Z8350 2 Cores), 5V (x5-Z8350 4 Cores/x7-E3950)
    num_cpu_cores = 0
    cpu_frequency_in_gigahertz = 0
    average_power_consumption_in_watts = 0
    battery_capacity_in_milliampere_hours = 0
    battery_voltage_in_volts = 0
    match device_name:
        case "Intel Atom x5-Z8350 2 Cores":
            num_cpu_cores = 2
            cpu_frequency_in_gigahertz = 1.0
            average_power_consumption_in_watts = 3
            battery_capacity_in_milliampere_hours = 2500
            battery_voltage_in_volts = 3.7
        case "Intel Atom x5-Z8350 4 Cores":
            num_cpu_cores = 4
            cpu_frequency_in_gigahertz = 1.44
            average_power_consumption_in_watts = 6
            battery_capacity_in_milliampere_hours = 5000
            battery_voltage_in_volts = 5
        case "Intel Atom x7-E3950":
            num_cpu_cores = 4
            cpu_frequency_in_gigahertz = 2.0
            average_power_consumption_in_watts = 10
            battery_capacity_in_milliampere_hours = 5000
            battery_voltage_in_volts = 5
    battery_stored_energy_in_joules = calculate_battery_stored_energy(battery_voltage_in_volts,
                                                                      battery_capacity_in_milliampere_hours,
                                                                      "joules")
    battery_duration_in_seconds = calculate_battery_duration(battery_stored_energy_in_joules,
                                                             average_power_consumption_in_watts,
                                                             "seconds")
    performance_dict = {"device_name": device_name,
                        "num_cpu_cores": num_cpu_cores,
                        "cpu_frequency_in_gigahertz": cpu_frequency_in_gigahertz,
                        "average_power_consumption_in_watts": average_power_consumption_in_watts,
                        "battery_capacity_in_milliampere_hours": battery_capacity_in_milliampere_hours,
                        "battery_voltage_in_volts": battery_voltage_in_volts,
                        "battery_stored_energy_in_joules": battery_stored_energy_in_joules,
                        "battery_duration_in_seconds": battery_duration_in_seconds}
    return performance_dict


def emulate_qualcomm_snapdragon_performance(device_name: str) -> dict:
    # Qualcomm Snapdragon (e.g., Snapdragon 410E, 820E)
    # CPU Model: Snapdragon 410E (Quad-core ARM Cortex-A53), Snapdragon 820E (Quad-core Kryo)
    # Number of Cores: 4 cores (both models)
    # Frequency: 1.2 GHz (410E) / 2.2 GHz (820E)
    # Energy Consumption: ~2-4W (410E) / ~5-7W (820E)
    # Battery Specs (Li-ion battery):
    #   Capacity: 6,000 mAh
    #   Voltage: 3.7V
    num_cpu_cores = 0
    cpu_frequency_in_gigahertz = 0
    average_power_consumption_in_watts = 0
    battery_capacity_in_milliampere_hours = 0
    battery_voltage_in_volts = 0
    match device_name:
        case "Qualcomm Snapdragon 410E":
            num_cpu_cores = 4
            cpu_frequency_in_gigahertz = 1.2
            average_power_consumption_in_watts = 4
            battery_capacity_in_milliampere_hours = 6000
            battery_voltage_in_volts = 3.7
        case "Qualcomm Snapdragon 820E":
            num_cpu_cores = 4
            cpu_frequency_in_gigahertz = 2.2
            average_power_consumption_in_watts = 7
            battery_capacity_in_milliampere_hours = 6000
            battery_voltage_in_volts = 3.7
    battery_stored_energy_in_joules = calculate_battery_stored_energy(battery_voltage_in_volts,
                                                                      battery_capacity_in_milliampere_hours,
                                                                      "joules")
    battery_duration_in_seconds = calculate_battery_duration(battery_stored_energy_in_joules,
                                                             average_power_consumption_in_watts,
                                                             "seconds")
    performance_dict = {"device_name": device_name,
                        "num_cpu_cores": num_cpu_cores,
                        "cpu_frequency_in_gigahertz": cpu_frequency_in_gigahertz,
                        "average_power_consumption_in_watts": average_power_consumption_in_watts,
                        "battery_capacity_in_milliampere_hours": battery_capacity_in_milliampere_hours,
                        "battery_voltage_in_volts": battery_voltage_in_volts,
                        "battery_stored_energy_in_joules": battery_stored_energy_in_joules,
                        "battery_duration_in_seconds": battery_duration_in_seconds}
    return performance_dict


def emulate_google_coral_edge_tpu_performance(device_name: str) -> dict:
    # Google Coral Edge TPU
    # CPU Model: ARM Cortex-A53 for the host processor (used alongside a dedicated TPU for machine learning tasks)
    # Number of Cores: 4 cores (Cortex-A53)
    # Frequency: 1.2 GHz
    # Energy Consumption: ~2-4W (host CPU) + ~1-2W for Edge TPU (additional to CPU consumption during ML tasks)
    # Battery Specs (Li-ion battery):
    #   Capacity: 10,000 mAh
    #   Voltage: 3.7V
    num_cpu_cores = 0
    cpu_frequency_in_gigahertz = 0
    average_power_consumption_in_watts = 0
    battery_capacity_in_milliampere_hours = 0
    battery_voltage_in_volts = 0
    match device_name:
        case "Google Coral Edge TPU Cortex-A53":
            num_cpu_cores = 4
            cpu_frequency_in_gigahertz = 1.2
            average_power_consumption_in_watts = 4 + 2
            battery_capacity_in_milliampere_hours = 10000
            battery_voltage_in_volts = 3.7
    battery_stored_energy_in_joules = calculate_battery_stored_energy(battery_voltage_in_volts,
                                                                      battery_capacity_in_milliampere_hours,
                                                                      "joules")
    battery_duration_in_seconds = calculate_battery_duration(battery_stored_energy_in_joules,
                                                             average_power_consumption_in_watts,
                                                             "seconds")
    performance_dict = {"device_name": device_name,
                        "num_cpu_cores": num_cpu_cores,
                        "cpu_frequency_in_gigahertz": cpu_frequency_in_gigahertz,
                        "average_power_consumption_in_watts": average_power_consumption_in_watts,
                        "battery_capacity_in_milliampere_hours": battery_capacity_in_milliampere_hours,
                        "battery_voltage_in_volts": battery_voltage_in_volts,
                        "battery_stored_energy_in_joules": battery_stored_energy_in_joules,
                        "battery_duration_in_seconds": battery_duration_in_seconds}
    return performance_dict


def emulate_beaglebone_black_performance(device_name: str) -> dict:
    # BeagleBone Black
    # CPU Model: Texas Instruments AM335x (ARM Cortex-A8)
    # Number of Cores: 1 core
    # Frequency: 1.0 GHz
    # Energy Consumption: ~2W under typical use (can be lower with energy-saving modes)
    # Battery Specs (Li-ion battery):
    #   Capacity: 5,000 mAh
    #   Voltage: 3.7V
    num_cpu_cores = 0
    cpu_frequency_in_gigahertz = 0
    average_power_consumption_in_watts = 0
    battery_capacity_in_milliampere_hours = 0
    battery_voltage_in_volts = 0
    match device_name:
        case "BeagleBone Black Cortex-A8":
            num_cpu_cores = 1
            cpu_frequency_in_gigahertz = 1.0
            average_power_consumption_in_watts = 2
            battery_capacity_in_milliampere_hours = 3000
            battery_voltage_in_volts = 3.7
    battery_stored_energy_in_joules = calculate_battery_stored_energy(battery_voltage_in_volts,
                                                                      battery_capacity_in_milliampere_hours,
                                                                      "joules")
    battery_duration_in_seconds = calculate_battery_duration(battery_stored_energy_in_joules,
                                                             average_power_consumption_in_watts,
                                                             "seconds")
    performance_dict = {"device_name": device_name,
                        "num_cpu_cores": num_cpu_cores,
                        "cpu_frequency_in_gigahertz": cpu_frequency_in_gigahertz,
                        "average_power_consumption_in_watts": average_power_consumption_in_watts,
                        "battery_capacity_in_milliampere_hours": battery_capacity_in_milliampere_hours,
                        "battery_voltage_in_volts": battery_voltage_in_volts,
                        "battery_stored_energy_in_joules": battery_stored_energy_in_joules,
                        "battery_duration_in_seconds": battery_duration_in_seconds}
    return performance_dict


def emulate_intel_celeron_performance(device_name: str) -> dict:
    # Embedded edge devices with 2-core processors
    # CPU Model: Intel Celeron N3350
    # Number of Cores: 2
    # Frequency: 800 MHz
    # Energy Consumption: ~4W
    # Battery Specs (Li-ion battery):
    #   Capacity: 3,000 mAh
    #   Voltage: 5V
    num_cpu_cores = 0
    cpu_frequency_in_gigahertz = 0
    average_power_consumption_in_watts = 0
    battery_capacity_in_milliampere_hours = 0
    battery_voltage_in_volts = 0
    match device_name:
        case "Intel Celeron N3350":
            num_cpu_cores = 2
            cpu_frequency_in_gigahertz = 0.8
            average_power_consumption_in_watts = 4
            battery_capacity_in_milliampere_hours = 3000
            battery_voltage_in_volts = 5
    battery_stored_energy_in_joules = calculate_battery_stored_energy(battery_voltage_in_volts,
                                                                      battery_capacity_in_milliampere_hours,
                                                                      "joules")
    battery_duration_in_seconds = calculate_battery_duration(battery_stored_energy_in_joules,
                                                             average_power_consumption_in_watts,
                                                             "seconds")
    performance_dict = {"device_name": device_name,
                        "num_cpu_cores": num_cpu_cores,
                        "cpu_frequency_in_gigahertz": cpu_frequency_in_gigahertz,
                        "average_power_consumption_in_watts": average_power_consumption_in_watts,
                        "battery_capacity_in_milliampere_hours": battery_capacity_in_milliampere_hours,
                        "battery_voltage_in_volts": battery_voltage_in_volts,
                        "battery_stored_energy_in_joules": battery_stored_energy_in_joules,
                        "battery_duration_in_seconds": battery_duration_in_seconds}
    return performance_dict


def emulate_intel_core_i_series_performance(device_name: str) -> dict:
    # Intel Core i3/i5/i7 (Used in higher-end edge devices)
    # CPU Model: Intel Core i3/i5/i7 (10th generation)
    # Number of Cores: 4 (i3), 6 (i5), and 8 cores (i7)
    # Frequency: 3.6 GHz (i3), 2.4 GHz (i5), 3.1 GHz (i7)
    # Energy Consumption: ~15W (i3), ~25W (i5), ~45W (i7)
    # Battery Specs (Li-ion battery):
    #   Capacity: 50,000 mAh
    #   Voltage: 11.1V
    num_cpu_cores = 0
    cpu_frequency_in_gigahertz = 0
    average_power_consumption_in_watts = 0
    battery_capacity_in_milliampere_hours = 0
    battery_voltage_in_volts = 0
    match device_name:
        case "Intel Core i3":
            num_cpu_cores = 4
            cpu_frequency_in_gigahertz = 3.6
            average_power_consumption_in_watts = 15
            battery_capacity_in_milliampere_hours = 50000
            battery_voltage_in_volts = 11.1
        case "Intel Core i5":
            num_cpu_cores = 6
            cpu_frequency_in_gigahertz = 2.4
            average_power_consumption_in_watts = 25
            battery_capacity_in_milliampere_hours = 50000
            battery_voltage_in_volts = 11.1
        case "Intel Core i7":
            num_cpu_cores = 8
            cpu_frequency_in_gigahertz = 3.1
            average_power_consumption_in_watts = 45
            battery_capacity_in_milliampere_hours = 50000
            battery_voltage_in_volts = 11.1
    battery_stored_energy_in_joules = calculate_battery_stored_energy(battery_voltage_in_volts,
                                                                      battery_capacity_in_milliampere_hours,
                                                                      "joules")
    battery_duration_in_seconds = calculate_battery_duration(battery_stored_energy_in_joules,
                                                             average_power_consumption_in_watts,
                                                             "seconds")
    performance_dict = {"device_name": device_name,
                        "num_cpu_cores": num_cpu_cores,
                        "cpu_frequency_in_gigahertz": cpu_frequency_in_gigahertz,
                        "average_power_consumption_in_watts": average_power_consumption_in_watts,
                        "battery_capacity_in_milliampere_hours": battery_capacity_in_milliampere_hours,
                        "battery_voltage_in_volts": battery_voltage_in_volts,
                        "battery_stored_energy_in_joules": battery_stored_energy_in_joules,
                        "battery_duration_in_seconds": battery_duration_in_seconds}
    return performance_dict


def emulate_arm_cortex_m_series_performance(device_name: str) -> dict:
    # 8. ARM Cortex-M Series (Low-power, for simple edge devices)
    # CPU Model: ARM Cortex-M0, M3, M4, M7
    # Number of Cores: 1 core
    # Frequency: Up to 50 MHz-100 MHz (M0), Up to 100 MHz-200 MHz (M3), Up to 150 MHz-200 MHz (M4), Up to 400 MHz (M7)
    # Energy Consumption: ~0.00005 W to 0.0001 W (M0), ~0.0001 W to 0.0002 W (M3), ~0.0001 W to 0.0002 W (M4), ~0.0002 W to 0.0004 W (M7)
    # Battery Specs (Li-Po battery):
    #   Capacity: 100 mAh (M0), 200 mAh (M3), 300 mAh (M4), 500 mAh (M7)
    #   Voltage: 1.8V (M0), 2.0V (M3), 1.8V (M4), 1.8V (M7)
    num_cpu_cores = 0
    cpu_frequency_in_gigahertz = 0
    average_power_consumption_in_watts = 0
    battery_capacity_in_milliampere_hours = 0
    battery_voltage_in_volts = 0
    match device_name:
        case "ARM Cortex-M0":
            num_cpu_cores = 1
            cpu_frequency_in_gigahertz = 0.1
            average_power_consumption_in_watts = 0.0001
            battery_capacity_in_milliampere_hours = 100
            battery_voltage_in_volts = 1.8
        case "ARM Cortex-M3":
            num_cpu_cores = 1
            cpu_frequency_in_gigahertz = 0.2
            average_power_consumption_in_watts = 0.0002
            battery_capacity_in_milliampere_hours = 200
            battery_voltage_in_volts = 2.0
        case "ARM Cortex-M4":
            num_cpu_cores = 1
            cpu_frequency_in_gigahertz = 0.2
            average_power_consumption_in_watts = 0.0002
            battery_capacity_in_milliampere_hours = 300
            battery_voltage_in_volts = 1.8
        case "ARM Cortex-M7":
            num_cpu_cores = 1
            cpu_frequency_in_gigahertz = 0.4
            average_power_consumption_in_watts = 0.0004
            battery_capacity_in_milliampere_hours = 500
            battery_voltage_in_volts = 1.8
    battery_stored_energy_in_joules = calculate_battery_stored_energy(battery_voltage_in_volts,
                                                                      battery_capacity_in_milliampere_hours,
                                                                      "joules")
    battery_duration_in_seconds = calculate_battery_duration(battery_stored_energy_in_joules,
                                                             average_power_consumption_in_watts,
                                                             "seconds")
    performance_dict = {"device_name": device_name,
                        "num_cpu_cores": num_cpu_cores,
                        "cpu_frequency_in_gigahertz": cpu_frequency_in_gigahertz,
                        "average_power_consumption_in_watts": average_power_consumption_in_watts,
                        "battery_capacity_in_milliampere_hours": battery_capacity_in_milliampere_hours,
                        "battery_voltage_in_volts": battery_voltage_in_volts,
                        "battery_stored_energy_in_joules": battery_stored_energy_in_joules,
                        "battery_duration_in_seconds": battery_duration_in_seconds}
    return performance_dict


def emulate_microchip_sam_microcontrollers_performance(device_name: str) -> dict:
    # Microchip (formerly Atmel) SAM D and SAM E (Microcontrollers)
    # CPU Model: ARM Cortex-M0+ (SAM D), Cortex-M3 (SAM E)
    # Number of Cores: 1 core
    # Frequency: Up to 48 MHz (SAM D) / 120 MHz (SAM E)
    # Energy Consumption: ~0.1W (SAM D) / ~0.3W (SAM E)
    # Battery Specs (Li-ion battery):
    #   Capacity: 300 mAh
    #   Voltage: 3.7V
    num_cpu_cores = 0
    cpu_frequency_in_gigahertz = 0
    average_power_consumption_in_watts = 0
    battery_capacity_in_milliampere_hours = 0
    battery_voltage_in_volts = 0
    match device_name:
        case "Microchip SAM D":
            num_cpu_cores = 1
            cpu_frequency_in_gigahertz = 0.048
            average_power_consumption_in_watts = 0.1
            battery_capacity_in_milliampere_hours = 300
            battery_voltage_in_volts = 3.7
        case "Microchip SAM E":
            num_cpu_cores = 1
            cpu_frequency_in_gigahertz = 0.12
            average_power_consumption_in_watts = 0.3
            battery_capacity_in_milliampere_hours = 300
            battery_voltage_in_volts = 3.7
    battery_stored_energy_in_joules = calculate_battery_stored_energy(battery_voltage_in_volts,
                                                                      battery_capacity_in_milliampere_hours,
                                                                      "joules")
    battery_duration_in_seconds = calculate_battery_duration(battery_stored_energy_in_joules,
                                                             average_power_consumption_in_watts,
                                                             "seconds")
    performance_dict = {"device_name": device_name,
                        "num_cpu_cores": num_cpu_cores,
                        "cpu_frequency_in_gigahertz": cpu_frequency_in_gigahertz,
                        "average_power_consumption_in_watts": average_power_consumption_in_watts,
                        "battery_capacity_in_milliampere_hours": battery_capacity_in_milliampere_hours,
                        "battery_voltage_in_volts": battery_voltage_in_volts,
                        "battery_stored_energy_in_joules": battery_stored_energy_in_joules,
                        "battery_duration_in_seconds": battery_duration_in_seconds}
    return performance_dict


def generate_edge_devices() -> dict:
    edge_devices = {"raspberry_pi_3_device": emulate_raspberry_pi_performance("Raspberry Pi 3"),
                    "raspberry_pi_4_device": emulate_raspberry_pi_performance("Raspberry Pi 4"),
                    "nvidia_jetson_tx1_device": emulate_nvidia_jetson_performance("NVIDIA Jetson TX1"),
                    "nvidia_jetson_tx2_device": emulate_nvidia_jetson_performance("NVIDIA Jetson TX2"),
                    "nvidia_jetson_nano_device": emulate_nvidia_jetson_performance("NVIDIA Jetson Nano"),
                    "nvidia_jetson_xavier_device": emulate_nvidia_jetson_performance("NVIDIA Jetson Xavier"),
                    "intel_atom_x5_z8350_2c_device": emulate_intel_atom_performance("Intel Atom x5-Z8350 2 Cores"),
                    "intel_atom_x5_z8350_4c_device": emulate_intel_atom_performance("Intel Atom x5-Z8350 4 Cores"),
                    "intel_atom_x7_e3950_device": emulate_intel_atom_performance("Intel Atom x7-E3950"),
                    "qualcomm_snapdragon_410e_device": emulate_qualcomm_snapdragon_performance("Qualcomm Snapdragon 410E"),
                    "qualcomm_snapdragon_820e_device": emulate_qualcomm_snapdragon_performance("Qualcomm Snapdragon 820E"),
                    "google_coral_edge_tpu_cortex_a53_device": emulate_google_coral_edge_tpu_performance("Google Coral Edge TPU Cortex-A53"),
                    "beaglebone_black_cortex_a8_device": emulate_beaglebone_black_performance("BeagleBone Black Cortex-A8"),
                    "intel_celeron_n3350_device": emulate_intel_celeron_performance("Intel Celeron N3350"),
                    "intel_core_i3_device": emulate_intel_core_i_series_performance("Intel Core i3"),
                    "intel_core_i5_device": emulate_intel_core_i_series_performance("Intel Core i5"),
                    "intel_core_i7_device": emulate_intel_core_i_series_performance("Intel Core i7"),
                    "arm_cortex_m0_device": emulate_arm_cortex_m_series_performance("ARM Cortex-M0"),
                    "arm_cortex_m3_device": emulate_arm_cortex_m_series_performance("ARM Cortex-M3"),
                    "arm_cortex_m4_device": emulate_arm_cortex_m_series_performance("ARM Cortex-M4"),
                    "arm_cortex_m7_device": emulate_arm_cortex_m_series_performance("ARM Cortex-M7"),
                    "microchip_sam_d_microcontroller_device": emulate_microchip_sam_microcontrollers_performance("Microchip SAM D"),
                    "microchip_sam_e_microcontroller_device": emulate_microchip_sam_microcontrollers_performance("Microchip SAM E")}
    return edge_devices


def generate_edge_devices_configuration_file(output_file: Path) -> None:
    # Create the parents directories of the output file (if not exist yet).
    output_file.parent.mkdir(exist_ok=True, parents=True)
    # Emulate edge devices performance.
    edge_devices = generate_edge_devices()
    with open(file=output_file, mode="a", encoding="utf-8") as file:
        # Get the dictionary of edge devices settings from the first device.
        _, edge_device_settings_dict = next(iter(edge_devices.items()))
        # Set and write the header line.
        header_line = ",".join(list(edge_device_settings_dict.keys()))
        file.write(header_line + "\n")
        # Set and write the data lines.
        data_lines = []
        for _, edge_device_settings_dict in edge_devices.items():
            data_line = ",".join(str(value) for value in edge_device_settings_dict.values())
            data_lines.append(data_line)
        file.writelines("\n".join(str(data_line) for data_line in data_lines))
        file.write("\n")
