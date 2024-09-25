#!/bin/bash

# Get the number of provided arguments.
number_of_provided_arguments=$#

# Set the required arguments array.
required_arguments_array=("Number of Clients to Launch [Integer]"
                          "Client Config File [Path]"
                          "Client Implementation [String]"
                          "Client Dataset Root Folder [Path]")
number_of_required_arguments=${#required_arguments_array[@]}

# Set the optional arguments array.
optional_arguments_array=()
number_of_optional_arguments=${#optional_arguments_array[@]}

# Parse the provided arguments.
if [ $number_of_provided_arguments -lt "$number_of_required_arguments" ]; then
  if [ "$number_of_required_arguments" -gt 0 ]; then
    echo -e "Required Arguments ($number_of_required_arguments):"
    for i in $(seq 0 $(("$number_of_required_arguments"-1))); do
      echo "$((i+1))) ${required_arguments_array[$i]}"
    done
  fi
  if [ "$number_of_optional_arguments" -gt 0 ]; then
    echo -e "\nOptional Arguments ($number_of_optional_arguments):"
    for i in $(seq 0 $(("$number_of_optional_arguments"-1))); do
      echo "$((i+number_of_required_arguments+1))) ${optional_arguments_array[$i]}"
    done
  fi
  exit 1
fi

# Script arguments.
num_clients=${1}
client_config_file=${2}
client_implementation=${3}
client_dataset_root_folder=${4}

# Get the script file.
script_file="$(basename "$(test -L "$0" && readlink "$0" || echo "$0")")"

# Script started message.
echo "$(date +%F_%T) $script_file INFO: The '$script_file' script has started!"

# Get the start time.
start_time=$(date +%s)

# Launch the clients (background processes).
for ((client_idx = 0; client_idx < num_clients; client_idx++)); do

  # Set the dataset settings for the current client.
  dataset_root_folder=$client_dataset_root_folder'/partition_'$client_idx

  # Copy the base client config file for the current client.
  client_idx_config_file=${client_config_file/".cfg"/"_$client_idx.cfg"}
  cp "$client_config_file" "$client_idx_config_file"

  # Update the dataset root folder value in the current client's temporary config file.
  sed -i -e "s|dataset_root_folder =.*$|dataset_root_folder = $dataset_root_folder|g" "$client_idx_config_file"

  # Launch the current client using its particular config file.
  python3 main.py launch_client --id "$client_idx" --config-file "$client_idx_config_file" --implementation "$client_implementation" &

  # Wait briefly.
  sleep 1

done

# Print the number of clients launched.
if [ "$num_clients" -eq 0 ] || [ "$num_clients" -gt 1 ]; then
  echo "$(date +%F_%T) $script_file INFO: Launched $num_clients Clients!"
else
  echo "$(date +%F_%T) $script_file INFO: Launched $num_clients Client!"
fi

# Get the end time.
end_time=$(date +%s)

# Script ended message.
echo "$(date +%F_%T) $script_file INFO: The '$script_file' script has ended successfully!"
echo "$(date +%F_%T) $script_file INFO: Elapsed time: $((end_time - start_time)) seconds."

# Wait for the completion of the clients execution.
wait

# Remove the clients config temporary files.
for ((client_idx = 0; client_idx < num_clients; client_idx++)); do
  client_idx_config_file=${client_config_file/".cfg"/"_$client_idx.cfg"}
  rm -f "$client_idx_config_file"
  rm -f "$client_idx_config_file""-e"
done

# Exit.
exit 0
