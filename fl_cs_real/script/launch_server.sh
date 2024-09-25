#!/bin/bash

# Get the number of provided arguments.
number_of_provided_arguments=$#

# Set the required arguments array.
required_arguments_array=("Server ID [Integer]"
                          "Server Config File [Path]"
                          "Server Implementation [String]")
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
server_id=${1}
server_config_file=${2}
server_implementation=${3}

# Get the script file.
script_file="$(basename "$(test -L "$0" && readlink "$0" || echo "$0")")"

# Script started message.
echo "$(date +%F_%T) $script_file INFO: The '$script_file' script has started!"

# Get the start time.
start_time=$(date +%s)

# Launch the server (background process).
python3 main.py launch_server --id "$server_id" --config-file "$server_config_file" --implementation "$server_implementation" &

# Print the server launched notice.
echo "$(date +%F_%T) $script_file INFO: Launched the Server!"

# Get the end time.
end_time=$(date +%s)

# Script ended message.
echo "$(date +%F_%T) $script_file INFO: The '$script_file' script has ended successfully!"
echo "$(date +%F_%T) $script_file INFO: Elapsed time: $((end_time - start_time)) seconds."

# Wait for the completion of the server execution.
wait

# Exit.
exit 0
