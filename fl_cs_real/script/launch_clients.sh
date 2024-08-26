#!/bin/bash
# Script begin.

# Get the number of provided arguments.
number_of_provided_arguments=$#

# Set the required arguments array.
required_arguments_array=("Number of Clients to Launch [Integer]"
                          "Config File [Path]"
                          "Implementation [String]")
number_of_required_arguments=${#required_arguments_array[@]}

# Set the optional arguments array.
optional_arguments_array=()
number_of_optional_arguments=${#optional_arguments_array[@]}

# Parse the provided arguments.
if [ $number_of_provided_arguments -lt $number_of_required_arguments ]; then
    if [ $number_of_required_arguments -gt 0 ]; then
        echo -e "Required Arguments ($number_of_required_arguments):"
        for i in $(seq 0 $(($number_of_required_arguments-1))); do
            echo "$(($i+1))) ${required_arguments_array[$i]}"
        done
    fi
    if [ $number_of_optional_arguments -gt 0 ]; then
        echo -e "\nOptional Arguments ($number_of_optional_arguments):"
        for i in $(seq 0 $(($number_of_optional_arguments-1))); do
            echo "$(($i+$number_of_required_arguments+1))) ${optional_arguments_array[$i]}"
        done
    fi
    exit 1
fi

# Script arguments.
num_clients=${1}
config_file=${2}
implementation=${3}

# Launch the clients (background processes).
for ((client_id = 0; client_id < $num_clients; client_id++)); do
    python3 main.py launch_client --id $client_id --config-file $config_file --implementation $implementation &
    sleep 1
done

# Print the number of clients launched.
if [ $num_clients -eq 0 ] || [ $num_clients -gt 1 ]; then
    echo "Launched $num_clients Clients!"
else
    echo "Launched $num_clients Client!"
fi

# Wait for all background processes to finish.
wait

# Script end.
exit 0
