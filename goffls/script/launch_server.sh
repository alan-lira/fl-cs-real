#!/bin/bash
# Script begin.

# Get the number of provided arguments.
number_of_provided_arguments=$#

# Set the required arguments array.
required_arguments_array=("Config File [Path]"
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
config_file=${1}
implementation=${2}

# Set the server's id.
server_id="0"

# Launch the server (background processes).
python3 main.py launch_server --id $server_id --config-file $config_file --implementation $implementation &

# Print the server launched notice.
echo "Launched the Server!"

# Wait for all background processes to finish.
wait

# Script end.
exit 0
