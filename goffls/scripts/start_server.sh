# Script Begin.

# Get Number of Provided Arguments.
number_of_provided_arguments=$#

# Set Required Arguments Array.
required_arguments_array=("Config File [Path]"
                          "Implementation [String]")
number_of_required_arguments=${#required_arguments_array[@]}

# Set Optional Arguments Array.
optional_arguments_array=()
number_of_optional_arguments=${#optional_arguments_array[@]}

# Parse Provided Arguments.
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

# Script Arguments.
config_file=${1}
implementation=${2}

# Set Server's ID.
server_id="0"

# Launch the Server (Background Process).
python3 main.py launch_server --id $server_id --config-file $config_file --implementation $implementation &

# Print the Server Launched Notice.
echo "Launched the Server!"

# Wait for All Background Processes to Finish.
wait

# Script End.
exit 0
