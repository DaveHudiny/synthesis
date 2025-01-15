print_help() {
    echo "Usage: runner_interpretation.sh [options] <input_folder> <log_folder>"
    echo "Options:"
    echo "  --broad-interpretation <input_folder> <log_folder>   Run broad interpretation"
}

broad_interpretation_body() {
    model=$1
    fsc_size=$2
    rnn_le=$3
    input_folder=$4
    rnn_less_log_folder=$5
    rnn_log_folder=$6

    echo "Model: $model"
    echo "FSC Size: $fsc_size"
    echo "RNN Less: $rnn_le"
    entry=$input_folder/$model
    log_folder_path=$rnn_less_log_folder
    log_file_suffix=""

    if [ $rnn_le != "True" ]; then
        log_folder_path=$rnn_log_folder
        log_file_suffix="-$entry"
    fi

    log_file=$log_folder_path/$model-fsc_size_$fsc_size.log

    # Check, if there is a log file already. If so, create a new one
    if [ -f $log_file ]; then
        log_file=$log_folder_path/$model-fsc_size_$fsc_size-$(date +%Y%m%d%H%M%S).log
    fi

    error_log_file=$log_file.error
    echo $log_file

    rnn_less_flag=""
    if [ $rnn_le == "True" ]; then
        rnn_less_flag="--rnn-less"
    fi

    python3 paynt.py --fsc-synthesis --storm-pomdp --reinforcement-learning --fsc-size $fsc_size $rnn_less_flag $entry > $log_file 2> $error_log_file
}

broad_interpretation() {
    trap "kill 0; exit" SIGINT SIGTERM
    input_folder=$1
    log_folder=$2
    rnn_less_log_folder=$log_folder/rnn_less
    rnn_log_folder=$log_folder/rnn
    mkdir -p $rnn_less_log_folder
    mkdir -p $rnn_log_folder
    fsc_sizes=(1)
    rnn_less=("False")
    for fsc_size in ${fsc_sizes[@]}; do
        for rnn_le in ${rnn_less[@]}; do
            for model in $(ls $1); do
                # if drone or evade in model name, skip them 
                if [[ $model == *"drone"* ]] || [[ $model == *"evade"* ]]; then
                    continue
                fi
                # if fsc_size is 1, skip also models geo, intercept, maze, network-3-8-20 and network-5-10-8
                if [ $fsc_size == 1 ]; then
                    if [[ $model == *"geo"* ]] || [[ $model == *"intercept"* ]] || [[ $model == *"maze"* ]] || [[ $model == *"network-3-8-20"* ]] || [[ $model == *"network-5-10-8"* ]]; then
                        continue
                    fi
                fi
                broad_interpretation_body $model $fsc_size $rnn_le $input_folder $rnn_less_log_folder $rnn_log_folder
            done
        done
    done

    wait
}

if [ $1 == "--help" ]; then
    print_help
    exit 1
fi

if [ $# -lt 2 ]; then
    echo "Error: Not enough arguments"
    print_help
    exit 1
fi

# runner_interpretation.sh --broad-interpretation <input_folder> <log_folder>
if [ $1 == "--broad-interpretation" ]; then
    input_folder=$2
    log_folder=$3
    source prerequisites/venv/bin/activate
    broad_interpretation $input_folder $log_folder
    deactivate
    exit 0
fi

exit 0
