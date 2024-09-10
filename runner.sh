run_fsc_synthesis() {
    source prerequisites/venv/bin/activate
    echo "Running Paynt with --fsc-synthesis"
    for entry in `ls $1`; do
        if [ -d $1/$entry ]; then
            echo "Running Paynt on $entry"
            timeout $2 python3 paynt.py --fsc-synthesis --project $1/$entry > $1/$entry/paynt-fsc-synthesis.log
        fi
    done
}

run_saynt() {
    source prerequisites/venv/bin/activate
    echo "Running SAYNT"
    for entry in `ls $1`; do
        if [ -d $1/$entry ]; then
            echo "Running Paynt on $entry"
            python3 paynt.py --fsc-synthesis --storm-pomdp --iterative-storm 400 30 10 $1/$entry > $1/$entry/probab_sampling_fix_rand.txt
        fi
    done
}

run_with_dictionary() {
    source prerequisites/venv/bin/activate
    echo "Running Paynt with --dictionary"
    counter=0
    for entry in `ls $1`; do
        if [ -d $1/$entry ]; then
            echo "Running Paynt on model $entry"
            folders_found=($(find ./$2 -type d -regex ".*$entry\_\(DQN\|PPO\|DDQN\|STOCHASTIC_PPO\).*" ))
            for folder in "${folders_found[@]}"; do
                if [ -f "$folder/labels.pickle" ] && [ -f "$folder/obs_action_dict.pickle" ] && [ -f "$folder/memory_dict.pickle" ]; then
                    size=$(wc -c < "$folder/obs_action_dict.pickle")
                    if [ "$size" -eq 0 ]; then
                        echo "Obs-Action dictionary in folder $folder is empty, skipping."
                        continue
                    fi
                    echo "   with dictionary from $folder"
                    cp $folder/labels.pickle ./labels.pickle
                    cp $folder/obs_action_dict.pickle ./obs_action_dict.pickle
                    cp $folder/memory_dict.pickle ./memory_dict.pickle
                    echo "Running timeout $3 python3 paynt.py --fsc-synthesis --storm-pomdp --project $1/$entry > $folder/paynt-dict-not-prunning-memory-really.log"
                    timeout $3 python3 paynt.py --fsc-synthesis --storm-pomdp --project $1/$entry > "$folder/paynt-dict-not-prunning-memory-really.log"
                    echo "Finished Paynt on model $entry with dictionary from $folder"
                    counter=$((counter+1))
                fi
            done
        fi
    done
    echo "counter: $counter"
}

if [ ! -d "prerequisites/venv" ]; then
    echo "Virtual environment not found. Please run install.sh first."
    exit 1
fi

print_help() {
    echo "Simple runner for experiments with PAYNT"
    echo ""
    echo "Usage 1: $0 --fsc-synthesis <path-to-models>"
    echo "  - performs FSC synthesis on all models in the given directory."
    echo ""
    echo "Usage 2: $0 <path-to-models> <path-to-dictionary> <timeout>"
    echo "  - runs PAYNT on all models in the given directory with the given dictionary and timeout."
    echo ""
    echo "Usage 3: $0 --saynt <path-to-models>"
    echo "  - runs SAYNT on all models in the given directory."
    echo ""
    echo "Usage 4: $0 --help"
    echo "  - prints this help message."
}

if [ "$#" -lt 1 ]; then
    print_help
    exit 1
fi

if [ $1 == "--help" ]; then
    print_help
    exit 1
fi

if [ $1 == "--saynt" ]; then
    if [ "$#" -ne 2 ]; then
        echo "Usage: $0 --saynt <path-to-models>"
        exit 1
    fi
    run_saynt $2
elif [ $1 == "--fsc-synthesis" ]; then
    if [ "$#" -ne 3 ]; then
        echo "Usage: $0 --fsc-synthesis <path-to-models> <timeout>"
        exit 1
    fi
    run_fsc_synthesis $2 $3
elif [ "$#" -ne 3 ]; then
    echo "Usage: $0 <path-to-models> <path-to-dictionary> <timeout>"
    exit 1
else
    run_with_dictionary $1 $2 $3
fi

exit 0