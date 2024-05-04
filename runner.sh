run_fsc_synthesis() {
    source prerequisites/venv/bin/activate
    echo "Running Paynt with --fsc-synthesis"
    for entry in `ls $1`; do
        if [ -d $1/$entry ]; then
            echo "Running Paynt on $entry"
            python3 paynt.py --fsc-synthesis --project $1/$entry > $1/$entry/paynt-fsc-synthesis.log
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
                    echo "Running timeout $3 python3 paynt.py --fsc-synthesis --storm-pomdp --project $1/$entry > $folder/paynt-dict-not-prunning-memory.log"
                    timeout $3 python3 paynt.py --fsc-synthesis --storm-pomdp --project $1/$entry > "$folder/paynt-dict-not-prunning-memory.log"
                    echo "Finished Paynt on model $entry with dictionary from $folder"
                    counter=$((counter+1))
                fi
            done
        fi
    done
    echo "counter: $counter"
}

# run_fsc_synthesis $1
run_with_dictionary $1 $2 $3
