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
            python3 paynt.py --fsc-synthesis --storm-pomdp --iterative-storm 400 30 10 $1/$entry > $1/$entry/saynt_rewards.txt
        fi
    done
}

run_saynt_bc(){
    source prerequisites/venv/bin/activate
    echo "Running SAYNT with behavioral cloning"
    
    # sub_methods=("only_pretrained" "only_duplex" "only_duplex_critic" "complete")
    sub_methods=("SAYNT_Pretraining_batch256")
    for entry in `ls $1`; do
        if [ -d $1/$entry ]; then
            echo "Running Paynt on $entry"
            
            for sub_method in "${sub_methods[@]}"; do
                echo "Running Paynt with --sub_method=$sub_method on $entry"
                python3 paynt.py --fsc-synthesis --storm-pomdp --iterative-storm 400 30 10 --reinforcement-learning --model-name $entry --agent_task behavioral_cloning --sub_method $sub_method $1/$entry > $1/$entry/offline_pretrain_$sub_method.fixed.txt
            done
        fi
    done
}

run_paynt_imitation(){
    source prerequisites/venv/bin/activate
    echo "Running PAYNT with imitation learning"
    # methods=("BC" "R_Shaping" "Jumpstarts")
    methods=("BC")
    sub_methods=("longer_trajectories" "continuous_training")
    num_iterations=10
    for iteration in $(seq 1 $num_iterations); do
        echo "Iteration $iteration"
        for entry in `ls $1`; do
            if [ -d $1/$entry ]; then
                # Skip if the model is not refuel-20 or rocks-16
                if [ "$entry" != "refuel-20" ] && [ "$entry" != "rocks-16" ]; then
                    continue
                fi
                echo "Running Paynt on $entry"
                for method in "${methods[@]}"; do
                    # echo "Running Paynt with --imitation-learning=$method on $entry"
                    # echo "python3 paynt.py --fsc-synthesis --storm-pomdp --iterative-storm 400 30 10 --reinforcement-learning --model-name $entry --rl-method $method $1/$entry > $1/$entry/imitation_$method.$iteration.txt"
                    # python3 paynt.py --fsc-synthesis --storm-pomdp --iterative-storm 400 30 10 --reinforcement-learning --model-name $entry --rl-method $method $1/$entry > $1/$entry/imitation_$method.$iteration.txt
                    for sub_method in "${sub_methods[@]}"; do
                        echo "Running Paynt with --imitation-learning=$method --sub_method=$sub_method on $entry"
                        python3 paynt.py --fsc-synthesis --storm-pomdp --iterative-storm 400 30 10 --reinforcement-learning --model-name $entry --rl-method $method --sub-method $sub_method $1/$entry > $1/$entry/imitation_$method.$sub_method.$iteration.txt
                    done
                done
            fi
        done  
    done  
}

run_paynt_shaping(){
    source prerequisites/venv/bin/activate
    echo "Running PAYNT with shaping"
    methods=("R_Shaping")
    num_iterations=10
    sub_method=("longer_trajectories")
    for iteration in $(seq 1 $num_iterations); do
        echo "Iteration $iteration"
        for entry in `ls $1`; do
            if [ -d $1/$entry ]; then
                # Skip if the model is not refuel-20 or rocks-16
                if [ "$entry" != "refuel-20" ] && [ "$entry" != "rocks-16" ]; then
                    continue
                fi
                echo "Running Paynt on $entry"
                for method in "${methods[@]}"; do
                    echo "Running Paynt with --imitation-learning=$method --sub_method=$sub_method on $entry"
                    python3 paynt.py --fsc-synthesis --storm-pomdp --iterative-storm 400 30 10 --reinforcement-learning --model-name $entry --rl-method $method --sub-method $sub_method $1/$entry > $1/$entry/imitation_$method.$sub_method.$iteration.txt
                done
            fi
        done  
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

run_paynt_with_rl_hints() {
    source prerequisites/venv/bin/activate
    echo "Running Paynt with --rl-hints"
    sub_methods=("without_memory", "with_memory")
    for entry in `ls $1`; do
        if [ -d $1/$entry ]; then
            echo "Running Paynt on model $entry"
            echo "Running Paynt with --rl-hints=without_memory on $entry"
            timeout $2 python3 paynt.py --fsc-synthesis --storm-pomdp --reinforcement-learning $1/$entry > $1/$entry/paynt-rl-hints-without-memory.log
            
            echo "Running Paynt with --rl-hints=with_memory on $entry"
            timeout $2 python3 paynt.py --fsc-synthesis --storm-pomdp --reinforcement-learning --rl-load-memory-flag $1/$entry > $1/$entry/paynt-rl-hints-with-memory.log
        fi
    done
}

run_paynt_with_stochastic_hints() {
    source prerequisites/venv/bin/activate
    echo "Running Paynt with --rl-hints"
    sub_methods=("without_memory", "with_memory")
    for entry in `ls $1`; do
        if [ -d $1/$entry ]; then
            echo "Running Paynt on model $entry"
            echo "Running Paynt with --rl-hints=without_memory on $entry"
            timeout $2 python3 paynt.py --fsc-synthesis --storm-pomdp --greedy --reinforcement-learning --greedy $1/$entry > $1/$entry/paynt-rl-hints-without-memory-stochastic.log 
            
            echo "Running Paynt with --rl-hints=with_memory on $entry"
            timeout $2 python3 paynt.py --fsc-synthesis --storm-pomdp --reinforcement-learning --greedy --rl-load-memory-flag --greedy $1/$entry > $1/$entry/paynt-rl-hints-with-memory-stochastic.log
        fi
    done
}

run_paynt_with_pruning_rl_hints(){
    source prerequisites/venv/bin/activate
    echo "Running Paynt with --pruning"
    sub_methods=("without_memory", "with_memory")
    for entry in `ls $1`; do
        if [ -d $1/$entry ]; then
            echo "Running Paynt on model $entry"
            echo "Running Paynt with --pruning=without_memory on $entry"
            timeout $2 python3 paynt.py --fsc-synthesis --storm-pomdp --reinforcement-learning --prune-storm $1/$entry > $1/$entry/paynt-pruning-rl-hints-without-memory.log
            
            echo "Running Paynt with --pruning=with_memory on $entry"
            timeout $2 python3 paynt.py --fsc-synthesis --storm-pomdp --reinforcement-learning --prune-storm --rl-load-memory-flag $1/$entry > $1/$entry/paynt-pruning-rl-hints-with-memory.log
        fi
    done
}

create_save_file(){

    save_file=$1/$2/paynt-$3-$index.log
    index=1
    while [ -f $save_file ]; do
        save_file=$1/$2/paynt-$3-$index.log
        index=$((index+1))
    done
    echo $save_file
}

run_paynt_loop_rl(){
    # Example python3 paynt.py --fsc-synthesis --storm-pomdp --reinforcement-learning rl_src/models/rocks-16/ --loop --rl-method=R_Shaping --greedy --fsc-time-in-loop 30--fsc-time-in-loop --rl-training-iters 500
    source prerequisites/venv/bin/activate

    trap "echo 'Terminating script...'; kill 0; exit" SIGINT SIGTERM
    for entry in `ls $1`; do
        if [ -d $1/$entry ]; then
            echo "Running Paynt on model $entry"
            echo "Running Paynt with --loop-rl on $entry with parameters $2 $3"
            # create save file
            # save_file=$( create_save_file $1 $entry "loop-rl" )
            # echo "save file: $save_file"
            # python3 paynt.py --fsc-synthesis --storm-pomdp --reinforcement-learning $1/$entry --loop --rl-method=R_Shaping --greedy --fsc-time-in-loop $2 --model-name $entry --rl-training-iters $3 > $save_file

            save_file=$( create_save_file $1 $entry "loop-rl-memory" )
            python3 paynt.py --fsc-synthesis --storm-pomdp --reinforcement-learning $1/$entry --loop --rl-method=R_Shaping --greedy --model-name $entry --fsc-time-in-loop $2 --rl-training-iters $3 --rl-load-memory-flag > $save_file
            # pid2=$!
            # wait $pid
            # wait $pid2 
        fi
    done

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
    echo "Usage 4: $0 --saynt_bc"
    echo "  - runs SAYNT with various types of behavioral cloning."
    echo "Usage %: $0 --help"
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
elif [ $1 == "--saynt-bc" ]; then
    if [ "$#" -ne 2 ]; then
        echo "Usage: $0 --saynt-bc <path-to-models>"
        exit 1
    fi
    run_saynt_bc $2
elif [ $1 == "--shaping" ]; then
    if [ "$#" -ne 2 ]; then
        echo "Usage: $0 --shaping <path-to-models>"
        exit 1
    fi
    run_paynt_shaping $2 
elif [ $1 == "--imitation-learning" ]; then
    if [ "$#" -ne 2 ]; then
        echo "Usage: $0 --imitation <path-to-models>"
        exit 1
    fi
    run_paynt_imitation $2
elif [ $1 == "--fsc-synthesis" ]; then
    if [ "$#" -ne 3 ]; then
        echo "Usage: $0 --fsc-synthesis <path-to-models> <timeout>"
        exit 1
    fi
    run_fsc_synthesis $2 $3
elif [ $1 == "--rl-hints" ]; then
    if [ "$#" -ne 3 ]; then
        echo "Usage: $0 --rl-hints <path-to-models> <timeout>"
        exit 1
    fi
    run_paynt_with_rl_hints $2 $3
elif [ $1 == "--stochastic-hints" ]; then
    if [ "$#" -ne 3 ]; then
        echo "Usage: $0 --stochastic-hints <path-to-models> <timeout>"
        exit 1
    fi
    run_paynt_with_stochastic_hints $2 $3
elif [ $1 == "--pruning-rl-hints" ]; then
    if [ "$#" -ne 3 ]; then
        echo "Usage: $0 --pruning-rl-hints <path-to-models> <timeout>"
        exit 1
    fi
    run_paynt_with_pruning_rl_hints $2 $3
elif [ $1 == "--loop-rl" ]; then
    if [ "$#" -ne 4 ]; then
        echo "Usage: $0 --loop-rl <path-to-models> <fsc-timeout> <rl-iters>"
        exit 1
    fi
    run_paynt_loop_rl $2 $3 $4
elif [ "$#" -ne 3 ]; then
    echo "Usage: $0 <path-to-models> <path-to-dictionary> <timeout>"
    exit 1
else
    run_with_dictionary $1 $2 $3
fi

exit 0