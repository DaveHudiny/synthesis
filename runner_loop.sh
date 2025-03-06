source prerequisites/venv/bin/activate

run_loop(){
    # Go through all models in the folder and runs paynt on the model
    folders=("rl_src/models_paynt_experiments/")
    # already_performed_experiments=("evade" "grid-large-10-5" "grid-large-30-5" "intercept" "intercept-n7-r1" "mba" "mba-small")
    already_performed_experiments=("evade-n12" "drone-2-6-1" "geo-2-8" "maze-10")
    experiment_folder="experiments_bc_loop"
    if ! [ -d $experiment_folder ]; then
        mkdir $experiment_folder
    fi


    for folder in $folders; do
        for entry in `ls $folder`; do
            echo Running loopy PAYNT on $folder/$entry
            #check, if the entry is folder
            if [[ " ${already_performed_experiments[@]} " =~ " ${entry} " ]]; then
                echo "Skipping $entry"
                continue
            fi
            if ! [ -d $folder/$entry ]; then
                continue
            fi
            model_name=$entry
            model_path=$folder/$entry
            if [ -f "./$experiment_folder/$model_name.log" ]; then
                current_time=$(date +%Y%m%d%H%M%S)
                log_file_name="./$experiment_folder/$model_name-$current_time.log"
                err_log_file_name="./$experiment_folder/$model_name-$current_time.err.log"
            else
                log_file_name="./$experiment_folder/$model_name.log"
                err_log_file_name="./$experiment_folder/$model_name.err.log"
            fi
            python3 paynt.py --fsc-synthesis --storm-pomdp --iterative-storm 600 30 30 --reinforcement-learning --model-name $model_name --loop --rl-method BC $model_path > $log_file_name 2> $err_log_file_name

        done

    done
}


run_loop