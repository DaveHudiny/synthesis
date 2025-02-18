source prerequisites/venv/bin/activate

run_loop(){
    # Go through all models in the folder and runs paynt on the model
    folders=("rl_src/models/" "rl_src/models_large/")
    already_performed_experiments=("evade" "grid-large-10-5" "grid-large-30-5" "intercept" "intercept-n7-r1" "mba" "mba-small")
    if ! [ -d "./experiments_looper/" ]; then
        mkdir "experiments_looper"
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
            if [ -f "./experiments_looper/$model_name.log" ]; then
                current_time=$(date +%Y%m%d%H%M%S)
                log_file_name="./experiments_looper/$model_name-$current_time.log"
                err_log_file_name="./experiments_looper/$model_name-$current_time.err.log"
            else
                log_file_name="./experiments_looper/$model_name.log"
                err_log_file_name="./experiments_looper/$model_name.err.log"
            fi
            python3 paynt.py --fsc-synthesis --storm-pomdp --reinforcement-learning --model-name $model_name $model_path --loop > $log_file_name 2> $err_log_file_name

        done

    done
}


run_loop