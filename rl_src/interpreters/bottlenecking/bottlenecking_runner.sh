source ../../../prerequisites/venv/bin/activate

models_folder_paths=("../../models" "../../models_large")
bottlneck_widths=(3)

for models_folder in "${models_folder_paths[@]}"
do
    for model in $(ls $models_folder)
    do
        if [[ $model != *"rocks"* ]] && [[ $models_folder != *"models_large"* ]]
        then
            echo "Model $model is not a rocks model"
            continue
        else
            echo "Processing model $model"
            # continue
        fi; 

        for bottleneck_width in "${bottlneck_widths[@]}"
        do
            python3 quantized_bottleneck_extractor.py $models_folder/$model $bottleneck_width
        done
    done
done