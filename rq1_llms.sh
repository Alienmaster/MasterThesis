#!/bin/bash

# List of values for the model and split token
models=("google/gemma-1.1-7b-it <start_of_turn>model" "meta-llama/Meta-Llama-3-8B-Instruct assistant<|end_header_id|>" "mistralai/Mistral-7B-Instruct-v0.2 [/inst]")
# "google/gemma-1.1-7b-it <start_of_turn>model"
# "meta-llama/Llama-2-13b-chat-hf [/inst]"
# "meta-llama/Meta-Llama-3-8B-Instruct assistant<|end_header_id|>"
# "mistralai/Mistral-7B-Instruct-v0.2 [/inst]"
# "mistralai/Mixtral-8x7B-Instruct-v0.1 [/inst]"

# List of values for the dataset and split
datasets=("germeval test_syn" "omp Full" "schmidt test" "wikipedia Full")
# "germeval test_syn"
# "omp Full"
# "wikipedia Full"
# "schmidt test"

for model in "${models[@]}"
do
    for dataset in "${datasets[@]}"
    do
        date
        echo "Running script with model: $model and dataset: $dataset"
        time python rq1_llms.py "$model" "$dataset"
        echo "Script run with model: $model and dataset: $dataset completed"
    done
done
