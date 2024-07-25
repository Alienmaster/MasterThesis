# training_sizes=("32" "64" "128" "256" "512" "1024" "2048" "4096" "8192" "16201") # GE
# training_sizes=("32" "64" "128" "256" "512" "1024" "1799") # OMP
training_sizes=("32" "64" "128" "256" "512" "1024" "1428") # Schmidt
models=("google/gemma-7b" "meta-llama/Meta-Llama-3-8B" "mistralai/Mistral-7B-v0.1")
# "google/gemma-7b"
# "meta-llama/Meta-Llama-3-8B"
# "mistralai/Mistral-7B-v0.1"

for size in "${training_sizes[@]}"
do
    for model in "${models[@]}"
    do
        date
        echo "Running script with model: $model and size: $size"
        time python rq22.py "$size" "$model" 0
        time python rq22.py "$size" "$model" 1
        time python rq22.py "$size" "$model" 2
        echo "Script run with model: $model and size: $size completed"
    done
done