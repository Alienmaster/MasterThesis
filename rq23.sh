sizes=("32" "64" "128" "256" "512" "1024" "2048" "4096")

for size in "${sizes[@]}"
do
    date
    echo "Running script with: $size"
    time python rq23.py "$size" 1
    echo "Script run with size: $size"
    date
    echo "Running script with: $size"
    time python rq23.py "$size" 2
    echo "Script run with size: $size"
    date
    echo "Running script with: $size"
    time python rq23.py "$size" 3
    echo "Script run with size: $size"
done
