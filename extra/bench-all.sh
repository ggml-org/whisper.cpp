#!/bin/bash

# Helper script to run the bench tool on all models and print the results in share-able format

printf "Usage: ./bench.sh [n_threads]\n"

if [ -z "$1" ]; then
    n_threads=4
else
    n_threads=$1
fi

models=( "tiny" "base" "small" "medium" "large" )

printf "\n"
printf "Running benchmark for all models\n"
printf "This can take a while!\n"
printf "\n"

printf "| CPU | OS | Config | Model | Threads | Total Load | Real Load | Total Encode | Real Encode | Commit |\n"
printf "| --- | -- | ------ | ----- | ------- | ---------- | --------- | ------------ | ----------- | ------ |\n"

for model in "${models[@]}"; do
    # run once to heat-up the cache
    ./bench -m ./models/ggml-$model.bin -t $n_threads 2>/dev/null 1>/dev/null

    # actual run
    # store stderr output in a variable in order to parse it later
    output=$(./bench -m ./models/ggml-$model.bin -t $n_threads 2>&1)

    # parse the output:
    total_load_time=$(echo "$output" | grep "load time" | awk '{print $5}')
    real_load_time=$(echo "$output" | grep "load time" | awk '{print $8}')
    total_encode_time=$(echo "$output" | grep "encode time" | awk '{print $5}')
    real_encode_time=$(echo "$output" | grep "encode time" | awk '{print $8}')
    system_info=$(echo "$output" | grep "system_info")
    n_threads=$(echo "$output" | grep "system_info" | awk '{print $4}')

    # floor to milliseconds
    total_load_time=${total_load_time%.*}
    real_load_time=${real_load_time%.*}
    total_encode_time=${total_encode_time%.*}
    real_encode_time=${real_encode_time%.*}

    config=""

    if [[ $system_info == *"AVX2 = 1"* ]]; then
        config="$config AVX2"
    fi

    if [[ $system_info == *"NEON = 1"* ]]; then
        config="$config NEON"
    fi

    if [[ $system_info == *"BLAS = 1"* ]]; then
        config="$config BLAS"
    fi

    commit=$(git rev-parse --short HEAD)

    printf "| <todo> | <todo> | $config | $model | $n_threads | $total_load_time | $real_load_time | $total_encode_time | $real_encode_time | $commit |\n"
done

