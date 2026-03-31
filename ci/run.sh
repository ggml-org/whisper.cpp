#!/bin/bash
#
# sample usage:
#
# mkdir tmp
#
# # CPU-only build
# bash ./ci/run.sh ./tmp/results ./tmp/mnt
#
# # with CUDA support
# GG_BUILD_CUDA=1 bash ./ci/run.sh ./tmp/results ./tmp/mnt
#
# # with SYCL support
# GG_BUILD_SYCL=1 bash ./ci/run.sh ./tmp/results ./tmp/mnt

if [ -z "$2" ]; then
    echo "usage: $0 <output-dir> <mnt-dir>"
    exit 1
fi

mkdir -p "$1"
mkdir -p "$2"

OUT=$(realpath "$1")
MNT=$(realpath "$2")

rm -vf $OUT/*.log
rm -vf $OUT/*.exit
rm -vf $OUT/*.md

sd=`dirname $0`
cd $sd/../
SRC=`pwd`

ALL_MODELS=( "tiny.en" "tiny" "base.en" "base" "small.en" "small" "medium.en" "medium" "large-v1" "large-v2" "large-v3" "large-v3-turbo" )
BENCH_N_THREADS=4
BENCH_ENCODER_ONLY=0
BENCH_FLASH_ATTN=0

# check for user-specified models first. if not specified, use fast models
if [ ! -z ${GG_BUILD_TEST_MODELS} ]; then
    IFS=',' read -r -a MODELS <<< "${GG_BUILD_TEST_MODELS}"
else
    if [ ! -z ${GG_BUILD_LOW_PERF} ]; then
        MODELS=( "tiny" "base" "small" )
    else
        MODELS=("${ALL_MODELS[@]}")
    fi
fi

CMAKE_EXTRA="-DWHISPER_FATAL_WARNINGS=ON"

if [ ! -z ${GG_BUILD_METAL} ]; then
    CMAKE_EXTRA="${CMAKE_EXTRA} -DGGML_METAL=ON"
fi

if [ ! -z ${GG_BUILD_CUDA} ]; then
    CMAKE_EXTRA="${CMAKE_EXTRA} -DGGML_CUDA=ON"

    if command -v nvidia-smi >/dev/null 2>&1; then
        CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d '.')
        if [[ -n "$CUDA_ARCH" && "$CUDA_ARCH" =~ ^[0-9]+$ ]]; then
            CMAKE_EXTRA="${CMAKE_EXTRA} -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}"
        else
            echo "Warning: Using fallback CUDA architectures"
            CMAKE_EXTRA="${CMAKE_EXTRA} -DCMAKE_CUDA_ARCHITECTURES=61;70;75;80;86;89"
        fi
    else
        echo "Error: nvidia-smi not found, cannot build with CUDA"
        exit 1
    fi
fi

if [ ! -z ${GG_BUILD_ROCM} ]; then
    CMAKE_EXTRA="${CMAKE_EXTRA} -DGGML_HIP=ON"
    if [ -z ${GG_BUILD_AMDGPU_TARGETS} ]; then
        echo "Missing GG_BUILD_AMDGPU_TARGETS, please set it to your GPU architecture (e.g. gfx90a, gfx1100, etc.)"
        exit 1
    fi

    CMAKE_EXTRA="${CMAKE_EXTRA} -DAMDGPU_TARGETS=${GG_BUILD_AMDGPU_TARGETS}"
fi

if [ ! -z ${GG_BUILD_SYCL} ]; then
    if [ -z ${ONEAPI_ROOT} ]; then
        echo "Not detected ONEAPI_ROOT, please install oneAPI base toolkit and enable it by:"
        echo "source /opt/intel/oneapi/setvars.sh"
        exit 1
    fi
    # Use only main GPU
    export ONEAPI_DEVICE_SELECTOR="level_zero:0"
    # Enable sysman for correct memory reporting
    export ZES_ENABLE_SYSMAN=1
    # to circumvent precision issues on CPY operations
    export SYCL_PROGRAM_COMPILE_OPTIONS="-cl-fp32-correctly-rounded-divide-sqrt"
    CMAKE_EXTRA="${CMAKE_EXTRA} -DGGML_SYCL=1 -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DGGML_SYCL_F16=ON"
fi

if [ ! -z ${GG_BUILD_VULKAN} ]; then
    CMAKE_EXTRA="${CMAKE_EXTRA} -DGGML_VULKAN=1"

    # if on Mac, disable METAL
    if [[ "$OSTYPE" == "darwin"* ]]; then
        CMAKE_EXTRA="${CMAKE_EXTRA} -DGGML_METAL=OFF -DGGML_BLAS=OFF"
    fi

fi

if [ ! -z ${GG_BUILD_WEBGPU} ]; then
    CMAKE_EXTRA="${CMAKE_EXTRA} -DGGML_WEBGPU=1"
fi

if [ ! -z ${GG_BUILD_MUSA} ]; then
    # Use qy1 by default (MTT S80)
    MUSA_ARCH=${MUSA_ARCH:-21}
    CMAKE_EXTRA="${CMAKE_EXTRA} -DGGML_MUSA=ON -DMUSA_ARCHITECTURES=${MUSA_ARCH}"
fi

if [ ! -z ${GG_BUILD_NO_SVE} ]; then
    # arm 9 and newer enables sve by default, adjust these flags depending on the cpu used
    CMAKE_EXTRA="${CMAKE_EXTRA} -DGGML_NATIVE=OFF -DGGML_CPU_ARM_ARCH=armv8.5-a+fp16+i8mm"
fi

## helpers

# download a file if it does not exist or if it is outdated
function gg_wget {
    local out=$1
    local url=$2

    local cwd=`pwd`

    mkdir -p $out
    cd $out

    # should not re-download if file is the same
    wget -nv -N $url

    cd $cwd
}

function gg_download_model {
    local model_name=$1
    local model_file="$MNT/models/ggml-${model_name}.bin"

    if [ ! -f ${model_file} ]; then
        local cwd=`pwd`
        mkdir -p "$MNT/models"
        cd "$MNT/models"
        bash "$cwd/models/download-ggml-model.sh" ${model_name} .
        cd "$cwd"
    fi
}

function gg_printf {
    printf -- "$@" >> $OUT/README.md
}

# Helper function to check command exit status
function gg_check_last_command_status {
    local exit_file=$1
    local command_name=$2

    local exit_status=$?
    echo "$exit_status" > "$exit_file"

    if [ $exit_status -ne 0 ]; then
        echo "Error: Command $command_name failed with exit status $exit_status"
        return 1
    fi

    return 0
}

# Usage: gg_run <test_name> [additional_args...]
#
# Parameters:
#   test_name       - Name of the test to run (calls gg_run_<test_name>)
#   additional_args - Any additional arguments to pass to the test function (first argument is appended to the log filename)
function gg_run {
    ci=$1

    if [ $# -gt 1 ]; then
        ci="${ci}_${2}"
    fi

    set -o pipefail
    set -x

    gg_run_$1 "$@" | tee $OUT/$ci.log
    cur=$?
    echo "$cur" > $OUT/$ci.exit

    set +x
    set +o pipefail

    gg_sum_$1 "$@"

    ret=$((ret | cur))
}

function gg_check_build_requirements {
    if ! command -v cmake &> /dev/null; then
        gg_printf 'cmake not found, please install'
    fi

    if ! command -v make &> /dev/null; then
        gg_printf 'make not found, please install'
    fi
}

## ci

function gg_run_ctest {
    mode=$2

    cd ${SRC}

    rm -rf build-ci-${mode} && mkdir build-ci-${mode} && cd build-ci-${mode}

    set -e

    gg_check_build_requirements

    (time cmake -DCMAKE_BUILD_TYPE=${mode} ${CMAKE_EXTRA} .. ) 2>&1 | tee -a $OUT/${ci}-cmake.log
    (time make -j$(nproc)                                    ) 2>&1 | tee -a $OUT/${ci}-make.log

    (time ctest --output-on-failure -L main -E test-opt ) 2>&1 | tee -a $OUT/${ci}-ctest.log

    set +e
}

function gg_sum_ctest {
    mode=$2

    gg_printf '### %s\n\n' "${ci}"

    gg_printf 'Runs ctest in '${mode}' mode\n'
    gg_printf '- status: %s\n' "$(cat $OUT/${ci}.exit)"
    gg_printf '```\n'
    gg_printf '%s\n' "$(cat $OUT/${ci}-ctest.log)"
    gg_printf '```\n'
}

function gg_run_bench {
    cd ${SRC}

    # set flash attention flag if enabled
    fattn="-nfa"
    if [ "$BENCH_FLASH_ATTN" -eq 1 ]; then
        fattn="-fa"
    fi

    # run memcpy benchmark if not encoder-only mode
    if [ "$BENCH_ENCODER_ONLY" -eq 0 ]; then
        echo "Running memcpy benchmark"
        (time ./build-ci-release/bin/whisper-bench -w 1 -t $BENCH_N_THREADS 2>&1) | tee -a $OUT/${ci}-memcpy.log
        gg_check_last_command_status "$OUT/${ci}-memcpy.exit" "memcpy benchmark"

        echo "Running ggml_mul_mat benchmark with $BENCH_N_THREADS threads"
        (time ./build-ci-release/bin/whisper-bench -w 2 -t $BENCH_N_THREADS 2>&1) | tee -a $OUT/${ci}-mul_mat.log
        gg_check_last_command_status "$OUT/${ci}-mul_mat.exit" "ggml_mul_mat benchmark"
    fi

    echo "Running benchmark for all models"

    # generate header for the benchmark table
    {
        printf "| %16s | %13s | %3s | %3s | %7s | %7s | %7s | %7s | %7s |\n" "Config" "Model" "Th" "FA" "Enc." "Dec." "Bch5" "PP" "Commit"
        printf "| %16s | %13s | %3s | %3s | %7s | %7s | %7s | %7s | %7s |\n" "---" "---" "---" "---" "---" "---" "---" "---" "---"
    } | tee -a $OUT/${ci}-models-table.log

    res=0

    # run benchmark for each model
    for model in "${MODELS[@]}"; do
        echo "Benchmarking model: $model"

        # run the benchmark and capture output
        output=$(./build-ci-release/bin/whisper-bench -m $MNT/models/ggml-$model.bin -t $BENCH_N_THREADS $fattn 2>&1)
        ret=$?

        # save the raw output
        echo "$output" > $OUT/${ci}-bench-$model.log

        if [ $ret -eq 0 ]; then
            # parse the benchmark results
            encode_time=$(echo "$output" | grep "encode time" | awk '{print $11}')
            decode_time=$(echo "$output" | grep "decode time" | awk '{print $11}')
            batchd_time=$(echo "$output" | grep "batchd time" | awk '{print $11}')
            prompt_time=$(echo "$output" | grep "prompt time" | awk '{print $11}')
            system_info=$(echo "$output" | grep "system_info")
            actual_threads=$(echo "$output" | grep "system_info" | awk '{print $4}')

            # determine configuration
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
            if [[ $system_info == *"COREML = 1"* ]]; then
                config="$config COREML"
            fi
            if [[ $system_info == *"CUDA = 1"* ]]; then
                config="$config CUDA"
            fi
            if [[ $system_info == *"METAL = 1"* ]]; then
                config="$config METAL"
            fi

            # get commit hash
            commit=$(git rev-parse --short HEAD)

            # add row to benchmark table
            printf "| %16s | %13s | %3s | %3s | %7s | %7s | %7s | %7s | %7s |\n" \
                "$config" "$model" "$actual_threads" "$BENCH_FLASH_ATTN" "$encode_time" "$decode_time" "$batchd_time" "$prompt_time" "$commit" \
                | tee -a $OUT/${ci}-models-table.log
        else
            echo "Benchmark failed for model: $model" | tee -a $OUT/${ci}-bench-errors.log
            res=1
        fi
    done

    return $res
}

function gg_sum_bench {
    gg_printf '### %s\n\n' "${ci}"

    gg_printf 'Whisper Benchmark Results\n'
    gg_printf '- status: %s\n' "$(cat $OUT/${ci}.exit)"

    # show memcpy and ggml_mul_mat benchmark results if available
    if [ "$BENCH_ENCODER_ONLY" -eq 0 ]; then
        if [ -f "$OUT/${ci}-memcpy.log" ]; then
            gg_printf '#### memcpy Benchmark\n\n'
            gg_printf '```\n%s\n```\n\n' "$(cat $OUT/${ci}-memcpy.log)"
        fi

        if [ -f "$OUT/${ci}-mul_mat.log" ]; then
            gg_printf '#### ggml_mul_mat Benchmark\n\n'
            gg_printf '```\n%s\n```\n\n' "$(cat $OUT/${ci}-mul_mat.log)"
        fi
    fi

    # show model benchmark results
    gg_printf '#### Model Benchmarks\n\n'
    if [ -f "$OUT/${ci}-models-table.log" ]; then
        gg_printf '%s\n\n' "$(cat $OUT/${ci}-models-table.log)"
    else
        gg_printf 'No model benchmark results available.\n\n'
    fi

    # show any errors that occurred
    if [ -f "$OUT/${ci}-bench-errors.log" ]; then
        gg_printf '#### Benchmark Errors\n\n'
        gg_printf '```\n%s\n```\n\n' "$(cat $OUT/${ci}-bench-errors.log)"
    fi
}

ret=0

for model in "${MODELS[@]}"; do
    test $ret -eq 0 && gg_download_model ${model}
done

test $ret -eq 0 && gg_run ctest debug
test $ret -eq 0 && gg_run ctest release

test $ret -eq 0 && gg_run bench

cat $OUT/README.md

exit $ret
