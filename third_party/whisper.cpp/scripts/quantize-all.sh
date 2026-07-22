#!/bin/bash

printf "Usage: $0 <upload>"

if [ $# -ne 1 ]; then
    printf "\nError: Invalid number of arguments\n"
    exit 1
fi

qtype0="q5_0"
qtype1="q5_1"
upload="$1"
declare -a filedex

cd `dirname $0`
cd ../

for i in `ls ./models | grep ^ggml-.*.bin | grep -v "\-q"`; do
    m="models/$i"
    if [ -f "$m" ]; then
        if [ "${m##*.}" == "bin" ]; then
            ./build/bin/whisper-quantize "${m}" "${m::${#m}-4}-${qtype1}.bin" ${qtype1};
            ./build/bin/whisper-quantize "${m}" "${m::${#m}-4}-${qtype0}.bin" ${qtype0};
            filedex+=( "${m::${#m}-4}-${qtype1}.bin" "${m::${#m}-4}-${qtype0}.bin" )
        fi
    fi
done



if [ "$upload" == "1" ]; then
    for i in ${!filedex[@]}; do
        if [ "${filedex[$i]:9:8}" != "for-test" ]; then
            scp ${filedex[$i]} root@linode0:/mnt/Data/ggml/ggml-model-${filedex[$i]:9}
        fi
    done
fi
