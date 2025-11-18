# Benchmark word-level timestamps for different models
#
# This script takes two arguments
# - an audio file
# - [optional] path to a font file

# I'm using "/usr/share/fonts/truetype/freefont/FreeMono.ttf" on Ubuntu

if [ -z "$1" ]; then
    echo "Usage: $0 <audio file> [font file]"
    exit 1
fi

#TODO: Make this a command line parameter
#models="base small large"
#models="tiny.en tiny base.en base small.en small medium.en medium large-v1 large"
models="tiny.en base.en small.en medium.en large"

DURATION=$(ffprobe -i $1 -show_entries format=duration -v quiet -of csv="p=0")
DURATION=$(printf "%.2f" $DURATION)
echo "Input file duration: ${DURATION}s"

for model in $models; do
    echo "Running $model"
    COMMAND="./build/bin/whisper-cli -m models/ggml-$model.bin -owts -f $1 -of $1.$model"

    if [ ! -z "$2" ]; then
        COMMAND="$COMMAND -fp $2"
    fi
    #TODO: Surface errors better
    # TIMEFMT is for zsh, TIMEFORMAT is for bash
    EXECTIME=$({ TIMEFMT="%E";TIMEFORMAT=%E; time $COMMAND >/dev/null 2>&1; } 2>&1)

    # Slightly different formats between zsh and bash
    if [ "${EXECTIME: -1}" == "s" ]; then
        EXECTIME=${EXECTIME::-1}
    fi

    RATIO=$(echo "$DURATION / $EXECTIME" | bc -l)
    RATIO=$(printf "%.2f" $RATIO)

    echo "Execution time: ${EXECTIME}s (${RATIO}x realtime)"

    # If the file already exists, delete it
    if [ -f $1.mp4 ]; then
        rm $1.mp4
    fi

    bash $1.$model.wts >/dev/null 2>&1
    mv $1.mp4 $1.$model.mp4

    ffmpeg -y -f lavfi -i color=c=black:s=1200x50:d=$DURATION -vf "drawtext=fontfile=$2:fontsize=36:x=10:y=(h-text_h)/2:text='ggml-$model - ${EXECTIME}s (${RATIO}x realtime)':fontcolor=lightgrey" $1.$model.info.mp4 >/dev/null 2>&1
done

COMMAND="ffmpeg -y"
for model in $models; do
    COMMAND="$COMMAND -i $1.$model.info.mp4 -i $1.$model.mp4"
done
COMMAND="$COMMAND -filter_complex \""
COUNT=0
for model in $models; do
    COMMAND="$COMMAND[${COUNT}:v][$(($COUNT+1)):v]"
    COUNT=$((COUNT+2))
done
COMMAND="$COMMAND vstack=inputs=${COUNT}[v]\" -map \"[v]\" -map 1:a $1.all.mp4 >/dev/null 2>&1"

echo $COMMAND

# Run the command
eval $COMMAND
