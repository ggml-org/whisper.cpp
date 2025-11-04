import os
import subprocess
import re
import csv
import wave
import contextlib
import argparse


# Custom action to handle comma-separated list
class ListAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, [int(val) for val in values.split(",")])


parser = argparse.ArgumentParser(description="Benchmark the speech recognition model")

# Define the argument to accept a list
parser.add_argument(
    "-t",
    "--threads",
    dest="threads",
    action=ListAction,
    default=[4],
    help="List of thread counts to benchmark (comma-separated, default: 4)",
)

parser.add_argument(
    "-p",
    "--processors",
    dest="processors",
    action=ListAction,
    default=[1],
    help="List of processor counts to benchmark (comma-separated, default: 1)",
)


parser.add_argument(
    "-f",
    "--filename",
    type=str,
    default="./samples/jfk.wav",
    help="Relative path of the file to transcribe (default: ./samples/jfk.wav)",
)

# Parse the command line arguments
args = parser.parse_args()

sample_file = args.filename

threads = args.threads
processors = args.processors

# Define the models, threads, and processor counts to benchmark
models = [
    "ggml-tiny.en.bin",
    "ggml-tiny.bin",
    "ggml-base.en.bin",
    "ggml-base.bin",
    "ggml-small.en.bin",
    "ggml-small.bin",
    "ggml-medium.en.bin",
    "ggml-medium.bin",
    "ggml-large-v1.bin",
    "ggml-large-v2.bin",
    "ggml-large-v3.bin",
    "ggml-large-v3-turbo.bin",
]


metal_device = ""

# Initialize a dictionary to hold the results
results = {}

gitHashHeader = "Commit"
modelHeader = "Model"
hardwareHeader = "Hardware"
recordingLengthHeader = "Recording Length (seconds)"
threadHeader = "Thread"
processorCountHeader = "Processor Count"
loadTimeHeader = "Load Time (ms)"
sampleTimeHeader = "Sample Time (ms)"
encodeTimeHeader = "Encode Time (ms)"
decodeTimeHeader = "Decode Time (ms)"
sampleTimePerRunHeader = "Sample Time per Run (ms)"
encodeTimePerRunHeader = "Encode Time per Run (ms)"
decodeTimePerRunHeader = "Decode Time per Run (ms)"
totalTimeHeader = "Total Time (ms)"


def check_file_exists(file: str) -> bool:
    return os.path.isfile(file)


def get_git_short_hash() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode()
            .strip()
        )
    except subprocess.CalledProcessError as e:
        return ""


def wav_file_length(file: str = sample_file) -> float:
    with contextlib.closing(wave.open(file, "r")) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        return duration


def extract_metrics(output: str, label: str) -> tuple[float, float]:
    match = re.search(rf"{label} \s*=\s*(\d+\.\d+)\s*ms\s*/\s*(\d+)\s*runs", output)
    time = float(match.group(1)) if match else None
    runs = float(match.group(2)) if match else None
    return time, runs


def extract_device(output: str) -> str:
    match = re.search(r"picking default device: (.*)", output)
    device = match.group(1) if match else "Not found"
    return device


# Check if the sample file exists
if not check_file_exists(sample_file):
    raise FileNotFoundError(f"Sample file {sample_file} not found")

recording_length = wav_file_length()


# Check that all models exist
# Filter out models from list that are not downloaded
filtered_models = []
for model in models:
    if check_file_exists(f"models/{model}"):
        filtered_models.append(model)
    else:
        print(f"Model {model} not found, removing from list")

models = filtered_models

# Loop over each combination of parameters
for model in filtered_models:
    for thread in threads:
        for processor_count in processors:
            # Construct the command to run
            cmd = f"./build/bin/whisper-cli -m models/{model} -t {thread} -p {processor_count} -f {sample_file}"
            # Run the command and get the output
            process = subprocess.Popen(
                cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
            )

            output = ""
            while process.poll() is None:
                output += process.stdout.read().decode()

            # Parse the output
            load_time_match = re.search(r"load time\s*=\s*(\d+\.\d+)\s*ms", output)
            load_time = float(load_time_match.group(1)) if load_time_match else None

            metal_device = extract_device(output)
            sample_time, sample_runs = extract_metrics(output, "sample time")
            encode_time, encode_runs = extract_metrics(output, "encode time")
            decode_time, decode_runs = extract_metrics(output, "decode time")

            total_time_match = re.search(r"total time\s*=\s*(\d+\.\d+)\s*ms", output)
            total_time = float(total_time_match.group(1)) if total_time_match else None

            model_name = model.replace("ggml-", "").replace(".bin", "")

            print(
                f"Ran model={model_name} threads={thread} processor_count={processor_count}, took {total_time}ms"
            )
            # Store the times in the results dictionary
            results[(model_name, thread, processor_count)] = {
                loadTimeHeader: load_time,
                sampleTimeHeader: sample_time,
                encodeTimeHeader: encode_time,
                decodeTimeHeader: decode_time,
                sampleTimePerRunHeader: round(sample_time / sample_runs, 2),
                encodeTimePerRunHeader: round(encode_time / encode_runs, 2),
                decodeTimePerRunHeader: round(decode_time / decode_runs, 2),
                totalTimeHeader: total_time,
            }

# Write the results to a CSV file
with open("benchmark_results.csv", "w", newline="") as csvfile:
    fieldnames = [
        gitHashHeader,
        modelHeader,
        hardwareHeader,
        recordingLengthHeader,
        threadHeader,
        processorCountHeader,
        loadTimeHeader,
        sampleTimeHeader,
        encodeTimeHeader,
        decodeTimeHeader,
        sampleTimePerRunHeader,
        encodeTimePerRunHeader,
        decodeTimePerRunHeader,
        totalTimeHeader,
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()

    shortHash = get_git_short_hash()
    # Sort the results by total time in ascending order
    sorted_results = sorted(results.items(), key=lambda x: x[1].get(totalTimeHeader, 0))
    for params, times in sorted_results:
        row = {
            gitHashHeader: shortHash,
            modelHeader: params[0],
            hardwareHeader: metal_device,
            recordingLengthHeader: recording_length,
            threadHeader: params[1],
            processorCountHeader: params[2],
        }
        row.update(times)
        writer.writerow(row)
