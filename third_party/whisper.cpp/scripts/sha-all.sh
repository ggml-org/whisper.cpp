#!/bin/bash

# Compute the SHA1 of all model files in ./models/ggml-*.bin

for f in ./models/ggml-*.bin; do
    shasum "$f" -a 1
done
