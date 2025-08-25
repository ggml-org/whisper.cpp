#!/bin/bash

set -e

## Test initialize request
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | \
    build/bin/whisper-mcp-server -m models/ggml-base.en.bin

## Test tools list request
echo '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}' | \
    build/bin/whisper-mcp-server -m models/ggml-base.en.bin

## Test transcribe
#echo '{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"transcribe","arguments":{"file":"samples/jfk.wav","language":"en"}}}' | \
    #build/bin/whisper-mcp-server -m models/ggml-base.en.bin
