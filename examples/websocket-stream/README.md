# WebSocket Whisper Stream Example

This example demonstrates a WebSocket-based real-time audio transcription service using the Whisper model. The server captures audio from clients, processes it using the Whisper model, and sends transcriptions back through WebSocket connections.

## Features

- Real-time audio transcription
- WebSocket communication for audio and transcription data
- Configurable parameters for model, language, and processing settings
- Integration with backend services via HTTP requests

## Usage

Run the server with the following command:

```bash
./build/bin/whisper-websocket-stream -m ./models/ggml-large-v3-turbo.bin -t 8 --host 0.0.0.0 --port 9002 --forward-url http://localhost:8080/completion
```

### Parameters

- `-m` or `--model`: Path to the Whisper model file.
- `-t` or `--threads`: Number of threads for processing.
- `-H` or `--host`: Hostname or IP address to bind the server to.
- `-p` or `--port`: Port number for the server.
- `-f` or `--forward-url`: URL to forward transcriptions to a backend service.
- `-nm` or `--max-messages`: Maximum number of messages before sending to the backend.
- `-l` or `--language`: Spoken language for transcription.
- `-vth` or `--vad-thold`: Voice activity detection threshold.
- `-tr` or `--translate`: Enable translation to English.
- `-ng` or `--no-gpu`: Disable GPU usage.
- `-bs` or `--beam-size`: Beam size for beam search.

## Building

To build the server, follow these steps:

```bash
# Install dependencies
git clone --depth 1 https://github.com/machinezone/IXWebSocket/
cd IXWebSocket
mkdir -p build && cd build && cmake -GNinja .. && sudo ninja -j$((npoc)) install
# Build the project
#cuda is optional
git clone --depth 1 https://github.com/ggml-org/whisper.cpp
cd whisper.cpp
mkdir -p build && cd build
cmake -GNinja -DCMAKE_BUILD_TYPE=Release -DWEBSOCKET=ON -DGGML_CUDA ..
ninja -j$((npoc))

# Run the server
./bin/whisper-websocket-stream --help
```

## Client Integration

Clients can connect to the WebSocket server and send audio data. The server processes the audio and sends transcriptions back through the WebSocket connection.

### Example Client Code (JavaScript)

```javascript
const socket = new WebSocket('ws://localhost:9002');

socket.onopen = () => {
    console.log('Connected to WebSocket server');
};

socket.onmessage = (event) => {
    console.log('Transcription:', event.data);
};

socket.onclose = () => {
    console.log('Disconnected from WebSocket server');
};

// Function to send audio data to the server
function sendAudioData(audioData) {
    socket.send(audioData);
}
```

## Backend Integration

The server can forward transcriptions to a backend service via HTTP requests. Configure the `forward_url` parameter to specify the backend service URL.

## Dependencies
- whisper.cpp
- ixwebsocket for WebSocket communication
- libcurl for HTTP requests
```
