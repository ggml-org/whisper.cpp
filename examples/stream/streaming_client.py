import asyncio
import websockets
import ffmpeg
import json

# SOCKET_SERVER_URI = "ws://localhost:3000"
SOCKET_SERVER_URI = "ws://localhost:8080"

print("Starting ffmpeg command")
# Convert the audio file to 8kHz and perform mu-law encoding
input_audio = ffmpeg.input("./jfk.wav")
output_audio = ffmpeg.output(
    input_audio, "pipe:", format="mulaw", acodec="pcm_mulaw", ar="8000"
)
out, _ = ffmpeg.run(output_audio, capture_stdout=True, capture_stderr=True)
print("Finished ffmpeg command")

# Convert the audio data to bytes
audio_bytes = out


async def client():
    try:
        # Connect to the specific server
        print("Attempting to connect to server")
        async with websockets.connect(SOCKET_SERVER_URI) as websocket:
            print("Connected to server")
            # Loop playback on the audio file forever until the script is killed
            # while True:
            # Send the audio data as WebSocket messages
            for i in range(0, len(audio_bytes), 1024):
                # We are creating a message in the same format that Twilio uses.
                # The payload is the audio data encoded in base64.
                message = {
                    "event": "media",
                    "sequenceNumber": str(i // 1024 + 1),
                    "media": {
                        "track": "outbound",
                        "chunk": str(i // 1024 + 1),
                        "timestamp": str(i // 1024 * 125),
                        "payload": audio_bytes[i : i + 1024].hex(),
                    },
                    "streamSid": "MZ18ad3ab5a668481ce02b83e7395059f0",
                }
                # This sends the message over the WebSocket connection.
                await websocket.send(json.dumps(message))
                print("Message sent")
    except Exception as e:
        print(f"Exception occurred: {e.__class__}")
        print(e)


asyncio.get_event_loop().run_until_complete(client())
