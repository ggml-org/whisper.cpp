import asyncio
import websockets
import time
import json
import wave
import numpy as np

# --- CONFIGURATION ---
SERVER_IP = "IPADDRESS" # Your server's public IP
AUDIO_FILE = "WAV FILE 16BIT"     # Must be in the same directory
CONCURRENT_CLIENTS = 1
# ---------------------

SERVER_URL = f"ws://{SERVER_IP}/stream" # Connect to NGINX endpoint
CHUNK_SECONDS = 0.2     # Send audio chunks every 200ms

async def load_audio_data(file_path):
    """Loads a WAV file and ensures it's 16-bit 16kHz mono PCM."""
    # (Same as previous script - ensures correct audio format)
    try:
        with wave.open(file_path, "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            audio_data = wf.readframes(n_frames)
            if framerate != 16000: raise ValueError(f"Need 16kHz")
            if sampwidth != 2: raise ValueError(f"Need 16-bit")
            if n_channels == 2:
                samples = np.frombuffer(audio_data, dtype=np.int16)
                audio_data = samples[::2].tobytes()
            print(f"Loaded audio: {len(audio_data)} bytes, {n_frames / framerate:.2f} seconds")
            return audio_data, framerate
    except FileNotFoundError:
        print(f"Error: Audio file '{file_path}' not found.")
        return None, None
    except Exception as e: print(f"Error loading audio: {e}"); return None, None

async def receive_transcriptions(websocket, metrics):
    """Task to handle receiving messages from the server."""
    start_time = time.time()
    first_message_received = False
    # NO accumulated_text NEEDED
    
    try:
        async for message in websocket:
            if not first_message_received:
                metrics["ttft_ms"] = (time.time() - start_time) * 1000
                first_message_received = True

            try:
                response = json.loads(message)
                segment_text = response.get('text', '')

                # --- *** CORRECTED LOGIC *** ---
                # The server sends the *full* transcription every time.
                # We simply store the latest version we receive.
                
                # Store the current text. This will be overwritten by the next
                # intermediate message, which is fine.
                metrics["final_transcription"] = segment_text.strip()

                if response.get("is_final", False):
                    # This is the FINAL message. Store it one last time and exit.
                    print(f"Client {metrics['id']} Final: {metrics['final_transcription']}")
                    break # We are done
                else:
                    # This is an INTERMEDIATE message.
                    print(f"Client {metrics['id']} Intermediate: {metrics['final_transcription']}")
                # --- *** END CORRECTED LOGIC *** ---

            except json.JSONDecodeError:
                print(f"Client {metrics['id']} RX non-JSON: {message}")
                break
            except Exception as e:
                print(f"Client {metrics['id']} error processing message: {e}")
                metrics["error"] = f"Processing error: {e}"
                break

    except websockets.exceptions.ConnectionClosedOK:
        pass # Normal closure
    except websockets.exceptions.ConnectionClosedError as e:
        metrics["error"] = f"Connection closed error: {e}"
    except Exception as e:
        metrics["error"] = f"Receive loop error: {e}"

    # Fallback in case loop breaks before 'is_final' is received
    # This is fine, as metrics["final_transcription"] will hold the *last*
    # intermediate text we received.
    if response.get("is_final") is None:
         print(f"Client {metrics['id']} Ended Prematurely, Last Text: {metrics.get('final_transcription', '')}")

async def run_client(client_id, audio_data, framerate):
    """Simulates a single client connecting and streaming audio chunks."""
    chunk_size = int(framerate * CHUNK_SECONDS * 2) # Bytes per chunk

    metrics = {
        "id": client_id,
        "ttft_ms": None,
        "total_time_ms": None,
        "error": None,
        "final_transcription": None
    }

    start_time = time.time()

    try:
        async with websockets.connect(SERVER_URL, ping_interval=None) as websocket: # ping_interval=None can help stability
            # Start a background task to receive messages
            receive_task = asyncio.create_task(receive_transcriptions(websocket, metrics))

            # 1. Stream the audio in chunks
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i : i + chunk_size]
                await websocket.send(chunk)
                await asyncio.sleep(CHUNK_SECONDS) # Simulate real-time capture

            # 2. Send end-of-stream signal (might vary by server implementation)
            # Common methods: Send empty bytes, or a specific JSON message.
            # Check the fork's documentation/code for how it signals end-of-stream.
            # Let's assume empty bytes for now:
            await websocket.send(b"")

            # Wait for the receiving task to finish (gets final transcription or error)
            await asyncio.wait_for(receive_task, timeout=30.0) # 30 sec timeout

    except websockets.exceptions.InvalidStatusCode as e:
         metrics["error"] = f"Connection failed: HTTP {e.status_code}" # Catch 404s etc.
    except Exception as e:
        metrics["error"] = f"Connection/Send error: {e}"
        if 'receive_task' in locals() and not receive_task.done():
             receive_task.cancel() # Clean up receiver if connect/send failed

    metrics["total_time_ms"] = (time.time() - start_time) * 1000
    return metrics

async def main():
    # (Same audio loading as before)
    audio_data, framerate = await load_audio_data(AUDIO_FILE)
    if not audio_data: return

    print(f"--- Starting WebSocket Streaming Load Test ---")
    print(f"Target: {SERVER_URL}")
    print(f"Simulating: {CONCURRENT_CLIENTS} clients")
    print(f"Audio: {AUDIO_FILE}")
    print("---------------------------------------------")

    start_total_time = time.time()
    tasks = [run_client(i, audio_data, framerate) for i in range(CONCURRENT_CLIENTS)]
    results = await asyncio.gather(*tasks)
    end_total_time = time.time()

    successes = [r for r in results if r["error"] is None and r["final_transcription"] is not None]
    errors = [r for r in results if r["error"] is not None or r["final_transcription"] is None]

    print(f"\n--- Test Complete ---")
    print(f"Total time for all clients: {(end_total_time - start_total_time):.2f}s")
    print(f"Successful clients: {len(successes)} / {CONCURRENT_CLIENTS}")
    print(f"Failed clients: {len(errors)}")

    if successes:
        valid_ttft = [r["ttft_ms"] for r in successes if r["ttft_ms"] is not None]
        all_total_times = [r["total_time_ms"] for r in successes] # Get all total times

        avg_ttft = sum(valid_ttft) / len(valid_ttft) if valid_ttft else 0
        avg_total = sum(all_total_times) / len(successes) # Avg total time
        max_ttft = max(valid_ttft) if valid_ttft else 0
        max_total = max(all_total_times) # Max total time

        # --- ADD MINIMUM CALCULATIONS ---
        min_ttft = min(valid_ttft) if valid_ttft else 0
        min_total = min(all_total_times)
        # --- END ADDITIONS ---


        print("\n--- Latency (Successful Clients) ---")
        print(f"Avg. Time to First Transcription (TTFT): {avg_ttft:.0f} ms")
        print(f"Avg. Total Transcription Time: {avg_total:.0f} ms")
        print(f"Max TTFT: {max_ttft:.0f} ms")
        print(f"Max Total Time: {max_total:.0f} ms")
        # --- ADD MINIMUM PRINT STATEMENTS ---
        print(f"Min TTFT: {min_ttft:.0f} ms")
        print(f"Min Total Time: {min_total:.0f} ms")
        # --- END ADDITIONS ---


    if errors:
        print("\n--- Errors ---")
        for i, r in enumerate(errors[:5]):
            print(f"  Client {r['id']}: {r['error']} (Total time: {r['total_time_ms']:.0f}ms)")

if __name__ == "__main__":
    asyncio.run(main())
