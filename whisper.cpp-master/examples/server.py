import http.server
import socketserver
import os
import sys
from pathlib import Path
import urllib.parse

SCRIPT_DIR = Path(__file__).parent.absolute()
DIRECTORY = os.path.join(SCRIPT_DIR, "../build-em/bin")
DIRECTORY = os.path.abspath(DIRECTORY)

# The context root we want for all applications
CONTEXT_ROOT = "/whisper.cpp"

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def do_GET(self):
        # Redirect root to the context root
        if self.path == '/':
            self.send_response(302)
            self.send_header('Location', CONTEXT_ROOT + '/')
            self.end_headers()
            return

        # Handle requests under the context root
        if self.path.startswith(CONTEXT_ROOT):
            # Remove the context root prefix to get the actual path
            actual_path = self.path[len(CONTEXT_ROOT):]

            if not actual_path:
                self.send_response(302)
                self.send_header('Location', CONTEXT_ROOT + '/')
                self.end_headers()
                return

            if '.worker.js' in actual_path:
                worker_file = os.path.basename(actual_path)
                worker_path = os.path.join(DIRECTORY, worker_file)

                if os.path.exists(worker_path):
                    print(f"Found worker file: {worker_path}")
                    self.path = '/' + worker_file
                else:
                    print(f"Worker file not found: {worker_path}")

            elif actual_path == '/':
                self.path = '/whisper.wasm/index.html'
            elif actual_path.startswith('/bench.wasm/') or actual_path.startswith('/command.wasm/') or actual_path.startswith('/stream.wasm/'):
                # Keep the path as is, just remove the context root
                self.path = actual_path
            # For all other paths under the context root
            else:
                # Check if this is a request to a file in whisper.wasm
                potential_file = os.path.join(DIRECTORY, 'whisper.wasm', actual_path.lstrip('/'))
                if os.path.exists(potential_file) and not os.path.isdir(potential_file):
                    self.path = '/whisper.wasm' + actual_path
                else:
                    # Try to resolve the file from the base directory
                    potential_file = os.path.join(DIRECTORY, actual_path.lstrip('/'))
                    if os.path.exists(potential_file):
                        self.path = actual_path

        # For direct requests to worker files (without context root as these
        # are in the build-em/bin directory
        elif '.worker.js' in self.path:
            worker_file = os.path.basename(self.path)
            worker_path = os.path.join(DIRECTORY, worker_file)

            if os.path.exists(worker_path):
                self.path = '/' + worker_file

        # Handle coi-serviceworker.js separately
        if 'coi-serviceworker.js' in self.path:
            worker_file = "coi-serviceworker.js"
            worker_path = os.path.join(SCRIPT_DIR, worker_file)
            if os.path.exists(worker_path):
                self.send_response(200)
                self.send_header('Content-type', 'application/javascript')
                self.end_headers()
                with open(worker_path, 'rb') as file:
                    self.wfile.write(file.read())
                return
            else:
                print(f"Warning: Could not find {worker_path}")

        return super().do_GET()

    def end_headers(self):
        # Add required headers for SharedArrayBuffer
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

PORT = 8000

# Enable address reuse
class CustomServer(socketserver.TCPServer):
    allow_reuse_address = True

try:
    with CustomServer(("", PORT), CustomHTTPRequestHandler) as httpd:
        print(f"Serving directory '{DIRECTORY}' at http://localhost:{PORT}")
        print(f"Application context root: http://localhost:{PORT}{CONTEXT_ROOT}/")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")
            # Force complete exit
            sys.exit(0)
except OSError as e:
    print(f"Error: {e}")
    sys.exit(1)
