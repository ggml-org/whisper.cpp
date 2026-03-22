package whisper

import low "github.com/ggerganov/whisper.cpp/bindings/go"

// DisableLogs disables all C-side logging from whisper.cpp and ggml.
// Call once early in your program before creating models/contexts.
func DisableLogs() {
	low.DisableLogs()
}
