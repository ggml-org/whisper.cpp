package whisper_test

import (
	"os"
	"testing"
)

const (
	ModelPath  = "../../models/ggml-tiny.en.bin"
	SamplePath = "../../samples/jfk.wav"
)

func TestMain(m *testing.M) {
	// whisper.DisableLogs()
	os.Exit(m.Run())
}
