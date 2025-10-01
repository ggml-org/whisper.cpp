package whisper_test

import (
	"os"
	"testing"
)

const (
	ModelPath              = "../../models/ggml-small.en.bin"
	ModelTinydiarizePath   = "../../models/ggml-small.en-tdrz.bin"
	SamplePath             = "../../samples/jfk.wav"
	MultiSpeakerSamplePath = "../../samples/a13.wav"
)

func TestMain(m *testing.M) {
	// whisper.DisableLogs()
	os.Exit(m.Run())
}
