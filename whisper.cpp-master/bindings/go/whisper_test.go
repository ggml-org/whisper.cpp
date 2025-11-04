package whisper_test

import (
	"os"
	"runtime"
	"testing"
	"time"

	// Packages
	whisper "github.com/ggerganov/whisper.cpp/bindings/go"
	wav "github.com/go-audio/wav"
	assert "github.com/stretchr/testify/assert"
)

const (
	ModelPath  = "models/ggml-small.en.bin"
	SamplePath = "samples/jfk.wav"
)

func Test_Whisper_000(t *testing.T) {
	assert := assert.New(t)
	if _, err := os.Stat(ModelPath); os.IsNotExist(err) {
		t.Skip("Skipping test, model not found:", ModelPath)
	}
	ctx := whisper.Whisper_init(ModelPath)
	assert.NotNil(ctx)
	ctx.Whisper_free()
}

func Test_Whisper_001(t *testing.T) {
	assert := assert.New(t)
	if _, err := os.Stat(ModelPath); os.IsNotExist(err) {
		t.Skip("Skipping test, model not found:", ModelPath)
	}
	if _, err := os.Stat(SamplePath); os.IsNotExist(err) {
		t.Skip("Skipping test, sample not found:", SamplePath)
	}

	// Open samples
	fh, err := os.Open(SamplePath)
	assert.NoError(err)
	defer fh.Close()

	// Read samples
	d := wav.NewDecoder(fh)
	buf, err := d.FullPCMBuffer()
	assert.NoError(err)

	// Run whisper
	ctx := whisper.Whisper_init(ModelPath)
	assert.NotNil(ctx)
	defer ctx.Whisper_free()
	params := ctx.Whisper_full_default_params(whisper.SAMPLING_GREEDY)
	data := buf.AsFloat32Buffer().Data
	err = ctx.Whisper_full(params, data, nil, nil, nil)
	assert.NoError(err)

	// Print out tokens
	num_segments := ctx.Whisper_full_n_segments()
	assert.GreaterOrEqual(num_segments, 1)
	for i := 0; i < num_segments; i++ {
		str := ctx.Whisper_full_get_segment_text(i)
		assert.NotEmpty(str)
		t0 := time.Duration(ctx.Whisper_full_get_segment_t0(i)) * time.Millisecond
		t1 := time.Duration(ctx.Whisper_full_get_segment_t1(i)) * time.Millisecond
		t.Logf("[%6s->%-6s] %q", t0, t1, str)
	}
}

func Test_Whisper_002(t *testing.T) {
	assert := assert.New(t)
	for i := 0; i < whisper.Whisper_lang_max_id(); i++ {
		str := whisper.Whisper_lang_str(i)
		assert.NotEmpty(str)
		t.Log(str)
	}
}

func Test_Whisper_003(t *testing.T) {
	threads := runtime.NumCPU()
	assert := assert.New(t)
	if _, err := os.Stat(ModelPath); os.IsNotExist(err) {
		t.Skip("Skipping test, model not found:", ModelPath)
	}
	if _, err := os.Stat(SamplePath); os.IsNotExist(err) {
		t.Skip("Skipping test, sample not found:", SamplePath)
	}

	// Open samples
	fh, err := os.Open(SamplePath)
	assert.NoError(err)
	defer fh.Close()

	// Read samples
	d := wav.NewDecoder(fh)
	buf, err := d.FullPCMBuffer()
	assert.NoError(err)

	// Make the model
	ctx := whisper.Whisper_init(ModelPath)
	assert.NotNil(ctx)
	defer ctx.Whisper_free()

	// Get MEL
	assert.NoError(ctx.Whisper_pcm_to_mel(buf.AsFloat32Buffer().Data, threads))

	// Get Languages
	languages, err := ctx.Whisper_lang_auto_detect(0, threads)
	assert.NoError(err)
	for i, p := range languages {
		t.Logf("%s: %f", whisper.Whisper_lang_str(i), p)
	}
}
