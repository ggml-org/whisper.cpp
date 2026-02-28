package whisper_test

import (
	"errors"
	"os"
	"runtime"
	"sync"
	"testing"
	"time"

	// Packages
	whisper "github.com/ggerganov/whisper.cpp/bindings/go"
	wav "github.com/go-audio/wav"
	assert "github.com/stretchr/testify/assert"
)

const (
	ModelPath  = "models/ggml-tiny.en.bin"
	SamplePath = "samples/jfk.wav"
)

func TestMain(m *testing.M) {
	// whisper.DisableLogs() // temporarily disabled to see error messages
	os.Exit(m.Run())
}

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
	defer func() { _ = fh.Close() }()

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
	defer func() { _ = fh.Close() }()

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

func Test_Whisper_State_Init_Free(t *testing.T) {
	assert := assert.New(t)
	if _, err := os.Stat(ModelPath); os.IsNotExist(err) {
		t.Skip("Skipping test, model not found:", ModelPath)
	}

	ctx := whisper.Whisper_init(ModelPath)
	assert.NotNil(ctx)
	defer ctx.Whisper_free()

	state := ctx.Whisper_init_state()
	assert.NotNil(state)
	state.Whisper_free_state()
}

func Test_Whisper_Full_With_State(t *testing.T) {
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
	defer func() { _ = fh.Close() }()

	// Read samples
	d := wav.NewDecoder(fh)
	buf, err := d.FullPCMBuffer()
	assert.NoError(err)
	data := buf.AsFloat32Buffer().Data

	ctx := whisper.Whisper_init(ModelPath)
	assert.NotNil(ctx)
	defer ctx.Whisper_free()

	state := ctx.Whisper_init_state()
	assert.NotNil(state)
	defer state.Whisper_free_state()

	params := ctx.Whisper_full_default_params(whisper.SAMPLING_GREEDY)
	// Run using state
	err = ctx.Whisper_full_with_state(state, params, data, nil, nil, nil)
	assert.NoError(err)

	// Validate results are stored in state
	nSegments := ctx.Whisper_full_n_segments_from_state(state)
	assert.GreaterOrEqual(nSegments, 1)
	text := ctx.Whisper_full_get_segment_text_from_state(state, 0)
	assert.NotEmpty(text)
}

func Test_Whisper_Lang_Auto_Detect_With_State(t *testing.T) {
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
	defer func() { _ = fh.Close() }()

	// Read samples
	d := wav.NewDecoder(fh)
	buf, err := d.FullPCMBuffer()
	assert.NoError(err)
	data := buf.AsFloat32Buffer().Data

	ctx := whisper.Whisper_init(ModelPath)
	assert.NotNil(ctx)
	defer ctx.Whisper_free()

	state := ctx.Whisper_init_state()
	assert.NotNil(state)
	defer state.Whisper_free_state()

	threads := runtime.NumCPU()
	// Prepare mel into state then detect
	assert.NoError(ctx.Whisper_pcm_to_mel_with_state(state, data, threads))
	probs, err := ctx.Whisper_lang_auto_detect_with_state(state, 0, threads)
	assert.NoError(err)
	assert.Equal(whisper.Whisper_lang_max_id()+1, len(probs))
}

func Test_Whisper_Concurrent_With_State(t *testing.T) {
	assert := assert.New(t)
	if _, err := os.Stat(ModelPath); os.IsNotExist(err) {
		t.Skip("Skipping test, model not found:", ModelPath)
	}
	if _, err := os.Stat(SamplePath); os.IsNotExist(err) {
		t.Skip("Skipping test, sample not found:", SamplePath)
	}

	// Load audio once
	fh, err := os.Open(SamplePath)
	assert.NoError(err)
	defer func() { _ = fh.Close() }()
	dec := wav.NewDecoder(fh)
	buf, err := dec.FullPCMBuffer()
	assert.NoError(err)
	data := buf.AsFloat32Buffer().Data

	ctx := whisper.Whisper_init(ModelPath)
	assert.NotNil(ctx)
	defer ctx.Whisper_free()

	// Each goroutine has its own state
	state1 := ctx.Whisper_init_state()
	state2 := ctx.Whisper_init_state()
	assert.NotNil(state1)
	assert.NotNil(state2)
	defer state1.Whisper_free_state()
	defer state2.Whisper_free_state()

	params := ctx.Whisper_full_default_params(whisper.SAMPLING_GREEDY)

	var wg sync.WaitGroup
	var mu sync.Mutex // guard calls into shared ctx, per upstream note not thread-safe for same context
	errs := make(chan error, 2)

	worker := func(state *whisper.State) {
		defer wg.Done()
		mu.Lock()
		err := ctx.Whisper_full_with_state(state, params, data, nil, nil, nil)
		if err == nil {
			n := ctx.Whisper_full_n_segments_from_state(state)
			if n <= 0 {
				err = errors.New("no segments")
			} else {
				_ = ctx.Whisper_full_get_segment_text_from_state(state, 0)
			}
		}
		mu.Unlock()
		errs <- err
	}

	wg.Add(2)
	go worker(state1)
	go worker(state2)
	wg.Wait()
	close(errs)

	for e := range errs {
		assert.NoError(e)
	}
}
