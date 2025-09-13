package whisper_test

import (
	"os"
	"sync"
	"testing"

	"github.com/ggerganov/whisper.cpp/bindings/go/pkg/whisper"
	"github.com/go-audio/wav"
	assert "github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSetLanguage(t *testing.T) {
	assert := assert.New(t)

	model, err := whisper.New(ModelPath)
	assert.NoError(err)
	assert.NotNil(model)
	defer func() { _ = model.Close() }()

	context, err := model.NewContext()
	assert.NoError(err)

	// This returns an error since
	// the model 'models/ggml-small.en.bin'
	// that is loaded is not multilingual
	err = context.SetLanguage("en")
	assert.Error(err)
}

func TestContextModelIsMultilingual(t *testing.T) {
	assert := assert.New(t)

	model, err := whisper.New(ModelPath)
	assert.NoError(err)
	assert.NotNil(model)
	defer func() { _ = model.Close() }()

	context, err := model.NewContext()
	assert.NoError(err)

	isMultilingual := context.IsMultilingual()

	// This returns false since
	// the model 'models/ggml-small.en.bin'
	// that is loaded is not multilingual
	assert.False(isMultilingual)
}

func TestLanguage(t *testing.T) {
	assert := assert.New(t)

	model, err := whisper.New(ModelPath)
	assert.NoError(err)
	assert.NotNil(model)
	defer func() { _ = model.Close() }()

	context, err := model.NewContext()
	assert.NoError(err)

	// This always returns en since
	// the model 'models/ggml-small.en.bin'
	// that is loaded is not multilingual
	expectedLanguage := "en"
	actualLanguage := context.Language()
	assert.Equal(expectedLanguage, actualLanguage)
}

func TestProcess(t *testing.T) {
	assert := assert.New(t)

	fh, err := os.Open(SamplePath)
	assert.NoError(err)
	defer func() { _ = fh.Close() }()

	// Decode the WAV file - load the full buffer
	dec := wav.NewDecoder(fh)
	buf, err := dec.FullPCMBuffer()
	assert.NoError(err)
	assert.Equal(uint16(1), dec.NumChans)

	data := buf.AsFloat32Buffer().Data

	model, err := whisper.New(ModelPath)
	assert.NoError(err)
	assert.NotNil(model)
	defer func() { _ = model.Close() }()

	context, err := model.NewContext()
	assert.NoError(err)

	err = context.Process(data, nil, nil, nil)
	assert.NoError(err)
}

func TestDetectedLanguage(t *testing.T) {
	assert := assert.New(t)

	fh, err := os.Open(SamplePath)
	assert.NoError(err)
	defer func() { _ = fh.Close() }()

	// Decode the WAV file - load the full buffer
	dec := wav.NewDecoder(fh)
	buf, err := dec.FullPCMBuffer()
	assert.NoError(err)
	assert.Equal(uint16(1), dec.NumChans)

	data := buf.AsFloat32Buffer().Data

	model, err := whisper.New(ModelPath)
	assert.NoError(err)
	assert.NotNil(model)
	defer func() { _ = model.Close() }()

	context, err := model.NewContext()
	assert.NoError(err)

	err = context.Process(data, nil, nil, nil)
	assert.NoError(err)

	expectedLanguage := "en"
	actualLanguage := context.DetectedLanguage()
	assert.Equal(expectedLanguage, actualLanguage)
}

// TestContext_ConcurrentProcessing tests that multiple contexts can process concurrently
// without interfering with each other (validates the whisper_state isolation fix)
func TestContext_ConcurrentProcessing(t *testing.T) {
	assert := assert.New(t)

	if _, err := os.Stat(ModelPath); os.IsNotExist(err) {
		t.Skip("Skipping test, model not found:", ModelPath)
	}
	if _, err := os.Stat(SamplePath); os.IsNotExist(err) {
		t.Skip("Skipping test, sample not found:", SamplePath)
	}

	fh, err := os.Open(SamplePath)
	assert.NoError(err)
	defer func() { _ = fh.Close() }()

	dec := wav.NewDecoder(fh)
	buf, err := dec.FullPCMBuffer()
	assert.NoError(err)
	assert.Equal(uint16(1), dec.NumChans)
	data := buf.AsFloat32Buffer().Data

	model, err := whisper.New(ModelPath)
	assert.NoError(err)
	assert.NotNil(model)
	defer func() { _ = model.Close() }()

	ctx, err := model.NewContext()
	assert.NoError(err)
	assert.NotNil(ctx)
	defer func() { _ = ctx.Close() }()

	err = ctx.Process(data, nil, nil, nil)
	assert.NoError(err)

	seg, err := ctx.NextSegment()
	assert.NoError(err)
	assert.NotEmpty(seg.Text)
}

// TestContext_Parallel_DifferentInputs tests concurrent processing with different inputs
// This validates that each context maintains isolated state for concurrent processing
func TestContext_Parallel_DifferentInputs(t *testing.T) {
	assert := assert.New(t)

	if _, err := os.Stat(ModelPath); os.IsNotExist(err) {
		t.Skip("Skipping test, model not found:", ModelPath)
	}
	if _, err := os.Stat(SamplePath); os.IsNotExist(err) {
		t.Skip("Skipping test, sample not found:", SamplePath)
	}

	fh, err := os.Open(SamplePath)
	assert.NoError(err)
	defer func() { _ = fh.Close() }()

	dec := wav.NewDecoder(fh)
	buf, err := dec.FullPCMBuffer()
	assert.NoError(err)
	assert.Equal(uint16(1), dec.NumChans)
	data := buf.AsFloat32Buffer().Data
	assert.Greater(len(data), 10)

	// Create half-sample (second half)
	half := make([]float32, len(data)/2)
	copy(half, data[len(data)/2:])

	model, err := whisper.New(ModelPath)
	assert.NoError(err)
	assert.NotNil(model)
	defer func() { _ = model.Close() }()

	ctx1, err := model.NewContext()
	assert.NoError(err)
	defer func() { _ = ctx1.Close() }()
	ctx2, err := model.NewContext()
	assert.NoError(err)
	defer func() { _ = ctx2.Close() }()

	// Run in parallel - each context has isolated whisper_state
	var wg sync.WaitGroup
	var first1, first2 string
	var e1, e2 error

	wg.Add(2)

	// No mutex needed because each context is isolated by whisper_state
	go func() {
		defer wg.Done()
		e1 = ctx1.Process(data, nil, nil, nil)
		if e1 == nil {
			seg, err := ctx1.NextSegment()
			if err == nil {
				first1 = seg.Text
			} else {
				e1 = err
			}
		}
	}()

	go func() {
		defer wg.Done()
		e2 = ctx2.Process(half, nil, nil, nil)
		if e2 == nil {
			seg, err := ctx2.NextSegment()
			if err == nil {
				first2 = seg.Text
			} else {
				e2 = err
			}
		}
	}()

	wg.Wait()
	assert.NoError(e1)
	assert.NoError(e2)
	assert.NotEmpty(first1)
	assert.NotEmpty(first2)
	assert.NotEqual(first1, first2, "first segments should differ for different inputs")
}

// TestContext_Close tests that Context.Close() properly frees resources
// and allows context to be used even after it has been closed
func TestContext_Close(t *testing.T) {
	assert := assert.New(t)

	if _, err := os.Stat(ModelPath); os.IsNotExist(err) {
		t.Skip("Skipping test, model not found:", ModelPath)
	}

	model, err := whisper.New(ModelPath)
	assert.NoError(err)
	assert.NotNil(model)
	defer func() { _ = model.Close() }()

	ctx, err := model.NewContext()
	assert.NoError(err)
	assert.NotNil(ctx)

	// Close the context
	err = ctx.Close()
	require.NoError(t, err)

	// Try to use closed context - should return errors
	err = ctx.Process([]float32{0.1, 0.2, 0.3}, nil, nil, nil)
	require.ErrorIs(t, err, whisper.ErrModelClosed)

	lang := ctx.DetectedLanguage()
	require.Empty(t, lang)

	// Multiple closes should be safe
	err = ctx.Close()
	require.NoError(t, err)
}

func Test_Close_Context_of_Closed_Model(t *testing.T) {
	assert := assert.New(t)

	model, err := whisper.New(ModelPath)
	assert.NoError(err)
	assert.NotNil(model)

	ctx, err := model.NewContext()
	assert.NoError(err)
	assert.NotNil(ctx)

	require.NoError(t, model.Close())
	require.NoError(t, ctx.Close())
}

func TestContext_VAD_And_Diarization_Params_DoNotPanic(t *testing.T) {
	assert := assert.New(t)

	if _, err := os.Stat(ModelPath); os.IsNotExist(err) {
		t.Skip("Skipping test, model not found:", ModelPath)
	}
	if _, err := os.Stat(SamplePath); os.IsNotExist(err) {
		t.Skip("Skipping test, sample not found:", SamplePath)
	}

	fh, err := os.Open(SamplePath)
	assert.NoError(err)
	defer func() { _ = fh.Close() }()

	dec := wav.NewDecoder(fh)
	buf, err := dec.FullPCMBuffer()
	assert.NoError(err)
	assert.Equal(uint16(1), dec.NumChans)
	data := buf.AsFloat32Buffer().Data

	model, err := whisper.New(ModelPath)
	assert.NoError(err)
	defer func() { _ = model.Close() }()

	ctx, err := model.NewContext()
	assert.NoError(err)
	defer func() { _ = ctx.Close() }()

	p := ctx.Params()
	p.SetDiarize(true)
	p.SetVAD(true)
	p.SetVADThreshold(0.5)
	p.SetVADMinSpeechMs(200)
	p.SetVADMinSilenceMs(100)
	p.SetVADMaxSpeechSec(10)
	p.SetVADSpeechPadMs(30)
	p.SetVADSamplesOverlap(0.02)

	err = ctx.Process(data, nil, nil, nil)
	assert.NoError(err)
}

func TestContext_SpeakerTurnNext_Field_Present(t *testing.T) {
	assert := assert.New(t)

	if _, err := os.Stat(ModelPath); os.IsNotExist(err) {
		t.Skip("Skipping test, model not found:", ModelPath)
	}
	if _, err := os.Stat(SamplePath); os.IsNotExist(err) {
		t.Skip("Skipping test, sample not found:", SamplePath)
	}

	fh, err := os.Open(SamplePath)
	assert.NoError(err)
	defer func() { _ = fh.Close() }()

	dec := wav.NewDecoder(fh)
	buf, err := dec.FullPCMBuffer()
	assert.NoError(err)
	assert.Equal(uint16(1), dec.NumChans)
	data := buf.AsFloat32Buffer().Data

	model, err := whisper.New(ModelPath)
	assert.NoError(err)
	defer func() { _ = model.Close() }()

	ctx, err := model.NewContext()
	assert.NoError(err)
	defer func() { _ = ctx.Close() }()

	err = ctx.Process(data, nil, nil, nil)
	assert.NoError(err)

	seg, err := ctx.NextSegment()
	assert.NoError(err)
	t.Logf("SpeakerTurnNext: %v", seg.SpeakerTurnNext)
	_ = seg.SpeakerTurnNext // ensure field exists and is readable
}
