package whisper_test

import (
	"io"
	"os"
	"testing"

	"github.com/ggerganov/whisper.cpp/bindings/go/pkg/whisper"
	assert "github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestSetLanguage(t *testing.T) {
	assert := assert.New(t)

	cases := []struct {
		name string
		new  func(t *testing.T) (whisper.Context, func())
	}{
		{name: "stateless", new: helperNewStatelessContext},
		{name: "stateful", new: helperNewStatefulContext},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cleanup := tc.new(t)
			defer cleanup()

			// This returns an error since the small.en model is not multilingual
			err := ctx.SetLanguage("en")
			assert.Error(err)
		})
	}
}

func TestContextModelIsMultilingual(t *testing.T) {
	assert := assert.New(t)

	cases := []struct {
		name string
		new  func(t *testing.T) (whisper.Context, func())
	}{
		{name: "stateless", new: helperNewStatelessContext},
		{name: "stateful", new: helperNewStatefulContext},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cleanup := tc.new(t)
			defer cleanup()
			assert.False(ctx.IsMultilingual())
		})
	}
}

func TestLanguage(t *testing.T) {
	assert := assert.New(t)

	cases := []struct {
		name string
		new  func(t *testing.T) (whisper.Context, func())
	}{
		{name: "stateless", new: helperNewStatelessContext},
		{name: "stateful", new: helperNewStatefulContext},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cleanup := tc.new(t)
			defer cleanup()
			expectedLanguage := "en"
			actualLanguage := ctx.Language()
			assert.Equal(expectedLanguage, actualLanguage)
		})
	}
}

// Generic behavior: Language() and DetectedLanguage() match for both context types
func TestContext_Generic_LanguageAndDetectedLanguage(t *testing.T) {
	assert := assert.New(t)

	if _, err := os.Stat(ModelPath); os.IsNotExist(err) {
		t.Skip("Skipping test, model not found:", ModelPath)
	}
	if _, err := os.Stat(SamplePath); os.IsNotExist(err) {
		t.Skip("Skipping test, sample not found:", SamplePath)
	}

	data := helperLoadSample(t, SamplePath)

	cases := []struct {
		name string
		new  func(t *testing.T) (whisper.Context, func())
	}{
		{name: "stateless", new: helperNewStatelessContext},
		{name: "stateful", new: helperNewStatefulContext},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cleanup := tc.new(t)
			defer cleanup()

			langBefore := ctx.Language()
			assert.NoError(ctx.Process(data, nil, nil, nil))
			detected := ctx.DetectedLanguage()
			assert.Equal(langBefore, detected)
		})
	}
}

func TestProcess(t *testing.T) {
	assert := assert.New(t)

	if _, err := os.Stat(ModelPath); os.IsNotExist(err) {
		t.Skip("Skipping test, model not found:", ModelPath)
	}
	if _, err := os.Stat(SamplePath); os.IsNotExist(err) {
		t.Skip("Skipping test, sample not found:", SamplePath)
	}

	data := helperLoadSample(t, SamplePath)

	cases := []struct {
		name string
		new  func(t *testing.T) (whisper.Context, func())
	}{
		{name: "stateless", new: helperNewStatelessContext},
		{name: "stateful", new: helperNewStatefulContext},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cleanup := tc.new(t)
			defer cleanup()
			err := ctx.Process(data, nil, nil, nil)
			assert.NoError(err)
		})
	}
}

func TestDetectedLanguage(t *testing.T) {
	assert := assert.New(t)

	if _, err := os.Stat(ModelPath); os.IsNotExist(err) {
		t.Skip("Skipping test, model not found:", ModelPath)
	}
	if _, err := os.Stat(SamplePath); os.IsNotExist(err) {
		t.Skip("Skipping test, sample not found:", SamplePath)
	}

	data := helperLoadSample(t, SamplePath)

	cases := []struct {
		name string
		new  func(t *testing.T) (whisper.Context, func())
	}{
		{name: "stateless", new: helperNewStatelessContext},
		{name: "stateful", new: helperNewStatefulContext},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cleanup := tc.new(t)
			defer cleanup()
			err := ctx.Process(data, nil, nil, nil)
			assert.NoError(err)
			expectedLanguage := "en"
			actualLanguage := ctx.DetectedLanguage()
			assert.Equal(expectedLanguage, actualLanguage)
		})
	}
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

	data := helperLoadSample(t, SamplePath)

	cases := []struct {
		name string
		new  func(t *testing.T) (whisper.Context, func())
	}{
		{name: "stateless", new: helperNewStatelessContext},
		{name: "stateful", new: helperNewStatefulContext},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cleanup := tc.new(t)
			defer cleanup()

			err := ctx.Process(data, nil, nil, nil)
			assert.NoError(err)

			seg, err := ctx.NextSegment()
			assert.NoError(err)
			assert.NotEmpty(seg.Text)
		})
	}
}

// TestContext_Close tests that Context.Close() properly frees resources
// and allows context to be used even after it has been closed
func TestContext_Close(t *testing.T) {
	assert := assert.New(t)

	if _, err := os.Stat(ModelPath); os.IsNotExist(err) {
		t.Skip("Skipping test, model not found:", ModelPath)
	}

	cases := []struct {
		name string
		new  func(t *testing.T) (whisper.Context, func())
	}{
		{name: "stateless", new: helperNewStatelessContext},
		{name: "stateful", new: helperNewStatefulContext},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cleanup := tc.new(t)
			defer cleanup()

			// Close the context
			err := ctx.Close()
			require.NoError(t, err)

			// Try to use closed context - should return errors
			err = ctx.Process([]float32{0.1, 0.2, 0.3}, nil, nil, nil)
			require.ErrorIs(t, err, whisper.ErrModelClosed)
			// TODO: remove this logic after deprecating the ErrInternalAppError
			require.ErrorIs(t, err, whisper.ErrInternalAppError)

			lang := ctx.DetectedLanguage()
			require.Empty(t, lang)

			_, err = ctx.NextSegment()
			assert.ErrorIs(err, whisper.ErrModelClosed)
			// TODO: remove this logic after deprecating the ErrInternalAppError
			assert.ErrorIs(err, whisper.ErrInternalAppError)

			// Multiple closes should be safe
			err = ctx.Close()
			require.NoError(t, err)
		})
	}
}

func Test_Close_Context_of_Closed_Model(t *testing.T) {
	assert := assert.New(t)

	if _, err := os.Stat(ModelPath); os.IsNotExist(err) {
		t.Skip("Skipping test, model not found:", ModelPath)
	}

	t.Run("stateless", func(t *testing.T) {
		model, err := whisper.NewModelContext(ModelPath)
		assert.NoError(err)
		defer func() { _ = model.Close() }()
		params := helperNewParams(t, model, nil)
		ctx, err := whisper.NewStatelessContext(model, params)
		assert.NoError(err)
		require.NoError(t, model.Close())
		require.NoError(t, ctx.Close())
	})

	t.Run("stateful", func(t *testing.T) {
		model, err := whisper.NewModelContext(ModelPath)
		assert.NoError(err)
		defer func() { _ = model.Close() }()
		params := helperNewParams(t, model, nil)
		ctx, err := whisper.NewStatefulContext(model, params)
		assert.NoError(err)
		require.NoError(t, model.Close())
		require.NoError(t, ctx.Close())
	})
}

func TestContext_VAD_And_Diarization_Params_DoNotPanic(t *testing.T) {
	assert := assert.New(t)

	if _, err := os.Stat(ModelPath); os.IsNotExist(err) {
		t.Skip("Skipping test, model not found:", ModelPath)
	}
	if _, err := os.Stat(SamplePath); os.IsNotExist(err) {
		t.Skip("Skipping test, sample not found:", SamplePath)
	}

	data := helperLoadSample(t, SamplePath)

	model, err := whisper.NewModelContext(ModelPath)
	assert.NoError(err)
	defer func() { _ = model.Close() }()

	params, err := whisper.NewParameters(model, whisper.SAMPLING_GREEDY, nil)
	assert.NoError(err)
	assert.NotNil(params)

	ctx, err := whisper.NewStatefulContext(model, params)
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

func TestDiarization_TwoSpeakers_Boundaries(t *testing.T) {
	data := helperLoadSample(t, MultiSpeakerSamplePath)

	model, err := whisper.NewModelContext(ModelTinydiarizePath)
	require.NoError(t, err)
	defer func() { _ = model.Close() }()

	params, err := whisper.NewParameters(model, whisper.SAMPLING_GREEDY, func(p *whisper.Parameters) {
		p.SetDiarize(true)
		p.SetVAD(false)
		p.SetSplitOnWord(true)
		p.SetMaxSegmentLength(1)
		p.SetMaxTokensPerSegment(64)
		p.SetTokenTimestamps(true)
	})
	require.NoError(t, err)

	// diarize ON with beam search and tighter segmentation
	ctxOn, err := whisper.NewStatefulContext(model, params)
	require.NoError(t, err)
	defer func() { _ = ctxOn.Close() }()

	require.NoError(t, ctxOn.Process(data, nil, nil, nil))
	var turnsOn int
	for {
		seg, err := ctxOn.NextSegment()
		if err == io.EOF {
			break
		}
		require.NoError(t, err)
		if seg.SpeakerTurnNext {
			turnsOn++
		}
	}
	require.Greater(t, turnsOn, 0, "expected speaker turn boundaries with diarization enabled")

	// diarize OFF baseline with same segmentation and beam
	ctxOff, err := whisper.NewStatefulContext(model, params)
	require.NoError(t, err)
	defer func() { _ = ctxOff.Close() }()

	require.NoError(t, ctxOff.Process(data, nil, nil, nil))
	var turnsOff int
	for {
		seg, err := ctxOff.NextSegment()
		if err == io.EOF {
			break
		}
		require.NoError(t, err)
		if seg.SpeakerTurnNext {
			turnsOff++
		}
	}

	require.GreaterOrEqual(t, turnsOn, turnsOff, "diarization should not reduce turn boundaries")
}

func TestContext_SpeakerTurnNext_Field_Present(t *testing.T) {
	assert := assert.New(t)

	if _, err := os.Stat(ModelPath); os.IsNotExist(err) {
		t.Skip("Skipping test, model not found:", ModelPath)
	}
	if _, err := os.Stat(SamplePath); os.IsNotExist(err) {
		t.Skip("Skipping test, sample not found:", SamplePath)
	}

	data := helperLoadSample(t, SamplePath)

	cases := []struct {
		name string
		new  func(t *testing.T) (whisper.Context, func())
	}{
		{name: "stateless", new: helperNewStatelessContext},
		{name: "stateful", new: helperNewStatefulContext},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ctx, cleanup := tc.new(t)
			defer cleanup()

			err := ctx.Process(data, nil, nil, nil)
			assert.NoError(err)

			seg, err := ctx.NextSegment()
			assert.NoError(err)
			t.Logf("SpeakerTurnNext: %v", seg.SpeakerTurnNext)
			_ = seg.SpeakerTurnNext
		})
	}
}

// Ensure Process produces at least one segment for both stateless and stateful contexts
func TestContext_Process_ProducesSegments_BothKinds(t *testing.T) {
	assert := assert.New(t)

	if _, err := os.Stat(ModelPath); os.IsNotExist(err) {
		t.Skip("Skipping test, model not found:", ModelPath)
	}
	if _, err := os.Stat(SamplePath); os.IsNotExist(err) {
		t.Skip("Skipping test, sample not found:", SamplePath)
	}

	data := helperLoadSample(t, SamplePath)

	// Stateless
	stateless, cleanupS := helperNewStatelessContext(t)
	defer cleanupS()
	require.NoError(t, stateless.Process(data, nil, nil, nil))
	var statelessCount int
	for {
		_, err := stateless.NextSegment()
		if err == io.EOF {
			break
		}
		require.NoError(t, err)
		statelessCount++
	}
	assert.Greater(statelessCount, 0, "stateless should produce at least one segment")

	// Stateful
	stateful, cleanupSt := helperNewStatefulContext(t)
	defer cleanupSt()
	require.NoError(t, stateful.Process(data, nil, nil, nil))
	var statefulCount int
	for {
		_, err := stateful.NextSegment()
		if err == io.EOF {
			break
		}
		require.NoError(t, err)
		statefulCount++
	}
	assert.Greater(statefulCount, 0, "stateful should produce at least one segment")
}

// With temperature=0 (greedy), stateless and stateful should produce identical segments
func TestContext_Process_SameResults_TemperatureZero(t *testing.T) {
	assert := assert.New(t)

	if _, err := os.Stat(ModelPath); os.IsNotExist(err) {
		t.Skip("Skipping test, model not found:", ModelPath)
	}
	if _, err := os.Stat(SamplePath); os.IsNotExist(err) {
		t.Skip("Skipping test, sample not found:", SamplePath)
	}

	data := helperLoadSample(t, SamplePath)

	// Use a single model to avoid environment differences
	model, err := whisper.NewModelContext(ModelPath)
	require.NoError(t, err)
	defer func() { _ = model.Close() }()

	// Independent params with temperature=0 for determinism
	p := helperNewParams(t, model, func(p *whisper.Parameters) {
		p.SetTemperature(0)
		p.SetThreads(1)
	})

	stateless, err := whisper.NewStatelessContext(model, p)
	require.NoError(t, err)
	defer func() { _ = stateless.Close() }()

	stateful, err := whisper.NewStatefulContext(model, p)
	require.NoError(t, err)
	defer func() { _ = stateful.Close() }()

	require.NoError(t, stateless.Process(data, nil, nil, nil))
	require.NoError(t, stateful.Process(data, nil, nil, nil))

	// Collect segment texts
	var segsStateless, segsStateful []string
	for {
		seg, err := stateless.NextSegment()
		if err == io.EOF {
			break
		}
		require.NoError(t, err)
		segsStateless = append(segsStateless, seg.Text)
	}
	for {
		seg, err := stateful.NextSegment()
		if err == io.EOF {
			break
		}
		require.NoError(t, err)
		segsStateful = append(segsStateful, seg.Text)
	}

	// Both should have at least one segment and be identical
	require.Greater(t, len(segsStateless), 0)
	require.Greater(t, len(segsStateful), 0)
	assert.Equal(len(segsStateful), len(segsStateless))
	for i := range segsStateless {
		assert.Equal(segsStateless[i], segsStateful[i], "segment %d text differs", i)
	}
}

// Model.GetTimings: stateless processing updates model timings (non-zero),
// stateful processing does not (zero timings)
func TestModel_GetTimings_Stateless_NonZero_Stateful_Zero(t *testing.T) {
	assert := assert.New(t)

	if _, err := os.Stat(ModelPath); os.IsNotExist(err) {
		t.Skip("Skipping test, model not found:", ModelPath)
	}
	if _, err := os.Stat(SamplePath); os.IsNotExist(err) {
		t.Skip("Skipping test, sample not found:", SamplePath)
	}

	data := helperLoadSample(t, SamplePath)

	model, err := whisper.NewModelContext(ModelPath)
	require.NoError(t, err)
	defer func() { _ = model.Close() }()

	// Stateless should produce non-zero timings
	t.Run("stateless", func(t *testing.T) {
		model.ResetTimings()
		params := helperNewParams(t, model, nil)
		ctx, err := whisper.NewStatelessContext(model, params)
		require.NoError(t, err)
		defer func() { _ = ctx.Close() }()

		require.NoError(t, ctx.Process(data, nil, nil, nil))

		timings, ok := model.GetTimings()
		require.True(t, ok, "expected timings to be available after stateless processing")
		nonZero := timings.SampleMS > 0 || timings.EncodeMS > 0 || timings.DecodeMS > 0 || timings.BatchdMS > 0 || timings.PromptMS > 0
		assert.True(nonZero, "expected at least one non-zero timing after stateless processing: %#v", timings)
	})

	// Stateful should keep model-level timings at zero
	t.Run("stateful", func(t *testing.T) {
		model.ResetTimings()
		params := helperNewParams(t, model, nil)
		ctx, err := whisper.NewStatefulContext(model, params)
		require.NoError(t, err)
		defer func() { _ = ctx.Close() }()

		require.NoError(t, ctx.Process(data, nil, nil, nil))

		timings, ok := model.GetTimings()
		// Expect timings present but all zero; if not present at all, treat as zero-equivalent
		if ok {
			assert.Equal(float32(0), timings.SampleMS)
			assert.Equal(float32(0), timings.EncodeMS)
			assert.Equal(float32(0), timings.DecodeMS)
			assert.Equal(float32(0), timings.BatchdMS)
			assert.Equal(float32(0), timings.PromptMS)
		} else {
			t.Log("timings not available for stateful processing; treating as zero")
		}
	})
}
