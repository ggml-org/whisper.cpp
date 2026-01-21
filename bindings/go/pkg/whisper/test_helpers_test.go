package whisper_test

import (
	"os"
	"testing"

	whisper "github.com/ggerganov/whisper.cpp/bindings/go/pkg/whisper"
	wav "github.com/go-audio/wav"
)

func helperLoadSample(tb testing.TB, path string) []float32 {
	tb.Helper()
	fh, err := os.Open(path)
	if err != nil {
		tb.Fatalf("open sample: %v", err)
	}
	defer func() { _ = fh.Close() }()

	dec := wav.NewDecoder(fh)
	buf, err := dec.FullPCMBuffer()
	if err != nil {
		tb.Fatalf("decode wav: %v", err)
	}
	if dec.NumChans != 1 {
		tb.Fatalf("expected mono wav, got channels=%d", dec.NumChans)
	}
	return buf.AsFloat32Buffer().Data
}

// helperLoadSampleWithMeta loads wav and returns samples with sample rate and channels
func helperLoadSampleWithMeta(tb testing.TB, path string) ([]float32, int, int) {
	tb.Helper()
	fh, err := os.Open(path)
	if err != nil {
		tb.Fatalf("open sample: %v", err)
	}
	defer func() { _ = fh.Close() }()

	dec := wav.NewDecoder(fh)
	buf, err := dec.FullPCMBuffer()
	if err != nil {
		tb.Fatalf("decode wav: %v", err)
	}
	if dec.NumChans != 1 {
		tb.Fatalf("expected mono wav, got channels=%d", dec.NumChans)
	}
	return buf.AsFloat32Buffer().Data, int(dec.SampleRate), int(dec.NumChans)
}

func helperNewModel(t *testing.T) (whisper.Model, func()) {
	t.Helper()
	if _, err := os.Stat(ModelPath); os.IsNotExist(err) {
		t.Skip("Skipping test, model not found:", ModelPath)
	}
	model, err := whisper.New(ModelPath)
	if err != nil {
		t.Fatalf("load model: %v", err)
	}
	return model, func() { _ = model.Close() }
}

func helperNewModelContext(t *testing.T) (*whisper.ModelContext, func()) {
	t.Helper()
	if _, err := os.Stat(ModelPath); os.IsNotExist(err) {
		t.Skip("Skipping test, model not found:", ModelPath)
	}
	model, err := whisper.NewModelContext(ModelPath)
	if err != nil {
		t.Fatalf("load model ctx: %v", err)
	}
	return model, func() { _ = model.Close() }
}

func helperNewParams(t *testing.T, model *whisper.ModelContext, configure whisper.ParamsConfigure) *whisper.Parameters {
	t.Helper()
	params, err := whisper.NewParameters(model, whisper.SAMPLING_GREEDY, configure)
	if err != nil {
		t.Fatalf("new params: %v", err)
	}
	return params
}

func helperProcessOnce(t *testing.T, ctx whisper.Context, data []float32) {
	t.Helper()
	if err := ctx.Process(data, nil, nil, nil); err != nil {
		t.Fatalf("process: %v", err)
	}
}

func helperFirstSegmentText(t *testing.T, ctx whisper.Context) string {
	t.Helper()
	seg, err := ctx.NextSegment()
	if err != nil {
		t.Fatalf("next segment: %v", err)
	}
	return seg.Text
}

// helperNewStatelessContext creates a fresh stateless context and returns a cleanup func
func helperNewStatelessContext(t *testing.T) (whisper.Context, func()) {
	t.Helper()
	model, closeModel := helperNewModelContext(t)
	params := helperNewParams(t, model, nil)
	ctx, err := whisper.NewStatelessContext(model, params)
	if err != nil {
		t.Fatalf("new stateless context: %v", err)
	}
	cleanup := func() {
		_ = ctx.Close()
		closeModel()
	}
	return ctx, cleanup
}

// helperNewStatefulContext creates a fresh stateful context and returns a cleanup func
func helperNewStatefulContext(t *testing.T) (whisper.Context, func()) {
	t.Helper()
	model, closeModel := helperNewModelContext(t)
	params := helperNewParams(t, model, nil)
	ctx, err := whisper.NewStatefulContext(model, params)
	if err != nil {
		t.Fatalf("new stateful context: %v", err)
	}
	cleanup := func() {
		_ = ctx.Close()
		closeModel()
	}
	return ctx, cleanup
}
