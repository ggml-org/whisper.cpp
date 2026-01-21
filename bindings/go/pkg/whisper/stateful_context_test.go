package whisper_test

import (
	"os"
	"sync"
	"testing"

	whisper "github.com/ggerganov/whisper.cpp/bindings/go/pkg/whisper"
	assert "github.com/stretchr/testify/assert"
)

// Stateful-specific: parallel processing supported
func TestContext_Parallel_DifferentInputs_Stateful(t *testing.T) {
	assert := assert.New(t)

	if _, err := os.Stat(ModelPath); os.IsNotExist(err) {
		t.Skip("Skipping test, model not found:", ModelPath)
	}
	if _, err := os.Stat(SamplePath); os.IsNotExist(err) {
		t.Skip("Skipping test, sample not found:", SamplePath)
	}

	data := helperLoadSample(t, SamplePath)
	assert.Greater(len(data), 10)

	// Create half-sample (second half)
	half := make([]float32, len(data)/2)
	copy(half, data[len(data)/2:])

	model, err := whisper.NewModelContext(ModelPath)
	assert.NoError(err)
	defer func() { _ = model.Close() }()

	params1 := helperNewParams(t, model, nil)
	params2 := helperNewParams(t, model, nil)

	ctx1, err := whisper.NewStatefulContext(model, params1)
	assert.NoError(err)
	defer func() { _ = ctx1.Close() }()
	ctx2, err := whisper.NewStatefulContext(model, params2)
	assert.NoError(err)
	defer func() { _ = ctx2.Close() }()

	var wg sync.WaitGroup
	var first1, first2 string
	var e1, e2 error
	wg.Add(2)

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
