package whisper_test

import (
	"sync"
	"testing"

	whisper "github.com/ggerganov/whisper.cpp/bindings/go/pkg/whisper"
	assert "github.com/stretchr/testify/assert"
)

// Ensure stateless contexts cannot process in parallel without isolation
func TestStatelessContext_NotParallelSafe(t *testing.T) {
	data := helperLoadSample(t, SamplePath)

	model, closeModel := helperNewModelContext(t)
	defer closeModel()

	params := helperNewParams(t, model, nil)

	// Create two stateless contexts sharing the same underlying model context
	ctx1, err := whisper.NewStatelessContext(model, params)
	assert.NoError(t, err)
	defer func() { _ = ctx1.Close() }()

	ctx2, err := whisper.NewStatelessContext(model, params)
	assert.NoError(t, err)
	defer func() { _ = ctx2.Close() }()

	// Run both in parallel - expect a panic or error from underlying whisper_full
	// We capture panics to assert the behavior.
	var wg sync.WaitGroup
	wg.Add(2)

	var err1, err2 error

	go func() {
		defer wg.Done()
		err1 = ctx1.Process(data, nil, nil, nil)
	}()

	go func() {
		defer wg.Done()
		err2 = ctx2.Process(data, nil, nil, nil)
	}()

	wg.Wait()

	// At least one should return ErrStatelessBusy
	if err1 != whisper.ErrStatelessBusy && err2 != whisper.ErrStatelessBusy {
		t.Fatalf("expected ErrStatelessBusy when processing in parallel with StatelessContext, got err1=%v err2=%v", err1, err2)
	}
}
