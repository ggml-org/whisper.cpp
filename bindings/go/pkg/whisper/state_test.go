package whisper_test

import (
	"os"
	"sync"
	"testing"

	"github.com/ggerganov/whisper.cpp/bindings/go/pkg/whisper"
	"github.com/go-audio/wav"
	assert "github.com/stretchr/testify/assert"
)

func TestState_Process(t *testing.T) {
	assert := assert.New(t)

	if _, err := os.Stat(ModelPath); os.IsNotExist(err) {
		t.Skip("Skipping test, model not found:", ModelPath)
	}
	if _, err := os.Stat(SamplePath); os.IsNotExist(err) {
		t.Skip("Skipping test, sample not found:", SamplePath)
	}

	fh, err := os.Open(SamplePath)
	assert.NoError(err)
	defer fh.Close()

	dec := wav.NewDecoder(fh)
	buf, err := dec.FullPCMBuffer()
	assert.NoError(err)
	assert.Equal(uint16(1), dec.NumChans)
	data := buf.AsFloat32Buffer().Data

	model, err := whisper.New(ModelPath)
	assert.NoError(err)
	assert.NotNil(model)
	defer model.Close()

	st, err := whisper.NewState(model)
	assert.NoError(err)
	assert.NotNil(st)
	defer func() { _ = st.Close() }()

	err = st.Process(data, nil, nil, nil)
	assert.NoError(err)

	seg, err := st.NextSegment()
	assert.NoError(err)
	assert.NotEmpty(seg.Text)
}

func TestState_Parallel_DifferentInputs(t *testing.T) {
	assert := assert.New(t)

	if _, err := os.Stat(ModelPath); os.IsNotExist(err) {
		t.Skip("Skipping test, model not found:", ModelPath)
	}
	if _, err := os.Stat(SamplePath); os.IsNotExist(err) {
		t.Skip("Skipping test, sample not found:", SamplePath)
	}

	fh, err := os.Open(SamplePath)
	assert.NoError(err)
	defer fh.Close()

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
	defer model.Close()

	st1, err := whisper.NewState(model)
	assert.NoError(err)
	st2, err := whisper.NewState(model)
	assert.NoError(err)
	defer func() { _ = st1.Close() }()
	defer func() { _ = st2.Close() }()

	// Run in parallel, but guard core call to respect context safety
	var wg sync.WaitGroup
	var first1, first2 string
	var e1, e2 error

	wg.Add(2)

	// No mutex needed because each state is isolated
	go func() {
		defer wg.Done()
		e1 = st1.Process(data, nil, nil, nil)
		if e1 == nil {
			seg, err := st1.NextSegment()
			if err == nil {
				first1 = seg.Text
			} else {
				e1 = err
			}
		}
	}()

	go func() {
		defer wg.Done()
		e2 = st2.Process(half, nil, nil, nil)
		if e2 == nil {
			seg, err := st2.NextSegment()
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
