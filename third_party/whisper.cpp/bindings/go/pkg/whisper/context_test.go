package whisper_test

import (
	"os"
	"strings"
	"testing"

	"github.com/ggerganov/whisper.cpp/bindings/go/pkg/whisper"
	"github.com/go-audio/wav"
	assert "github.com/stretchr/testify/assert"
)

func TestSetLanguage(t *testing.T) {
	assert := assert.New(t)

	model, err := whisper.New(ModelPath)
	assert.NoError(err)
	assert.NotNil(model)
	defer model.Close()

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
	defer model.Close()

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
	defer model.Close()

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
	defer fh.Close()

	// Decode the WAV file - load the full buffer
	dec := wav.NewDecoder(fh)
	buf, err := dec.FullPCMBuffer()
	assert.NoError(err)
	assert.Equal(uint16(1), dec.NumChans)

	data := buf.AsFloat32Buffer().Data

	model, err := whisper.New(ModelPath)
	assert.NoError(err)
	assert.NotNil(model)
	defer model.Close()

	context, err := model.NewContext()
	assert.NoError(err)

	err = context.Process(data, nil, nil, nil)
	assert.NoError(err)
}

func TestProcessMaxTokensPerSegment(t *testing.T) {
	assert := assert.New(t)

	if _, err := os.Stat(ModelPath); os.IsNotExist(err) {
		t.Skip("Skipping test, model not found:", ModelPath)
	}

	fh, err := os.Open(SamplePath)
	assert.NoError(err)
	defer fh.Close()

	// Decode the WAV file - load the full buffer
	dec := wav.NewDecoder(fh)
	buf, err := dec.FullPCMBuffer()
	assert.NoError(err)
	assert.Equal(uint16(1), dec.NumChans)

	data := buf.AsFloat32Buffer().Data

	model, err := whisper.New(ModelPath)
	assert.NoError(err)
	assert.NotNil(model)
	defer model.Close()

	context, err := model.NewContext()
	assert.NoError(err)

	context.SetMaxTokensPerSegment(5)

	err = context.Process(data, nil, nil, nil)
	assert.NoError(err)

	var text strings.Builder
	nSegments := 0
	for {
		segment, err := context.NextSegment()
		if err != nil {
			break
		}
		nSegments++
		text.WriteString(segment.Text)
	}

	assert.Greater(nSegments, 1)
	assert.Contains(text.String(), "country")
}

func TestDetectedLanguage(t *testing.T) {
	assert := assert.New(t)

	fh, err := os.Open(SamplePath)
	assert.NoError(err)
	defer fh.Close()

	// Decode the WAV file - load the full buffer
	dec := wav.NewDecoder(fh)
	buf, err := dec.FullPCMBuffer()
	assert.NoError(err)
	assert.Equal(uint16(1), dec.NumChans)

	data := buf.AsFloat32Buffer().Data

	model, err := whisper.New(ModelPath)
	assert.NoError(err)
	assert.NotNil(model)
	defer model.Close()

	context, err := model.NewContext()
	assert.NoError(err)

	err = context.Process(data, nil, nil, nil)
	assert.NoError(err)

	expectedLanguage := "en"
	actualLanguage := context.DetectedLanguage()
	assert.Equal(expectedLanguage, actualLanguage)
}
