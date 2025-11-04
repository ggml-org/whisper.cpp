package whisper_test

import (
	"testing"

	"github.com/ggerganov/whisper.cpp/bindings/go/pkg/whisper"
	assert "github.com/stretchr/testify/assert"
)

func TestNew(t *testing.T) {
	assert := assert.New(t)
	t.Run("valid model path", func(t *testing.T) {
		model, err := whisper.New(ModelPath)
		assert.NoError(err)
		assert.NotNil(model)
		defer model.Close()

	})

	t.Run("invalid model path", func(t *testing.T) {
		invalidModelPath := "invalid-model-path.bin"
		model, err := whisper.New(invalidModelPath)
		assert.Error(err)
		assert.Nil(model)
	})
}

func TestClose(t *testing.T) {
	assert := assert.New(t)

	model, err := whisper.New(ModelPath)
	assert.NoError(err)
	assert.NotNil(model)

	err = model.Close()
	assert.NoError(err)
}

func TestNewContext(t *testing.T) {
	assert := assert.New(t)

	model, err := whisper.New(ModelPath)
	assert.NoError(err)
	assert.NotNil(model)
	defer model.Close()

	context, err := model.NewContext()
	assert.NoError(err)
	assert.NotNil(context)
}

func TestIsMultilingual(t *testing.T) {
	assert := assert.New(t)

	model, err := whisper.New(ModelPath)
	assert.NoError(err)
	assert.NotNil(model)
	defer model.Close()

	isMultilingual := model.IsMultilingual()

	// This returns false since
	// the model 'models/ggml-small.en.bin'
	// that is loaded is not multilingual
	assert.False(isMultilingual)
}

func TestLanguages(t *testing.T) {
	assert := assert.New(t)

	model, err := whisper.New(ModelPath)
	assert.NoError(err)
	assert.NotNil(model)
	defer model.Close()

	expectedLanguages := []string{
		"en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl",
		"ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi", "he", "uk",
		"el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr",
		"bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn",
		"sr", "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne",
		"mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si", "km", "sn",
		"yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi",
		"lo", "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my",
		"bo", "tl", "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su",
	}

	actualLanguages := model.Languages()

	assert.Equal(expectedLanguages, actualLanguages)
}
