package whisper

import (
	"fmt"
	"os"
	"runtime"

	// Bindings
	whisper "github.com/ggerganov/whisper.cpp/bindings/go"
)

///////////////////////////////////////////////////////////////////////////////
// TYPES

type model struct {
	path string
	ctx  *whisper.Context
}

// Make sure model adheres to the interface
var _ Model = (*model)(nil)

///////////////////////////////////////////////////////////////////////////////
// LIFECYCLE

func New(path string) (Model, error) {
	model := new(model)
	if _, err := os.Stat(path); err != nil {
		return nil, err
	} else if ctx := whisper.Whisper_init(path); ctx == nil {
		return nil, ErrUnableToLoadModel
	} else {
		model.ctx = ctx
		model.path = path
	}

	// Return success
	return model, nil
}

func (model *model) Close() error {
	if model.ctx != nil {
		model.ctx.Whisper_free()
	}

	// Release resources
	model.ctx = nil

	// Return success
	return nil
}

///////////////////////////////////////////////////////////////////////////////
// STRINGIFY

func (model *model) String() string {
	str := "<whisper.model"
	if model.ctx != nil {
		str += fmt.Sprintf(" model=%q", model.path)
	}
	return str + ">"
}

///////////////////////////////////////////////////////////////////////////////
// PUBLIC METHODS

// Return true if model is multilingual (language and translation options are supported)
func (model *model) IsMultilingual() bool {
	return model.ctx.Whisper_is_multilingual() != 0
}

// Return all recognized languages. Initially it is set to auto-detect
func (model *model) Languages() []string {
	result := make([]string, 0, whisper.Whisper_lang_max_id())
	for i := 0; i < whisper.Whisper_lang_max_id(); i++ {
		str := whisper.Whisper_lang_str(i)
		if model.ctx.Whisper_lang_id(str) >= 0 {
			result = append(result, str)
		}
	}
	return result
}

func (model *model) NewContext() (Context, error) {
	if model.ctx == nil {
		return nil, ErrInternalAppError
	}

	// Create new context
	params := model.ctx.Whisper_full_default_params(whisper.SAMPLING_GREEDY)
	params.SetTranslate(false)
	params.SetPrintSpecial(false)
	params.SetPrintProgress(false)
	params.SetPrintRealtime(false)
	params.SetPrintTimestamps(false)
	params.SetThreads(runtime.NumCPU())
	params.SetNoContext(true)

	// Return new context
	return newContext(model, params)
}
