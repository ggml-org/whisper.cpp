package whisper

import (
	"fmt"
	"os"

	// Bindings
	whisper "github.com/ggerganov/whisper.cpp/bindings/go"
)

type model struct {
	path            string
	ctx             *whisperCtx
	tokenIdentifier *tokenIdentifier
}

// Make sure model adheres to the interface
var _ Model = (*model)(nil)

// Deprecated: Use NewModel instead
func New(path string) (Model, error) {
	return NewModel(path)
}

// NewModel creates a new model without initializing the context
func NewModel(
	path string,
) (*model, error) {
	model := new(model)
	if _, err := os.Stat(path); err != nil {
		return nil, err
	} else if ctx := whisper.Whisper_init(path); ctx == nil {
		return nil, ErrUnableToLoadModel
	} else {
		model.ctx = newWhisperCtx(ctx)
		model.tokenIdentifier = newTokenIdentifier(model.ctx)
		model.path = path
	}

	// Return success
	return model, nil
}

func (model *model) Close() error {
	return model.ctx.Close()
}

func (model *model) WhisperContext() WhisperContext {
	return model.ctx
}

func (model *model) whisperContext() *whisperCtx {
	return model.ctx
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
	ctx, err := model.ctx.unsafeContext()
	if err != nil {
		return false
	}

	return ctx.Whisper_is_multilingual() != 0
}

// Return all recognized languages. Initially it is set to auto-detect
func (model *model) Languages() []string {
	ctx, err := model.ctx.unsafeContext()
	if err != nil {
		return nil
	}

	result := make([]string, 0, whisper.Whisper_lang_max_id())
	for i := 0; i < whisper.Whisper_lang_max_id(); i++ {
		str := whisper.Whisper_lang_str(i)
		if ctx.Whisper_lang_id(str) >= 0 {
			result = append(result, str)
		}
	}

	return result
}

// NewContext creates a new speech-to-text context.
// Each context is backed by an isolated whisper_state for safe concurrent processing.
func (model *model) NewContext() (Context, error) {
	// Create new context with default params
	params, err := NewParameters(model, SAMPLING_GREEDY, nil)
	if err != nil {
		return nil, err
	}

	// Return new context (now state-backed)
	return NewContext(
		model,
		params,
	)
}

// PrintTimings prints the model performance timings to stdout.
func (model *model) PrintTimings() {
	ctx, err := model.ctx.unsafeContext()
	if err != nil {
		return
	}

	ctx.Whisper_print_timings()
}

// ResetTimings resets the model performance timing counters.
func (model *model) ResetTimings() {
	ctx, err := model.ctx.unsafeContext()
	if err != nil {
		return
	}

	ctx.Whisper_reset_timings()
}

// WhisperContext returns the low-level whisper context, or error if the model is closed.
func (model *model) TokenIdentifier() TokenIdentifier {
	return model.tokenIdentifier
}
