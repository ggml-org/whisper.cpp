package whisper

import (
	"fmt"
	"os"

	// Bindings
	whisper "github.com/ggerganov/whisper.cpp/bindings/go"
)

type ModelContext struct {
	path  string
	ca    *ctxAccessor
	tokId *tokenIdentifier
}

// Make sure model adheres to the interface
var _ Model = (*ModelContext)(nil)

// Deprecated: Use NewModelContext instead
func New(path string) (Model, error) {
	return NewModelContext(path)
}

// NewModelContext creates a new model context
func NewModelContext(
	path string,
) (*ModelContext, error) {
	model := new(ModelContext)
	if _, err := os.Stat(path); err != nil {
		return nil, err
	} else if ctx := whisper.Whisper_init(path); ctx == nil {
		return nil, ErrUnableToLoadModel
	} else {
		model.ca = newCtxAccessor(ctx)
		model.tokId = newTokenIdentifier(model.ca)
		model.path = path
	}

	// Return success
	return model, nil
}

func (model *ModelContext) Close() error {
	return model.ca.close()
}

func (model *ModelContext) ctxAccessor() *ctxAccessor {
	return model.ca
}

func (model *ModelContext) String() string {
	str := "<whisper.model"
	if model.ca != nil {
		str += fmt.Sprintf(" model=%q", model.path)
	}

	return str + ">"
}

// Return true if model is multilingual (language and translation options are supported)
func (model *ModelContext) IsMultilingual() bool {
	ctx, err := model.ca.context()
	if err != nil {
		return false
	}

	return ctx.Whisper_is_multilingual() != 0
}

// Return all recognized languages. Initially it is set to auto-detect
func (model *ModelContext) Languages() []string {
	ctx, err := model.ca.context()
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
func (model *ModelContext) NewContext() (Context, error) {
	// Create new context with default params
	params, err := NewParameters(model, SAMPLING_GREEDY, nil)
	if err != nil {
		return nil, err
	}

	// Return new context (now state-backed)
	return NewStatefulContext(
		model,
		params,
	)
}

// PrintTimings prints the model performance timings to stdout.
func (model *ModelContext) PrintTimings() {
	ctx, err := model.ca.context()
	if err != nil {
		return
	}

	ctx.Whisper_print_timings()
}

// ResetTimings resets the model performance timing counters.
func (model *ModelContext) ResetTimings() {
	ctx, err := model.ca.context()
	if err != nil {
		return
	}

	ctx.Whisper_reset_timings()
}

func (model *ModelContext) tokenIdentifier() *tokenIdentifier {
	return model.tokId
}
