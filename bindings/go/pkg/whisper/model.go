package whisper

import (
	"fmt"
	"os"

	// Bindings
	low "github.com/ggerganov/whisper.cpp/bindings/go"
)

type ModelContext struct {
	path  string
	ca    *ctxAccessor
	tokId *tokenIdentifier
}

// Make sure model adheres to the interface
var _ Model = (*ModelContext)(nil)

// Timings is a compact, high-level timing snapshot in milliseconds
type Timings struct {
	SampleMS float32
	EncodeMS float32
	DecodeMS float32
	BatchdMS float32
	PromptMS float32
}

// Deprecated: Use NewModelContext instead
func New(path string) (Model, error) {
	return NewModelContext(path)
}

// NewModelContext creates a new model context

func NewModelContext(
	path string,
) (*ModelContext, error) {
	return NewModelContextWithParams(
		path,
		NewModelContextParams(),
	)
}

// NewModelContextWithParams creates a new model context with custom initialization params
func NewModelContextWithParams(
	path string,
	params ModelContextParams,
) (*ModelContext, error) {
	model := new(ModelContext)
	if _, err := os.Stat(path); err != nil {
		return nil, err
	}

	ctx := low.Whisper_init_with_params(path, params.toLow())
	if ctx == nil {
		return nil, ErrUnableToLoadModel
	}

	model.ca = newCtxAccessor(ctx)
	model.tokId = newTokenIdentifier(model.ca)
	model.path = path

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

	result := make([]string, 0, low.Whisper_lang_max_id())
	for i := 0; i < low.Whisper_lang_max_id(); i++ {
		str := low.Whisper_lang_str(i)
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

	// Return new context (stateless for backward compatibility with timings)
	return NewStatelessContext(
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

// GetTimings returns a compact snapshot of model-level processing timings.
//
// Behavior notes:
//   - Stateless contexts (created via ModelContext.NewContext or NewStatelessContext)
//     update model-level timings during Process. After a stateless Process call,
//     the returned timings are expected to be non-zero (ok == true).
//   - Stateful contexts (created via NewStatefulContext) use a per-state backend
//     and do not affect model-level timings. After a stateful Process call,
//     the returned timings are expected to be zero values (fields equal 0) or
//     the call may return ok == false depending on the underlying implementation.
//
// Use ResetTimings before measurement to clear previous values.
func (model *ModelContext) GetTimings() (Timings, bool) {
	ctx, err := model.ca.context()
	if err != nil {
		return Timings{}, false
	}
	if t, ok := ctx.Whisper_get_timings_go(); ok {
		return Timings{
			SampleMS: t.SampleMS,
			EncodeMS: t.EncodeMS,
			DecodeMS: t.DecodeMS,
			BatchdMS: t.BatchdMS,
			PromptMS: t.PromptMS,
		}, true
	}
	return Timings{}, false
}

func (model *ModelContext) tokenIdentifier() *tokenIdentifier {
	return model.tokId
}
