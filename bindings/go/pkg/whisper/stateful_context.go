package whisper

import (
	"fmt"
	"io"
	"runtime"
	"strings"
	"time"

	// Bindings
	whisper "github.com/ggerganov/whisper.cpp/bindings/go"
)

type StatefulContext struct {
	n      int
	model  *ModelContext
	st     *whisperState
	params *Parameters
}

// NewStatefulContext creates a new stateful context
func NewStatefulContext(model *ModelContext, params *Parameters) (*StatefulContext, error) {
	if model == nil {
		return nil, errModelRequired
	}

	if params == nil {
		return nil, errParametersRequired
	}

	c := new(StatefulContext)
	c.model = model
	c.params = params

	// allocate isolated state per context
	ctx, err := model.ctxAccessor().context()
	if err != nil {
		return nil, err
	}

	st := ctx.Whisper_init_state()
	if st == nil {
		return nil, errUnableToCreateState
	}

	c.st = newWhisperState(st)

	// Return success
	return c, nil
}

// DetectedLanguage returns the detected language for the current context data
func (context *StatefulContext) DetectedLanguage() string {
	ctx, err := context.model.ctxAccessor().context()
	if err != nil {
		return ""
	}

	st, err := context.st.unsafeState()
	if err != nil {
		return ""
	}

	return whisper.Whisper_lang_str(
		ctx.Whisper_full_lang_id_from_state(
			st,
		),
	)
}

// Close frees the whisper state and marks the context as closed.
func (context *StatefulContext) Close() error {
	return context.st.close()
}

// Params returns a high-level parameters wrapper
func (context *StatefulContext) Params() *Parameters {
	return context.params
}

// ResetTimings resets the model performance timing counters.
// Deprecated: Use Model.ResetTimings() instead - these are model-level performance metrics.
func (context *StatefulContext) ResetTimings() {
	context.model.ResetTimings()
}

// PrintTimings prints the model performance timings to stdout.
// Deprecated: Use Model.PrintTimings() instead - these are model-level performance metrics.
func (context *StatefulContext) PrintTimings() {
	context.model.PrintTimings()
}

// SystemInfo returns the system information
func (context *StatefulContext) SystemInfo() string {
	return fmt.Sprintf("system_info: n_threads = %d / %d | %s\n",
		context.params.Threads(),
		runtime.NumCPU(),
		whisper.Whisper_print_system_info(),
	)
}

// Use mel data at offset_ms to try and auto-detect the spoken language
// Make sure to call whisper_pcm_to_mel() or whisper_set_mel() first.
// Returns the probabilities of all languages for this context's state.
func (context *StatefulContext) WhisperLangAutoDetect(offset_ms int, n_threads int) ([]float32, error) {
	ctx, err := context.model.ctxAccessor().context()
	if err != nil {
		return nil, err
	}

	st, err := context.st.unsafeState()
	if err != nil {
		return nil, err
	}

	langProbs, err := ctx.Whisper_lang_auto_detect_with_state(st, offset_ms, n_threads)
	if err != nil {
		return nil, err
	}

	return langProbs, nil
}

// Process new sample data and return any errors
func (context *StatefulContext) Process(
	data []float32,
	callEncoderBegin EncoderBeginCallback,
	callNewSegment SegmentCallback,
	callProgress ProgressCallback,
) error {
	ctx, err := context.model.ctxAccessor().context()
	if err != nil {
		return err
	}

	// If the callback is defined then we force on single_segment mode
	if callNewSegment != nil {
		context.params.SetSingleSegment(true)
	}

	lowLevelParams, err := context.params.unsafeParams()
	if err != nil {
		return err
	}

	st, err := context.st.unsafeState()
	if err != nil {
		return err
	}

	if err := ctx.Whisper_full_with_state(st, *lowLevelParams, data, callEncoderBegin,
		func(new int) {
			if callNewSegment != nil {
				num_segments := ctx.Whisper_full_n_segments_from_state(st)
				s0 := num_segments - new
				for i := s0; i < num_segments; i++ {
					callNewSegment(toSegmentFromState(ctx, st, i))
				}
			}
		}, func(progress int) {
			if callProgress != nil {
				callProgress(progress)
			}
		}); err != nil {
		return err
	}

	// Return success
	return nil
}

// NextSegment returns the next segment from the context buffer
func (context *StatefulContext) NextSegment() (Segment, error) {
	ctx, err := context.model.ctxAccessor().context()
	if err != nil {
		return Segment{}, err
	}

	st, err := context.st.unsafeState()
	if err != nil {
		return Segment{}, err
	}

	if context.n >= ctx.Whisper_full_n_segments_from_state(st) {
		return Segment{}, io.EOF
	}

	result := toSegmentFromState(ctx, st, context.n)
	context.n++

	return result, nil
}

func (context *StatefulContext) IsMultilingual() bool {
	return context.model.IsMultilingual()
}

// Token helpers
// Deprecated: Use Model.IsText() instead - token checking is model-specific.
func (context *StatefulContext) IsText(t Token) bool {
	result, _ := context.model.tokenIdentifier().IsText(t)
	return result
}

// Deprecated: Use Model.IsBEG() instead - token checking is model-specific.
func (context *StatefulContext) IsBEG(t Token) bool {
	result, _ := context.model.tokenIdentifier().IsBEG(t)
	return result
}

// Deprecated: Use Model.IsSOT() instead - token checking is model-specific.
func (context *StatefulContext) IsSOT(t Token) bool {
	result, _ := context.model.tokenIdentifier().IsSOT(t)
	return result
}

// Deprecated: Use Model.IsEOT() instead - token checking is model-specific.
func (context *StatefulContext) IsEOT(t Token) bool {
	result, _ := context.model.tokenIdentifier().IsEOT(t)
	return result
}

// Deprecated: Use Model.IsPREV() instead - token checking is model-specific.
func (context *StatefulContext) IsPREV(t Token) bool {
	result, _ := context.model.tokenIdentifier().IsPREV(t)
	return result
}

// Deprecated: Use Model.IsSOLM() instead - token checking is model-specific.
func (context *StatefulContext) IsSOLM(t Token) bool {
	result, _ := context.model.tokenIdentifier().IsSOLM(t)
	return result
}

// Deprecated: Use Model.IsNOT() instead - token checking is model-specific.
func (context *StatefulContext) IsNOT(t Token) bool {
	result, _ := context.model.tokenIdentifier().IsNOT(t)
	return result
}

func (context *StatefulContext) SetLanguage(lang string) error {
	if context.model.ctxAccessor().isClosed() {
		// TODO: remove this logic after deprecating the ErrInternalAppError
		return ErrModelClosed
	}

	if !context.model.IsMultilingual() {
		return ErrModelNotMultilingual
	}

	return context.params.SetLanguage(lang)
}

// Deprecated: Use Model.IsLANG() instead - token checking is model-specific.
func (context *StatefulContext) IsLANG(t Token, lang string) bool {
	result, _ := context.model.tokenIdentifier().IsLANG(t, lang)
	return result
}

// State-backed helper functions
func toSegmentFromState(ctx *whisper.Context, st *whisper.State, n int) Segment {
	return Segment{
		Num:             n,
		Text:            strings.TrimSpace(ctx.Whisper_full_get_segment_text_from_state(st, n)),
		Start:           time.Duration(ctx.Whisper_full_get_segment_t0_from_state(st, n)) * time.Millisecond * 10,
		End:             time.Duration(ctx.Whisper_full_get_segment_t1_from_state(st, n)) * time.Millisecond * 10,
		Tokens:          toTokensFromState(ctx, st, n),
		SpeakerTurnNext: ctx.Whisper_full_get_segment_speaker_turn_next_from_state(st, n),
	}
}

func toTokensFromState(ctx *whisper.Context, st *whisper.State, n int) []Token {
	result := make([]Token, ctx.Whisper_full_n_tokens_from_state(st, n))

	for i := 0; i < len(result); i++ {
		data := ctx.Whisper_full_get_token_data_from_state(st, n, i)
		result[i] = Token{
			Id:    int(ctx.Whisper_full_get_token_id_from_state(st, n, i)),
			Text:  ctx.Whisper_full_get_token_text_from_state(st, n, i),
			P:     ctx.Whisper_full_get_token_p_from_state(st, n, i),
			Start: time.Duration(data.T0()) * time.Millisecond * 10,
			End:   time.Duration(data.T1()) * time.Millisecond * 10,
		}
	}

	return result
}

// Deprecated: Use Params().Language() instead
func (context *StatefulContext) Language() string {
	return context.params.Language()
}

// Deprecated: Use Params().SetAudioCtx() instead
func (context *StatefulContext) SetAudioCtx(n uint) {
	context.params.SetAudioCtx(n)
}

// SetBeamSize implements Context.
// Deprecated: Use Params().SetBeamSize() instead
func (context *StatefulContext) SetBeamSize(v int) {
	context.params.SetBeamSize(v)
}

// SetDuration implements Context.
// Deprecated: Use Params().SetDuration() instead
func (context *StatefulContext) SetDuration(v time.Duration) {
	context.params.SetDuration(v)
}

// SetEntropyThold implements Context.
// Deprecated: Use Params().SetEntropyThold() instead
func (context *StatefulContext) SetEntropyThold(v float32) {
	context.params.SetEntropyThold(v)
}

// SetInitialPrompt implements Context.
// Deprecated: Use Params().SetInitialPrompt() instead
func (context *StatefulContext) SetInitialPrompt(v string) {
	context.params.SetInitialPrompt(v)
}

// SetMaxContext implements Context.
// Deprecated: Use Params().SetMaxContext() instead
func (context *StatefulContext) SetMaxContext(v int) {
	context.params.SetMaxContext(v)
}

// SetMaxSegmentLength implements Context.
// Deprecated: Use Params().SetMaxSegmentLength() instead
func (context *StatefulContext) SetMaxSegmentLength(v uint) {
	context.params.SetMaxSegmentLength(v)
}

// SetMaxTokensPerSegment implements Context.
// Deprecated: Use Params().SetMaxTokensPerSegment() instead
func (context *StatefulContext) SetMaxTokensPerSegment(v uint) {
	context.params.SetMaxTokensPerSegment(v)
}

// SetOffset implements Context.
// Deprecated: Use Params().SetOffset() instead
func (context *StatefulContext) SetOffset(v time.Duration) {
	context.params.SetOffset(v)
}

// SetSplitOnWord implements Context.
// Deprecated: Use Params().SetSplitOnWord() instead
func (context *StatefulContext) SetSplitOnWord(v bool) {
	context.params.SetSplitOnWord(v)
}

// SetTemperature implements Context.
// Deprecated: Use Params().SetTemperature() instead
func (context *StatefulContext) SetTemperature(v float32) {
	context.params.SetTemperature(v)
}

// SetTemperatureFallback implements Context.
// Deprecated: Use Params().SetTemperatureFallback() instead
func (context *StatefulContext) SetTemperatureFallback(v float32) {
	context.params.SetTemperatureFallback(v)
}

// SetThreads implements Context.
// Deprecated: Use Params().SetThreads() instead
func (context *StatefulContext) SetThreads(v uint) {
	context.params.SetThreads(v)
}

// SetTokenSumThreshold implements Context.
// Deprecated: Use Params().SetTokenSumThreshold() instead
func (context *StatefulContext) SetTokenSumThreshold(v float32) {
	context.params.SetTokenSumThreshold(v)
}

// SetTokenThreshold implements Context.
// Deprecated: Use Params().SetTokenThreshold() instead
func (context *StatefulContext) SetTokenThreshold(v float32) {
	context.params.SetTokenThreshold(v)
}

// SetTokenTimestamps implements Context.
// Deprecated: Use Params().SetTokenTimestamps() instead
func (context *StatefulContext) SetTokenTimestamps(v bool) {
	context.params.SetTokenTimestamps(v)
}

// SetTranslate implements Context.
// Deprecated: Use Params().SetTranslate() instead
func (context *StatefulContext) SetTranslate(v bool) {
	context.params.SetTranslate(v)
}

// Make stateful context compatible with the old deprecated interface for
// the simple migration into multi-threaded processing.
var _ Context = (*StatefulContext)(nil)
