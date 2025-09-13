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

type context struct {
	n      int
	model  *model
	st     WhisperState
	params *parameters
	Parameters
}

func newContext(model *model, params *parameters) (Context, error) {
	c := new(context)
	c.model = model

	c.params = params
	c.Parameters = c.params

	// allocate isolated state per context
	ctx, err := model.whisperContext().unsafeContext()
	if err != nil {
		return nil, err
	}

	st := ctx.Whisper_init_state()
	if st == nil {
		return nil, ErrUnableToCreateState
	}

	c.st = newWhisperState(st)

	// Return success
	return c, nil
}

// DetectedLanguage returns the detected language for the current context data
func (context *context) DetectedLanguage() string {
	ctx, err := context.model.whisperContext().unsafeContext()
	if err != nil {
		return ""
	}

	st, err := context.st.UnsafeState()
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
func (context *context) Close() error {
	return context.st.Close()
}

// Params returns a high-level parameters wrapper
func (context *context) Params() Parameters {
	return context.params
}

// ResetTimings resets the model performance timing counters.
// Deprecated: Use Model.ResetTimings() instead - these are model-level performance metrics.
func (context *context) ResetTimings() {
	context.model.ResetTimings()
}

// PrintTimings prints the model performance timings to stdout.
// Deprecated: Use Model.PrintTimings() instead - these are model-level performance metrics.
func (context *context) PrintTimings() {
	context.model.PrintTimings()
}

// SystemInfo returns the system information
func (context *context) SystemInfo() string {
	return fmt.Sprintf("system_info: n_threads = %d / %d | %s\n",
		context.params.Threads(),
		runtime.NumCPU(),
		whisper.Whisper_print_system_info(),
	)
}

// Use mel data at offset_ms to try and auto-detect the spoken language
// Make sure to call whisper_pcm_to_mel() or whisper_set_mel() first.
// Returns the probabilities of all languages for this context's state.
func (context *context) WhisperLangAutoDetect(offset_ms int, n_threads int) ([]float32, error) {
	ctx, err := context.model.whisperContext().unsafeContext()
	if err != nil {
		return nil, err
	}

	st, err := context.st.UnsafeState()
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
func (context *context) Process(
	data []float32,
	callEncoderBegin EncoderBeginCallback,
	callNewSegment SegmentCallback,
	callProgress ProgressCallback,
) error {
	ctx, err := context.model.whisperContext().unsafeContext()
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

	st, err := context.st.UnsafeState()
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
func (context *context) NextSegment() (Segment, error) {
	ctx, err := context.model.whisperContext().unsafeContext()
	if err != nil {
		return Segment{}, err
	}

	st, err := context.st.UnsafeState()
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

func (context *context) IsMultilingual() bool {
	return context.model.IsMultilingual()
}

// Token helpers
// Deprecated: Use Model.IsText() instead - token checking is model-specific.
func (context *context) IsText(t Token) bool {
	result, _ := context.model.TokenIdentifier().IsText(t)
	return result
}

// Deprecated: Use Model.IsBEG() instead - token checking is model-specific.
func (context *context) IsBEG(t Token) bool {
	result, _ := context.model.TokenIdentifier().IsBEG(t)
	return result
}

// Deprecated: Use Model.IsSOT() instead - token checking is model-specific.
func (context *context) IsSOT(t Token) bool {
	result, _ := context.model.TokenIdentifier().IsSOT(t)
	return result
}

// Deprecated: Use Model.IsEOT() instead - token checking is model-specific.
func (context *context) IsEOT(t Token) bool {
	result, _ := context.model.TokenIdentifier().IsEOT(t)
	return result
}

// Deprecated: Use Model.IsPREV() instead - token checking is model-specific.
func (context *context) IsPREV(t Token) bool {
	result, _ := context.model.TokenIdentifier().IsPREV(t)
	return result
}

// Deprecated: Use Model.IsSOLM() instead - token checking is model-specific.
func (context *context) IsSOLM(t Token) bool {
	result, _ := context.model.TokenIdentifier().IsSOLM(t)
	return result
}

// Deprecated: Use Model.IsNOT() instead - token checking is model-specific.
func (context *context) IsNOT(t Token) bool {
	result, _ := context.model.TokenIdentifier().IsNOT(t)
	return result
}

func (context *context) SetLanguage(lang string) error {
	if context.model.whisperContext().IsClosed() {
		// TODO: remove this logic after deprecating the ErrInternalAppError
		return ErrModelClosed
	}

	if !context.model.IsMultilingual() {
		return ErrModelNotMultilingual
	}

	return context.params.SetLanguage(lang)
}

// Deprecated: Use Model.IsLANG() instead - token checking is model-specific.
func (context *context) IsLANG(t Token, lang string) bool {
	result, _ := context.model.TokenIdentifier().IsLANG(t, lang)
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

func (context *context) Model() Model {
	return context.model
}

var _ Context = (*context)(nil)
