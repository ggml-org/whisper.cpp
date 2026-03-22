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

type StatelessContext struct {
	n      int
	model  *ModelContext
	params *Parameters
	closed bool
}

// NewStatelessContext creates a new stateless context backed by the model's context
func NewStatelessContext(model *ModelContext, params *Parameters) (*StatelessContext, error) {
	if model == nil {
		return nil, errModelRequired
	}

	if params == nil {
		return nil, errParametersRequired
	}

	// Ensure model context is available
	if _, err := model.ctxAccessor().context(); err != nil {
		return nil, err
	}

	c := new(StatelessContext)
	c.model = model
	c.params = params

	return c, nil
}

// DetectedLanguage returns the detected language for the current context data
func (context *StatelessContext) DetectedLanguage() string {
	if context.closed {
		return ""
	}
	ctx, err := context.model.ctxAccessor().context()
	if err != nil {
		return ""
	}
	return whisper.Whisper_lang_str(ctx.Whisper_full_lang_id())
}

// Close marks the context as closed.
func (context *StatelessContext) Close() error {
	context.closed = true
	return nil
}

// Params returns a high-level parameters wrapper
func (context *StatelessContext) Params() *Parameters {
	return context.params
}

// ResetTimings resets the model performance timing counters.
// Deprecated: Use Model.ResetTimings() instead - these are model-level performance metrics.
func (context *StatelessContext) ResetTimings() {
	context.model.ResetTimings()
}

// PrintTimings prints the model performance timings to stdout.
// Deprecated: Use Model.PrintTimings() instead - these are model-level performance metrics.
func (context *StatelessContext) PrintTimings() {
	context.model.PrintTimings()
}

// SystemInfo returns the system information
func (context *StatelessContext) SystemInfo() string {
	return fmt.Sprintf("system_info: n_threads = %d / %d | %s\n",
		context.params.Threads(),
		runtime.NumCPU(),
		whisper.Whisper_print_system_info(),
	)
}

// Use mel data at offset_ms to try and auto-detect the spoken language
// Make sure to call whisper_pcm_to_mel() or whisper_set_mel() first.
// Returns the probabilities of all languages for this context.
func (context *StatelessContext) WhisperLangAutoDetect(offset_ms int, n_threads int) ([]float32, error) {
	if context.closed {
		return nil, ErrModelClosed
	}
	ctx, err := context.model.ctxAccessor().context()
	if err != nil {
		return nil, err
	}
	langProbs, err := ctx.Whisper_lang_auto_detect(offset_ms, n_threads)
	if err != nil {
		return nil, err
	}
	return langProbs, nil
}

// Process new sample data and return any errors
func (context *StatelessContext) Process(
	data []float32,
	callEncoderBegin EncoderBeginCallback,
	callNewSegment SegmentCallback,
	callProgress ProgressCallback,
) error {
	if context.closed {
		return ErrModelClosed
	}
	ctx, err := context.model.ctxAccessor().context()
	if err != nil {
		return err
	}
	// Concurrency guard: prevent concurrent stateless processing on shared model ctx
	k := modelKey(context.model)
	if !gate().Acquire(k) {
		return ErrStatelessBusy
	}
	defer gate().Release(k)

	// If the callback is defined then we force on single_segment mode
	if callNewSegment != nil {
		context.params.SetSingleSegment(true)
	}

	lowLevelParams, err := context.params.unsafeParams()
	if err != nil {
		return err
	}

	if err := ctx.Whisper_full(*lowLevelParams, data, callEncoderBegin,
		func(new int) {
			if callNewSegment != nil {
				num_segments := ctx.Whisper_full_n_segments()
				s0 := num_segments - new
				for i := s0; i < num_segments; i++ {
					callNewSegment(toSegmentFromContext(ctx, i))
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
func (context *StatelessContext) NextSegment() (Segment, error) {
	if context.closed {
		return Segment{}, ErrModelClosed
	}
	ctx, err := context.model.ctxAccessor().context()
	if err != nil {
		return Segment{}, err
	}

	if context.n >= ctx.Whisper_full_n_segments() {
		return Segment{}, io.EOF
	}

	result := toSegmentFromContext(ctx, context.n)
	context.n++

	return result, nil
}

func (context *StatelessContext) IsMultilingual() bool {
	return context.model.IsMultilingual()
}

// Token helpers
// Deprecated: Use Model.IsText() instead - token checking is model-specific.
func (context *StatelessContext) IsText(t Token) bool {
	result, _ := context.model.tokenIdentifier().IsText(t)
	return result
}

// Deprecated: Use Model.IsBEG() instead - token checking is model-specific.
func (context *StatelessContext) IsBEG(t Token) bool {
	result, _ := context.model.tokenIdentifier().IsBEG(t)
	return result
}

// Deprecated: Use Model.IsSOT() instead - token checking is model-specific.
func (context *StatelessContext) IsSOT(t Token) bool {
	result, _ := context.model.tokenIdentifier().IsSOT(t)
	return result
}

// Deprecated: Use Model.IsEOT() instead - token checking is model-specific.
func (context *StatelessContext) IsEOT(t Token) bool {
	result, _ := context.model.tokenIdentifier().IsEOT(t)
	return result
}

// Deprecated: Use Model.IsPREV() instead - token checking is model-specific.
func (context *StatelessContext) IsPREV(t Token) bool {
	result, _ := context.model.tokenIdentifier().IsPREV(t)
	return result
}

// Deprecated: Use Model.IsSOLM() instead - token checking is model-specific.
func (context *StatelessContext) IsSOLM(t Token) bool {
	result, _ := context.model.tokenIdentifier().IsSOLM(t)
	return result
}

// Deprecated: Use Model.IsNOT() instead - token checking is model-specific.
func (context *StatelessContext) IsNOT(t Token) bool {
	result, _ := context.model.tokenIdentifier().IsNOT(t)
	return result
}

func (context *StatelessContext) SetLanguage(lang string) error {
	if context.closed || context.model.ctxAccessor().isClosed() {
		return ErrModelClosed
	}

	if !context.model.IsMultilingual() {
		return ErrModelNotMultilingual
	}

	return context.params.SetLanguage(lang)
}

// Deprecated: Use Model.IsLANG() instead - token checking is model-specific.
func (context *StatelessContext) IsLANG(t Token, lang string) bool {
	result, _ := context.model.tokenIdentifier().IsLANG(t, lang)
	return result
}

// Context-backed helper functions
func toSegmentFromContext(ctx *whisper.Context, n int) Segment {
	return Segment{
		Num:             n,
		Text:            strings.TrimSpace(ctx.Whisper_full_get_segment_text(n)),
		Start:           time.Duration(ctx.Whisper_full_get_segment_t0(n)) * time.Millisecond * 10,
		End:             time.Duration(ctx.Whisper_full_get_segment_t1(n)) * time.Millisecond * 10,
		Tokens:          toTokensFromContext(ctx, n),
		SpeakerTurnNext: false, // speaker turn available only with state-backed accessors
	}
}

func toTokensFromContext(ctx *whisper.Context, n int) []Token {
	result := make([]Token, ctx.Whisper_full_n_tokens(n))

	for i := 0; i < len(result); i++ {
		data := ctx.Whisper_full_get_token_data(n, i)
		result[i] = Token{
			Id:    int(ctx.Whisper_full_get_token_id(n, i)),
			Text:  ctx.Whisper_full_get_token_text(n, i),
			P:     ctx.Whisper_full_get_token_p(n, i),
			Start: time.Duration(data.T0()) * time.Millisecond * 10,
			End:   time.Duration(data.T1()) * time.Millisecond * 10,
		}
	}

	return result
}

// Deprecated: Use Params().Language() instead
func (context *StatelessContext) Language() string {
	return context.params.Language()
}

// Deprecated: Use Params().SetAudioCtx() instead
func (context *StatelessContext) SetAudioCtx(n uint) {
	context.params.SetAudioCtx(n)
}

// SetBeamSize implements Context.
// Deprecated: Use Params().SetBeamSize() instead
func (context *StatelessContext) SetBeamSize(v int) {
	context.params.SetBeamSize(v)
}

// SetDuration implements Context.
// Deprecated: Use Params().SetDuration() instead
func (context *StatelessContext) SetDuration(v time.Duration) {
	context.params.SetDuration(v)
}

// SetEntropyThold implements Context.
// Deprecated: Use Params().SetEntropyThold() instead
func (context *StatelessContext) SetEntropyThold(v float32) {
	context.params.SetEntropyThold(v)
}

// SetInitialPrompt implements Context.
// Deprecated: Use Params().SetInitialPrompt() instead
func (context *StatelessContext) SetInitialPrompt(v string) {
	context.params.SetInitialPrompt(v)
}

// SetMaxContext implements Context.
// Deprecated: Use Params().SetMaxContext() instead
func (context *StatelessContext) SetMaxContext(v int) {
	context.params.SetMaxContext(v)
}

// SetMaxSegmentLength implements Context.
// Deprecated: Use Params().SetMaxSegmentLength() instead
func (context *StatelessContext) SetMaxSegmentLength(v uint) {
	context.params.SetMaxSegmentLength(v)
}

// SetMaxTokensPerSegment implements Context.
// Deprecated: Use Params().SetMaxTokensPerSegment() instead
func (context *StatelessContext) SetMaxTokensPerSegment(v uint) {
	context.params.SetMaxTokensPerSegment(v)
}

// SetOffset implements Context.
// Deprecated: Use Params().SetOffset() instead
func (context *StatelessContext) SetOffset(v time.Duration) {
	context.params.SetOffset(v)
}

// SetSplitOnWord implements Context.
// Deprecated: Use Params().SetSplitOnWord() instead
func (context *StatelessContext) SetSplitOnWord(v bool) {
	context.params.SetSplitOnWord(v)
}

// SetTemperature implements Context.
// Deprecated: Use Params().SetTemperature() instead
func (context *StatelessContext) SetTemperature(v float32) {
	context.params.SetTemperature(v)
}

// SetTemperatureFallback implements Context.
// Deprecated: Use Params().SetTemperatureFallback() instead
func (context *StatelessContext) SetTemperatureFallback(v float32) {
	context.params.SetTemperatureFallback(v)
}

// SetThreads implements Context.
// Deprecated: Use Params().SetThreads() instead
func (context *StatelessContext) SetThreads(v uint) {
	context.params.SetThreads(v)
}

// SetTokenSumThreshold implements Context.
// Deprecated: Use Params().SetTokenSumThreshold() instead
func (context *StatelessContext) SetTokenSumThreshold(v float32) {
	context.params.SetTokenSumThreshold(v)
}

// SetTokenThreshold implements Context.
// Deprecated: Use Params().SetTokenThreshold() instead
func (context *StatelessContext) SetTokenThreshold(v float32) {
	context.params.SetTokenThreshold(v)
}

// SetTokenTimestamps implements Context.
// Deprecated: Use Params().SetTokenTimestamps() instead
func (context *StatelessContext) SetTokenTimestamps(v bool) {
	context.params.SetTokenTimestamps(v)
}

// SetTranslate implements Context.
// Deprecated: Use Params().SetTranslate() instead
func (context *StatelessContext) SetTranslate(v bool) {
	context.params.SetTranslate(v)
}

// VAD methods - implement Context interface
// Deprecated: Use Params().SetVAD() instead
func (context *StatelessContext) SetVAD(v bool) {
	context.params.SetVAD(v)
}

// Deprecated: Use Params().SetVADModelPath() instead
func (context *StatelessContext) SetVADModelPath(path string) {
	context.params.SetVADModelPath(path)
}

// Deprecated: Use Params().SetVADThreshold() instead
func (context *StatelessContext) SetVADThreshold(t float32) {
	context.params.SetVADThreshold(t)
}

// Deprecated: Use Params().SetVADMinSpeechMs() instead
func (context *StatelessContext) SetVADMinSpeechMs(ms int) {
	context.params.SetVADMinSpeechMs(ms)
}

// Deprecated: Use Params().SetVADMinSilenceMs() instead
func (context *StatelessContext) SetVADMinSilenceMs(ms int) {
	context.params.SetVADMinSilenceMs(ms)
}

// Deprecated: Use Params().SetVADMaxSpeechSec() instead
func (context *StatelessContext) SetVADMaxSpeechSec(s float32) {
	context.params.SetVADMaxSpeechSec(s)
}

// Deprecated: Use Params().SetVADSpeechPadMs() instead
func (context *StatelessContext) SetVADSpeechPadMs(ms int) {
	context.params.SetVADSpeechPadMs(ms)
}

// Deprecated: Use Params().SetVADSamplesOverlap() instead
func (context *StatelessContext) SetVADSamplesOverlap(sec float32) {
	context.params.SetVADSamplesOverlap(sec)
}

var _ Context = (*StatelessContext)(nil)
