package whisper

import (
	"runtime"
	"time"

	// Bindings
	whisper "github.com/ggerganov/whisper.cpp/bindings/go"
)

// Parameters is a high-level wrapper that implements the Parameters interface
// and delegates to the underlying low-level whisper.Params.
type Parameters struct {
	p *whisper.Params
}

func defaultParamsConfigure(params *Parameters) {
	params.SetTranslate(false)
	params.SetPrintSpecial(false)
	params.SetPrintProgress(false)
	params.SetPrintRealtime(false)
	params.SetPrintTimestamps(false)
	// Default behavior backward compatibility
	params.SetThreads(uint(runtime.NumCPU()))
	params.SetNoContext(true)
}

func NewParameters(
	model *ModelContext,
	sampling SamplingStrategy,
	configure ParamsConfigure,
) (*Parameters, error) {
	ctx, err := model.ca.context()
	if err != nil {
		return nil, ErrModelClosed
	}

	p := ctx.Whisper_full_default_params(whisper.SamplingStrategy(sampling))
	safeParams := &Parameters{
		p: &p,
	}

	defaultParamsConfigure(safeParams)

	if configure != nil {
		configure(safeParams)
	}

	return safeParams, nil
}

func (w *Parameters) SetTranslate(v bool)              { w.p.SetTranslate(v) }
func (w *Parameters) SetSplitOnWord(v bool)            { w.p.SetSplitOnWord(v) }
func (w *Parameters) SetThreads(v uint)                { w.p.SetThreads(int(v)) }
func (w *Parameters) SetOffset(d time.Duration)        { w.p.SetOffset(int(d.Milliseconds())) }
func (w *Parameters) SetDuration(d time.Duration)      { w.p.SetDuration(int(d.Milliseconds())) }
func (w *Parameters) SetTokenThreshold(t float32)      { w.p.SetTokenThreshold(t) }
func (w *Parameters) SetTokenSumThreshold(t float32)   { w.p.SetTokenSumThreshold(t) }
func (w *Parameters) SetMaxSegmentLength(n uint)       { w.p.SetMaxSegmentLength(int(n)) }
func (w *Parameters) SetTokenTimestamps(b bool)        { w.p.SetTokenTimestamps(b) }
func (w *Parameters) SetMaxTokensPerSegment(n uint)    { w.p.SetMaxTokensPerSegment(int(n)) }
func (w *Parameters) SetAudioCtx(n uint)               { w.p.SetAudioCtx(int(n)) }
func (w *Parameters) SetMaxContext(n int)              { w.p.SetMaxContext(n) }
func (w *Parameters) SetBeamSize(n int)                { w.p.SetBeamSize(n) }
func (w *Parameters) SetEntropyThold(t float32)        { w.p.SetEntropyThold(t) }
func (w *Parameters) SetInitialPrompt(prompt string)   { w.p.SetInitialPrompt(prompt) }
func (w *Parameters) SetCarryInitialPrompt(v bool)     { w.p.SetCarryInitialPrompt(v) }
func (w *Parameters) SetTemperature(t float32)         { w.p.SetTemperature(t) }
func (w *Parameters) SetTemperatureFallback(t float32) { w.p.SetTemperatureFallback(t) }
func (w *Parameters) SetNoContext(v bool)              { w.p.SetNoContext(v) }
func (w *Parameters) SetPrintSpecial(v bool)           { w.p.SetPrintSpecial(v) }
func (w *Parameters) SetPrintProgress(v bool)          { w.p.SetPrintProgress(v) }
func (w *Parameters) SetPrintRealtime(v bool)          { w.p.SetPrintRealtime(v) }
func (w *Parameters) SetPrintTimestamps(v bool)        { w.p.SetPrintTimestamps(v) }
func (w *Parameters) SetDebugMode(v bool)              { w.p.SetDebugMode(v) }

// Diarization (tinydiarize)
func (w *Parameters) SetDiarize(v bool) { w.p.SetDiarize(v) }

// Voice Activity Detection (VAD)
func (w *Parameters) SetVAD(v bool)                    { w.p.SetVAD(v) }
func (w *Parameters) SetVADModelPath(p string)         { w.p.SetVADModelPath(p) }
func (w *Parameters) SetVADThreshold(t float32)        { w.p.SetVADThreshold(t) }
func (w *Parameters) SetVADMinSpeechMs(ms int)         { w.p.SetVADMinSpeechMs(ms) }
func (w *Parameters) SetVADMinSilenceMs(ms int)        { w.p.SetVADMinSilenceMs(ms) }
func (w *Parameters) SetVADMaxSpeechSec(s float32)     { w.p.SetVADMaxSpeechSec(s) }
func (w *Parameters) SetVADSpeechPadMs(ms int)         { w.p.SetVADSpeechPadMs(ms) }
func (w *Parameters) SetVADSamplesOverlap(sec float32) { w.p.SetVADSamplesOverlap(sec) }

func (w *Parameters) SetLanguage(lang string) error {
	if lang == "auto" {
		return w.p.SetLanguage(-1)
	}
	id := whisper.Whisper_lang_id_str(lang)
	if id < 0 {
		return ErrUnsupportedLanguage
	}
	return w.p.SetLanguage(id)
}

func (w *Parameters) SetSingleSegment(v bool) {
	w.p.SetSingleSegment(v)
}

// Getter methods for Parameters interface
func (w *Parameters) Language() string {
	id := w.p.Language()
	if id == -1 {
		return "auto"
	}

	return whisper.Whisper_lang_str(id)
}

func (w *Parameters) Threads() int {
	return w.p.Threads()
}

func (w *Parameters) unsafeParams() (*whisper.Params, error) {
	return w.p, nil
}
