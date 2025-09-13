package whisper

import (
	"time"

	// Bindings
	whisper "github.com/ggerganov/whisper.cpp/bindings/go"
)

// parameters is a high-level wrapper that implements the Parameters interface
// and delegates to the underlying low-level whisper.Params.
type parameters struct {
	p *whisper.Params
}

func newParameters(whisperParams *whisper.Params) Parameters {
	return &parameters{
		p: whisperParams,
	}
}

func (w *parameters) SetTranslate(v bool)              { w.p.SetTranslate(v) }
func (w *parameters) SetSplitOnWord(v bool)            { w.p.SetSplitOnWord(v) }
func (w *parameters) SetThreads(v uint)                { w.p.SetThreads(int(v)) }
func (w *parameters) SetOffset(d time.Duration)        { w.p.SetOffset(int(d.Milliseconds())) }
func (w *parameters) SetDuration(d time.Duration)      { w.p.SetDuration(int(d.Milliseconds())) }
func (w *parameters) SetTokenThreshold(t float32)      { w.p.SetTokenThreshold(t) }
func (w *parameters) SetTokenSumThreshold(t float32)   { w.p.SetTokenSumThreshold(t) }
func (w *parameters) SetMaxSegmentLength(n uint)       { w.p.SetMaxSegmentLength(int(n)) }
func (w *parameters) SetTokenTimestamps(b bool)        { w.p.SetTokenTimestamps(b) }
func (w *parameters) SetMaxTokensPerSegment(n uint)    { w.p.SetMaxTokensPerSegment(int(n)) }
func (w *parameters) SetAudioCtx(n uint)               { w.p.SetAudioCtx(int(n)) }
func (w *parameters) SetMaxContext(n int)              { w.p.SetMaxContext(n) }
func (w *parameters) SetBeamSize(n int)                { w.p.SetBeamSize(n) }
func (w *parameters) SetEntropyThold(t float32)        { w.p.SetEntropyThold(t) }
func (w *parameters) SetInitialPrompt(prompt string)   { w.p.SetInitialPrompt(prompt) }
func (w *parameters) SetTemperature(t float32)         { w.p.SetTemperature(t) }
func (w *parameters) SetTemperatureFallback(t float32) { w.p.SetTemperatureFallback(t) }
func (w *parameters) SetNoContext(v bool)              { w.p.SetNoContext(v) }
func (w *parameters) SetPrintSpecial(v bool)           { w.p.SetPrintSpecial(v) }
func (w *parameters) SetPrintProgress(v bool)          { w.p.SetPrintProgress(v) }
func (w *parameters) SetPrintRealtime(v bool)          { w.p.SetPrintRealtime(v) }
func (w *parameters) SetPrintTimestamps(v bool)        { w.p.SetPrintTimestamps(v) }

// Diarization (tinydiarize)
func (w *parameters) SetDiarize(v bool) { w.p.SetDiarize(v) }

// Voice Activity Detection (VAD)
func (w *parameters) SetVAD(v bool)                    { w.p.SetVAD(v) }
func (w *parameters) SetVADModelPath(p string)         { w.p.SetVADModelPath(p) }
func (w *parameters) SetVADThreshold(t float32)        { w.p.SetVADThreshold(t) }
func (w *parameters) SetVADMinSpeechMs(ms int)         { w.p.SetVADMinSpeechMs(ms) }
func (w *parameters) SetVADMinSilenceMs(ms int)        { w.p.SetVADMinSilenceMs(ms) }
func (w *parameters) SetVADMaxSpeechSec(s float32)     { w.p.SetVADMaxSpeechSec(s) }
func (w *parameters) SetVADSpeechPadMs(ms int)         { w.p.SetVADSpeechPadMs(ms) }
func (w *parameters) SetVADSamplesOverlap(sec float32) { w.p.SetVADSamplesOverlap(sec) }

func (w *parameters) SetLanguage(lang string) error {
	if lang == "auto" {
		return w.p.SetLanguage(-1)
	}
	id := whisper.Whisper_lang_id_str(lang)
	if id < 0 {
		return ErrUnsupportedLanguage
	}
	return w.p.SetLanguage(id)
}

func (w *parameters) SetSingleSegment(v bool) {
	w.p.SetSingleSegment(v)
}

// Getter methods for Parameters interface
func (w *parameters) Language() string {
	id := w.p.Language()
	if id == -1 {
		return "auto"
	}

	return whisper.Whisper_lang_str(id)
}

func (w *parameters) Threads() int {
	return w.p.Threads()
}

func (w *parameters) UnsafeParams() *whisper.Params {
	return w.p
}

var _ Parameters = &parameters{}
