package whisper

import (
	"fmt"
)

///////////////////////////////////////////////////////////////////////////////
// CGO

/*
#include <whisper.h>
*/
import "C"

///////////////////////////////////////////////////////////////////////////////
// PUBLIC METHODS

func (p *Params) SetTranslate(v bool) {
	p.translate = toBool(v)
}

func (p *Params) SetSplitOnWord(v bool) {
	p.split_on_word = toBool(v)
}

func (p *Params) SetNoContext(v bool) {
	p.no_context = toBool(v)
}

func (p *Params) SetSingleSegment(v bool) {
	p.single_segment = toBool(v)
}

func (p *Params) SetPrintSpecial(v bool) {
	p.print_special = toBool(v)
}

func (p *Params) SetPrintProgress(v bool) {
	p.print_progress = toBool(v)
}

func (p *Params) SetPrintRealtime(v bool) {
	p.print_realtime = toBool(v)
}

func (p *Params) SetPrintTimestamps(v bool) {
	p.print_timestamps = toBool(v)
}

// Set language id
func (p *Params) SetLanguage(lang int) error {
	if lang == -1 {
		p.language = nil
		return nil
	}
	str := C.whisper_lang_str(C.int(lang))
	if str == nil {
		return ErrInvalidLanguage
	} else {
		p.language = str
	}
	return nil
}

// Get language id
func (p *Params) Language() int {
	if p.language == nil {
		return -1
	}
	return int(C.whisper_lang_id(p.language))
}

// Threads available
func (p *Params) Threads() int {
	return int(p.n_threads)
}

// Set number of threads to use
func (p *Params) SetThreads(threads int) {
	p.n_threads = C.int(threads)
}

// Set start offset in ms
func (p *Params) SetOffset(offset_ms int) {
	p.offset_ms = C.int(offset_ms)
}

// Set audio duration to process in ms
func (p *Params) SetDuration(duration_ms int) {
	p.duration_ms = C.int(duration_ms)
}

// Set timestamp token probability threshold (~0.01)
func (p *Params) SetTokenThreshold(t float32) {
	p.thold_pt = C.float(t)
}

// Set timestamp token sum probability threshold (~0.01)
func (p *Params) SetTokenSumThreshold(t float32) {
	p.thold_ptsum = C.float(t)
}

// Set max segment length in characters
func (p *Params) SetMaxSegmentLength(n int) {
	p.max_len = C.int(n)
}

func (p *Params) SetTokenTimestamps(b bool) {
	p.token_timestamps = toBool(b)
}

// Set max tokens per segment (0 = no limit)
func (p *Params) SetMaxTokensPerSegment(n int) {
	p.max_tokens = C.int(n)
}

// Set audio encoder context
func (p *Params) SetAudioCtx(n int) {
	p.audio_ctx = C.int(n)
}

func (p *Params) SetMaxContext(n int) {
	p.n_max_text_ctx = C.int(n)
}

func (p *Params) SetBeamSize(n int) {
	p.beam_search.beam_size = C.int(n)
}

func (p *Params) SetEntropyThold(t float32) {
	p.entropy_thold = C.float(t)
}

func (p *Params) SetTemperature(t float32) {
	p.temperature = C.float(t)
}

// Sets the fallback temperature incrementation
// Pass -1.0 to disable this feature
func (p *Params) SetTemperatureFallback(t float32) {
	p.temperature_inc = C.float(t)
}

// Set initial prompt
func (p *Params) SetInitialPrompt(prompt string) {
	p.initial_prompt = C.CString(prompt)
}

///////////////////////////////////////////////////////////////////////////////
// PRIVATE METHODS

func toBool(v bool) C.bool {
	if v {
		return C.bool(true)
	}
	return C.bool(false)
}

///////////////////////////////////////////////////////////////////////////////
// STRINGIFY

func (p *Params) String() string {
	str := "<whisper.params"
	str += fmt.Sprintf(" strategy=%v", p.strategy)
	str += fmt.Sprintf(" n_threads=%d", p.n_threads)
	if p.language != nil {
		str += fmt.Sprintf(" language=%s", C.GoString(p.language))
	}
	str += fmt.Sprintf(" n_max_text_ctx=%d", p.n_max_text_ctx)
	str += fmt.Sprintf(" offset_ms=%d", p.offset_ms)
	str += fmt.Sprintf(" duration_ms=%d", p.duration_ms)
	str += fmt.Sprintf(" audio_ctx=%d", p.audio_ctx)
	str += fmt.Sprintf(" initial_prompt=%s", C.GoString(p.initial_prompt))
	str += fmt.Sprintf(" entropy_thold=%f", p.entropy_thold)
	str += fmt.Sprintf(" temperature=%f", p.temperature)
	str += fmt.Sprintf(" temperature_inc=%f", p.temperature_inc)
	str += fmt.Sprintf(" beam_size=%d", p.beam_search.beam_size)
	if p.translate {
		str += " translate"
	}
	if p.no_context {
		str += " no_context"
	}
	if p.single_segment {
		str += " single_segment"
	}
	if p.print_special {
		str += " print_special"
	}
	if p.print_progress {
		str += " print_progress"
	}
	if p.print_realtime {
		str += " print_realtime"
	}
	if p.print_timestamps {
		str += " print_timestamps"
	}
	if p.token_timestamps {
		str += " token_timestamps"
	}

	return str + ">"
}
