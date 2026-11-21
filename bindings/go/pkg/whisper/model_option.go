package whisper

import whisper "github.com/ggerganov/whisper.cpp/bindings/go"

type ContextParams = whisper.ContextParams

type (
	modelOption     interface{ apply(*ContextParams) }
	modelOptionFunc func(*ContextParams)
)

func (fn modelOptionFunc) apply(to *ContextParams) {
	fn(to)
}

func WithUseGPU(v bool) modelOption {
	return modelOptionFunc(func(p *ContextParams) {
		p.SetUseGPU(v)
	})
}

func WithUseFlashAttention(v bool) modelOption {
	return modelOptionFunc(func(p *ContextParams) {
		p.SetUseFlashAttention(v)
	})
}
