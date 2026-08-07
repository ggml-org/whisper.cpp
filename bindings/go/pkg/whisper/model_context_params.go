package whisper

import (
	low "github.com/ggerganov/whisper.cpp/bindings/go"
)

type ModelContextParams struct {
	p low.ContextParams
}

func NewModelContextParams() ModelContextParams {
	return ModelContextParams{
		p: low.Whisper_context_default_params(),
	}
}

func (p *ModelContextParams) SetUseGPU(v bool) {
	p.p.SetUseGPU(v)
}

func (p *ModelContextParams) SetGPUDevice(n int) {
	p.p.SetGPUDevice(n)
}

func (p *ModelContextParams) toLow() low.ContextParams {
	return p.p
}
