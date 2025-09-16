package whisper

type (
	contextParamsOption     interface{ apply(*ContextParams) }
	contextParamsOptionFunc func(*ContextParams)
)

func (fn contextParamsOptionFunc) apply(to *ContextParams) {
	fn(to)
}

func WithUseGPU(v bool) contextParamsOption {
	return contextParamsOptionFunc(func(p *ContextParams) {
		p.SetUseGPU(v)
	})
}

func (p *ContextParams) UseGPU() bool {
	return bool(p.use_gpu)
}

func (p *ContextParams) SetUseGPU(v bool) {
	p.use_gpu = toBool(v)
}
