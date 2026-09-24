package whisper

func (p *ContextParams) UseGPU() bool {
	return bool(p.use_gpu)
}

func (p *ContextParams) SetUseGPU(v bool) {
	p.use_gpu = toBool(v)
}

func (p *ContextParams) UseFlashAttention() bool {
	return bool(p.flash_attn)
}

func (p *ContextParams) SetUseFlashAttention(v bool) {
	p.flash_attn = toBool(v)
}
