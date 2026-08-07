package whisper

import whisper "github.com/ggerganov/whisper.cpp/bindings/go"

type ctxAccessor struct {
	ctx *whisper.Context
}

func newCtxAccessor(ctx *whisper.Context) *ctxAccessor {
	return &ctxAccessor{
		ctx: ctx,
	}
}

func (ctx *ctxAccessor) close() error {
	if ctx.ctx == nil {
		return nil
	}

	ctx.ctx.Whisper_free()
	ctx.ctx = nil

	return nil
}

func (ctx *ctxAccessor) isClosed() bool {
	return ctx.ctx == nil
}

func (ctx *ctxAccessor) context() (*whisper.Context, error) {
	if ctx.isClosed() {
		return nil, ErrModelClosed
	}

	return ctx.ctx, nil
}
