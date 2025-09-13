package whisper

import whisper "github.com/ggerganov/whisper.cpp/bindings/go"

type whisperCtx struct {
	ctx *whisper.Context
}

func newWhisperCtx(ctx *whisper.Context) *whisperCtx {
	return &whisperCtx{
		ctx: ctx,
	}
}

func (ctx *whisperCtx) close() error {
	if ctx.ctx == nil {
		return nil
	}

	ctx.ctx.Whisper_free()
	ctx.ctx = nil

	return nil
}

func (ctx *whisperCtx) isClosed() bool {
	return ctx.ctx == nil
}

func (ctx *whisperCtx) unsafeContext() (*whisper.Context, error) {
	if ctx.isClosed() {
		return nil, ErrModelClosed
	}

	return ctx.ctx, nil
}
