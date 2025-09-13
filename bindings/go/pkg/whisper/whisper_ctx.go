package whisper

import whisper "github.com/ggerganov/whisper.cpp/bindings/go"

type WhisperContext interface {
	// Close closes the whisper context
	Close() error

	// IsClosed returns true if the whisper context is closed
	IsClosed() bool
}

type whisperCtx struct {
	ctx *whisper.Context
}

func newWhisperCtx(ctx *whisper.Context) *whisperCtx {
	return &whisperCtx{
		ctx: ctx,
	}
}

func (ctx *whisperCtx) Close() error {
	if ctx.ctx == nil {
		return nil
	}

	ctx.ctx.Whisper_free()
	ctx.ctx = nil

	return nil
}

func (ctx *whisperCtx) IsClosed() bool {
	return ctx.ctx == nil
}

func (ctx *whisperCtx) unsafeContext() (*whisper.Context, error) {
	if ctx.IsClosed() {
		return nil, ErrModelClosed
	}

	return ctx.ctx, nil
}

var _ WhisperContext = (*whisperCtx)(nil)
