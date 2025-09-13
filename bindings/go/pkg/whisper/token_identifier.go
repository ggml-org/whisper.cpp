package whisper

import whisper "github.com/ggerganov/whisper.cpp/bindings/go"

type tokenIdentifier struct {
	ctx *whisperCtx
}

func newTokenIdentifier(whisperContext *whisperCtx) *tokenIdentifier {
	return &tokenIdentifier{
		ctx: whisperContext,
	}
}

// Token type checking methods (model-specific vocabulary)
func (ti *tokenIdentifier) IsBEG(t Token) (bool, error) {
	ctx, err := ti.ctx.UnsafeContext()
	if err != nil {
		return false, err
	}

	return whisper.Token(t.Id) == ctx.Whisper_token_beg(), nil
}

func (ti *tokenIdentifier) IsEOT(t Token) (bool, error) {
	ctx, err := ti.ctx.UnsafeContext()
	if err != nil {
		return false, err
	}

	return whisper.Token(t.Id) == ctx.Whisper_token_eot(), nil
}

func (ti *tokenIdentifier) IsSOT(t Token) (bool, error) {
	ctx, err := ti.ctx.UnsafeContext()
	if err != nil {
		return false, err
	}

	return whisper.Token(t.Id) == ctx.Whisper_token_sot(), nil
}

func (ti *tokenIdentifier) IsPREV(t Token) (bool, error) {
	ctx, err := ti.ctx.UnsafeContext()
	if err != nil {
		return false, err
	}

	return whisper.Token(t.Id) == ctx.Whisper_token_prev(), nil
}

func (ti *tokenIdentifier) IsSOLM(t Token) (bool, error) {
	ctx, err := ti.ctx.UnsafeContext()
	if err != nil {
		return false, err
	}

	return whisper.Token(t.Id) == ctx.Whisper_token_solm(), nil
}

func (ti *tokenIdentifier) IsNOT(t Token) (bool, error) {
	ctx, err := ti.ctx.UnsafeContext()
	if err != nil {
		return false, err
	}

	return whisper.Token(t.Id) == ctx.Whisper_token_not(), nil
}

func (ti *tokenIdentifier) IsLANG(t Token, lang string) (bool, error) {
	ctx, err := ti.ctx.UnsafeContext()
	if err != nil {
		return false, err
	}

	if id := ctx.Whisper_lang_id(lang); id >= 0 {
		return whisper.Token(t.Id) == ctx.Whisper_token_lang(id), nil
	}

	return false, nil
}

func (ti *tokenIdentifier) IsText(t Token) (bool, error) {
	// Check if it's any of the special tokens
	if isBeg, _ := ti.IsBEG(t); isBeg {
		return false, nil
	}

	if isSot, _ := ti.IsSOT(t); isSot {
		return false, nil
	}

	ctx, err := ti.ctx.UnsafeContext()
	if err != nil {
		return false, err
	}

	if whisper.Token(t.Id) >= ctx.Whisper_token_eot() {
		return false, nil
	}

	if isPrev, _ := ti.IsPREV(t); isPrev {
		return false, nil
	}

	if isSolm, _ := ti.IsSOLM(t); isSolm {
		return false, nil
	}

	if isNot, _ := ti.IsNOT(t); isNot {
		return false, nil
	}

	return true, nil
}
