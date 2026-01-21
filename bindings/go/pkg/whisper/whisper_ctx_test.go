package whisper

import (
	"os"
	"testing"

	w "github.com/ggerganov/whisper.cpp/bindings/go"
	assert "github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const testModelPathCtx = "../../models/ggml-tiny.en.bin"

func TestWhisperCtx_NilWrapper(t *testing.T) {
	wctx := newCtxAccessor(nil)

	assert.True(t, wctx.isClosed())

	raw, err := wctx.context()
	assert.Nil(t, raw)
	require.ErrorIs(t, err, ErrModelClosed)

	require.NoError(t, wctx.close())
	// idempotent
	require.NoError(t, wctx.close())
}

func TestWhisperCtx_Lifecycle(t *testing.T) {
	if _, err := os.Stat(testModelPathCtx); os.IsNotExist(err) {
		t.Skip("Skipping test, model not found:", testModelPathCtx)
	}

	raw := w.Whisper_init(testModelPathCtx)
	require.NotNil(t, raw)

	wctx := newCtxAccessor(raw)
	assert.False(t, wctx.isClosed())

	got, err := wctx.context()
	require.NoError(t, err)
	require.NotNil(t, got)

	// close frees underlying ctx and marks closed
	require.NoError(t, wctx.close())
	assert.True(t, wctx.isClosed())

	got, err = wctx.context()
	assert.Nil(t, got)
	require.ErrorIs(t, err, ErrModelClosed)

	// idempotent
	require.NoError(t, wctx.close())
	// no further free; raw already freed by wctx.Close()
}

func TestWhisperCtx_FromModelLifecycle(t *testing.T) {
	if _, err := os.Stat(testModelPathCtx); os.IsNotExist(err) {
		t.Skip("Skipping test, model not found:", testModelPathCtx)
	}

	modelNew, err := New(testModelPathCtx)
	require.NoError(t, err)
	require.NotNil(t, modelNew)

	model := modelNew.(*ModelContext)

	wc := model.ctxAccessor()
	require.NotNil(t, wc)

	// Should be usable before model.Close
	raw, err := wc.context()
	require.NoError(t, err)
	require.NotNil(t, raw)

	// Close model should close underlying context
	require.NoError(t, model.Close())

	assert.True(t, wc.isClosed())
	raw, err = wc.context()
	assert.Nil(t, raw)
	require.ErrorIs(t, err, ErrModelClosed)

	// Idempotent close on wrapper
	require.NoError(t, wc.close())
}
