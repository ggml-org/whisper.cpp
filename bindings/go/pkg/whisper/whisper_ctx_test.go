package whisper

import (
	"os"
	"testing"

	w "github.com/ggerganov/whisper.cpp/bindings/go"
	assert "github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const testModelPathCtx = "../../models/ggml-small.en.bin"

func TestWhisperCtx_NilWrapper(t *testing.T) {
	wctx := newWhisperCtx(nil)

	assert.True(t, wctx.IsClosed())

	raw, err := wctx.unsafeContext()
	assert.Nil(t, raw)
	require.ErrorIs(t, err, ErrModelClosed)

	require.NoError(t, wctx.Close())
	// idempotent
	require.NoError(t, wctx.Close())
}

func TestWhisperCtx_Lifecycle(t *testing.T) {
	if _, err := os.Stat(testModelPathCtx); os.IsNotExist(err) {
		t.Skip("Skipping test, model not found:", testModelPathCtx)
	}

	raw := w.Whisper_init(testModelPathCtx)
	require.NotNil(t, raw)

	wctx := newWhisperCtx(raw)
	assert.False(t, wctx.IsClosed())

	got, err := wctx.unsafeContext()
	require.NoError(t, err)
	require.NotNil(t, got)

	// close frees underlying ctx and marks closed
	require.NoError(t, wctx.Close())
	assert.True(t, wctx.IsClosed())

	got, err = wctx.unsafeContext()
	assert.Nil(t, got)
	require.ErrorIs(t, err, ErrModelClosed)

	// idempotent
	require.NoError(t, wctx.Close())
	// no further free; raw already freed by wctx.Close()
}

func TestWhisperCtx_FromModelLifecycle(t *testing.T) {
	if _, err := os.Stat(testModelPathCtx); os.IsNotExist(err) {
		t.Skip("Skipping test, model not found:", testModelPathCtx)
	}

	modelNew, err := New(testModelPathCtx)
	require.NoError(t, err)
	require.NotNil(t, modelNew)

	model := modelNew.(*model)

	wc := model.whisperContext()
	require.NotNil(t, wc)

	// Should be usable before model.Close
	raw, err := wc.unsafeContext()
	require.NoError(t, err)
	require.NotNil(t, raw)

	// Close model should close underlying context
	require.NoError(t, model.Close())

	assert.True(t, wc.IsClosed())
	raw, err = wc.unsafeContext()
	assert.Nil(t, raw)
	require.ErrorIs(t, err, ErrModelClosed)

	// Idempotent close on wrapper
	require.NoError(t, wc.Close())
}
