package whisper

import (
	"os"
	"testing"

	w "github.com/ggerganov/whisper.cpp/bindings/go"
	assert "github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

const testModelPathState = "../../models/ggml-tiny.en.bin"

func TestWhisperState_NilWrapper(t *testing.T) {
	ws := newWhisperState(nil)

	state, err := ws.unsafeState()
	assert.Nil(t, state)
	require.ErrorIs(t, err, ErrModelClosed)

	require.NoError(t, ws.close())
	// idempotent
	require.NoError(t, ws.close())
}

func TestWhisperState_Lifecycle(t *testing.T) {
	if _, err := os.Stat(testModelPathState); os.IsNotExist(err) {
		t.Skip("Skipping test, model not found:", testModelPathState)
	}

	ctx := w.Whisper_init(testModelPathState)
	require.NotNil(t, ctx)
	defer ctx.Whisper_free()

	state := ctx.Whisper_init_state()
	require.NotNil(t, state)

	ws := newWhisperState(state)

	got, err := ws.unsafeState()
	require.NoError(t, err)
	require.NotNil(t, got)

	// close frees underlying state and marks closed
	require.NoError(t, ws.close())

	got, err = ws.unsafeState()
	assert.Nil(t, got)
	require.ErrorIs(t, err, ErrModelClosed)

	// idempotent
	require.NoError(t, ws.close())
}
