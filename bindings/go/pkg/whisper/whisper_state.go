package whisper

import whisper "github.com/ggerganov/whisper.cpp/bindings/go"

type WhisperState interface {
	Close() error
	UnsafeState() (*whisper.State, error)
}

type whisperState struct {
	state *whisper.State
}

func newWhisperState(state *whisper.State) WhisperState {
	return &whisperState{
		state: state,
	}
}

func (s *whisperState) Close() error {
	if s.state == nil {
		return nil
	}

	s.state.Whisper_free_state()
	s.state = nil

	return nil
}

func (s *whisperState) UnsafeState() (*whisper.State, error) {
	if s.state == nil {
		return nil, ErrModelClosed
	}

	return s.state, nil
}
