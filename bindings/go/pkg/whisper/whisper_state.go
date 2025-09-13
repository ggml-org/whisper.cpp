package whisper

import whisper "github.com/ggerganov/whisper.cpp/bindings/go"

type whisperState struct {
	state *whisper.State
}

func newWhisperState(state *whisper.State) *whisperState {
	return &whisperState{
		state: state,
	}
}

func (s *whisperState) close() error {
	if s.state == nil {
		return nil
	}

	s.state.Whisper_free_state()
	s.state = nil

	return nil
}

func (s *whisperState) unsafeState() (*whisper.State, error) {
	if s.state == nil {
		return nil, ErrModelClosed
	}

	return s.state, nil
}
