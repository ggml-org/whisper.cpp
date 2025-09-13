package whisper

import (
	"io"
	"strings"
	"time"

	// Bindings
	whisper "github.com/ggerganov/whisper.cpp/bindings/go"
)

// state embeds context behavior and carries a low-level state pointer
// for isolated processing results.
type state struct {
	*context
	st *whisper.State
}

// NewState creates a new per-request State from a Model without changing the Model interface.
func NewState(m Model) (State, error) {
	impl, ok := m.(*model)
	if !ok {
		return nil, ErrInternalAppError
	}
	params := impl.ctx.Whisper_full_default_params(whisper.SAMPLING_GREEDY)
	params.SetTranslate(false)
	params.SetPrintSpecial(false)
	params.SetPrintProgress(false)
	params.SetPrintRealtime(false)
	params.SetPrintTimestamps(false)
	return newState(impl, params)
}

// internal constructor used by model.NewState
func newState(model *model, params whisper.Params) (State, error) {
	ctx := &context{model: model, params: params}
	st := model.ctx.Whisper_init_state()
	if st == nil {
		return nil, ErrUnableToCreateState
	}
	return &state{context: ctx, st: st}, nil
}

// Process using an isolated state for concurrency
func (s *state) Process(
	data []float32,
	callEncoderBegin EncoderBeginCallback,
	callNewSegment SegmentCallback,
	callProgress ProgressCallback,
) error {
	if s.model.ctx == nil || s.st == nil {
		return ErrInternalAppError
	}
	if callNewSegment != nil {
		s.params.SetSingleSegment(true)
	}
	if err := s.model.ctx.Whisper_full_with_state(s.st, s.params, data, callEncoderBegin,
		func(new int) {
			if callNewSegment != nil {
				num_segments := s.model.ctx.Whisper_full_n_segments_from_state(s.st)
				s0 := num_segments - new
				for i := s0; i < num_segments; i++ {
					callNewSegment(toSegmentFromState(s.model.ctx, s.st, i))
				}
			}
		}, func(progress int) {
			if callProgress != nil {
				callProgress(progress)
			}
		}); err != nil {
		return err
	}
	return nil
}

// Return the next segment of tokens for state
func (s *state) NextSegment() (Segment, error) {
	if s.model.ctx == nil {
		return Segment{}, ErrInternalAppError
	}
	if s.n >= s.model.ctx.Whisper_full_n_segments_from_state(s.st) {
		return Segment{}, io.EOF
	}
	result := toSegmentFromState(s.model.ctx, s.st, s.n)
	s.n++
	return result, nil
}

func (s *state) Close() error {
	if s.st != nil {
		s.st.Whisper_free_state()
		s.st = nil
	}
	return nil
}

// Helpers specific to state-based results
func toSegmentFromState(ctx *whisper.Context, st *whisper.State, n int) Segment {
	return Segment{
		Num:    n,
		Text:   stringsTrim(ctx.Whisper_full_get_segment_text_from_state(st, n)),
		Start:  duration10x(ctx.Whisper_full_get_segment_t0_from_state(st, n)),
		End:    duration10x(ctx.Whisper_full_get_segment_t1_from_state(st, n)),
		Tokens: toTokensFromState(ctx, st, n),
	}
}

func toTokensFromState(ctx *whisper.Context, st *whisper.State, n int) []Token {
	result := make([]Token, ctx.Whisper_full_n_tokens_from_state(st, n))
	for i := 0; i < len(result); i++ {
		data := ctx.Whisper_full_get_token_data_from_state(st, n, i)
		result[i] = Token{
			Id:    int(ctx.Whisper_full_get_token_id_from_state(st, n, i)),
			Text:  ctx.Whisper_full_get_token_text_from_state(st, n, i),
			P:     ctx.Whisper_full_get_token_p_from_state(st, n, i),
			Start: duration10x(data.T0()),
			End:   duration10x(data.T1()),
		}
	}
	return result
}

// small shared helpers to avoid importing time/strings here unnecessarily
func stringsTrim(s string) string          { return strings.TrimSpace(s) }
func duration10x(ms10 int64) time.Duration { return time.Duration(ms10) * time.Millisecond * 10 }
