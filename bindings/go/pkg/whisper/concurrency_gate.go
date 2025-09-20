package whisper

import (
	"sync"
	"sync/atomic"

	// Bindings
	whisper "github.com/ggerganov/whisper.cpp/bindings/go"
)

// Gate provides a simple acquire/release contract per key.
// The default implementation is a single-entry lock per key (limit=1).
type Gate interface {
	// Acquire returns true if the key was acquired; false if already held
	Acquire(key any) bool
	// Release releases the key if currently held
	Release(key any)
}

// singleFlightGate is a minimal lock with limit=1 per key
type singleFlightGate struct {
	m sync.Map // key -> *int32 (0 available, 1 held)
}

func (g *singleFlightGate) Acquire(key any) bool {
	ptr, _ := g.m.LoadOrStore(key, new(int32))
	busy := ptr.(*int32)
	return atomic.CompareAndSwapInt32(busy, 0, 1)
}

func (g *singleFlightGate) Release(key any) {
	if v, ok := g.m.Load(key); ok {
		atomic.StoreInt32(v.(*int32), 0)
	}
}

var defaultGate Gate = &singleFlightGate{}

// SetGate allows applications to override the default gate (e.g., for custom policies)
// Passing nil resets to the default singleFlightGate.
func SetGate(g Gate) {
	if g == nil {
		defaultGate = &singleFlightGate{}
		return
	}
	defaultGate = g
}

func gate() Gate { return defaultGate }

// modelKey derives a stable key per underlying model context for guarding stateless ops
func modelKey(model *ModelContext) *whisper.Context {
	if model == nil || model.ctxAccessor() == nil {
		return nil
	}
	ctx, _ := model.ctxAccessor().context()
	return ctx
}
