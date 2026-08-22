# Go bindings for Whisper

This package provides Go bindings for whisper.cpp. They have been tested on:

  * Darwin (OS X) 12.6 on x64_64
  * Debian Linux on arm64
  * Fedora Linux on x86_64

The "low level" bindings are in the `bindings/go` directory and there is a more
Go-style package in the `bindings/go/pkg/whisper` directory.

Legacy stateless example (single worker). For the recommended stateful API and
concurrency-safe usage, see "New high-level API" below. Note: `Model.NewContext()`
returns a stateless context for backward compatibility and is not safe for parallel
`Process` calls (may return `ErrStatelessBusy`).

```go
import (
	"github.com/ggerganov/whisper.cpp/bindings/go/pkg/whisper"
)

func main() {
	var modelpath string // Path to the model
	var samples []float32 // Samples to process

	// Load the model
	model, err := whisper.New(modelpath)
	if err != nil {
		panic(err)
	}
	defer model.Close()

	// Process samples
	context, err := model.NewContext()
	if err != nil {
		panic(err)
	}
	if err := context.Process(samples, nil, nil, nil); err != nil {
		return err
	}

	// Print out the results
	for {
		segment, err := context.NextSegment()
		if err != nil {
			break
		}
		fmt.Printf("[%6s->%6s] %s\n", segment.Start, segment.End, segment.Text)
	}
}
```

## Building & Testing

In order to build, you need to have the Go compiler installed. You can get it from [here](https://golang.org/dl/). Run the tests with:

```bash
git clone https://github.com/ggml-org/whisper.cpp.git
cd whisper.cpp/bindings/go
make test
```

This will compile a static `libwhisper.a` in a `build` folder, download a model file, then run the tests. To build the examples:

```bash
make examples
```

To build using cuda support add `GGML_CUDA=1`:

```bash
GGML_CUDA=1 make examples
```

The examples are placed in the `build` directory. Once built, you can download all the models with the following command:

```bash
./build/go-model-download -out models
```

And you can then test a model against samples with the following command:

```bash
./build/go-whisper -model models/ggml-tiny.en.bin samples/jfk.wav
```

## Using the bindings

To use the bindings in your own software,

  1. Import `github.com/ggerganov/whisper.cpp/bindings/go/pkg/whisper` (or `github.com/ggerganov/whisper.cpp/bindings/go` into your package;
  2. Compile `libwhisper.a` (you can use `make whisper` in the `bindings/go` directory);
  3. Link your go binary against whisper by setting the environment variables `C_INCLUDE_PATH` and `LIBRARY_PATH`
     to point to the `whisper.h` file directory and `libwhisper.a` file directory respectively.

Look at the `Makefile` in the `bindings/go` directory for an example.

The API Documentation:

  * https://pkg.go.dev/github.com/ggerganov/whisper.cpp/bindings/go
  * https://pkg.go.dev/github.com/ggerganov/whisper.cpp/bindings/go/pkg/whisper

Getting help:

  * Follow the discussion for the go bindings [here](https://github.com/ggml-org/whisper.cpp/discussions/312)

## New high-level API (stateful and stateless contexts)

The `pkg/whisper` package now exposes two context kinds:

- StatefulContext: recommended for concurrency. Each context owns its own whisper_state.
- StatelessContext: shares the model context. Simpler, but not suitable for parallel `Process` calls.

### Quick start: stateful context (recommended)

```go
package main

import (
    "fmt"
    whisper "github.com/ggerganov/whisper.cpp/bindings/go/pkg/whisper"
)

func main() {
    // Load model
    model, err := whisper.NewModelContext("./models/ggml-small.en.bin")
    if err != nil {
        panic(err)
    }
    defer model.Close()

    // Configure parameters (optional: provide a config func)
    params, err := whisper.NewParameters(model, whisper.SAMPLING_GREEDY, func(p *whisper.Parameters) {
        p.SetThreads(4)
        p.SetLanguage("en") // or "auto"
        p.SetTranslate(false)
    })
    if err != nil {
        panic(err)
    }

    // Create stateful context (safe for running in parallel goroutines)
    ctx, err := whisper.NewStatefulContext(model, params)
    if err != nil {
        panic(err)
    }
    defer ctx.Close()

    // Your 16-bit mono PCM at 16kHz as float32 samples
    var samples []float32

    // Process. Callbacks are optional.
    if err := ctx.Process(samples, nil, nil, nil); err != nil {
        panic(err)
    }

    // Read segments
    for {
        seg, err := ctx.NextSegment()
        if err != nil {
            break
        }
        fmt.Printf("[%v -> %v] %s\n", seg.Start, seg.End, seg.Text)
    }
}
```

### Quick start: stateless context (single worker)

```go
// Load model as above
model, _ := whisper.NewModelContext("./models/ggml-small.en.bin")
defer model.Close()

params, _ := whisper.NewParameters(model, whisper.SAMPLING_GREEDY, nil)
ctx, _ := whisper.NewStatelessContext(model, params)
defer ctx.Close()

if err := ctx.Process(samples, nil, nil, nil); err != nil { panic(err) }
for {
    seg, err := ctx.NextSegment()
    if err != nil { break }
    fmt.Println(seg.Text)
}
```

### Deprecations and migration notes

- The `Context` interface setters are deprecated (SetThreads, SetLanguage, etc.). Use `Parameters` via `NewParameters` and pass it when creating a context.
- `Model.NewContext()` remains for backward compatibility and returns a stateless context by default. Prefer `NewStatefulContext` for concurrency.
- Stateless contexts share the model context. A concurrency gate prevents overlapping `Process` calls and will return `ErrStatelessBusy` if another `Process` is in flight.
- For parallel processing, create one `StatefulContext` per goroutine.

## Benchmarks

Benchmarks live in `pkg/whisper` and compare CPU vs GPU, stateful vs stateless, threads, and callback modes.

### Prerequisites

- Model: `models/ggml-small.en.bin` (or your choice).
- Sample: `samples/jfk.wav`.
- Build the C libs once (also downloads a model for examples):

```bash
cd bindings/go
make examples
# optionally: ./build/go-model-download -out models
```

### Run benchmarks

```bash
cd bindings/go/pkg/whisper
make benchmark
```

### What the benchmarks measure

- Variants: device (cpu/gpu) x context kind (stateless/stateful) x threads {1,2,4, NumCPU} x callback mode (NoCallback, WithSegmentCallback).
- Standard Go benchmark outputs: ns/op, B/op, allocs/op. We also set bytes per op to sample bytes.
- Custom metric `ms_process`: wall time per `Process` iteration, reported via `b.ReportMetric`.
- When `printTimings` is enabled, model-level timings are printed for NoCallback runs using `model.PrintTimings()`.

## License

The license for the Go bindings is the same as the license for the rest of the whisper.cpp project, which is the MIT License. See the `LICENSE` file for more details.

