package whisper_test

import (
	"fmt"
	"io"
	"math"
	"os"
	"runtime"
	"testing"
	"time"

	whisper "github.com/ggerganov/whisper.cpp/bindings/go/pkg/whisper"
	"github.com/go-audio/audio"
	wav "github.com/go-audio/wav"
)

func processAndExtractSegmentsSequentially(ctx whisper.Context, samples []float32) ([]whisper.Segment, error) {
	if err := ctx.Process(samples, nil, nil, nil); err != nil {
		return nil, err
	}

	var segments []whisper.Segment
	for {
		seg, err := ctx.NextSegment()
		if err == io.EOF {
			break
		} else if err != nil {
			return nil, err
		}

		segments = append(segments, seg)
	}

	return segments, nil
}

func processAndExtractSegmentsWithCallback(ctx whisper.Context, samples []float32) ([]whisper.Segment, error) {
	segments := make([]whisper.Segment, 0)

	if err := ctx.Process(samples, nil, func(seg whisper.Segment) {
		segments = append(segments, seg)
	}, nil); err != nil {
		return nil, err
	}

	return segments, nil
}

// benchProcessVariants runs the common benchmark matrix across context kinds,
// thread sets, and callback modes, for given samples. If singleIteration is true
// it runs only one iteration regardless of b.N. If printTimings is true,
// model timings and custom ms_process metric are reported for NoCallback runs.
func benchProcessVariants(
	b *testing.B,
	samples []float32,
	singleIteration bool,
	printTimings bool,
	useGPU bool,
) {
	threadSets := []uint{1, 2, 4, uint(runtime.NumCPU())}

	device := "cpu"
	if useGPU {
		device = "gpu"
	}

	// Initialize model per device mode
	mp := whisper.NewModelContextParams()
	mp.SetUseGPU(useGPU)
	model, err := whisper.NewModelContextWithParams(ModelPath, mp)
	if err != nil {
		b.Fatalf("load model (%s): %v", device, err)
	}
	defer func() { _ = model.Close() }()

	// Context kinds: stateless and stateful
	ctxKinds := []struct {
		name string
		new  func() (whisper.Context, error)
	}{
		{
			name: "stateless",
			new: func() (whisper.Context, error) {
				params, err := whisper.NewParameters(model, whisper.SAMPLING_GREEDY, func(p *whisper.Parameters) {})
				if err != nil {
					return nil, err
				}
				return whisper.NewStatelessContext(model, params)
			},
		},
		{
			name: "stateful",
			new: func() (whisper.Context, error) {
				params, err := whisper.NewParameters(model, whisper.SAMPLING_GREEDY, nil)
				if err != nil {
					return nil, err
				}
				return whisper.NewStatefulContext(model, params)
			},
		},
	}

	for _, kind := range ctxKinds {
		b.Run(device+"/"+kind.name, func(b *testing.B) {
			for _, threads := range threadSets {
				b.Run(fmt.Sprintf("threads=%d/NoCallback", threads), func(b *testing.B) {
					b.ReportAllocs()
					b.SetBytes(int64(len(samples) * 4))
					ctx, err := kind.new()
					if err != nil {
						b.Fatalf("new %s context: %v", kind.name, err)
					}
					defer func() { _ = ctx.Close() }()
					ctx.SetThreads(threads)

					iters := b.N
					if singleIteration {
						iters = 1
					}

					b.ResetTimer()
					for i := 0; i < iters; i++ {
						model.ResetTimings()
						start := time.Now()

						segments, err := processAndExtractSegmentsSequentially(ctx, samples)
						if err != nil {
							b.Fatalf("process and extract segments sequentially: %v", err)
						}

						b.Logf("segments: %+v", segments)

						elapsed := time.Since(start)

						if printTimings {
							model.PrintTimings()
						}

						b.ReportMetric(float64(elapsed.Milliseconds()), "ms_process")
					}
				})

				b.Run(fmt.Sprintf("threads=%d/WithSegmentCallback", threads), func(b *testing.B) {
					b.ReportAllocs()
					b.SetBytes(int64(len(samples) * 4))
					ctx, err := kind.new()
					if err != nil {
						b.Fatalf("new %s context: %v", kind.name, err)
					}
					defer func() { _ = ctx.Close() }()
					ctx.SetThreads(threads)

					iters := b.N
					if singleIteration {
						iters = 1
					}

					b.ResetTimer()
					for i := 0; i < iters; i++ {
						start := time.Now()
						model.ResetTimings()

						// Passing a segment callback forces single-segment mode and exercises token extraction
						segments, err := processAndExtractSegmentsWithCallback(ctx, samples)
						if err != nil {
							b.Fatalf("process with callback: %v", err)
						}

						b.Logf("segments: %+v", segments)

						elapsed := time.Since(start)
						if printTimings {
							model.PrintTimings()
						}

						b.ReportMetric(float64(elapsed.Milliseconds()), "ms_process")
					}
				})
			}
		})
	}
}

// BenchmarkContextProcess runs the high-level Context.Process across
// different thread counts, with and without segment callbacks.
func BenchmarkContextProcessCPU(b *testing.B) {
	if _, err := os.Stat(ModelPath); os.IsNotExist(err) {
		b.Skipf("model not found: %s", ModelPath)
	}
	if _, err := os.Stat(SamplePath); os.IsNotExist(err) {
		b.Skipf("sample not found: %s", SamplePath)
	}

	// Load audio once (reuse helper)
	data := helperLoadSample(b, SamplePath)

	benchProcessVariants(b, data, false, true, false)
}

// BenchmarkContextProcessBig runs one single iteration over a big input
// (the short sample concatenated 10x) to simulate long audio processing.
// This is complementary to BenchmarkContextProcess which runs many iterations
// over the short sample.
func BenchmarkContextProcessBigCPU(b *testing.B) {
	if _, err := os.Stat(ModelPath); os.IsNotExist(err) {
		b.Skipf("model not found: %s", ModelPath)
	}
	if _, err := os.Stat(SamplePath); os.IsNotExist(err) {
		b.Skipf("sample not found: %s", SamplePath)
	}

	// Load audio once (reuse helper with meta)
	data, sampleRate, numChans := helperLoadSampleWithMeta(b, SamplePath)

	// Build big dataset: input concatenated 10x
	bigData := make([]float32, len(data)*10)
	for i := 0; i < 10; i++ {
		copy(bigData[i*len(data):(i+1)*len(data)], data)
	}

	// Write the big dataset to a wav file for inspection
	outPath := "../../samples/benchmark_out.wav"
	fout, err := os.Create(outPath)
	if err != nil {
		b.Fatalf("create output wav: %v", err)
	}
	enc := wav.NewEncoder(fout, sampleRate, 16, numChans, 1)
	intBuf := &audio.IntBuffer{
		Format:         &audio.Format{NumChannels: numChans, SampleRate: sampleRate},
		SourceBitDepth: 16,
		Data:           make([]int, len(bigData)),
	}
	for i, s := range bigData {
		v := int(math.Round(float64(s) * 32767.0))
		if v > 32767 {
			v = 32767
		} else if v < -32768 {
			v = -32768
		}
		intBuf.Data[i] = v
	}
	if err := enc.Write(intBuf); err != nil {
		_ = fout.Close()
		b.Fatalf("encode wav: %v", err)
	}
	if err := enc.Close(); err != nil {
		_ = fout.Close()
		b.Fatalf("close encoder: %v", err)
	}
	_ = fout.Close()

	benchProcessVariants(b, bigData, true, true, false)
}

// GPU variants reuse model-level GPU enablement via model params
func BenchmarkContextProcessGPU(b *testing.B) {
	if _, err := os.Stat(ModelPath); os.IsNotExist(err) {
		b.Skipf("model not found: %s", ModelPath)
	}
	if _, err := os.Stat(SamplePath); os.IsNotExist(err) {
		b.Skipf("sample not found: %s", SamplePath)
	}

	data := helperLoadSample(b, SamplePath)

	benchProcessVariants(b, data, false, true, true)
}

func BenchmarkContextProcessBigGPU(b *testing.B) {
	if _, err := os.Stat(ModelPath); os.IsNotExist(err) {
		b.Skipf("model not found: %s", ModelPath)
	}
	if _, err := os.Stat(SamplePath); os.IsNotExist(err) {
		b.Skipf("sample not found: %s", SamplePath)
	}

	data, _, _ := helperLoadSampleWithMeta(b, SamplePath)

	bigData := make([]float32, len(data)*10)
	for i := 0; i < 10; i++ {
		copy(bigData[i*len(data):(i+1)*len(data)], data)
	}

	benchProcessVariants(b, bigData, true, true, true)
}
