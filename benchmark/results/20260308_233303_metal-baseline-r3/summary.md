| Variant | Model | Audio Length | Runs | Init Mean | First Inference Mean | Runtime Median | Throughput | Std Dev | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| metal-baseline-r3 | ggml-small.en.bin | 30.000s | 5 | 196.96 ms | 0.782 s | 1.074 s | 27.795 audio-s/s | 0.011 s | tokens/s unavailable; encode mean=384.04 ms; decode mean=100.71 ms; wer median=0.0000; cer median=0.0000 |
| metal-baseline-r3 | ggml-small.en.bin | 300.000s | 5 | 203.26 ms | 0.882 s | 16.694 s | 17.968 audio-s/s | 0.008 s | tokens/s unavailable; encode mean=5795.70 ms; decode mean=8229.23 ms; wer median=0.0000; cer median=0.0000 |
| metal-baseline-r3 | ggml-small.en.bin | 1800.000s | 5 | 286.82 ms | 1.923 s | 116.002 s | 14.476 audio-s/s | 50.668 s | tokens/s unavailable; encode mean=48315.56 ms; decode mean=72152.48 ms; wer median=0.0000; cer median=0.0000 |
