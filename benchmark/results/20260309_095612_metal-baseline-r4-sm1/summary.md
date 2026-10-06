| Variant | Model | Audio Length | Runs | Init Mean | First Inference Mean | Runtime Median | Throughput | Std Dev | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| metal-baseline-r4-sm1 | ggml-small.en.bin | 30.000s | 5 | 554.89 ms | 4.332 s | 4.659 s | 6.480 audio-s/s | 0.320 s | tokens/s unavailable; encode mean=2680.76 ms; decode mean=799.27 ms; wer median=0.0000; cer median=0.0000 |
| metal-baseline-r4-sm1 | ggml-small.en.bin | 300.000s | 5 | 616.22 ms | 5.254 s | 130.465 s | 2.343 audio-s/s | 9.070 s | tokens/s unavailable; encode mean=43349.16 ms; decode mean=74948.45 ms; wer median=0.0000; cer median=0.0000 |
