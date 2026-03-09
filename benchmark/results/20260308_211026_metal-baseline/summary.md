| Variant | Model | Audio Length | Runs | Init Mean | First Inference Mean | Runtime Median | Throughput | Std Dev | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| metal-baseline | ggml-small.en.bin | 30.000s | 5 | 317.42 ms | 2.332 s | 2.611 s | 11.485 audio-s/s | 0.004 s | tokens/s unavailable; encode mean=1611.59 ms; decode mean=258.39 ms; wer median=0.0000; cer median=0.0000 |
| metal-baseline | ggml-small.en.bin | 300.000s | 5 | 371.39 ms | 3.505 s | 80.664 s | 3.924 audio-s/s | 7.152 s | tokens/s unavailable; encode mean=31327.92 ms; decode mean=38565.07 ms; wer median=0.0000; cer median=0.0000 |
| metal-baseline | ggml-small.en.bin | 1800.000s | 5 | 269.85 ms | 1.790 s | 119.858 s | 14.971 audio-s/s | 1.195 s | tokens/s unavailable; encode mean=46698.06 ms; decode mean=58974.11 ms; wer median=0.0000; cer median=0.0000 |
