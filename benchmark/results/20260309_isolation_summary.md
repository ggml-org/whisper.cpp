# Isolation Benchmark Summary (2026-03-09)

| Audio | Variant | Runtime Median (s) | Delta vs Baseline | Throughput (audio-s/s) | Delta Throughput | Std Dev (s) | Correctness |
|---|---|---:|---:|---:|---:|---:|---|
| short | Baseline (r3) | 1.0740 | 0.00% | 27.7949 | 0.00% | 0.0114 | pass |
| short | Option A (ne11_mm_min 8->6 on Apple7+) | 1.0769 | -0.27% | 27.9115 | +0.42% | 0.0049 | pass |
| short | Option B (ne21_mm_id_min 32->24 on Apple7+) | 1.0754 | -0.13% | 27.9177 | +0.44% | 0.0020 | pass |
| medium | Baseline (r3) | 16.6943 | 0.00% | 17.9677 | 0.00% | 0.0075 | pass |
| medium | Option A (ne11_mm_min 8->6 on Apple7+) | 16.9041 | -1.26% | 19.3549 | +7.72% | 2.7487 | pass |
| medium | Option B (ne21_mm_id_min 32->24 on Apple7+) | 10.7435 | +35.65% | 27.9097 | +55.33% | 0.0125 | pass |
| long | Baseline (r3) | 116.0024 | 0.00% | 14.4765 | 0.00% | 50.6682 | pass |
| long | Option A (ne11_mm_min 8->6 on Apple7+) | 79.5619 | +31.41% | 22.5231 | +55.58% | 2.0796 | pass |
| long | Option B (ne21_mm_id_min 32->24 on Apple7+) | 75.9855 | +34.50% | 23.6331 | +63.25% | 0.4570 | pass |
