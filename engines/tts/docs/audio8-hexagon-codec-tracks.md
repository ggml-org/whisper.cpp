# Audio8 Hexagon codec experiments, October 14

## Original experiment protocol

Only optimized Hexagon was measured for these experiments. Saved OpenCL measurements are reused; no fresh GPU or unoptimized Hexagon performance series was run. Each input has one warmup and three timed fresh-process captures, with medians excluding loading and output file IO. Inputs rotate across timed rounds. Decoding is greedy, seed 42, four CPU threads, six HVX workers, natural EOS, cap 512 frames, and profiling off.

Before each capture the runner settles for at least 50 seconds and requires three consecutive ready snapshots: unplugged, screen asleep/dozing, thermal status 0, and HAL skin at most 38.5 C. End snapshots must remain unplugged, asleep/dozing and at thermal status 0. Runtime/model/script hashes, output geometry, code/WAV hashes, thermal/power/battery and CPU frequency snapshots are retained. Actual DSP/DDR clocks are unrecorded; snapshots do not monitor the display continuously.

4x2, LM repacking, supported default flash decoding and vectorized softmax remain enabled. The earlier experimental HVX blocked reuse and residual fusion switches are 0. Normal backend operation fusion remains enabled; existing HMX matmul/ADD fusion can apply in Plan B.

Plan A is opt-in F32 multi-tap convolution tiling in VTCM (`AUDIO8_CODEC_HEXAGON_FP16=0`, `AUDIO8_CODEC_HEXAGON_CONV_TILE=1`). Plan B independently permits FP16 activation/HMX synthesis products; it was measured with `AUDIO8_CODEC_HEXAGON_FP16=1` and is now the default when that variable is unset. FP16 takes precedence while enabled. LM, latent transformer and small channel tails retain their existing precision.

## Plan A result

| Input | Previous saved codec s | Plan A codec s | Plan A codec range s | Frames |
| --- | ---: | ---: | ---: | ---: |
| S1 | 2.784 | 2.936 | 2.733–5.351 | 69 |
| S2 | 1.495 | 2.935 | 2.157–3.358 | 49 |
| S3 | 3.013 | 3.377 | 3.025–3.449 | 74 |
| S4 | 4.302 | 5.673 | 3.353–7.309 | 81 |
| S5 | 3.508 | 5.574 | 5.000–8.925 | 88 |

Plan A showed no codec improvement against the saved previous results. All 20 captures produced byte-identical codes and WAVs to preceding Hexagon, so this comparison retains the same generated workload. Separate batches and substantial timing spread do not establish the cause of slowdown. The prototype removes full-size tap intermediates but retains F32 dot-product computation.

## Focused correctness

The F32 kernel regression passed ordinary, dilated and transposed DDR cases at K=96, 192 and 384: full output byte equality with original Hexagon matmul/ADD and sampled independent CPU arithmetic at worker/tile boundaries. A separate fixed-code 81-frame S4 profile confirmed 11 tiled operations and an identical WAV.

Plan B uses fixed identical codes and unclipped float PCM, without alignment, gain normalization or lag search. The three-frame CPU comparison passed cosine 0.999998349 and NMSE 0.00000330349 over 6,144 samples. The 81-frame S4 comparison with preceding Hexagon passed cosine 0.999999634 and NMSE 0.000000731047 over 165,888 samples. Both were finite and satisfy cosine >=0.9999 and NMSE <=0.0002. A separate profile confirmed 117 HMX matmuls. These focused codec checks do not replace the archived full LM comparisons, whose strict gates do not universally pass.

## Full inference comparison

Latest median full inference seconds, excluding model loading and output file writing. S4 uses the five timed default-path Hexagon recheck below; S1/S2/S3/S5 retain their original three-run Plan B medians. OpenCL reuses the saved three-run GPU reference for every prompt. Totals use the recorded full-inference timer, not sums of independent stage medians.

RTF = full inference seconds / exact generated WAV duration. Lower is better; RTF <1 is faster than real time. Each backend uses its own exact WAV duration rather than a rounded CLI duration or the other backend's audio length.

| Input | OpenCL GPU s | Hexagon s | OpenCL RTF | Hexagon RTF | Frames GPU/Hexagon |
| --- | ---: | ---: | ---: | ---: | ---: |
| S1 | 3.546 | 3.057 | 1.005 | 0.954 | 76/69 |
| S2 | 2.392 | 1.972 | 1.030 | 0.866 | 50/49 |
| S3 | 3.374 | 3.497 | 1.009 | 1.017 | 72/74 |
| S4 | 3.147 | 3.448 | 1.011 | 0.917 | 67/81 |
| S5 | 4.373 | 4.209 | 1.002 | 1.030 | 94/88 |

Hexagon has lower full-inference prompt medians for S1, S2 and S5. S3 and S4 remain slower in latency. S4 now measures 3.448 s rather than the original 4.116 s; its output is unchanged. Its NPU RTF is lower than OpenCL because the generated NPU audio is longer (3.761633 versus 3.111474 s). Different trajectories/frame counts, separate batches and timing spread prevent identical-workload or universal speedup claims.

### Default revision

Plan B is now the default for eligible Hexagon decoder synthesis matmuls. Leave `AUDIO8_CODEC_HEXAGON_FP16` unset, or set it to `1`, for the measured HMX path. Set it to `0` before loading the decoder to restore F32 synthesis. The default revision changes only application selection; it retains the measured graph, precision eligibility, HMX arithmetic and small-tail/latent/LM precision. The default selection was subsequently rechecked with the five S4 timings below. No new heavy full-LM CPU correctness suite was run.

## S4 default-path recheck, five timed runs

Input: "Testing one, two, three, four, five." One completed warmup (2.698 s) and five accepted timed fresh-process Hexagon captures. The continuation preserved the original three timings and added two; there was no replacement or discarded valid run. Before each capture: at least 50 s settling, three ready snapshots, unplugged, asleep/dozing, thermal status 0, skin <=38.5 C. After: unplugged, asleep/dozing and thermal status 0. Profiling is off. Runtime/model/original-runner/continuation hashes and output geometry all validate.

The new default-on CLI leaves `AUDIO8_CODEC_HEXAGON_FP16` unset, confirms FP16 synthesis in its log, and produces byte-identical codes and WAVs to the earlier explicit Plan B capture in all six captures. LM repacking and 4x2 are enabled, flash decode uses its default, and vectorized softmax is automatic. Normal HMX ADD fusion is enabled; the earlier HVX blocked-reuse/residual-fusion experiments and F32 convolution tile remain disabled.

| Timed run | Full s | LM s | Codec s | RTF |
| --- | ---: | ---: | ---: | ---: |
| 1 | 3.448 | 2.203 | 1.246 | 0.917 |
| 2 | 4.241 | 3.034 | 1.208 | 1.127 |
| 3 | 3.122 | 2.399 | 0.723 | 0.830 |
| 4 | 3.030 | 2.486 | 0.544 | 0.805 |
| 5 | 3.779 | 2.711 | 1.068 | 1.005 |
| Median | 3.448 | 2.486 | 1.068 | 0.917 |

All outputs contain 81 frames, 165,888 samples at 44,100 Hz: 3.761632653 s of audio. Full inference ranges 3.030-4.241 s. Independent stage medians do not necessarily sum to the median total.

The latest S4 LM median (2.486 s) remains above saved OpenCL (2.064 s), while codec (1.068 s) is close to OpenCL (1.088 s). The actual DSP/DDR clocks are unrecorded; the recheck does not establish the cause of prior variation. All five condition checks pass, with 0 rejected captures. This is output equivalence/performance validation, without a heavy new CPU suite.

Evidence: `codec-default-s4-summary.json`, `codec-default-latest-comparison-rtf.csv`, full raw captures and both immutable runner manifests.

Default-path recheck runtime SHA256:

- `audio8-cli`: `78f5a56438d76c886723575458fef907b669db203acf1d0f06ef6c2619f5c8e3`
- `libqvac-speech-ggml-hexagon.so`: `dc18666d9f8ad4fa4f5b32bc23a2831a5b3975964338a6dd069be95e84b5a692`
- `libggml-htp-v79.so`: `9f6f13875ceb7db16db116f5fa25d14842a83e48f47e177638ebc034b54acbc2`

## Original LM and codec comparison

Times are seconds, independent medians of three valid timed observations. Stage medians need not sum to the median total. GPU figures reuse the existing three-run OpenCL reference. GPU/Hexagon natural frame counts differ; this is a prompt comparison.

### LM

| Input | Saved OpenCL LM s | Plan A Hexagon LM s | Plan B Hexagon LM s | Frames GPU/Hexagon |
| --- | ---: | ---: | ---: | ---: |
| S1 | 2.323 | 2.611 | 2.094 | 76/69 |
| S2 | 1.562 | 1.471 | 1.331 | 50/49 |
| S3 | 2.216 | 2.544 | 2.549 | 72/74 |
| S4 | 2.064 | 2.591 | 3.084 | 67/81 |
| S5 | 2.869 | 2.917 | 3.464 | 94/88 |

### Codec

| Input | Saved OpenCL codec s | Plan A Hexagon codec s | Plan B Hexagon codec s | OpenCL / Plan B time |
| --- | ---: | ---: | ---: | ---: |
| S1 | 1.221 | 2.936 | 0.728 | 1.68x |
| S2 | 0.835 | 2.935 | 0.627 | 1.33x |
| S3 | 1.163 | 3.377 | 0.948 | 1.23x |
| S4 | 1.088 | 5.673 | 0.786 | 1.39x |
| S5 | 1.504 | 5.574 | 0.863 | 1.74x |

Plan B codec medians are 2.39x–5.48x faster than the saved preceding Hexagon codec medians, and 3.56x–7.22x faster than Plan A. They are lower than saved OpenCL codec times for every prompt. These are separate batches with timing spread and different GPU trajectories; the ratios do not establish identical-workload kernel speedups. HMX selects dense matrix execution with F16 activation/output staging and existing ADD fusion. Plan A keeps HVX F32 dot products and also increased reported graph scratch. The specific cause of its regression is not established.

Plan B LM arithmetic is unchanged. Its measured LM medians remain above saved OpenCL for S3/S4/S5; residual prompt-level gaps are in LM execution rather than codec. The measured HMX mode is now the default for eligible Hexagon synthesis matmuls; the F32 tiled prototype remains off by default.

### Plan B ranges and waveform comparisons

All 20 Plan B captures produce identical codes to preceding Hexagon, and byte-identical WAVs within each prompt. Every timed before/after condition and runtime/model/script manifest passes validation, with zero rejected captures. The table compares PCM16 output to the preceding Hexagon WAV, without alignment or gain normalization. All five pass cosine >=0.9999 and NMSE <=0.0002.

| Input | LM range s | Codec range s | PCM cosine | PCM NMSE |
| --- | ---: | ---: | ---: | ---: |
| S1 | 2.079–2.868 | 0.293–0.963 | 0.999999745 | 5.10499704e-07 |
| S2 | 1.276–1.545 | 0.622–0.641 | 0.999999596 | 8.08785667e-07 |
| S3 | 2.542–2.717 | 0.881–1.068 | 0.999999772 | 4.5547583e-07 |
| S4 | 2.448–3.431 | 0.754–1.032 | 0.999999626 | 7.48552204e-07 |
| S5 | 2.505–3.639 | 0.745–1.101 | 0.999999330 | 1.34052731e-06 |

Plan A reported scratch allocations (preceding/Plan A MB): S1 207/233, S2 147/165, S3 222/250, S4 243/273, S5 264/297. This is an allocation observation, not a proven slowdown mechanism. Plan B retains the preceding graph scratch sizes.

### Reproduction

Use matching companion host and v79 DSP libraries. For Plan A, set `AUDIO8_CODEC_HEXAGON_FP16=0` and `AUDIO8_CODEC_HEXAGON_CONV_TILE=1`. The explicit Plan B setting below reproduces the measured default HMX path:

```sh
AUDIO8_CODEC_HEXAGON_FP16=1 AUDIO8_CODEC_HEXAGON_CONV_TILE=0 \
GGML_HEXAGON_F16_F32_BLOCKED=0 GGML_HEXAGON_F16_F32_FUSE_ADD=0 \
audio8-cli --lm lm.gguf --codec-decoder decoder.gguf --backend hexagon \
  --greedy --seed 42 --threads 4 --max-frames 512 --verbose \
  --text "Testing one, two, three, four, five." --out S4.wav
```

Keep flash decode enabled by default (unset `AUDIO8_LM_HEXAGON_FLASH_ATTN`) and profiling unset for timing. Vectorized short softmax, LM repacking and 4x2 retain their defaults.

## Provenance

Local evidence: `codec-track1-summary.json`, `codec-track1-comparison.csv`, `codec-track2-correctness.json`, `codec-track2-summary.json`, `codec-track2-comparison.csv`, frozen runtime/model/script manifests and complete raw captures. The saved GPU reference is `opencl-vs-hexagon-v2-comparison.csv`. The application implementation is `2face157ffb1e107c93f2e771d37671d8e0af1c3`, and backend implementation is `39dcb25702163f5ca0e2b1b22e541a0f12eb18bd`. Android/host application builds and v73/v75/v79/v81 DSP builds pass.

The GPU reference uses internal F16 activation packing and F16 output tiles with F32 accumulation in the measured Adreno XMEM kernel. Different native GPU codes and frame counts make GPU comparisons prompt-level. HMX also stages F16 activations/output tiles; numerical behavior and throughput are measured independently.


Plan A frozen runtime SHA256:

- `audio8-cli`: `0f67f35715aed37b7f2a8038e590b5be76eae8dd0ce469c38642ececcf308396`
- `audio8-codec-bench`: `e76c7476c4f528a1ce0d62e49eb25ae5ebff5c30944679b8148eaff53d20d673`
- `libqvac-speech-ggml-hexagon.so`: `dc18666d9f8ad4fa4f5b32bc23a2831a5b3975964338a6dd069be95e84b5a692`
- `libggml-htp-v79.so`: `9f6f13875ceb7db16db116f5fa25d14842a83e48f47e177638ebc034b54acbc2`

Plan B frozen runtime SHA256:

- `audio8-cli`: `5ebb30d95cc2fef8cae32d0fedf52dd189493ecc96e3b7b66205bd27ff226269`
- `audio8-codec-bench`: `9fa3b0ce80e9bc1a21eb305172a20bd6e8fd9fc8d03a40e86f65badabe1d761e`
- `libqvac-speech-ggml-hexagon.so`: `dc18666d9f8ad4fa4f5b32bc23a2831a5b3975964338a6dd069be95e84b5a692`
- `libggml-htp-v79.so`: `9f6f13875ceb7db16db116f5fa25d14842a83e48f47e177638ebc034b54acbc2`
