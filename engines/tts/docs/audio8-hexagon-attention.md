# Audio8 Hexagon F32 decode attention and short softmax

## Configuration and protocol

Measured on the OnePlus CPH2723, Hexagon v79, with six HVX workers and four CPU threads. LM projection repacking and the 4x2 codec panel are enabled in both variants. Blocked codec reuse and matmul-plus-residual fusion are disabled. The baseline uses separate LM attention and the preceding short-softmax implementation. The candidate enables maskless F32 flash attention in both decoders and the final vectorized short-softmax kernel.

The full S1–S5 suite has three warmups and five timed fresh-process runs per variant per prompt (80 captures). Order alternates, seed is 42, decoding is greedy and the safety cap is 512 frames. Runs finish at natural EOS; model loading is excluded. Profiling is disabled. Each capture starts with the phone asleep/dozing, thermal status 0 and HAL skin temperature at most 38.5 C; the end snapshot must also report thermal status 0. All repeat code and WAV hashes are identical within each variant/prompt. Actual DSP/memory clocks are not recorded, and these controls do not eliminate the observed timing spread.

## Latest prompt comparison, October 14

These are the latest median full inference times, excluding model loading and
output file writing. S1/S2 retain the five timed observations from the original
suite below. S3/S5 use three timed observations per variant from the follow-up
batch, and S4 uses three from its separate recheck. Each follow-up variant has
completed warmups, alternating order, six HVX workers and four CPU threads.
Timed captures start at thermal status 0 with the screen asleep/dozing and the
phone unplugged; start skin is at most 38.5 C. End snapshots also have thermal
status 0 and an asleep/dozing screen. The S3/S5 recheck initially used a 38 C
limit before returning to the original 38.5 C limit. Actual timed starts were
37.783-38.337 C for S3/S5 and 34.567-35.220 C for S4.

| Input | Baseline s | Flash + softmax s | Baseline / candidate | Frames (baseline/candidate) |
| --- | ---: | ---: | ---: | ---: |
| S1 | 6.474 | 5.200 | 1.25x | 71/69 |
| S2 | 4.130 | 3.539 | 1.17x | 57/49 |
| S3 | 6.717 | 4.931 | 1.36x | 67/74 |
| S4 | 5.144 | 6.804 | 0.76x | 80/81 |
| S5 | 7.103 | 6.626 | 1.07x | 91/88 |

Independent LM and codec medians from those same observations:

| Input | Baseline LM s | Candidate LM s | Baseline codec s | Candidate codec s |
| --- | ---: | ---: | ---: | ---: |
| S1 | 2.572 | 2.251 | 3.358 | 2.785 |
| S2 | 1.967 | 1.354 | 2.081 | 1.495 |
| S3 | 3.361 | 1.918 | 3.575 | 3.013 |
| S4 | 2.916 | 2.502 | 1.962 | 4.302 |
| S5 | 3.296 | 2.461 | 3.689 | 3.508 |

The latest LM medians improve for all five prompts, while S4 full inference is
slower because of codec synthesis. Its candidate codec times are 5.019, 1.743
and 4.302 s with identical codes and WAV bytes at 81 frames. These observations
do not establish a kernel cause for that variation. Different natural codes
and frame counts, separate batches and unrecorded actual DSP/memory clocks
limit causal performance claims. All follow-up outputs match the original
output for their variant and prompt. No new heavy CPU correctness suite was
run for the rechecks.

Evidence is retained in `native-v2-summary.json`,
`s3-s5-recheck-v1-summary.json` and `s4-recheck-v1-summary.json` in the local
artifact directory. The original full suite remains below for comparison.

## Original full inference suite

Median seconds from five timed runs. Frame counts are baseline/candidate. Different generated trajectories make these prompt-level comparisons, rather than identical-workload kernel speedups.

| Input | Baseline s | Flash + softmax s | Ratio | Frames | Baseline range s | Candidate range s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S1 | 6.474 | 5.200 | 1.25x | 71/69 | 5.556–8.501 | 4.255–6.202 |
| S2 | 4.130 | 3.539 | 1.17x | 57/49 | 3.284–6.870 | 2.559–5.697 |
| S3 | 5.270 | 7.896 | 0.67x | 67/74 | 4.584–7.888 | 5.473–8.606 |
| S4 | 9.548 | 9.282 | 1.03x | 80/81 | 9.529–9.596 | 9.268–9.300 |
| S5 | 8.202 | 8.319 | 0.99x | 91/88 | 5.786–9.232 | 5.323–10.604 |

Stage medians are computed independently and need not sum to the median total. Per-frame LM times remain descriptive because prefixes and token values also differ.

| Input | Baseline LM s | Candidate LM s | Baseline codec s | Candidate codec s | Baseline LM ms/frame | Candidate LM ms/frame |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S1 | 2.572 | 2.251 | 3.358 | 2.785 | 36.22 | 32.62 |
| S2 | 1.967 | 1.354 | 2.081 | 1.495 | 34.50 | 27.62 |
| S3 | 2.075 | 2.557 | 2.692 | 5.339 | 30.97 | 34.55 |
| S4 | 2.454 | 2.104 | 7.093 | 7.180 | 30.68 | 25.97 |
| S5 | 3.342 | 2.770 | 4.860 | 5.928 | 36.72 | 31.47 |

Inputs:

- **S1:** The quick brown fox jumps over the lazy dog.
- **S2:** She sells seashells by the seashore.
- **S3:** In a world where machines think, humans still dream.
- **S4:** Testing one, two, three, four, five.
- **S5:** The year was 2026, and the signal was finally clear.

## Short-softmax probes

Three alternating captures per runtime; each backend probe warms up and repeats the operation for at least one second. All have 14 heads, thermal status 0 before/after and starting skin at most 38.5 C. Values are medians.

| Shape | Previous us | Final us | Speedup |
| --- | ---: | ---: | ---: |
| 9 keys, one query, no mask | 4.16 | 3.54 | 1.18x |
| 30 keys, 30 queries, F32 mask | 78.53 | 53.76 | 1.46x |
| 95 keys, one query, no mask | 6.60 | 4.15 | 1.59x |

An initial padded-copy/scalar-reciprocal prototype regressed these probes and was replaced. Earlier matched-v1 replay measurements used that superseded prototype and are not the final candidate results. Flash decode removes the standalone softmax; the remaining application benefit of short softmax primarily concerns masked prefill.

## Final operator profile

Separate PROFILE=1 captures use the same fixed-entry 66-frame S1 replay. Each runtime produces byte-identical codes with profiling enabled/disabled. These event costs describe the execution paths; they are not unprofiled full inference times. Leaf and enclosing batch costs must not be added together.

| Component | Baseline profile ms | Candidate profile ms | Baseline calls | Candidate calls |
| --- | ---: | ---: | ---: | ---: |
| Attention core | 301.860 | 75.868 | 12672 | 4272 |
| Layout CONT | 55.981 | 0.897 | 4404 | 228 |
| KV copies | 12.689 | 8.613 | 8448 | 8448 |

The candidate records 4,200 `hvx-f32-decode` calls. Attention core accounts for 4.88% of leaf cycles; Q8 LM projection operations account for 76.13%. The largest remaining leaf operation families are:

| Operation and shape | Kernel | Candidate leaf cycles |
| --- | --- | ---: |
| MUL_MAT `896:9728 x 896:1 -> 9728:1` (q8_0 x f32 -> f32) | hvx-tiled | 41.14% |
| MUL_MAT+ADD `4864:896 x 4864:1 x 896:1 -> 896:1` (q8_0 x f32 x f32 -> f32) | hvx-tiled | 21.97% |
| MUL_MAT `896:4096 x 896:1 -> 4096:1` (f32 x f32 -> f32) | hvx-tiled | 8.73% |
| MUL_MAT+ADD `896:896 x 896:1 x 896:1 -> 896:1` (q8_0 x f32 x f32 -> f32) | hvx-tiled | 4.79% |

## Correctness against CPU

The final runtime passes 84 short-softmax, 39 decode-attention, three large-logit and 20 production cache CPU-reference cases. The cache cases cover masked prefill, grouped heads, two layers, resets and prefixes spanning 128 keys. Host cache, model loading, LM fusion and memory-fit checks pass; Android and DSP v73/v75/v79/v81 builds pass.

Full trajectories use archived CPU histories and CPU fast inputs through CPU EOS, with every raw reference tensor hash verified. They cover 352 frames and 3,907 tensor comparisons. Strict gates require cosine >= 0.9999 and NMSE <= 0.0002. **These strict LM gates do not universally pass.**

| Input | CPU frames | Comparisons | Previous numerical misses | Final numerical misses | Previous CPU top1 differences | Final CPU top1 differences |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| S1 | 73 | 810 | 180 | 191 | 17 | 18 |
| S2 | 56 | 623 | 180 | 187 | 15 | 15 |
| S3 | 66 | 733 | 213 | 213 | 20 | 20 |
| S4 | 64 | 711 | 185 | 173 | 15 | 15 |
| S5 | 93 | 1030 | 276 | 262 | 37 | 37 |

Totals: 1,026 numerical misses and 105 CPU top1 differences, versus 1,034 and 104 previously. Of 3,525 top1 decisions, the final runtime differs in 98 fast and seven slow decisions (2.98% total versus 2.95% previously). Worst fast-decoder NMSE is unchanged in every prompt versus the preceding Hexagon baseline (overall maximum 0.009690). Direct comparison of final versus preceding Hexagon has 90 numerical misses and seven top1 differences under fixed histories. There are zero execution failures, zero fast-chain inconsistencies and no CPU EOS disagreements under fixed histories. All 25 fixed-code codec boundaries pass and are byte-identical to preceding Hexagon. This does not establish identical native codes, frame counts or perceptual audio equivalence.

## Output screening

Unprompted whisper.cpp Tiny English CPU recognition of one timed baseline and candidate WAV per prompt; original transcripts and ordinary word-level error rates are retained. A secondary normalization treats “sea shells”/“seashells” and “sea shore”/“seashore” as spelling equivalents. ASR screens intelligibility and does not replace numerical or listening checks.

| Input | Runtime | Transcript | Word error rate | Spelling-normalized rate |
| --- | --- | --- | ---: | ---: |
| S1 | baseline | The quick brown fox jumps over the lazy dog. | 0.0% | 0.0% |
| S1 | candidate | The quick brown fox jumps over the lazy dog. | 0.0% | 0.0% |
| S2 | baseline | She sells sea shells by the sea shore. | 66.7% | 0.0% |
| S2 | candidate | She sells seashells by the sea shore. | 33.3% | 0.0% |
| S3 | baseline | In a world where machines thank humans still dream. | 11.1% | 11.1% |
| S3 | candidate | in a world where machines think humans still dream. | 0.0% | 0.0% |
| S4 | baseline | Testing 1, 2, 3, 4, 5. | 0.0% | 0.0% |
| S4 | candidate | Testing 1, 2, 3, 4, 5. | 0.0% | 0.0% |
| S5 | baseline | The year was 2026 and the signal was finally clear. | 0.0% | 0.0% |
| S5 | candidate | The year was 2026 and the signal was finally clear. | 0.0% | 0.0% |

## Conclusion

The maskless F32 decode kernel removes separate QK, softmax, value reduction and most V-layout materialization. The final vectorized softmax improves isolated short-row probes. Full prompt timings retain substantial spread and different generated workloads, so the median ratios alone do not establish a universal end-to-end speedup.

The remaining gap to OpenCL requires work on LM projection execution, particularly the Q8 gated FFN and residual down-projection families identified above, alongside remaining codec execution. No fresh OpenCL timing claim is made here. The profile establishes costly operation families; cache/memory/compute mechanisms still require focused kernel profiling. Supported Hexagon backends now use flash decode by default, with `AUDIO8_LM_HEXAGON_FLASH_ATTN=0` selecting separate attention. Vectorized short softmax is automatic for applicable shapes.

## Reproduction and runtime provenance

Install the companion Hexagon backend containing `hvx-f32-decode`. Set its host and matching DSP library paths, then run:

```sh
export GGML_HEXAGON_NHVX=6 GGML_HEXAGON_HOSTBUF=1
export GGML_HEXAGON_OPSTAGE=3 GGML_HEXAGON_OPPOLL=1 GGML_HEXAGON_OPFUSION=1
export GGML_HEXAGON_F16_F32_PANEL=1 GGML_HEXAGON_F16_F32_BLOCKED=0 GGML_HEXAGON_F16_F32_FUSE_ADD=0
export AUDIO8_LM_HEXAGON_REPACK=1
unset AUDIO8_LM_HEXAGON_FLASH_ATTN GGML_HEXAGON_PROFILE GGML_HEXAGON_MM_SELECT TTS_CPP_FORCE_SCHED
audio8-cli --lm lm.gguf --codec-decoder decoder.gguf --backend hexagon \
  --greedy --seed 42 --threads 4 --max-frames 512 \
  --text "Testing one, two, three, four, five." --out S4.wav
```

For the baseline, set `AUDIO8_LM_HEXAGON_FLASH_ATTN=0` and use the preceding softmax runtime. Setting only the flag to zero on the new backend keeps vectorized softmax enabled. The measurements above used the equivalent explicit value `1`; the default change selects the same supported graph and cache layout. The recorded CLI hash identifies the measured explicit-enable build, before the default changed.

- Final v79 DSP SHA256: `b2387d338869dd6224c222fce1b8aafa7398532799c15ffd56f1922d7d93e946`.
- Baseline v79 DSP SHA256: `1f1c122e6e274acd2fdb2b0c490df7464fb34b3888c62a8c27b83616656fa4e2`.
- CLI SHA256: `b0e56e1385369c97a46a81f3244593f637e2432be0ed005e8e13861afaafcdf0`.
- LM SHA256: `f33c58b2fd46320c01544eec961112b4b9778106ce883e86914dd38854072d40`.
- Codec SHA256: `84d4b88fc5b2274c2cbfd95c4ce1bdae7820eb1e2464b7bd715d3fe370895432`.

Raw captures, summaries, scripts and source hashes are retained in the local `audio8-attention-build` artifact directory. The final device runtime manifest and a separately collected manifest are checked against the tested runtime hashes. Runtime selection throughout the suite uses `both-v2`.
