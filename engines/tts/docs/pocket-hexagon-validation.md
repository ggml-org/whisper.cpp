# PocketTTS Hexagon validation — 2026-10-16

The FP32 PocketTTS FlowLM and Mimi stages passed on the OnePlus 13s. Warm
speech generation was about 1.4 times as fast as CPU for the two tested prompts.
First-audio latency did not consistently improve. These results describe this
device and configuration; they do not establish performance on other phones.

## Source and environment

| Component | Pin/configuration |
| --- | --- |
| Fabric implementation | `0f19bfe470315e3fe18420423854025b8624ea9a` |
| Fabric base | `c9573d25dcf361a33baf56a473eea87c6d9997be` (includes merged OpenCL #340) |
| ggml with runtime-info API | `84d4f0da6669ee2cff7b809d06c8e597da6b03d8` |
| ggml base | `e1fb17af921c04a3004b7850021c4b44397a37dd` on `speech` |
| Device | OnePlus 13s CPH2723, SM8750, Android 16, Linux 6.6.118 |
| HTP | v79; six HVX units, one HMX unit, 8 MiB VTCM |
| Build | Release, Android arm64-v8a/API 28, NDK r28b/Clang 19 |
| DSP toolchain | Hexagon SDK 6.6.0.0, Hexagon Tools/Clang 19.0.07 |
| Precision | F32 compute, including explicit F32 matrix multiplication; HVX baseline |
| Synthesis | Prepared `voice.gguf`, two workers, two CPU threads per worker, one flow step, temperature 0.3, seed 1234, 24 kHz mono |
| HTP settings | Default batch 1024, queue depth 16, one session, OPSTAGE=3; PROFILE=0 for timing |
| OpenCL reference | Qualcomm Adreno 830, OpenCL 3.0; warmed kernel cache |

The dependency change adds runtime introspection, not new kernels. The original
Hexagon implementation did not depend on Fabric PR #338 or the ggml
affine-repack PR #123. PR #338 has since merged into master and is preserved by
the merge resolution below; ggml #123 remains unnecessary. Model assets
are the same converted checkpoint used for the preceding OpenCL validation:

| Asset | SHA-256 |
| --- | --- |
| `flow-lm.gguf` | `11ed51882574fc1b929990a873924b733d6f6b6aeee9ddd89cab5e4e5db5a839` |
| `mimi.gguf` | `fa9f5d0e6b6b7b160d13f302a196b183d192ef0c0b78186a203843254c775892` |
| `voice.gguf` | `e7a6d9fb89c295417cc406850c2cb6e0f07fdc790559d75654521c38c7296cf9` |
| `frontend.json` | `e1bf5c4d7384d8d1331806653753e708b3e1a0ddc38a38ada46472bb3b86a086` |

## Numerical and lifecycle checks

All 11 Pocket host test groups passed against the existing CPU-only ggml
installation, including absent-Hexagon behavior. Independent review repeated
the suite successfully. On the phone:

- Small F32/F16 and real F32/F16 FlowLM fixtures passed the unchanged oracle
  tolerances. Maximum observed error for real F32 FlowLM was `8.6993e-5`
  against the existing `1e-4` limit.
- Random Mimi encoder/decoder and real Mimi decoder fixtures passed, including
  attention-window boundaries, variable chunks and reset. Real decoder maximum
  error was `2.2054e-6` against the existing `1e-3` oracle limit.
- Full HTP engine tests passed batch/stream equivalence, EOS handling, resampling,
  cancellation, callback exceptions, reuse after failure, reentry rejection and
  simultaneous independent engine instances. Both stages reported `HTP0`.
- Runtime queries passed for effective OPSTAGE values 0–3 and for two sessions
  with profile-3 tracing. Changing environment variables after initialization
  did not change the reported retained session settings.
- Pocket rejected diagnostic execution modes 0–2, a 1 MiB RPC-buffer limit,
  and a 64 MiB mapping budget during preflight.
- Real FlowLM/Mimi parity, fit and complete engine lifecycle suites also passed
  on CPU and OpenCL with the final implementation.
- A separate PROFILE=1 speech run recorded 13,418 DSP operations in 127 batches,
  including F32 HVX matrix multiplication in FlowLM and Mimi. Profiling was
  disabled for the performance results below.

The ggml normal host suite passed 36 tests with six hardware-dependent skips.
Its portable runtime sizing and opcache tests executed successfully. Host skips
do not substitute for the actual DSP runs above.

## Performance

For each prompt, timing order was CPU-A, HTP-A, HTP-B, CPU-B. Each process
performed one untimed warmup followed by three measured runs, giving six CPU
and six HTP observations. OpenCL has three observations from one separate
process and is a reference rather than an equally interleaved comparison.
Battery temperature remained between 36.5 and 36.8 degrees C. Profiling was off.

| Prompt/audio duration | Backend | Median generation | Median first audio | Median RTF |
| --- | --- | ---: | ---: | ---: |
| Reminder / 3.04 s | CPU | 1.131 s | 0.150 s | 0.372 |
| Reminder / 3.04 s | Hexagon | 0.784 s | 0.228 s | 0.258 |
| Reminder / 3.04 s | OpenCL | 1.009 s | 0.073 s | 0.332 |
| Paragraph / 6.16 s | CPU | 2.279 s | 0.249 s | 0.370 |
| Paragraph / 6.16 s | Hexagon | 1.607 s | 0.247 s | 0.261 |
| Paragraph / 6.16 s | OpenCL | 2.123 s | 0.071 s | 0.345 |

Hexagon reduced generation time by about 31%/29%, or speedups of 1.44x/1.42x.
RTF is wall-clock generation time divided by generated audio duration; lower is
better. The short prompt's first audio was slower than CPU despite faster total
generation. OpenCL had the lowest first-audio latency in this configuration.

Separate fresh-process startup samples measured model/registry initialization
at 0.770 s CPU and 0.581 s HTP; the first synthesis then took 1.106 s and
0.789 s. These are single observations with existing OS caches, not cold-boot
or uncached-storage measurements.

Reproduce an individual warm case with the Android runtime libraries beside
the executable and the DSP library available through `ADSP_LIBRARY_PATH`:

```sh
GGML_HEXAGON_OPSTAGE=3 GGML_HEXAGON_PROFILE=0 ./pocket-cli \
  --model-dir bundle --backend hexagon --threads 2 --steps 1 \
  --temperature 0.3 --seed 1234 --runs 3 \
  --text 'Hello Alok. Your meeting starts in ten minutes.' \
  --output hexagon.wav --report hexagon.json
```

Use `--backend cpu` for the paired baseline. The longer prompt was:
“The research team announced a new speech model today. It can generate natural
English speech directly on your phone.” Build/runtime setup and opt-in fixtures
are described in [pocket-tts.md](pocket-tts.md).

## Audio quality and memory interpretation

All waveforms were finite, unclipped and equal in duration across backends.
Independent Whisper base.en CPU transcriptions matched across CPU, HTP and
OpenCL: the paragraph matched exactly; all three transcribed “Alok” as “Alach”
and “ten” as “10” in the reminder. This is a shared recognition discrepancy.
CPU/HTP waveform SNR was 71.43 dB for the reminder and 51.87 dB for the paragraph.
The user listened to the CPU/HTP reminder WAVs and confirmed both sounded equally
clear without clicks or distortion.

The reminder's HTP preflight projected `1,055,543,848` bytes including a 256 MiB
margin. It charged model/state buffers, overlapping graph arenas, RPC guard and
page overhead, `13,647,872` bytes of registry queues, and `36,192,413` bytes of
retained/prospective registry metadata. The selected limits were a 1 GiB payload
buffer cap and `3,355,443,200` bytes per-batch mapping budget. VTCM was reported
separately from shared OS RAM.

Observed peak host RSS was about 516–518 MiB CPU and 42 MiB HTP. **These figures
cannot be compared as total model memory:** FastRPC/DMA allocations are not
reliably represented by process RSS. The HTP estimate explicitly includes those
buffers; opaque driver/DSP heaps and allocation bookkeeping still require the
margin. The runtime query cannot guarantee physical-memory availability.

## Packed-buffer review regression

After merging master `c739b8fb`, review identified that ggml's configured
packing limit can exceed the runtime query's effective RPC payload cap.
With `GGML_HEXAGON_MBUF=3072`, two individually valid 1.125 GiB tensors could
be packed into one invalid 2.25 GiB buffer. The original phone preflight
accepted that configuration. Pocket now rejects it during backend validation,
before allocating model buffers, by comparing the retained buffer type's
packing limit with the runtime's effective cap. Both context buffers and graph
arenas use that buffer type; environment changes after initialization cannot
bypass the check.

The merged branch passed all 11 Pocket host tests and an Android arm64 Release
build. A metadata-only regression covers the two 1.125 GiB tensors without
allocating their payloads, unsafe 2048/3072 MiB packing limits, and safe
1024/2047 MiB and exact effective-cap boundaries. On the same OnePlus 13s:

- Both memory preflight and direct model loading rejected `MBUF=2048` and
  `MBUF=3072`, identifying the packing limit and effective cap in the error.
- `MBUF=2047` passed preflight and completed speech synthesis.
- With the default settings, real F32 FlowLM/Mimi parity, memory fit and the
  complete engine lifecycle suite passed with unchanged numerical tolerances.
- The graph-memory and backend-selection unit tests also passed on-device.

The original performance measurements above were not rerun for this
configuration check.

## Integration with merged GPU support

Master `1cf60941` includes Pocket's Metal/CUDA/Vulkan support from PR #338
and the CosyVoice Core ML sidecars from PR #341. The conflict resolution keeps
automatic GPU selection, backend-directory loading, device-memory telemetry,
GPU fixture registration and explicit Hexagon selection together. OpenCL-only
scratch accounting remains restricted to OpenCL; Hexagon retains its runtime
queue/guard/mapping accounting and packed-buffer configuration check.

All 11 Pocket host tests and the portable CosyVoice Core ML policy test passed;
the Android arm64 Release build passed. On the OnePlus 13s, explicit Hexagon
real FlowLM/Mimi parity, fit, engine lifecycle and packed-buffer regressions
passed. The merged `--gpu` path, pinned to OpenCL, also passed real FlowLM
F32/F16, Mimi, fit and the complete engine lifecycle suite. Numerical
tolerances were unchanged. These device checks cover OpenCL and Hexagon;
they do not constitute new Metal/CUDA/Vulkan hardware validation.

## Release handoff

Merge the ggml runtime-info dependency into `speech`, then the Fabric change;
advance the registry-vcpkg pins/features to include both, integrate QVAC/addon
support, validate packaged device execution and release for SDK consumption.
Packaging and SDK publication are follow-up work. Keep CPU as the default;
explicit HTP requests require the matching runtime-info API and DSP libraries.
