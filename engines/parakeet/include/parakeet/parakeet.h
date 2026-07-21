#pragma once

// Umbrella include for libparakeet (each header below can also be included alone).
//
//   <parakeet/export.h>      - PARAKEET_API visibility macro
//   <parakeet/cli.h>         - parakeet_cli_main C entry point
//   <parakeet/log.h>         - parakeet_log_set host log sink
//   <parakeet/engine.h>      - Engine + EngineOptions / EngineResult
//                                   (CTC, TDT, EOU, Sortformer behind one
//                                   class)
//   <parakeet/streaming.h>   - StreamingOptions / StreamingSegment /
//                                   StreamSession + cross-engine
//                                   StreamEvent + VadState +
//                                   StreamEventType
//   <parakeet/diarization.h> - DiarizationOptions / Result +
//                                   SortformerStreamingOptions /
//                                   SortformerStreamSession
//   <parakeet/attributed.h>  - transcribe_with_speakers + the
//                                   attributed-segment types it emits
//
// Engine families behind the umbrella `Engine` (auto-routed by GGUF
// metadata at load time):
//
//   - Parakeet-CTC 0.6B / 1.1B  -- English transcription
//   - Parakeet-TDT 0.6B-v3 / 1.1B -- multilingual transcription with
//     punctuation and capitalisation, RNN-T (LSTM prediction + joint MLP)
//   - Parakeet-EOU 120M (`parakeet_realtime_eou_120m-v1`) -- low-latency
//     streaming ASR with native `<EOU>` end-of-utterance token
//   - Sortformer 4-spk — offline diarization; v2 adds sliding-history streaming.

#include "export.h"
#include "cli.h"
#include "log.h"
#include "engine.h"
#include "streaming.h"
#include "diarization.h"
#include "attributed.h"
