; ============================================================================
; WhisperCpp - AutoHotkey v2 Bindings for whisper.cpp
; ============================================================================
; This class provides a simple interface to use whisper.cpp from AHK v2
;
; Requirements:
; 1. whisper.dll (Windows) compiled from whisper.cpp
; 2. A Whisper model file (e.g., ggml-base.en.bin)
;
; Usage Example:
;   whisper := WhisperCpp("whisper.dll")
;   whisper.LoadModel("models/ggml-base.en.bin")
;   result := whisper.TranscribeFile("audio.wav")
;   MsgBox(result.text)
;   whisper.Free()
; ============================================================================

class WhisperCpp {
    ; Properties
    dllPath := ""
    hDll := 0
    ctx := 0

    ; Constants
    static WHISPER_SAMPLE_RATE := 16000

    ; ========================================================================
    ; Constructor - Load the DLL
    ; ========================================================================
    __New(dllPath := "whisper.dll") {
        this.dllPath := dllPath
        this.hDll := DllCall("LoadLibrary", "Str", dllPath, "Ptr")

        if (!this.hDll) {
            throw Error("Failed to load whisper.dll from: " . dllPath)
        }
    }

    ; ========================================================================
    ; Load a Whisper model
    ; ========================================================================
    LoadModel(modelPath) {
        ; Get default context params
        cparams := this.GetDefaultContextParams()

        ; Call: whisper_init_from_file_with_params(const char* path, whisper_context_params params)
        this.ctx := DllCall(this.dllPath "\whisper_init_from_file_with_params",
            "AStr", modelPath,     ; const char* path_model
            "Int", cparams.use_gpu,         ; bool use_gpu
            "Int", cparams.flash_attn,      ; bool flash_attn
            "Int", cparams.gpu_device,      ; int gpu_device
            "Int", cparams.dtw_token_timestamps,  ; bool dtw_token_timestamps
            "Int", cparams.dtw_aheads_preset,     ; enum dtw_aheads_preset
            "Int", cparams.dtw_n_top,             ; int dtw_n_top
            "Ptr", 0,                              ; whisper_aheads* (NULL)
            "UInt", 0,                             ; size_t dtw_mem_size
            "Ptr")                                 ; Returns whisper_context*

        if (!this.ctx) {
            throw Error("Failed to load model from: " . modelPath)
        }

        return true
    }

    ; ========================================================================
    ; Get default context parameters
    ; ========================================================================
    GetDefaultContextParams() {
        ; For simplicity, return defaults as an object
        ; In a full implementation, you'd call whisper_context_default_params()
        return {
            use_gpu: false,
            flash_attn: false,
            gpu_device: 0,
            dtw_token_timestamps: false,
            dtw_aheads_preset: 0,
            dtw_n_top: 0
        }
    }

    ; ========================================================================
    ; Transcribe audio from a file
    ; ========================================================================
    TranscribeFile(audioPath, language := "en") {
        if (!this.ctx) {
            throw Error("No model loaded. Call LoadModel() first.")
        }

        ; Load audio file (you'd need to implement WAV loading or use FFmpeg)
        ; For demonstration purposes, this shows the API structure
        audioData := this.LoadAudioFile(audioPath)

        return this.Transcribe(audioData, language)
    }

    ; ========================================================================
    ; Transcribe audio from PCM float32 array
    ; ========================================================================
    Transcribe(audioSamples, language := "en") {
        if (!this.ctx) {
            throw Error("No model loaded. Call LoadModel() first.")
        }

        ; Get default whisper_full_params
        ; Strategy: 0 = GREEDY, 1 = BEAM_SEARCH
        params := this.GetDefaultFullParams(0)

        ; Call whisper_full(ctx, params, samples, n_samples)
        ; Note: This is a simplified version. In reality, you need to:
        ; 1. Create a proper whisper_full_params struct
        ; 2. Pass the float32 array correctly
        ; 3. Handle memory management

        result := DllCall(this.dllPath "\whisper_full",
            "Ptr", this.ctx,           ; whisper_context*
            "Ptr", params,              ; whisper_full_params* (struct pointer)
            "Ptr", audioSamples,        ; const float* samples
            "Int", audioSamples.Length, ; int n_samples
            "Int")                      ; Returns 0 on success

        if (result != 0) {
            throw Error("Transcription failed with code: " . result)
        }

        ; Extract results
        return this.GetTranscriptionResults()
    }

    ; ========================================================================
    ; Get transcription results after whisper_full()
    ; ========================================================================
    GetTranscriptionResults() {
        ; Get number of segments
        nSegments := DllCall(this.dllPath "\whisper_full_n_segments",
            "Ptr", this.ctx,
            "Int")

        segments := []
        fullText := ""

        ; Iterate through segments
        Loop nSegments {
            i := A_Index - 1  ; 0-based index

            ; Get segment text
            textPtr := DllCall(this.dllPath "\whisper_full_get_segment_text",
                "Ptr", this.ctx,
                "Int", i,
                "Ptr")

            text := StrGet(textPtr, "UTF-8")

            ; Get segment timestamps (in centiseconds)
            t0 := DllCall(this.dllPath "\whisper_full_get_segment_t0",
                "Ptr", this.ctx,
                "Int", i,
                "Int64")

            t1 := DllCall(this.dllPath "\whisper_full_get_segment_t1",
                "Ptr", this.ctx,
                "Int", i,
                "Int64")

            segment := {
                text: text,
                start: t0 / 100.0,  ; Convert to seconds
                end: t1 / 100.0
            }

            segments.Push(segment)
            fullText .= text
        }

        return {
            text: fullText,
            segments: segments
        }
    }

    ; ========================================================================
    ; Get default full params (simplified)
    ; ========================================================================
    GetDefaultFullParams(strategy := 0) {
        ; In a real implementation, you'd call whisper_full_default_params()
        ; and return a struct pointer. This is a placeholder.

        ; For now, return 0 to use C defaults
        ; A full implementation would need to build the struct properly
        return 0
    }

    ; ========================================================================
    ; Load audio file (STUB - needs implementation)
    ; ========================================================================
    LoadAudioFile(filePath) {
        ; This is a stub. In a real implementation you would:
        ; 1. Load WAV file and extract PCM data
        ; 2. Resample to 16kHz mono if needed
        ; 3. Convert to float32 array normalized to [-1.0, 1.0]
        ; 4. Return a buffer with the float32 samples

        ; You could use:
        ; - FFmpeg via command line
        ; - Windows Media Foundation API
        ; - A separate audio processing DLL

        throw Error("LoadAudioFile not implemented. Please implement audio loading.")
    }

    ; ========================================================================
    ; Free the model and cleanup
    ; ========================================================================
    Free() {
        if (this.ctx) {
            DllCall(this.dllPath "\whisper_free", "Ptr", this.ctx)
            this.ctx := 0
        }

        if (this.hDll) {
            DllCall("FreeLibrary", "Ptr", this.hDll)
            this.hDll := 0
        }
    }

    ; ========================================================================
    ; Get whisper.cpp version
    ; ========================================================================
    GetVersion() {
        versionPtr := DllCall(this.dllPath "\whisper_version", "Ptr")
        return StrGet(versionPtr, "UTF-8")
    }

    ; ========================================================================
    ; Destructor - ensure cleanup
    ; ========================================================================
    __Delete() {
        this.Free()
    }
}


; ============================================================================
; Example Usage
; ============================================================================
/*
; Initialize whisper
whisper := WhisperCpp("whisper.dll")

; Check version
MsgBox("Whisper.cpp version: " . whisper.GetVersion())

; Load model
whisper.LoadModel("models/ggml-base.en.bin")

; Transcribe audio
result := whisper.TranscribeFile("samples/jfk.wav", "en")

; Display results
MsgBox("Full text: " . result.text)

for segment in result.segments {
    OutputDebug(Format("[{1:6.2f}s -> {2:6.2f}s] {3}",
        segment.start, segment.end, segment.text))
}

; Cleanup
whisper.Free()
*/
