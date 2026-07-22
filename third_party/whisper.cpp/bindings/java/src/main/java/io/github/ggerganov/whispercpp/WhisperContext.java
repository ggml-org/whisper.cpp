package io.github.ggerganov.whispercpp;

import com.sun.jna.NativeLong;
import com.sun.jna.Structure;
import com.sun.jna.ptr.PointerByReference;
import com.sun.jna.Pointer;
import io.github.ggerganov.whispercpp.ggml.GgmlType;
import io.github.ggerganov.whispercpp.WhisperModel;
import io.github.ggerganov.whispercpp.params.WhisperContextParams;

import java.util.List;

public class WhisperContext extends Structure {
    public NativeLong t_load_us;
    public NativeLong t_start_us;

    /** weight type (FP32 / FP16 / QX) */
    public GgmlType wtype = GgmlType.GGML_TYPE_F16;
    /** intermediate type (FP32 or FP16) */
    public GgmlType itype = GgmlType.GGML_TYPE_F16;

    public WhisperContextParams.ByValue params;

    public Pointer model;
    public Pointer vocab;
    public Pointer state;

    /** populated by whisper_init_from_file_with_params() */
    public Pointer path_model;

    @Override
    protected List<String> getFieldOrder() {
        return List.of("t_load_us", "t_start_us", "wtype", "itype",
                "params", "model", "vocab", "state", "path_model");
    }
}
