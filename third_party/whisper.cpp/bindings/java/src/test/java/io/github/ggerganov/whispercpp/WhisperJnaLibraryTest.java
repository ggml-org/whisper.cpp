package io.github.ggerganov.whispercpp;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

class WhisperJnaLibraryTest {

    @Test
    void testWhisperPrint_system_info() {
        String systemInfo = WhisperCppJnaLibrary.instance.whisper_print_system_info();
        // eg: "AVX = 1 | AVX2 = 1 | AVX512 = 0 | FMA = 1 | NEON = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0
        //    | WASM_SIMD = 0 | BLAS = 0 | SSE3 = 1 | VSX = 0 | COREML = 0 | "
        System.out.println("System info: " + systemInfo);
        assertTrue(systemInfo.length() > 10);
    }
}
