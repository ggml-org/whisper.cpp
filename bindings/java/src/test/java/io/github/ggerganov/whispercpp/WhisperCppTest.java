package io.github.ggerganov.whispercpp;

import static org.junit.jupiter.api.Assertions.*;

import io.github.ggerganov.whispercpp.bean.WhisperSegment;
import io.github.ggerganov.whispercpp.params.CBool;
import io.github.ggerganov.whispercpp.params.WhisperFullParams;
import io.github.ggerganov.whispercpp.params.WhisperSamplingStrategy;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import javax.sound.sampled.AudioInputStream;
import javax.sound.sampled.AudioSystem;
import java.io.File;
import java.io.FileNotFoundException;
import java.util.List;

class WhisperCppTest {
    private static WhisperCpp whisper = new WhisperCpp();
    private static boolean modelInitialised = false;

    @BeforeAll
    static void init() throws FileNotFoundException {
        // By default, models are loaded from ~/.cache/whisper/ and are usually named "ggml-${name}.bin"
        // or you can provide the absolute path to the model file.
        //String modelName = "../../models/ggml-tiny.bin";
        String modelName = "../../models/ggml-tiny.en.bin";
        try {
            whisper.initContext(modelName);
            //whisper.getFullDefaultParams(WhisperSamplingStrategy.WHISPER_SAMPLING_GREEDY);
            //whisper.getJavaDefaultParams(WhisperSamplingStrategy.WHISPER_SAMPLING_BEAM_SEARCH);
            modelInitialised = true;
        } catch (FileNotFoundException ex) {
            System.out.println("Model " + modelName + " not found");
        }
    }

    @Test
    void testGetDefaultFullParams_BeamSearch() {
        // When
        WhisperFullParams params = whisper.getFullDefaultParams(WhisperSamplingStrategy.WHISPER_SAMPLING_BEAM_SEARCH);

        // Then
        assertEquals(WhisperSamplingStrategy.WHISPER_SAMPLING_BEAM_SEARCH.ordinal(), params.strategy);
        assertNotEquals(0, params.n_threads);
        assertEquals(16384, params.n_max_text_ctx);
        assertFalse(params.translate);
        assertEquals(0.01f, params.thold_pt);
        assertEquals(5, params.beam_search.beam_size);
        assertEquals(-1.0f, params.beam_search.patience);
    }

    @Test
    void testGetDefaultFullParams_Greedy() {
        // When
        WhisperFullParams params = whisper.getFullDefaultParams(WhisperSamplingStrategy.WHISPER_SAMPLING_GREEDY);

        // Then
        assertEquals(WhisperSamplingStrategy.WHISPER_SAMPLING_GREEDY.ordinal(), params.strategy);
        assertNotEquals(0, params.n_threads);
        assertEquals(16384, params.n_max_text_ctx);
        assertEquals(5, params.greedy.best_of);
    }

    @Test
    void testFullTranscribe() throws Exception {
        if (!modelInitialised) {
            System.out.println("Model not initialised, skipping test");
            return;
        }

        // Given
        File file = new File(System.getProperty("user.dir"), "../../samples/jfk.wav");
        AudioInputStream audioInputStream = AudioSystem.getAudioInputStream(file);

        byte[] b = new byte[audioInputStream.available()];
        float[] floats = new float[b.length / 2];

        //WhisperFullParams params = whisper.getFullDefaultParams(WhisperSamplingStrategy.WHISPER_SAMPLING_GREEDY);
        WhisperFullParams.ByValue params = whisper.getFullDefaultParams(WhisperSamplingStrategy.WHISPER_SAMPLING_BEAM_SEARCH);
        params.setProgressCallback((ctx, state, progress, user_data) -> System.out.println("progress: " + progress));
        params.print_progress = CBool.FALSE;
        //params.initial_prompt = "and so my fellow Americans um, like";


        try {
            audioInputStream.read(b);

            for (int i = 0, j = 0; i < b.length; i += 2, j++) {
                int intSample = (int) (b[i + 1]) << 8 | (int) (b[i]) & 0xFF;
                floats[j] = intSample / 32767.0f;
            }

            // When
            String result = whisper.fullTranscribe(params, floats);

            // Then
            System.err.println(result);
            assertEquals("And so my fellow Americans ask not what your country can do for you " +
                    "ask what you can do for your country.",
                    result.replace(",", ""));
        } finally {
            audioInputStream.close();
        }
    }

    @Test
    void testFullTranscribeWithTime() throws Exception {
        if (!modelInitialised) {
            System.out.println("Model not initialised, skipping test");
            return;
        }

        // Given
        File file = new File(System.getProperty("user.dir"), "../../samples/jfk.wav");
        AudioInputStream audioInputStream = AudioSystem.getAudioInputStream(file);

        byte[] b = new byte[audioInputStream.available()];
        float[] floats = new float[b.length / 2];

        //WhisperFullParams params = whisper.getFullDefaultParams(WhisperSamplingStrategy.WHISPER_SAMPLING_GREEDY);
        WhisperFullParams.ByValue params = whisper.getFullDefaultParams(WhisperSamplingStrategy.WHISPER_SAMPLING_BEAM_SEARCH);
        params.setProgressCallback((ctx, state, progress, user_data) -> System.out.println("progress: " + progress));
        params.print_progress = CBool.FALSE;
        //params.initial_prompt = "and so my fellow Americans um, like";

        try {
            audioInputStream.read(b);

            for (int i = 0, j = 0; i < b.length; i += 2, j++) {
                int intSample = (int) (b[i + 1]) << 8 | (int) (b[i]) & 0xFF;
                floats[j] = intSample / 32767.0f;
            }

            List<WhisperSegment> segments = whisper.fullTranscribeWithTime(params, floats);
            assertTrue(segments.size() > 0, "The size of segments should be greater than 0");
            for (WhisperSegment segment : segments) {
                System.out.println(segment);
            }
        } finally {
            audioInputStream.close();
        }
    }

}
