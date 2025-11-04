package io.github.ggerganov.whispercpp.params;
import com.sun.jna.*;
import java.util.Arrays;
import java.util.List;

public class WhisperAhead extends Structure {

    public int n_text_layer;

    public int n_head;

    public WhisperAhead() {
        super();
    }

    public WhisperAhead(int textLayer, int head) {
        super();
        this.n_text_layer = textLayer;
        this.n_head = head;
    }

    @Override
    protected List<String> getFieldOrder() {
        return Arrays.asList("n_text_layer", "n_head");
    }

    public static class ByReference extends WhisperAhead implements Structure.ByReference {}

    public static class ByValue extends WhisperAhead implements Structure.ByValue {}
}
