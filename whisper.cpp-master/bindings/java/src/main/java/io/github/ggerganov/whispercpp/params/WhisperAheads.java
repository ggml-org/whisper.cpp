package io.github.ggerganov.whispercpp.params;
import com.sun.jna.*;
import java.util.Arrays;
import java.util.List;

public class WhisperAheads extends Structure {
    public NativeLong n_heads;

    public Pointer heads;

    public WhisperAheads() {
        super();
    }

    /**
     * Create alignment heads from an array of WhisperAhead objects
     */
    public void setHeads(WhisperAhead[] aheadsArray) {
        this.n_heads = new NativeLong(aheadsArray.length);

        int structSize = aheadsArray[0].size();
        Memory mem = new Memory(structSize * aheadsArray.length);

        for (int i = 0; i < aheadsArray.length; i++) {
            aheadsArray[i].write();
            byte[] buffer = aheadsArray[i].getPointer().getByteArray(0, structSize);
            mem.write(i * structSize, buffer, 0, buffer.length);
        }

        this.heads = mem;
    }

    @Override
    protected List<String> getFieldOrder() {
        return Arrays.asList("n_heads", "heads");
    }

    public static class ByReference extends WhisperAheads implements Structure.ByReference {}

    public static class ByValue extends WhisperAheads implements Structure.ByValue {}
}
