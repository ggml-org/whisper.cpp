package io.github.ggerganov.whispercpp.params;

import com.sun.jna.IntegerType;

import java.util.function.BooleanSupplier;

public class CBool extends IntegerType implements BooleanSupplier {
    public static final int SIZE = 1;
    public static final CBool FALSE = new CBool(0);
    public static final CBool TRUE = new CBool(1);


    public CBool() {
        this(0);
    }

    public CBool(long value) {
        super(SIZE, value, true);
    }

    @Override
    public boolean getAsBoolean() {
        return intValue() == 1;
    }

    @Override
    public String toString() {
        return intValue() == 1 ? "true" : "false";
    }
}
