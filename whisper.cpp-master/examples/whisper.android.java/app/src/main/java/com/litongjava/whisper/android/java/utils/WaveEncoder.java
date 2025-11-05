package com.litongjava.whisper.android.java.utils;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.ShortBuffer;

public class WaveEncoder {

  public static float[] decodeWaveFile(File file) throws IOException {
    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    try (FileInputStream fis = new FileInputStream(file)) {
      byte[] buffer = new byte[1024];
      int bytesRead;
      while ((bytesRead = fis.read(buffer)) != -1) {
        baos.write(buffer, 0, bytesRead);
      }
    }
    ByteBuffer byteBuffer = ByteBuffer.wrap(baos.toByteArray());
    byteBuffer.order(ByteOrder.LITTLE_ENDIAN);

    int channel = byteBuffer.getShort(22);
    byteBuffer.position(44);

    ShortBuffer shortBuffer = byteBuffer.asShortBuffer();
    short[] shortArray = new short[shortBuffer.limit()];
    shortBuffer.get(shortArray);

    float[] output = new float[shortArray.length / channel];

    for (int index = 0; index < output.length; index++) {
      if (channel == 1) {
        output[index] = Math.max(-1f, Math.min(1f, shortArray[index] / 32767.0f));
      } else {
        output[index] = Math.max(-1f, Math.min(1f, (shortArray[2 * index] + shortArray[2 * index + 1]) / 32767.0f / 2.0f));
      }
    }
    return output;
  }

  public static void encodeWaveFile(File file, short[] data) throws IOException {
    try (FileOutputStream fos = new FileOutputStream(file)) {
      fos.write(headerBytes(data.length * 2));

      ByteBuffer buffer = ByteBuffer.allocate(data.length * 2);
      buffer.order(ByteOrder.LITTLE_ENDIAN);
      buffer.asShortBuffer().put(data);

      byte[] bytes = new byte[buffer.limit()];
      buffer.get(bytes);

      fos.write(bytes);
    }
  }

  private static byte[] headerBytes(int totalLength) {
    if (totalLength < 44)
      throw new IllegalArgumentException("Total length must be at least 44 bytes");

    ByteBuffer buffer = ByteBuffer.allocate(44);
    buffer.order(ByteOrder.LITTLE_ENDIAN);

    buffer.put((byte) 'R');
    buffer.put((byte) 'I');
    buffer.put((byte) 'F');
    buffer.put((byte) 'F');

    buffer.putInt(totalLength - 8);

    buffer.put((byte) 'W');
    buffer.put((byte) 'A');
    buffer.put((byte) 'V');
    buffer.put((byte) 'E');

    buffer.put((byte) 'f');
    buffer.put((byte) 'm');
    buffer.put((byte) 't');
    buffer.put((byte) ' ');

    buffer.putInt(16);
    buffer.putShort((short) 1);
    buffer.putShort((short) 1);
    buffer.putInt(16000);
    buffer.putInt(32000);
    buffer.putShort((short) 2);
    buffer.putShort((short) 16);

    buffer.put((byte) 'd');
    buffer.put((byte) 'a');
    buffer.put((byte) 't');
    buffer.put((byte) 'a');

    buffer.putInt(totalLength - 44);
    buffer.position(0);

    byte[] bytes = new byte[buffer.limit()];
    buffer.get(bytes);

    return bytes;
  }
}