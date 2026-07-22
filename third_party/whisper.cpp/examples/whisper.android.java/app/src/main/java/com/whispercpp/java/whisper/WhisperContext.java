package com.whispercpp.java.whisper;

import android.content.res.AssetManager;
import android.os.Build;
import android.util.Log;

import androidx.annotation.RequiresApi;

import com.litongjava.whisper.android.java.bean.WhisperSegment;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class WhisperContext {

  private static final String LOG_TAG = "LibWhisper";
  private long ptr;
  private final ExecutorService executorService;

  private WhisperContext(long ptr) {
    this.ptr = ptr;
    this.executorService = Executors.newSingleThreadExecutor();
  }

  public String transcribeData(float[] data) throws ExecutionException, InterruptedException {
    return executorService.submit(new Callable<String>() {
      @RequiresApi(api = Build.VERSION_CODES.O)
      @Override
      public String call() throws Exception {
        if (ptr == 0L) {
          throw new IllegalStateException();
        }
        int numThreads = WhisperCpuConfig.getPreferredThreadCount();
        Log.d(LOG_TAG, "Selecting " + numThreads + " threads");

        StringBuilder result = new StringBuilder();
        synchronized (this) {

          WhisperLib.fullTranscribe(ptr, numThreads, data);
          int textCount = WhisperLib.getTextSegmentCount(ptr);
          for (int i = 0; i < textCount; i++) {
            String sentence = WhisperLib.getTextSegment(ptr, i);
            result.append(sentence);
          }
        }
        return result.toString();
      }
    }).get();
  }

  public List<WhisperSegment> transcribeDataWithTime(float[] data) throws ExecutionException, InterruptedException {
    return executorService.submit(new Callable<List<WhisperSegment>>() {
      @RequiresApi(api = Build.VERSION_CODES.O)
      @Override
      public List<WhisperSegment> call() throws Exception {
        if (ptr == 0L) {
          throw new IllegalStateException();
        }
        int numThreads = WhisperCpuConfig.getPreferredThreadCount();
        Log.d(LOG_TAG, "Selecting " + numThreads + " threads");

        List<WhisperSegment> segments = new ArrayList<>();
        synchronized (this) {
//          StringBuilder result = new StringBuilder();
          WhisperLib.fullTranscribe(ptr, numThreads, data);
          int textCount = WhisperLib.getTextSegmentCount(ptr);
          for (int i = 0; i < textCount; i++) {
            long start = WhisperLib.getTextSegmentT0(ptr, i);
            String sentence = WhisperLib.getTextSegment(ptr, i);
            long end = WhisperLib.getTextSegmentT1(ptr, i);
//            result.append();
            segments.add(new WhisperSegment(start, end, sentence));

          }
//          return result.toString();
        }
        return segments;
      }
    }).get();
  }

  @RequiresApi(api = Build.VERSION_CODES.O)
  public String benchMemory(int nthreads) throws ExecutionException, InterruptedException {
    return executorService.submit(() -> WhisperLib.benchMemcpy(nthreads)).get();
  }

  @RequiresApi(api = Build.VERSION_CODES.O)
  public String benchGgmlMulMat(int nthreads) throws ExecutionException, InterruptedException {
    return executorService.submit(() -> WhisperLib.benchGgmlMulMat(nthreads)).get();
  }

  @RequiresApi(api = Build.VERSION_CODES.O)
  public void release() throws ExecutionException, InterruptedException {
    executorService.submit(() -> {
      if (ptr != 0L) {
        WhisperLib.freeContext(ptr);
        ptr = 0;
      }
    }).get();
  }

  @RequiresApi(api = Build.VERSION_CODES.O)
  public static WhisperContext createContextFromFile(String filePath) {
    long ptr = WhisperLib.initContext(filePath);
    if (ptr == 0L) {
      throw new RuntimeException("Couldn't create context with path " + filePath);
    }
    return new WhisperContext(ptr);
  }

  @RequiresApi(api = Build.VERSION_CODES.O)
  public static WhisperContext createContextFromInputStream(InputStream stream) {
    long ptr = WhisperLib.initContextFromInputStream(stream);
    if (ptr == 0L) {
      throw new RuntimeException("Couldn't create context from input stream");
    }
    return new WhisperContext(ptr);
  }

  @RequiresApi(api = Build.VERSION_CODES.O)
  public static WhisperContext createContextFromAsset(AssetManager assetManager, String assetPath) {
    long ptr = WhisperLib.initContextFromAsset(assetManager, assetPath);
    if (ptr == 0L) {
      throw new RuntimeException("Couldn't create context from asset " + assetPath);
    }
    return new WhisperContext(ptr);
  }

  @RequiresApi(api = Build.VERSION_CODES.O)
  public static String getSystemInfo() {
    return WhisperLib.getSystemInfo();
  }
}
