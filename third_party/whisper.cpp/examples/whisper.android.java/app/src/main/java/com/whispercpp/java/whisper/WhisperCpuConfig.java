package com.whispercpp.java.whisper;

import android.os.Build;

import androidx.annotation.RequiresApi;

public class WhisperCpuConfig {
  @RequiresApi(api = Build.VERSION_CODES.N)
  public static int getPreferredThreadCount() {
    return Math.max(CpuInfo.getHighPerfCpuCount(), 2);
  }
}
