package com.whispercpp.java.whisper;

import android.os.Build;
import android.util.Log;

import androidx.annotation.RequiresApi;

import java.io.File;
import java.nio.file.Path;

public class WhisperUtils {
  private static final String LOG_TAG = "LibWhisper";


  public static boolean isArmEabiV7a() {
    return Build.SUPPORTED_ABIS[0].equals("armeabi-v7a");
  }

  public static boolean isArmEabiV8a() {
    return Build.SUPPORTED_ABIS[0].equals("arm64-v8a");
  }

  @RequiresApi(api = Build.VERSION_CODES.O)
  public static String cpuInfo() {
    try {
      Path path = new File("/proc/cpuinfo").toPath();
      return new String(java.nio.file.Files.readAllBytes(path));
    } catch (Exception e) {
      Log.w(LOG_TAG, "Couldn't read /proc/cpuinfo", e);
      return null;
    }

  }
}