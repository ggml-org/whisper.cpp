package com.whispercpp.java.whisper;

import android.os.Build;
import android.util.Log;

import androidx.annotation.RequiresApi;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class CpuInfo {
  private static final String LOG_TAG = "WhisperCpuConfig";

  private List<String> lines;

  public CpuInfo(List<String> lines) {
    this.lines = lines;
  }

  @RequiresApi(api = Build.VERSION_CODES.N)
  public int getHighPerfCpuCount0() {
    try {
      return getHighPerfCpuCountByFrequencies();
    } catch (Exception e) {
      Log.d(LOG_TAG, "Couldn't read CPU frequencies", e);
      return getHighPerfCpuCountByVariant();
    }
  }

  @RequiresApi(api = Build.VERSION_CODES.N)
  private int getHighPerfCpuCountByFrequencies() {
    List<Integer> frequencies = getCpuValues("processor", line -> {
        try {
          return getMaxCpuFrequency(Integer.parseInt(line.trim()));
        } catch (IOException e) {
          e.printStackTrace();
        }
        return 0;
      }
    );
    Log.d(LOG_TAG, "Binned cpu frequencies (frequency, count): " + binnedValues(frequencies));
    return countDroppingMin(frequencies);
  }

  @RequiresApi(api = Build.VERSION_CODES.N)
  private int getHighPerfCpuCountByVariant() {
    List<Integer> variants = getCpuValues("CPU variant", line -> Integer.parseInt(line.trim().substring(line.indexOf("0x") + 2), 16));
    Log.d(LOG_TAG, "Binned cpu variants (variant, count): " + binnedValues(variants));
    return countKeepingMin(variants);
  }

  @RequiresApi(api = Build.VERSION_CODES.N)
  private Map<Integer, Integer> binnedValues(List<Integer> values) {
    Map<Integer, Integer> countMap = new HashMap<>();
    for (int value : values) {
      countMap.put(value, countMap.getOrDefault(value, 0) + 1);
    }
    return countMap;
  }

  @RequiresApi(api = Build.VERSION_CODES.N)
  private List<Integer> getCpuValues(String property, Mapper mapper) {
    List<Integer> values = new ArrayList<>();
    for (String line : lines) {
      if (line.startsWith(property)) {
        values.add(mapper.map(line.substring(line.indexOf(':') + 1)));
      }
    }
    values.sort(Integer::compareTo);
    return values;
  }

  @RequiresApi(api = Build.VERSION_CODES.N)
  private int countDroppingMin(List<Integer> values) {
    int min = values.stream().mapToInt(i -> i).min().orElse(Integer.MAX_VALUE);
    return (int) values.stream().filter(value -> value > min).count();
  }

  @RequiresApi(api = Build.VERSION_CODES.N)
  private int countKeepingMin(List<Integer> values) {
    int min = values.stream().mapToInt(i -> i).min().orElse(Integer.MAX_VALUE);
    return (int) values.stream().filter(value -> value.equals(min)).count();
  }

  @RequiresApi(api = Build.VERSION_CODES.N)
  public static int getHighPerfCpuCount() {
    try {
      return readCpuInfo().getHighPerfCpuCount0();
    } catch (Exception e) {
      Log.d(LOG_TAG, "Couldn't read CPU info", e);
      return Math.max(Runtime.getRuntime().availableProcessors() - 4, 0);
    }
  }

  private static CpuInfo readCpuInfo() throws IOException {
    try (BufferedReader reader = new BufferedReader(new FileReader("/proc/cpuinfo"))) {
      List<String> lines = new ArrayList<>();
      String line;
      while ((line = reader.readLine()) != null) {
        lines.add(line);
      }
      return new CpuInfo(lines);
    }
  }

  private static int getMaxCpuFrequency(int cpuIndex) throws IOException {
    String path = "/sys/devices/system/cpu/cpu" + cpuIndex + "/cpufreq/cpuinfo_max_freq";
    try (BufferedReader reader = new BufferedReader(new FileReader(path))) {
      return Integer.parseInt(reader.readLine());
    }
  }

  private interface Mapper {
    int map(String line);
  }
}
