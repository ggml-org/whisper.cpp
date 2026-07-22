package com.litongjava.whisper.android.java.app;

import android.app.Application;

import com.blankj.utilcode.util.Utils;

public class App extends Application {
  @Override
  public void onCreate() {
    super.onCreate();
    Utils.init(this);
  }
}
