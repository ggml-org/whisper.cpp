package com.litongjava.whisper.android.java.task;

import android.content.Context;
import android.os.Build;
import android.widget.TextView;

import com.blankj.utilcode.util.ThreadUtils;
import com.litongjava.jfinal.aop.Aop;
import com.litongjava.whisper.android.java.services.WhisperService;

import java.io.File;

public class TranscriptionTask extends ThreadUtils.Task<Object> {
  private final TextView tv;
  private final File sampleFile;

  public TranscriptionTask(TextView tv, File sampleFile) {
    this.tv = tv;
    this.sampleFile = sampleFile;

  }

  @Override
  public Object doInBackground() {
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
      Aop.get(WhisperService.class).transcribeSample(tv, sampleFile);
    }else{
      tv.append("not supported android devices");
    }
    return null;
  }

  @Override
  public void onSuccess(Object result) {
  }

  @Override
  public void onCancel() {
  }

  @Override
  public void onFail(Throwable t) {
  }
}