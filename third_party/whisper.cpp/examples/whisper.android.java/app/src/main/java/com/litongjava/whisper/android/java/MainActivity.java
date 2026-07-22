package com.litongjava.whisper.android.java;

import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.view.View;
import android.widget.TextView;

import com.blankj.utilcode.util.ThreadUtils;
import com.litongjava.android.view.inject.annotation.FindViewById;
import com.litongjava.android.view.inject.annotation.FindViewByIdLayout;
import com.litongjava.android.view.inject.annotation.OnClick;
import com.litongjava.android.view.inject.utils.ViewInjectUtils;
import com.litongjava.jfinal.aop.Aop;
import com.litongjava.jfinal.aop.AopManager;
import com.litongjava.whisper.android.java.services.WhisperService;
import com.litongjava.whisper.android.java.task.LoadModelTask;
import com.litongjava.whisper.android.java.task.TranscriptionTask;
import com.litongjava.whisper.android.java.utils.AssetUtils;
import com.whispercpp.java.whisper.WhisperLib;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;


@FindViewByIdLayout(R.layout.activity_main)
public class MainActivity extends AppCompatActivity {

  @FindViewById(R.id.sample_text)
  private TextView tv;

  Logger log = LoggerFactory.getLogger(this.getClass());
  private WhisperService whisperService = Aop.get(WhisperService.class);

  @RequiresApi(api = Build.VERSION_CODES.O)
  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);
    //setContentView(R.layout.activity_main);
    ViewInjectUtils.injectActivity(this, this);
    initAopBean();
    showSystemInfo();
  }

  private void initAopBean() {
    Handler mainHandler = new Handler(Looper.getMainLooper());
    AopManager.me().addSingletonObject(mainHandler);
  }

  @RequiresApi(api = Build.VERSION_CODES.O)
  @OnClick(R.id.loadModelBtn)
  public void loadModelBtn_OnClick(View v) {
    Context context = getBaseContext();
    ThreadUtils.executeByIo(new LoadModelTask(tv));
  }

  @OnClick(R.id.transcriptSampleBtn)
  public void transcriptSampleBtn_OnClick(View v) {
    Context context = getBaseContext();

    long start = System.currentTimeMillis();
    String sampleFilePath = "samples/jfk.wav";
    File filesDir = context.getFilesDir();
    File sampleFile = AssetUtils.copyFileIfNotExists(context, filesDir, sampleFilePath);
    long end = System.currentTimeMillis();
    String msg = "copy file:" + (end - start) + "ms";
    outputMsg(tv, msg);
    ThreadUtils.executeByIo(new TranscriptionTask(tv, sampleFile));
  }

  private void outputMsg(TextView tv, String msg) {
    tv.append(msg + "\n");
    log.info(msg);
  }


  @RequiresApi(api = Build.VERSION_CODES.O)
  @OnClick(R.id.systemInfoBtn)
  public void systemInfoBtn_OnClick(View v) {
    showSystemInfo();
  }

  @RequiresApi(api = Build.VERSION_CODES.O)
  public void showSystemInfo() {
    String systemInfo = WhisperLib.getSystemInfo();
    tv.append(systemInfo + "\n");
  }

  @OnClick(R.id.clearBtn)
  public void clearBtn_OnClick(View v) {
    tv.setText("");
  }

  @RequiresApi(api = Build.VERSION_CODES.O)
  @Override
  protected void onDestroy() {
    super.onDestroy();
    whisperService.release();
  }
}