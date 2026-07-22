package com.litongjava.whisper.android.java.single;

import android.app.Application;
import android.os.Build;
import android.os.Handler;

import androidx.annotation.RequiresApi;

import com.blankj.utilcode.util.ToastUtils;
import com.blankj.utilcode.util.Utils;
import com.litongjava.jfinal.aop.Aop;
import com.litongjava.whisper.android.java.bean.WhisperSegment;
import com.litongjava.whisper.android.java.utils.AssetUtils;
import com.whispercpp.java.whisper.WhisperContext;

import java.io.File;
import java.util.List;
import java.util.concurrent.ExecutionException;


@RequiresApi(api = Build.VERSION_CODES.O)
public enum LocalWhisper {
  INSTANCE;

  public static final String modelFilePath = "models/ggml-tiny.bin";
  private WhisperContext whisperContext;

  @RequiresApi(api = Build.VERSION_CODES.O)
  LocalWhisper() {
    Application context = Utils.getApp();
    File filesDir = context.getFilesDir();
    File modelFile = AssetUtils.copyFileIfNotExists(context, filesDir, modelFilePath);
    String realModelFilePath = modelFile.getAbsolutePath();
    whisperContext = WhisperContext.createContextFromFile(realModelFilePath);
  }

  public synchronized String transcribeData(float[] data) throws ExecutionException, InterruptedException {
    if(whisperContext==null){
        toastModelLoading();
        return null;
    }else{
      return whisperContext.transcribeData(data);
    }
  }

    private static void toastModelLoading() {
        Aop.get(Handler.class).post(()->{
          ToastUtils.showShort("please wait for model loading");
        });
    }

    public List<WhisperSegment> transcribeDataWithTime(float[] audioData) throws ExecutionException, InterruptedException {
    if(whisperContext==null){
        toastModelLoading();
      return null;
    }else{
      return whisperContext.transcribeDataWithTime(audioData);
    }
  }

  public void init() {
    //noting to do.but init
  }


}
