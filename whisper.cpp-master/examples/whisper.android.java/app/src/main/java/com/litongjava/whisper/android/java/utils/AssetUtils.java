package com.litongjava.whisper.android.java.utils;

import android.content.Context;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public class AssetUtils {
  private static Logger log = LoggerFactory.getLogger(AssetUtils.class);

  public static File copyFileIfNotExists(Context context, File distDir, String filename) {
    File dstFile = new File(distDir, filename);
    if (dstFile.exists()) {
      return dstFile;
    } else {
      File parentFile = dstFile.getParentFile();
      log.info("parentFile:{}", parentFile);
      if (!parentFile.exists()) {
        parentFile.mkdirs();
      }
      AssetUtils.copyFileFromAssets(context, filename, dstFile);
    }
    return dstFile;
  }

  public static void copyDirectoryFromAssets(Context appCtx, String srcDir, String dstDir) {
    if (srcDir.isEmpty() || dstDir.isEmpty()) {
      return;
    }
    try {
      if (!new File(dstDir).exists()) {
        new File(dstDir).mkdirs();
      }
      for (String fileName : appCtx.getAssets().list(srcDir)) {
        String srcSubPath = srcDir + File.separator + fileName;
        String dstSubPath = dstDir + File.separator + fileName;
        if (new File(srcSubPath).isDirectory()) {
          copyDirectoryFromAssets(appCtx, srcSubPath, dstSubPath);
        } else {
          copyFileFromAssets(appCtx, srcSubPath, dstSubPath);
        }
      }
    } catch (Exception e) {
      e.printStackTrace();
    }
  }

  public static void copyFileFromAssets(Context appCtx, String srcPath, String dstPath) {
    File dstFile = new File(dstPath);
    copyFileFromAssets(appCtx, srcPath, dstFile);
  }

  public static void copyFileFromAssets(Context appCtx, String srcPath, File dstFile) {
    if (srcPath.isEmpty()) {
      return;
    }
    InputStream is = null;
    OutputStream os = null;
    try {
      is = new BufferedInputStream(appCtx.getAssets().open(srcPath));

      os = new BufferedOutputStream(new FileOutputStream(dstFile));
      byte[] buffer = new byte[1024];
      int length = 0;
      while ((length = is.read(buffer)) != -1) {
        os.write(buffer, 0, length);
      }
    } catch (FileNotFoundException e) {
      e.printStackTrace();
    } catch (IOException e) {
      e.printStackTrace();
    } finally {
      try {
        os.close();
        is.close();
      } catch (IOException e) {
        e.printStackTrace();
      }
    }

  }
}