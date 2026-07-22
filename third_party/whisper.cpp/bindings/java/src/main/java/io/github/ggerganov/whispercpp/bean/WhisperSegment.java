package io.github.ggerganov.whispercpp.bean;

/**
 * Created by litonglinux@qq.com on 10/21/2023_7:48 AM
 */
public class WhisperSegment {
  private long start, end;
  private String sentence;

  public WhisperSegment() {
  }

  public WhisperSegment(long start, long end, String sentence) {
    this.start = start;
    this.end = end;
    this.sentence = sentence;
  }

  public long getStart() {
    return start;
  }

  public long getEnd() {
    return end;
  }

  public String getSentence() {
    return sentence;
  }

  public void setStart(long start) {
    this.start = start;
  }

  public void setEnd(long end) {
    this.end = end;
  }

  public void setSentence(String sentence) {
    this.sentence = sentence;
  }

  @Override
  public String toString() {
    return "[" + start + " --> " + end + "]:" + sentence;
  }
}
