const { join } = require('path');
const { whisper } = require('../../../build/Release/addon.node');
const { promisify } = require('util');

const whisperAsync = promisify(whisper);

const commonParams = {
  language: 'en',
  model: join(__dirname, '../../../models/ggml-base.en.bin'),
  fname_inp: join(__dirname, '../../../samples/jfk.wav'),
  use_gpu: true,
  flash_attn: false,
  no_prints: true,
  no_timestamps: false,
  detect_language: false,
  audio_ctx: 0,
  max_len: 0
};

describe('Whisper.cpp Node.js addon with VAD support', () => {
  test('Basic whisper transcription without VAD', async () => {
    const params = {
      ...commonParams,
      vad: false
    };

    const result = await whisperAsync(params);
    
    expect(typeof result).toBe('object');
    expect(Array.isArray(result.transcription)).toBe(true);
    expect(result.transcription.length).toBeGreaterThan(0);
    
    // Check that we got some transcription text
    const text = result.transcription.map(segment => segment[2]).join(' ');
    expect(text.length).toBeGreaterThan(0);
    expect(text.toLowerCase()).toContain('ask not');
  }, 30000);

  test('VAD parameters validation', async () => {
    // Test with invalid VAD model - should return empty transcription
    const invalidParams = {
      ...commonParams,
      vad: true,
      vad_model: 'non-existent-model.bin',
      vad_threshold: 0.5
    };

    // This should handle the error gracefully and return empty transcription
    const result = await whisperAsync(invalidParams);
    expect(typeof result).toBe('object');
    expect(Array.isArray(result.transcription)).toBe(true);
    // When VAD model doesn't exist, it should return empty transcription
    expect(result.transcription.length).toBe(0);
  }, 10000);

  test('VAD parameter parsing', async () => {
    // Test that VAD parameters are properly parsed (even if VAD model doesn't exist)
    const vadParams = {
      ...commonParams,
      vad: false, // Disabled so no model required
      vad_threshold: 0.7,
      vad_min_speech_duration_ms: 300,
      vad_min_silence_duration_ms: 150,
      vad_max_speech_duration_s: 45.0,
      vad_speech_pad_ms: 50,
      vad_samples_overlap: 0.15
    };

    const result = await whisperAsync(vadParams);
    
    expect(typeof result).toBe('object');
    expect(Array.isArray(result.transcription)).toBe(true);
  }, 30000);

  test('Progress callback with VAD disabled', async () => {
    let progressCalled = false;
    let lastProgress = 0;

    const params = {
      ...commonParams,
      vad: false,
      progress_callback: (progress) => {
        progressCalled = true;
        lastProgress = progress;
        expect(progress).toBeGreaterThanOrEqual(0);
        expect(progress).toBeLessThanOrEqual(100);
      }
    };

    const result = await whisperAsync(params);
    
    expect(progressCalled).toBe(true);
    expect(lastProgress).toBe(100);
    expect(typeof result).toBe('object');
  }, 30000);

  test('Language detection without VAD', async () => {
    const params = {
      ...commonParams,
      vad: false,
      detect_language: true,
      language: 'auto'
    };

    const result = await whisperAsync(params);
    
    expect(typeof result).toBe('object');
    expect(typeof result.language).toBe('string');
    expect(result.language.length).toBeGreaterThan(0);
  }, 30000);

  test('Basic transcription with all VAD parameters set', async () => {
    // Test with VAD disabled but all parameters set to ensure no crashes
    const params = {
      ...commonParams,
      vad: false, // Disabled so it works without VAD model
      vad_model: '', // Empty model path
      vad_threshold: 0.6,
      vad_min_speech_duration_ms: 200,
      vad_min_silence_duration_ms: 80,
      vad_max_speech_duration_s: 25.0,
      vad_speech_pad_ms: 40,
      vad_samples_overlap: 0.08
    };

    const result = await whisperAsync(params);
    
    expect(typeof result).toBe('object');
    expect(Array.isArray(result.transcription)).toBe(true);
    expect(result.transcription.length).toBeGreaterThan(0);
  }, 30000);
});

