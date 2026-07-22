import whisper_processor

try:
    result = whisper_processor.process_audio("./audio/wake_word_detected16k.wav", "base.en")
    print(result)
except Exception as e:
    print(f"Error: {e}")