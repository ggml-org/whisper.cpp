const path = require("path");
const { whisper } = require(path.join(
  __dirname,
  "../../../build/Release/addon.node"
));
const { promisify } = require("util");

const whisperAsync = promisify(whisper);

const whisperParamsMock = {
  language: "en",
  model: path.join(__dirname, "../../../models/ggml-base.en.bin"),
  fname_inp: path.join(__dirname, "../../../samples/jfk.wav"),
  use_gpu: true,
  flash_attn: false,
  no_prints: true,
  comma_in_time: false,
  translate: true,
  no_timestamps: false,
  detect_language: false,
  audio_ctx: 0,
  max_len: 0,
  prompt: "",
  print_progress: false,
  progress_callback: (progress) => {
    console.log(`Progress: ${progress}`);
  },
  max_context: -1
};

describe("Run whisper.node", () => {
    test("it should receive a non-empty value", async () => {
        let result = await whisperAsync(whisperParamsMock);
      console.log(result);

        expect(result['transcription'].length).toBeGreaterThan(0);
    }, 10000);
});

