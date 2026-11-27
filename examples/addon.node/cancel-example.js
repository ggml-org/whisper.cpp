// Demonstrates cancelling an in-flight transcription via AbortSignal (params.signal).
//
// Usage: node cancel-example.js [--model=path/to/model.bin]

const path = require("path");
const os = require("os");
const { promisify } = require("util");

const isWindows = os.platform() === "win32";
const buildPath = isWindows ? "../../build/bin/Release/addon.node" : "../../build/Release/addon.node";
const { whisper } = require(path.join(__dirname, buildPath));

const whisperAsync = promisify(whisper);

const modelArg = process.argv.find((a) => a.startsWith("--model="));
const model = modelArg
  ? modelArg.slice("--model=".length)
  : path.join(__dirname, "../../models/ggml-base.en.bin");

// Long synthetic audio (tone + noise) so the transcription runs long enough
// to be cancelled mid-flight.
function syntheticAudio(seconds) {
  const n = 16000 * seconds;
  const pcm = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    pcm[i] = 0.05 * Math.sin((2 * Math.PI * 440 * i) / 16000) + (Math.random() - 0.5) * 0.02;
  }
  return pcm;
}

const baseParams = {
  language: "en",
  model,
  use_gpu: true,
  no_prints: true,
  no_timestamps: false,
  comma_in_time: false,
};

async function cancelMidFlight() {
  console.log("--- test 1: cancel mid-transcription ---");
  const ac = new AbortController();
  const progressSeen = [];

  const t0 = Date.now();
  const promise = whisperAsync({
    ...baseParams,
    fname_inp: "",
    pcmf32: syntheticAudio(600),
    signal: ac.signal,
    progress_callback: (p) => {
      progressSeen.push(p);
      console.log(`progress: ${p}%`);
      if (!ac.signal.aborted) {
        console.log(">>> calling abort()");
        ac.abort();
      }
    },
  });

  const result = await promise;
  const elapsed = Date.now() - t0;

  console.log(`cancelled = ${result.cancelled}, segments = ${result.transcription.length}, elapsed = ${elapsed} ms`);
  if (result.cancelled !== true) throw new Error("FAIL: expected cancelled === true");
  if (progressSeen.includes(100)) throw new Error("FAIL: transcription ran to completion, was not cancelled");
  console.log("PASS\n");
}

async function preAbortedSignal() {
  console.log("--- test 2: already-aborted signal ---");
  const ac = new AbortController();
  ac.abort();

  const t0 = Date.now();
  const result = await whisperAsync({
    ...baseParams,
    fname_inp: "",
    pcmf32: syntheticAudio(600),
    signal: ac.signal,
  });
  const elapsed = Date.now() - t0;

  console.log(`cancelled = ${result.cancelled}, segments = ${result.transcription.length}, elapsed = ${elapsed} ms`);
  if (result.cancelled !== true) throw new Error("FAIL: expected cancelled === true");
  if (result.transcription.length !== 0) throw new Error("FAIL: expected no segments");
  console.log("PASS\n");
}

async function normalRun() {
  console.log("--- test 3: normal run without signal (regression) ---");
  const t0 = Date.now();
  const result = await whisperAsync({
    ...baseParams,
    fname_inp: path.join(__dirname, "../../samples/jfk.wav"),
  });
  const elapsed = Date.now() - t0;

  const text = result.transcription.map((s) => s[2]).join(" ");
  console.log(`cancelled = ${result.cancelled}, segments = ${result.transcription.length}, elapsed = ${elapsed} ms`);
  console.log(`text: ${text.trim()}`);
  if (result.cancelled !== false) throw new Error("FAIL: expected cancelled === false");
  if (!text.toLowerCase().includes("ask not")) throw new Error("FAIL: unexpected transcription");
  console.log("PASS\n");
}

(async () => {
  await cancelMidFlight();
  await preAbortedSignal();
  await normalRun();
  console.log("ALL TESTS PASSED");
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
