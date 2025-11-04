var factory = require('../bindings/javascript/whisper.js')

factory().then(function(whisper) {
    var fs = require('fs');

    // to avoid reading WAV files and depending on some 3rd-party package, we read
    // 32-bit float PCM directly. to genereate it:
    //
    //   $ ffmpeg -i samples/jfk.wav -f f32le -acodec pcm_f32le samples/jfk.pcmf32
    //
    let fname_wav   = "../samples/jfk.pcmf32";
    let fname_model = "../models/ggml-base.en.bin";

    // init whisper
    {
        // read binary data from file
        var model_data = fs.readFileSync(fname_model);
        if (model_data == null) {
            console.log("whisper: failed to read model file");
            process.exit(1);
        }

        // write binary data to WASM memory
        whisper.FS_createDataFile("/", "whisper.bin", model_data, true, true);

        // init the model
        var ret = whisper.init("whisper.bin");
        if (ret == false) {
            console.log('whisper: failed to init');
            process.exit(1);
        }
    }

    // transcribe wav file
    {
        // read raw binary data
        var pcm_data = fs.readFileSync(fname_wav);
        if (pcm_data == null) {
            console.log("whisper: failed to read wav file");
            process.exit(1);
        }

        // convert to 32-bit float array
        var pcm = new Float32Array(pcm_data.buffer);

        // transcribe
        var ret = whisper.full_default(pcm, "en", false);
        if (ret != 0) {
            console.log("whisper: failed to transcribe");
            process.exit(1);
        }
    }

    // free memory
    {
        whisper.free();
    }
});
