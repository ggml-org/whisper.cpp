# whisper.objc

Minimal Obj-C application for automatic offline speech recognition.
The inference runs locally, on-device.

https://user-images.githubusercontent.com/1991296/197385372-962a6dea-bca1-4d50-bf96-1d8c27b98c81.mp4

Real-time transcription demo:

https://user-images.githubusercontent.com/1991296/204126266-ce4177c6-6eca-4bd9-bca8-0e46d9da2364.mp4

## Usage

This example uses the whisper.xcframework which needs to be built first using the following command:
```bash
./build-xcframework.sh
```

A model is also required to be downloaded and can be done using the following command:
```bash
./models/download-ggml-model.sh base.en
```

If you don't want to convert a Core ML model, you can skip this step by creating dummy model:
```bash
mkdir models/ggml-base.en-encoder.mlmodelc
```

### Core ML support
1. Follow all the steps in the `Usage` section, including adding the ggml model file.  
The ggml model file is required as the Core ML model is only used for the encoder. The
decoder which is in the ggml model is still required.
2. Follow the [`Core ML support` section of readme](../../README.md#core-ml-support) to convert the
model.
3. Add the Core ML model (`models/ggml-base.en-encoder.mlmodelc/`) to `whisper.swiftui.demo/Resources/models` **via Xcode**.

When the example starts running you should now see that it is using the Core ML model:
```console
whisper_init_state: loading Core ML model from '/Library/Developer/CoreSimulator/Devices/25E8C27D-0253-4281-AF17-C3F2A4D1D8F4/data/Containers/Bundle/Application/3ADA7D59-7B9C-43B4-A7E1-A87183FC546A/whisper.swiftui.app/models/ggml-base.en-encoder.mlmodelc'
whisper_init_state: first run on a device may take a while ...
whisper_init_state: Core ML model loaded
```
