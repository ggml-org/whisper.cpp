# Convert the 30-second base.en encoder to an unquantized RKNN model for RK3588.
# Tested with openai-whisper==20231117, torch==2.4.0, onnx==1.17.0,
# and rknn-toolkit2==2.3.2. No Model Zoo checkout is required.
import argparse
import os


def convert_encoder(hparams, encoder, mname, output_dir):
    encoder.eval()
    encoder.requires_grad_(False)

    # Whisper's default 30-second window contains 3000 mel frames.
    input_shape = [1, hparams.n_mels, 3000]
    output_shape = [1, hparams.n_audio_ctx, hparams.n_audio_state]
    if input_shape != [1, 80, 3000] or output_shape != [1, 1500, 512]:
        raise ValueError("RKNN support requires base.en [1,80,3000] -> [1,1500,512]")

    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "ggml-" + mname + "-encoder")
    onnx_path = model_path + ".onnx"
    rknn_path = model_path + ".rknn"

    print("--> Exporting 30-second encoder to ONNX: " + onnx_path)
    mel = torch.zeros(input_shape)
    with torch.no_grad():
        torch.onnx.export(
            encoder,
            mel,
            onnx_path,
            input_names=["mel"],
            output_names=["output_features"],
            opset_version=12,
        )

    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise ValueError("Expected one ONNX input and one output")
    actual_input = [dim.dim_value for dim in model.graph.input[0].type.tensor_type.shape.dim]
    actual_output = [dim.dim_value for dim in model.graph.output[0].type.tensor_type.shape.dim]
    if actual_input != input_shape or actual_output != output_shape:
        raise ValueError(f"Unexpected ONNX shapes: {actual_input} -> {actual_output}")
    print(f"ONNX shape check passed: {actual_input} -> {actual_output}")

    rknn = RKNN(verbose=False)
    try:
        print("--> Configuring RKNN for rk3588")
        ret = rknn.config(target_platform="rk3588")
        if ret != 0:
            raise RuntimeError(f"RKNN config failed: {ret}")

        print("--> Loading ONNX model")
        ret = rknn.load_onnx(model=onnx_path)
        if ret != 0:
            raise RuntimeError(f"RKNN load_onnx failed: {ret}")

        print("--> Building floating-point RKNN model (no quantization)")
        ret = rknn.build(do_quantization=False)
        if ret != 0:
            raise RuntimeError(f"RKNN build failed: {ret}")

        print("--> Exporting RKNN model: " + rknn_path)
        ret = rknn.export_rknn(rknn_path)
        if ret != 0:
            raise RuntimeError(f"RKNN export_rknn failed: {ret}")
    finally:
        rknn.release()

    size = os.path.getsize(rknn_path)
    with open(rknn_path, "rb") as model_file:
        if size <= 4 or model_file.read(4) != b"RKNN":
            raise ValueError("Invalid RKNN file header")

    print(f"Done: {rknn_path} ({size} bytes)")
    print(f"Input/output shape check passed (ONNX): {actual_input} -> {actual_output}")
    print("RKNN magic check passed; verify runtime tensor shapes on an RK3588 device.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a Whisper encoder to RKNN for RK3588")
    parser.add_argument("--model", type=str, choices=["base.en"], required=True,
                        help="model to convert (currently base.en only)")
    parser.add_argument("--output-dir", type=str, default=os.path.dirname(os.path.abspath(__file__)),
                        help="output directory for ONNX and RKNN files (default: models directory)")
    args = parser.parse_args()

    try:
        import torch
        import onnx
        from whisper import load_model
        from rknn.api import RKNN
    except ImportError as error:
        parser.exit(1, f"Missing conversion dependency: {error}\n"
                    "Install openai-whisper==20231117 torch==2.4.0 onnx==1.17.0 "
                    "rknn-toolkit2==2.3.2 in a Python virtual environment.\n")

    print("--> Loading Whisper model: " + args.model + " (reuses the Whisper weight cache)")
    whisper = load_model(args.model, device="cpu")
    convert_encoder(whisper.dims, whisper.encoder, args.model, args.output_dir)
