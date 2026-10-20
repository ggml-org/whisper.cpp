import importlib.util
import sys
import types
import unittest
from pathlib import Path


def install_gguf_stub():
    gguf = types.ModuleType("gguf")
    gguf.GGMLQuantizationType = types.SimpleNamespace(
        BF16=0,
        Q8_0=1,
        Q5_0=2,
        Q4_0=3,
    )
    gguf.LlamaFileType = types.SimpleNamespace(
        ALL_F32=1,
        MOSTLY_F16=2,
        MOSTLY_Q8_0=3,
        MOSTLY_Q5_0=4,
        MOSTLY_Q4_0=5,
        MOSTLY_BF16=6,
    )
    sys.modules["gguf"] = gguf


def install_torch_stub():
    torch = types.ModuleType("torch")
    torch.Tensor = object
    sys.modules["torch"] = torch


def install_numpy_stub():
    numpy = types.ModuleType("numpy")
    numpy.ndarray = object
    numpy.float32 = object
    sys.modules["numpy"] = numpy


def install_yaml_stub():
    sys.modules["yaml"] = types.ModuleType("yaml")


try:
    import gguf
except ModuleNotFoundError:
    install_gguf_stub()

try:
    import torch
except ModuleNotFoundError:
    install_torch_stub()

try:
    import numpy
except ModuleNotFoundError:
    install_numpy_stub()

try:
    import yaml
except ModuleNotFoundError:
    install_yaml_stub()


SCRIPTS_DIR = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
SCRIPT = SCRIPTS_DIR / "convert-nemo-to-gguf.py"
SPEC = importlib.util.spec_from_file_location("convert_nemo_to_gguf", SCRIPT)
CONVERTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONVERTER)


def unified_config():
    return {
        "target": "nemo.collections.asr.models.rnnt_bpe_models.EncDecRNNTBPEModel",
        "model_defaults": {
            "enc_hidden": 1024,
            "pred_hidden": 640,
            "joint_hidden": 640,
        },
        "decoder": {
            "vocab_size": 1024,
            "prednet": {
                "pred_hidden": 640,
                "pred_rnn_layers": 2,
            },
        },
        "joint": {
            "num_classes": 1024,
            "jointnet": {
                "joint_hidden": 640,
            },
        },
        "decoding": {
            "greedy": {
                "max_symbols": 10,
            },
        },
        "loss": {
            "loss_name": "default",
        },
        "labels": ["<unk>", "word"],
    }


class RecordingWriter:
    def __init__(self):
        self.values = {}

    def add_uint32(self, name, value):
        self.values[name] = value

    def add_array(self, name, value):
        self.values[name] = value

    def add_string(self, name, value):
        self.values[name] = value

    def add_description(self, value):
        self.values["general.description"] = value


class FakeTensor:
    def __init__(self, shape):
        self.shape = shape


def transducer_state_dict():
    return {
        "decoder.prediction.embed.weight": "embed",
        "decoder.prediction.dec_rnn.lstm.weight_ih_l0": "w_ih_0",
        "decoder.prediction.dec_rnn.lstm.weight_hh_l0": "w_hh_0",
        "decoder.prediction.dec_rnn.lstm.bias_ih_l0": "b_ih_0",
        "decoder.prediction.dec_rnn.lstm.bias_hh_l0": "b_hh_0",
        "decoder.prediction.dec_rnn.lstm.weight_ih_l1": "w_ih_1",
        "decoder.prediction.dec_rnn.lstm.weight_hh_l1": "w_hh_1",
        "decoder.prediction.dec_rnn.lstm.bias_ih_l1": "b_ih_1",
        "decoder.prediction.dec_rnn.lstm.bias_hh_l1": "b_hh_1",
        "joint.enc.weight": "joint_enc_w",
        "joint.enc.bias": "joint_enc_b",
        "joint.pred.weight": "joint_pred_w",
        "joint.pred.bias": "joint_pred_b",
        "joint.joint_net.2.weight": FakeTensor((1025, 640)),
        "joint.joint_net.2.bias": FakeTensor((1025,)),
    }


def nemotron_config():
    config = unified_config()
    config.update(
        {
            "target": CONVERTER.NEMOTRON_TARGET,
            "num_prompts": 128,
            "encoder": {
                "n_layers": 24,
                "d_model": 1024,
                "n_heads": 8,
                "subsampling_factor": 8,
                "conv_kernel_size": 9,
                "att_context_style": "chunked_limited",
                "att_context_size": [
                    [56, 3],
                    [56, 0],
                    [56, 6],
                    [56, 13],
                ],
                "conv_context_size": "causal",
                "causal_downsampling": True,
                "conv_norm_type": "layer_norm",
            },
            "model_defaults": {
                "initialize_prompt_feature": True,
                "num_prompts": 128,
                "prompt_dictionary": {
                    "en-US": 0,
                    "en": 0,
                    "auto": 101,
                },
            },
            "decoder": {
                "vocab_size": 13087,
                "prednet": {
                    "pred_hidden": 640,
                    "pred_rnn_layers": 2,
                },
            },
            "joint": {
                "num_classes": 13087,
                "jointnet": {
                    "encoder_hidden": 1024,
                    "pred_hidden": 640,
                    "joint_hidden": 640,
                },
            },
        },
    )
    return config


def nemotron_state_dict():
    shapes = {
        "prompt_kernel.0.weight": (2048, 1152),
        "prompt_kernel.0.bias": (2048,),
        "prompt_kernel.2.weight": (1024, 2048),
        "prompt_kernel.2.bias": (1024,),
        "decoder.prediction.embed.weight": (13088, 640),
        "decoder.prediction.dec_rnn.lstm.weight_ih_l0": (2560, 640),
        "decoder.prediction.dec_rnn.lstm.weight_hh_l0": (2560, 640),
        "decoder.prediction.dec_rnn.lstm.bias_ih_l0": (2560,),
        "decoder.prediction.dec_rnn.lstm.bias_hh_l0": (2560,),
        "decoder.prediction.dec_rnn.lstm.weight_ih_l1": (2560, 640),
        "decoder.prediction.dec_rnn.lstm.weight_hh_l1": (2560, 640),
        "decoder.prediction.dec_rnn.lstm.bias_ih_l1": (2560,),
        "decoder.prediction.dec_rnn.lstm.bias_hh_l1": (2560,),
        "joint.enc.weight": (640, 1024),
        "joint.enc.bias": (640,),
        "joint.pred.weight": (640, 640),
        "joint.pred.bias": (640,),
        "joint.joint_net.2.weight": (13088, 640),
        "joint.joint_net.2.bias": (13088,),
    }
    return {
        name: FakeTensor(shape)
        for name, shape in shapes.items()
    }


def eou_config():
    config = unified_config()
    config["encoder"] = {
        "d_model": 512,
        "n_layers": 17,
        "n_heads": 8,
        "subsampling_factor": 8,
        "causal_downsampling": True,
        "conv_context_size": "causal",
        "conv_norm_type": "layer_norm",
        "att_context_style": "chunked_limited",
        "att_context_size": [70, 1],
    }
    config["decoder"]["vocab_size"] = 1026
    config["decoder"]["prednet"]["pred_rnn_layers"] = 1
    config["joint"]["num_classes"] = 1026
    config["labels"] = [f"token-{index}" for index in range(1024)] + [
        "<EOU>", "<EOB>"]
    return config


class ConverterRnntTests(unittest.TestCase):
    def test_detects_standard_rnnt(self):
        self.assertEqual(CONVERTER.detect_model_type(unified_config()), "rnnt")

    def test_keeps_tdt_detection(self):
        config = unified_config()
        config["model_defaults"]["tdt_durations"] = [0, 1, 2]
        self.assertEqual(CONVERTER.detect_model_type(config), "tdt")

    def test_keeps_eou_detection(self):
        config = unified_config()
        config["labels"] = ["<unk>", "<EOU>"]
        self.assertEqual(CONVERTER.detect_model_type(config), "eou")

    def test_eou_causal_subsampling_frequency_geometry(self):
        self.assertEqual([
            CONVERTER.subsampling_freq_bins(128, factor, True)
            for factor in (1, 2, 4, 8)
        ], [128, 65, 33, 17])
        self.assertEqual(
            CONVERTER.subsampling_freq_bins(128, 8, False), 16)

    def test_rejects_non_power_of_two_subsampling(self):
        with self.assertRaisesRegex(ValueError, "positive power of two"):
            CONVERTER.subsampling_freq_bins(128, 6, True)

    def test_eou_license_contract(self):
        writer = RecordingWriter()
        CONVERTER.write_eou_provenance(writer)
        self.assertEqual(writer.values["general.license"],
                         "NVIDIA Open Model License")
        self.assertEqual(
            writer.values["general.source.url"],
            "https://huggingface.co/nvidia/parakeet_realtime_eou_120m-v1")
        self.assertIn("NVIDIA Open Model License",
                      writer.values["general.description"])

    def test_validates_eou_checkpoint_contract(self):
        CONVERTER.validate_eou_contract(eou_config())

    def test_eou_required_tensor_set_covers_encoder_and_decoder(self):
        required = CONVERTER.eou_required_tensor_names(eou_config())
        self.assertIn("encoder.pre_encode.conv.0.weight", required)
        self.assertIn("encoder.layers.16.conv.batch_norm.weight", required)
        self.assertIn("decoder.prediction.dec_rnn.lstm.weight_ih_l0", required)
        self.assertIn("joint.joint_net.2.weight", required)

        state_dict = {name: object() for name in required}
        CONVERTER.validate_eou_contract(eou_config(), state_dict)
        del state_dict["encoder.layers.16.conv.batch_norm.weight"]
        with self.assertRaisesRegex(ValueError, "missing required tensors"):
            CONVERTER.validate_eou_contract(eou_config(), state_dict)

    def test_rejects_wrong_eou_attention_context(self):
        config = eou_config()
        config["encoder"]["att_context_size"] = [64, 1]
        with self.assertRaisesRegex(ValueError, r"\[70, 1\]"):
            CONVERTER.validate_eou_contract(config)

    def test_rejects_wrong_eou_token_ids(self):
        config = eou_config()
        config["labels"][1023], config["labels"][1024] = (
            config["labels"][1024], config["labels"][1023])
        with self.assertRaisesRegex(ValueError, "token id 1024"):
            CONVERTER.validate_eou_contract(config)

    def test_can_select_hybrid_rnnt_branch(self):
        config = unified_config()
        config["target"] = (
            "nemo.collections.asr.models.hybrid_rnnt_ctc_bpe_models."
            "EncDecHybridRNNTCTCBPEModel"
        )
        self.assertEqual(CONVERTER.detect_model_type(config), "ctc")
        self.assertEqual(
            CONVERTER.detect_model_type(config, "rnnt"), "rnnt")

    def test_validates_rnnt_contract(self):
        CONVERTER.validate_rnnt_contract(
            unified_config(), transducer_state_dict())

    def test_rejects_tdt_shaped_joint_for_rnnt(self):
        state_dict = transducer_state_dict()
        state_dict["joint.joint_net.2.weight"] = FakeTensor((1028, 640))
        with self.assertRaisesRegex(ValueError, "duration logits present"):
            CONVERTER.validate_rnnt_contract(unified_config(), state_dict)

    def test_uncapped_max_symbols_uses_runtime_guard(self):
        config = unified_config()
        config["decoding"]["greedy"]["max_symbols"] = None
        self.assertEqual(CONVERTER.rnnt_max_symbols(config), 10)

    def test_writes_rnnt_metadata_without_durations(self):
        writer = RecordingWriter()
        CONVERTER.write_transducer_metadata(
            writer, unified_config(), "rnnt")

        self.assertEqual(writer.values["parakeet.rnnt.vocab_size"], 1024)
        self.assertEqual(writer.values["parakeet.rnnt.blank_id"], 1024)
        self.assertEqual(
            writer.values["parakeet.rnnt.max_symbols_per_step"], 10)
        self.assertFalse(
            any("duration" in name for name in writer.values))

    def test_long_form_hint_frames(self):
        # 60 s at the 80 ms stride is 750 frames; a 25 % overlap is 94 frames each side.
        self.assertEqual(CONVERTER.long_form_hint_frames(160, 16000, 8, 60.0, 0.25), (750, 94))
        self.assertEqual(CONVERTER.long_form_hint_frames(160, 16000, 8, 60.0, None), (750, 0))

    def test_long_form_hint_keys_written_only_when_requested(self):
        writer = RecordingWriter()
        CONVERTER.write_long_form_hint(writer, 160, 16000, unified_config(), None, None)
        self.assertNotIn("parakeet.long_form.window_frames", writer.values)
        CONVERTER.write_long_form_hint(writer, 160, 16000, unified_config(), 60.0, 0.25)
        self.assertEqual(writer.values["parakeet.long_form.window_frames"], 750)
        self.assertEqual(writer.values["parakeet.long_form.context_frames"], 94)

    def test_tdt_durations_from_model_defaults(self):
        config = unified_config()
        config["model_defaults"]["tdt_durations"] = [0, 1, 2, 3, 4]
        config["model_defaults"]["num_tdt_durations"] = 5
        self.assertEqual(CONVERTER.tdt_durations(config), [0, 1, 2, 3, 4])

    def test_tdt_durations_reject_count_mismatch(self):
        config = unified_config()
        config["model_defaults"]["tdt_durations"] = [0, 1, 2]
        config["model_defaults"]["num_tdt_durations"] = 5
        with self.assertRaises(ValueError):
            CONVERTER.tdt_durations(config)

    def test_tdt_durations_from_loss_kwargs(self):
        config = unified_config()
        config["loss"] = {"loss_name": "tdt", "tdt_kwargs": {"durations": [0, 1, 2, 3]}}
        self.assertEqual(CONVERTER.tdt_durations(config), [0, 1, 2, 3])

    def test_tdt_durations_from_decoding(self):
        config = unified_config()
        config["decoding"]["durations"] = [0, 1, 2]
        self.assertEqual(CONVERTER.tdt_durations(config), [0, 1, 2])

    def test_tdt_durations_from_joint_extra_outputs(self):
        config = unified_config()
        config["joint"]["num_extra_outputs"] = 5
        self.assertEqual(CONVERTER.tdt_durations(config), [0, 1, 2, 3, 4])

    def test_tdt_durations_missing_everywhere(self):
        with self.assertRaises(KeyError):
            CONVERTER.tdt_durations(unified_config())

    def test_writes_tdt_metadata_from_loss_kwargs(self):
        writer = RecordingWriter()
        config = unified_config()
        config["loss"] = {"loss_name": "tdt", "tdt_kwargs": {"durations": [0, 1, 2, 3, 4]}}
        CONVERTER.write_transducer_metadata(writer, config, "tdt")

        self.assertEqual(writer.values["parakeet.tdt.num_durations"], 5)
        self.assertEqual(writer.values["parakeet.tdt.durations"], [0, 1, 2, 3, 4])

    def test_default_description_names_the_family(self):
        writer = RecordingWriter()
        CONVERTER.write_provenance(writer, "tdt")
        self.assertEqual(
            writer.values["general.description"],
            "NVIDIA Parakeet-TDT FastConformer ASR (CC-BY-4.0)")

    def test_description_override_replaces_the_family_text(self):
        writer = RecordingWriter()
        CONVERTER.write_provenance(
            writer, "tdt",
            "TranscribePsy-250M-EU-sub, FastConformer-TDT speech recognition")
        self.assertEqual(
            writer.values["general.description"],
            "TranscribePsy-250M-EU-sub, FastConformer-TDT speech recognition")

    def test_description_override_keeps_license_keys(self):
        writer = RecordingWriter()
        CONVERTER.write_provenance(writer, "nemotron", "custom")
        self.assertEqual(writer.values["general.description"], "custom")
        self.assertEqual(writer.values["general.license"], "OpenMDW-1.1")

    def test_writes_rnnt_tensor_namespace(self):
        names = []

        def record(name, value):
            names.append(name)

        CONVERTER.write_transducer_tensors(
            unified_config(),
            transducer_state_dict(),
            "rnnt",
            record,
            record,
        )

        self.assertIn("rnnt.predict.embed.weight", names)
        self.assertIn("rnnt.predict.lstm.1.w_hh", names)
        self.assertIn("rnnt.joint.out.weight", names)
        self.assertFalse(any(name.startswith("tdt.") for name in names))


def unified_streaming_config():
    config = unified_config()
    config["encoder"] = {
        "att_context_style": "chunked_limited_with_rc",
        "att_context_size": [-1, -1],
        "att_chunk_context_size": [[70], [13, 1, 2, 7], [0, 1, 2, 3, 4, 7, 13]],
        "conv_kernel_size": 9,
        "conv_context_style": "dcc",
        "causal_downsampling": False,
    }
    return config


class ConverterUnifiedStreamingTests(unittest.TestCase):
    def test_writes_unified_streaming_contexts(self):
        writer = RecordingWriter()

        written = CONVERTER.write_unified_streaming_metadata(
            writer, unified_streaming_config())

        self.assertTrue(written)
        self.assertEqual(
            writer.values["parakeet.unified.left_context_frames"], 70)
        self.assertEqual(
            writer.values["parakeet.unified.cache_time_steps"], 4)
        self.assertEqual(
            writer.values["parakeet.unified.allowed_chunk_frames"],
            [1, 2, 7, 13])
        self.assertEqual(
            writer.values["parakeet.unified.allowed_right_context_frames"],
            [0, 1, 2, 3, 4, 7, 13])

    def test_regular_rnnt_writes_no_streaming_contexts(self):
        writer = RecordingWriter()
        config = unified_config()
        config["encoder"] = {"att_context_style": "regular", "conv_kernel_size": 9}

        written = CONVERTER.write_unified_streaming_metadata(writer, config)

        self.assertFalse(written)
        self.assertFalse(
            any(name.startswith("parakeet.unified.") for name in writer.values))

    def test_rejects_multiple_left_contexts(self):
        config = unified_streaming_config()
        config["encoder"]["att_chunk_context_size"] = [[70, 35], [7], [7]]

        with self.assertRaises(ValueError):
            CONVERTER.unified_streaming_contexts(config["encoder"])


class ConverterNemotronTests(unittest.TestCase):
    def test_detects_prompt_conditioned_nemotron_narrowly(self):
        self.assertEqual(
            CONVERTER.detect_model_type(nemotron_config()),
            "nemotron",
        )

        config = nemotron_config()
        config["target"] = (
            "example.EncDecRNNTBPEModelWithPrompt"
        )
        self.assertEqual(CONVERTER.detect_model_type(config), "rnnt")

    def test_validates_checkpoint_derived_contract(self):
        CONVERTER.validate_nemotron_contract(
            nemotron_config(),
            nemotron_state_dict(),
        )

    def test_rejects_incompatible_prompt_projection_shape(self):
        state_dict = nemotron_state_dict()
        state_dict["prompt_kernel.0.weight"] = FakeTensor((2048, 1024))

        with self.assertRaisesRegex(
            ValueError,
            r"prompt_kernel\.0\.weight shape=\(2048, 1024\)",
        ):
            CONVERTER.validate_nemotron_contract(
                nemotron_config(),
                state_dict,
            )

    def test_writes_locale_and_streaming_metadata_in_source_order(self):
        writer = RecordingWriter()

        CONVERTER.write_nemotron_metadata(
            writer,
            nemotron_config(),
        )

        self.assertEqual(
            writer.values[
                "parakeet.nemotron.allowed_right_context_frames"
            ],
            [0, 1, 3, 6, 13],
        )
        self.assertEqual(
            writer.values["parakeet.nemotron.allowed_chunk_ms"],
            [80, 160, 320, 560, 1120],
        )
        self.assertEqual(
            writer.values["parakeet.nemotron.locale_aliases"],
            ["en-US", "en", "auto"],
        )
        self.assertEqual(
            writer.values["parakeet.nemotron.locale_prompt_ids"],
            [0, 0, 101],
        )
        self.assertEqual(
            writer.values["parakeet.nemotron.max_symbols_per_step"],
            10,
        )

    def test_writes_rnnt_and_prompt_tensors_under_nemotron_namespace(self):
        names = []

        def record(name, value):
            names.append(name)

        config = nemotron_config()
        state_dict = transducer_state_dict()
        state_dict.update(
            {
                "prompt_kernel.0.weight": "prompt_0_w",
                "prompt_kernel.0.bias": "prompt_0_b",
                "prompt_kernel.2.weight": "prompt_2_w",
                "prompt_kernel.2.bias": "prompt_2_b",
            },
        )

        CONVERTER.write_transducer_tensors(
            config,
            state_dict,
            "nemotron",
            record,
            record,
        )
        CONVERTER.write_nemotron_prompt_tensors(
            state_dict,
            record,
            record,
        )

        self.assertIn("nemotron.predict.lstm.1.w_hh", names)
        self.assertIn("nemotron.joint.out.weight", names)
        self.assertIn("nemotron.prompt.proj.0.weight", names)
        self.assertIn("nemotron.prompt.proj.2.bias", names)

    def test_resolves_default_320ms_context_pair(self):
        self.assertEqual(
            CONVERTER.resolve_attention_context(
                nemotron_config()["encoder"],
                default_right=CONVERTER.NEMOTRON_DEFAULT_ATT_CONTEXT_RIGHT,
            ),
            (56, 3),
        )

    def test_pins_default_right_context_when_another_pair_is_listed_first(self):
        encoder = nemotron_config()["encoder"]
        encoder["att_context_size"] = [
            [56, 0],
            [56, 3],
            [56, 6],
            [56, 13],
        ]
        self.assertEqual(
            CONVERTER.resolve_attention_context(
                encoder,
                default_right=CONVERTER.NEMOTRON_DEFAULT_ATT_CONTEXT_RIGHT,
            ),
            (56, 3),
        )

    def test_supports_float_and_block_quantized_output_routes(self):
        for quant in ("f16", "bf16", "q8_0", "q4_0"):
            with self.subTest(quant=quant):
                self.assertIn(quant, CONVERTER.FILE_TYPE_MAP)
        self.assertIn("q8_0", CONVERTER.QUANT_MAP)
        self.assertIn("q4_0", CONVERTER.QUANT_MAP)
        self.assertEqual(
            CONVERTER.QUANT_MAP["bf16"],
            CONVERTER.gguf.GGMLQuantizationType.BF16,
        )


if __name__ == "__main__":
    unittest.main()
