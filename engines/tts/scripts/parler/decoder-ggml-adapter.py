"""ggml CPU/Metal baseline; see decoder-native-adapter.py for JSON config."""
import importlib.util
from pathlib import Path


def create_runner(manifest, config=None):
    spec = importlib.util.spec_from_file_location("parler_ggml_native_adapter", Path(__file__).with_name("decoder-native-adapter.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.NativeRunner(manifest, config, coreml=False)
