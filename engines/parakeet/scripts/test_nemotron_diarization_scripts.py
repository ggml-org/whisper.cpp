"""Tests for Nemotron 3 Diarization model preparation scripts."""

import tempfile
import subprocess
import unittest
from pathlib import Path

import convert_nemotron_diarization as converter
import download_nemotron_diarization as downloader


class NemotronPreparationTests(unittest.TestCase):
    def test_downloads_pinned_gguf_and_source(self):
        calls = []

        def fake_download(**arguments):
            calls.append(arguments)
            return str(Path(arguments["local_dir"]) / arguments["filename"])

        with tempfile.TemporaryDirectory() as temporary:
            model_directory = Path(temporary) / "models"
            artifacts = downloader.download_model(
                model_directory, include_source=True, download=fake_download)

        self.assertEqual(
            [artifact.name for artifact in artifacts],
            [downloader.GGUF_FILENAME, downloader.NEMO_FILENAME],
        )
        self.assertEqual(calls[0]["repo_id"], downloader.MODEL_REPOSITORY)
        self.assertEqual(calls[0]["revision"], downloader.MODEL_REVISION)
        self.assertEqual(calls[1]["revision"], downloader.MODEL_REVISION)

    def test_conversion_preserves_official_gguf(self):
        calls = []

        def fake_run(command, **arguments):
            calls.append((command, arguments))

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = directory / downloader.NEMO_FILENAME
            output = directory / "converted" / converter.OUTPUT_FILENAME
            script = directory / "converter" / "convert_model.py"
            converter.convert_model(source, output, script, run=fake_run)
            resolved_source = source.resolve()
            resolved_output = output.resolve()

        command, arguments = calls[0]
        self.assertEqual(command[1], str(script))
        self.assertEqual(Path(command[2]), resolved_source)
        self.assertEqual(command[3:], ["--outfile", str(resolved_output), "--outtype", "q8_0"])
        self.assertEqual(arguments["cwd"], script.parent)
        self.assertTrue(arguments["check"])
        self.assertNotEqual(output.name, downloader.GGUF_FILENAME)

    def test_converts_each_supported_output_type(self):
        for outtype in converter.OUTPUT_TYPES:
            with self.subTest(outtype=outtype), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                calls = []
                output = directory / converter.output_filename(outtype)
                converter.convert_model(
                    directory / "source.nemo", output, directory / "convert_model.py",
                    run=lambda command, **kwargs: calls.append(command), outtype=outtype)
                self.assertEqual(calls[0][-2:], ["--outtype", outtype])
                self.assertIn(f".converted.{outtype}.", output.name)
                self.assertEqual(converter.parse_args(["--outtype", outtype]).outtype, outtype)

    def test_default_output_remains_q8(self):
        self.assertEqual(converter.parse_args([]).outtype, converter.OUTPUT_TYPE)
        self.assertEqual(converter.output_filename(converter.OUTPUT_TYPE), converter.OUTPUT_FILENAME)
        self.assertEqual(len({converter.output_filename(value) for value in converter.OUTPUT_TYPES}),
                         len(converter.OUTPUT_TYPES))

    def test_rejects_unsupported_type_before_running_converter(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with self.assertRaises(ValueError):
                converter.convert_model(
                    directory / "source.nemo", directory / "new" / "output.gguf",
                    directory / "convert_model.py", outtype="q2_k")
            self.assertFalse((directory / "new").exists())

    def test_uses_existing_converter_checkout(self):
        calls = []

        def fake_run(command, **arguments):
            calls.append(command)
            output = converter.CONVERTER_REVISION + "\n" if "rev-parse" in command else ""
            return subprocess.CompletedProcess(command, 0, output)

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / ".git").mkdir()
            script = directory / "convert_model.py"
            script.touch()
            self.assertEqual(converter.ensure_converter(directory, run=fake_run), script)
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0][-2:], ["rev-parse", "HEAD"])

    def test_accepts_pinned_converter_worktree(self):
        def fake_run(command, **arguments):
            output = converter.CONVERTER_REVISION + "\n" if "rev-parse" in command else ""
            return subprocess.CompletedProcess(command, 0, output)

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / ".git").write_text("gitdir: linked-worktree")
            script = directory / "convert_model.py"
            script.touch()
            self.assertEqual(converter.ensure_converter(directory, run=fake_run), script)

    def test_rejects_modified_converter_script(self):
        def fake_run(command, **arguments):
            if "rev-parse" in command:
                output = converter.CONVERTER_REVISION + "\n"
            elif "status" in command:
                output = " M convert_model.py\n"
            else:
                output = ""
            return subprocess.CompletedProcess(command, 0, output)

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / ".git").mkdir()
            (directory / "convert_model.py").touch()
            with self.assertRaises(RuntimeError):
                converter.ensure_converter(directory, run=fake_run)

    def test_updates_stale_converter_checkout(self):
        calls = []
        revision = "old-revision"

        def fake_run(command, **arguments):
            nonlocal revision
            calls.append(command)
            if "checkout" in command:
                revision = converter.CONVERTER_REVISION
            if "rev-parse" in command:
                return subprocess.CompletedProcess(command, 0, revision + "\n")
            return subprocess.CompletedProcess(command, 0, "")

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / ".git").mkdir()
            script = directory / "convert_model.py"
            script.touch()
            self.assertEqual(converter.ensure_converter(directory, run=fake_run), script)
        self.assertEqual([command[-2] for command in calls if "rev-parse" in command],
                         ["rev-parse", "rev-parse"])
        self.assertTrue(any("fetch" in command for command in calls))
        self.assertTrue(any("checkout" in command for command in calls))

    def test_rejects_non_git_converter_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "convert_model.py").touch()
            with self.assertRaises(RuntimeError):
                converter.ensure_converter(directory)

    def test_fetches_pinned_converter_source(self):
        calls = []

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "NeMo-Speech.cpp"
            script = directory / "convert_model.py"

            def fake_run(command, **arguments):
                calls.append(command)
                if "checkout" in command:
                    directory.mkdir(parents=True)
                    script.touch()
                if "rev-parse" in command:
                    return subprocess.CompletedProcess(
                        command, 0, converter.CONVERTER_REVISION + "\n")
                return subprocess.CompletedProcess(command, 0, "")

            self.assertEqual(converter.ensure_converter(directory, run=fake_run), script)

        self.assertEqual(calls[0][1], "clone")
        self.assertEqual(calls[1][-1], converter.CONVERTER_REVISION)
        self.assertEqual(calls[2][-1], converter.CONVERTER_REVISION)
        self.assertEqual(calls[3][-2:], ["rev-parse", "HEAD"])


if __name__ == "__main__":
    unittest.main()
