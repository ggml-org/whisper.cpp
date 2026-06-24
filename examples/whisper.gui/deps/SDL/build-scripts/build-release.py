#!/usr/bin/env python3

"""
This script is shared between SDL2, SDL3, and all satellite libraries.
Don't specialize this script for doing project-specific modifications.
Rather, modify release-info.json.
"""

import argparse
import collections
import dataclasses
from collections.abc import Callable
import contextlib
import datetime
import fnmatch
import glob
import io
import json
import logging
import multiprocessing
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import typing
import zipfile


logger = logging.getLogger(__name__)
GIT_HASH_FILENAME = ".git-hash"
REVISION_TXT = "REVISION.txt"


def safe_isotime_to_datetime(str_isotime: str) -> datetime.datetime:
    try:
        return datetime.datetime.fromisoformat(str_isotime)
    except ValueError:
        pass
    logger.warning("Invalid iso time: %s", str_isotime)
    if str_isotime[-6:-5] in ("+", "-"):
        # Commits can have isotime with invalid timezone offset (e.g. "2021-07-04T20:01:40+32:00")
        modified_str_isotime = str_isotime[:-6] + "+00:00"
        try:
            return datetime.datetime.fromisoformat(modified_str_isotime)
        except ValueError:
            pass
    raise ValueError(f"Invalid isotime: {str_isotime}")


def arc_join(*parts: list[str]) -> str:
    assert all(p[:1] != "/" and p[-1:] != "/" for p in parts), f"None of {parts} may start or end with '/'"
    return "/".join(p for p in parts if p)


@dataclasses.dataclass(frozen=True)
class VsArchPlatformConfig:
    arch: str
    configuration: str
    platform: str

    def extra_context(self):
        return {
            "ARCH": self.arch,
            "CONFIGURATION": self.configuration,
            "PLATFORM": self.platform,
        }


@contextlib.contextmanager
def chdir(path):
    original_cwd = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(original_cwd)


class Executer:
    def __init__(self, root: Path, dry: bool=False):
        self.root = root
        self.dry = dry

    def run(self, cmd, cwd=None, env=None):
        logger.info("Executing args=%r", cmd)
        sys.stdout.flush()
        if not self.dry:
            subprocess.check_call(cmd, cwd=cwd or self.root, env=env, text=True)

    def check_output(self, cmd, cwd=None, dry_out=None, env=None, text=True):
        logger.info("Executing args=%r", cmd)
        sys.stdout.flush()
        if self.dry:
            return dry_out
        return subprocess.check_output(cmd, cwd=cwd or self.root, env=env, text=text)


class SectionPrinter:
    @contextlib.contextmanager
    def group(self, title: str):
        print(f"{title}:")
        yield


class GitHubSectionPrinter(SectionPrinter):
    def __init__(self):
        super().__init__()
        self.in_group = False

    @contextlib.contextmanager
    def group(self, title: str):
        print(f"::group::{title}")
        assert not self.in_group, "Can enter a group only once"
        self.in_group = True
        yield
        self.in_group = False
        print("::endgroup::")


class VisualStudio:
    def __init__(self, executer: Executer, year: typing.Optional[str]=None):
        self.executer = executer
        self.vsdevcmd = self.find_vsdevcmd(year)
        self.msbuild = self.find_msbuild()

    @property
    def dry(self) -> bool:
        return self.executer.dry

    VS_YEAR_TO_VERSION = {
        "2022": 17,
        "2019": 16,
        "2017": 15,
        "2015": 14,
        "2013": 12,
    }

    def find_vsdevcmd(self, year: typing.Optional[str]=None) -> typing.Optional[Path]:
        vswhere_spec = ["-latest"]
        if year is not None:
            try:
                version = self.VS_YEAR_TO_VERSION[year]
            except KeyError:
                logger.error("Invalid Visual Studio year")
                return None
            vswhere_spec.extend(["-version", f"[{version},{version+1})"])
        vswhere_cmd = ["vswhere"] + vswhere_spec + ["-property", "installationPath"]
        vs_install_path = Path(self.executer.check_output(vswhere_cmd, dry_out="/tmp").strip())
        logger.info("VS install_path = %s", vs_install_path)
        assert vs_install_path.is_dir(), "VS installation path does not exist"
        vsdevcmd_path = vs_install_path / "Common7/Tools/vsdevcmd.bat"
        logger.info("vsdevcmd path = %s", vsdevcmd_path)
        if self.dry:
            vsdevcmd_path.parent.mkdir(parents=True, exist_ok=True)
            vsdevcmd_path.touch(exist_ok=True)
        assert vsdevcmd_path.is_file(), "vsdevcmd.bat batch file does not exist"
        return vsdevcmd_path

    def find_msbuild(self) -> typing.Optional[Path]:
        vswhere_cmd = ["vswhere", "-latest", "-requires", "Microsoft.Component.MSBuild", "-find", r"MSBuild\**\Bin\MSBuild.exe"]
        msbuild_path = Path(self.executer.check_output(vswhere_cmd, dry_out="/tmp/MSBuild.exe").strip())
        logger.info("MSBuild path = %s", msbuild_path)
        if self.dry:
            msbuild_path.parent.mkdir(parents=True, exist_ok=True)
            msbuild_path.touch(exist_ok=True)
        assert msbuild_path.is_file(), "MSBuild.exe does not exist"
        return msbuild_path

    def build(self, arch_platform: VsArchPlatformConfig, projects: list[Path]):
        assert projects, "Need at least one project to build"

        vsdev_cmd_str = f"\"{self.vsdevcmd}\" -arch={arch_platform.arch}"
        msbuild_cmd_str = " && ".join([f"\"{self.msbuild}\" \"{project}\" /m /p:BuildInParallel=true /p:Platform={arch_platform.platform} /p:Configuration={arch_platform.configuration}" for project in projects])
        bat_contents = f"{vsdev_cmd_str} && {msbuild_cmd_str}\n"
        bat_path = Path(tempfile.gettempdir()) / "cmd.bat"
        with bat_path.open("w") as f:
            f.write(bat_contents)

        logger.info("Running cmd.exe script (%s): %s", bat_path, bat_contents)
        cmd = ["cmd.exe", "/D", "/E:ON", "/V:OFF", "/S", "/C", f"CALL {str(bat_path)}"]
        self.executer.run(cmd)


class Archiver:
    def __init__(self, zip_path: typing.Optional[Path]=None, tgz_path: typing.Optional[Path]=None, txz_path: typing.Optional[Path]=None):
        self._zip_files = []
        self._tar_files = []
        self._added_files = set()
        if zip_path:
            self._zip_files.append(zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED))
        if tgz_path:
            self._tar_files.append(tarfile.open(tgz_path, "w:gz"))
        if txz_path:
            self._tar_files.append(tarfile.open(txz_path, "w:xz"))

    @property
    def added_files(self) -> set[str]:
        return self._added_files

    def add_file_data(self, arcpath: str, data: bytes, mode: int, time: datetime.datetime):
        for zf in self._zip_files:
            file_data_time = (time.year, time.month, time.day, time.hour, time.minute, time.second)
            zip_info = zipfile.ZipInfo(filename=arcpath, date_time=file_data_time)
            zip_info.external_attr = mode << 16
            zip_info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(zip_info, data=data)
        for tf in self._tar_files:
            tar_info = tarfile.TarInfo(arcpath)
            tar_info.type = tarfile.REGTYPE
            tar_info.mode = mode
            tar_info.size = len(data)
            tar_info.mtime = int(time.timestamp())
            tf.addfile(tar_info, fileobj=io.BytesIO(data))

        self._added_files.add(arcpath)

    def add_symlink(self, arcpath: str, target: str, time: datetime.datetime, files_for_zip):
        logger.debug("Adding symlink (target=%r) -> %s", target, arcpath)
        for zf in self._zip_files:
            file_data_time = (time.year, time.month, time.day, time.hour, time.minute, time.second)
            for f in files_for_zip:
                zip_info = zipfile.ZipInfo(filename=f["arcpath"], date_time=file_data_time)
                zip_info.external_attr = f["mode"] << 16
                zip_info.compress_type = zipfile.ZIP_DEFLATED
                zf.writestr(zip_info, data=f["data"])
        for tf in self._tar_files:
            tar_info = tarfile.TarInfo(arcpath)
            tar_info.type = tarfile.SYMTYPE
            tar_info.mode = 0o777
            tar_info.mtime = int(time.timestamp())
            tar_info.linkname = target
            tf.addfile(tar_info)

        self._added_files.update(f["arcpath"] for f in files_for_zip)

    def add_git_hash(self, arcdir: str, commit: str, time: datetime.datetime):
        arcpath = arc_join(arcdir, GIT_HASH_FILENAME)
        data = f"{commit}\n".encode()
        self.add_file_data(arcpath=arcpath, data=data, mode=0o100644, time=time)

    def add_file_path(self, arcpath: str, path: Path):
        assert path.is_file(), f"{path} should be a file"
        logger.debug("Adding %s -> %s", path, arcpath)
        for zf in self._zip_files:
            zf.write(path, arcname=arcpath)
        for tf in self._tar_files:
            tf.add(path, arcname=arcpath)

    def add_file_directory(self, arcdirpath: str, dirpath: Path):
        assert dirpath.is_dir()
        if arcdirpath and arcdirpath[-1:] != "/":
            arcdirpath += "/"
        for f in dirpath.iterdir():
            if f.is_file():
                arcpath = f"{arcdirpath}{f.name}"
                logger.debug("Adding %s to %s", f, arcpath)
                self.add_file_path(arcpath=arcpath, path=f)

    def close(self):
        # Archiver is intentionally made invalid after this function
        del self._zip_files
        self._zip_files = None
        del self._tar_files
        self._tar_files = None

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()


class NodeInArchive:
    def __init__(self, arcpath: str, path: typing.Optional[Path]=None, data: typing.Optional[bytes]=None, mode: typing.Optional[int]=None, symtarget: typing.Optional[str]=None, time: typing.Optional[datetime.datetime]=None, directory: bool=False):
        self.arcpath = arcpath
        self.path = path
        self.data = data
        self.mode = mode
        self.symtarget = symtarget
        self.time = time
        self.directory = directory

    @classmethod
    def from_fs(cls, arcpath: str, path: Path, mode: int=0o100644, time: typing.Optional[datetime.datetime]=None) -> "NodeInArchive":
        if time is None:
            time = datetime.datetime.fromtimestamp(os.stat(path).st_mtime)
        return cls(arcpath=arcpath, path=path, mode=mode)

    @classmethod
    def from_data(cls, arcpath: str, data: bytes, time: datetime.datetime) -> "NodeInArchive":
        return cls(arcpath=arcpath, data=data, time=time, mode=0o100644)

    @classmethod
    def from_text(cls, arcpath: str, text: str, time: datetime.datetime) -> "NodeInArchive":
        return cls.from_data(arcpath=arcpath, data=text.encode(), time=time)

    @classmethod
    def from_symlink(cls, arcpath: str, symtarget: str) -> "NodeInArchive":
        return cls(arcpath=arcpath, symtarget=symtarget)

    @classmethod
    def from_directory(cls, arcpath: str) -> "NodeInArchive":
        return cls(arcpath=arcpath, directory=True)

    def __repr__(self) -> str:
        return f"<{type(self).__name__}:arcpath={self.arcpath},path='{str(self.path)}',len(data)={len(self.data) if self.data else 'n/a'},directory={self.directory},symtarget={self.symtarget}>"


def configure_file(path: Path, context: dict[str, str]) -> bytes:
    text = path.read_text()
    return configure_text(text, context=context).encode()


def configure_text(text: str, context: dict[str, str]) -> str:
    original_text = text
    for txt, repl in context.items():
        text = text.replace(f"@<@{txt}@>@", repl)
    success = all(thing not in text for thing in ("@<@", "@>@"))
    if not success:
        raise ValueError(f"Failed to configure {repr(original_text)}")
    return text


def configure_text_list(text_list: list[str], context: dict[str, str]) -> list[str]:
    return [configure_text(text=e, context=context) for e in text_list]


class ArchiveFileTree:
    def __init__(self):
        self._tree: dict[str, NodeInArchive] = {}

    def add_file(self, file: NodeInArchive):
        self._tree[file.arcpath] = file

    def get_latest_mod_time(self) -> datetime.datetime:
        return max(item.time for item in self._tree.values() if item.time)

    def add_to_archiver(self, archive_base: str, archiver: Archiver):
        remaining_symlinks = set()
        added_files = dict()

        def calculate_symlink_target(s: NodeInArchive) -> str:
            dest_dir = os.path.dirname(s.arcpath)
            if dest_dir:
                dest_dir += "/"
            target = dest_dir + s.symtarget
            while True:
                new_target, n = re.subn(r"([^/]+/+[.]{2}/)", "", target)
                target = new_target
                if not n:
                    break
            return target

        # Add files in first pass
        for arcpath, node in self._tree.items():
            assert node is not None, f"{arcpath} -> node"
            if node.data is not None:
                archiver.add_file_data(arcpath=arc_join(archive_base, arcpath), data=node.data, time=node.time, mode=node.mode)
                assert node.arcpath is not None, f"{node=}"
                added_files[node.arcpath] = node
            elif node.path is not None:
                archiver.add_file_path(arcpath=arc_join(archive_base, arcpath), path=node.path)
                assert node.arcpath is not None, f"{node=}"
                added_files[node.arcpath] = node
            elif node.symtarget is not None:
                remaining_symlinks.add(node)
            elif node.directory:
                pass
            else:
                raise ValueError(f"Invalid Archive Node: {repr(node)}")

        assert None not in added_files

        # Resolve symlinks in second pass: zipfile does not support symlinks, so add files to zip archive
        while True:
            if not remaining_symlinks:
                break
            symlinks_this_time = set()
            extra_added_files = {}
            for symlink in remaining_symlinks:
                symlink_files_for_zip = {}
                symlink_target_path = calculate_symlink_target(symlink)
                if symlink_target_path in added_files:
                    symlink_files_for_zip[symlink.arcpath] = added_files[symlink_target_path]
                else:
                    symlink_target_path_slash = symlink_target_path + "/"
                    for added_file in added_files:
                        if added_file.startswith(symlink_target_path_slash):
                            path_in_symlink = symlink.arcpath + "/" + added_file.removeprefix(symlink_target_path_slash)
                            symlink_files_for_zip[path_in_symlink] = added_files[added_file]
                if symlink_files_for_zip:
                    symlinks_this_time.add(symlink)
                    extra_added_files.update(symlink_files_for_zip)
                    files_for_zip = [{"arcpath": f"{archive_base}/{sym_path}", "data": sym_info.data, "mode": sym_info.mode} for sym_path, sym_info in symlink_files_for_zip.items()]
                    archiver.add_symlink(arcpath=f"{archive_base}/{symlink.arcpath}", target=symlink.symtarget, time=symlink.time, files_for_zip=files_for_zip)
            # if not symlinks_this_time:
            #     logger.info("files added: %r", set(path for path in added_files.keys()))
            assert symlinks_this_time, f"No targets found for symlinks: {remaining_symlinks}"
            remaining_symlinks.difference_update(symlinks_this_time)
            added_files.update(extra_added_files)

    def add_directory_tree(self, arc_dir: str, path: Path, time: datetime.datetime):
        assert path.is_dir()
        for files_dir, _, filenames in os.walk(path):
            files_dir_path = Path(files_dir)
            rel_files_path = files_dir_path.relative_to(path)
            for filename in filenames:
                self.add_file(NodeInArchive.from_fs(arcpath=arc_join(arc_dir, str(rel_files_path), filename), path=files_dir_path / filename, time=time))

    def _add_files_recursively(self, arc_dir: str, paths: list[Path], time: datetime.datetime):
        logger.debug(f"_add_files_recursively({arc_dir=} {paths=})")
        for path in paths:
            arcpath = arc_join(arc_dir, path.name)
            if path.is_file():
                logger.debug("Adding %s as %s", path, arcpath)
                self.add_file(NodeInArchive.from_fs(arcpath=arcpath, path=path, time=time))
            elif path.is_dir():
                self._add_files_recursively(arc_dir=arc_join(arc_dir, path.name), paths=list(path.iterdir()), time=time)
            else:
                raise ValueError(f"Unsupported file type to add recursively: {path}")

    def add_file_mapping(self, arc_dir: str, file_mapping: dict[str, list[str]], file_mapping_root: Path, context: dict[str, str], time: datetime.datetime):
        for meta_rel_destdir, meta_file_globs in file_mapping.items():
            rel_destdir = configure_text(meta_rel_destdir, context=context)
            assert "@" not in rel_destdir, f"archive destination should not contain an @ after configuration ({repr(meta_rel_destdir)}->{repr(rel_destdir)})"
            for meta_file_glob in meta_file_globs:
                file_glob = configure_text(meta_file_glob, context=context)
                assert "@" not in rel_destdir, f"archive glob should not contain an @ after configuration ({repr(meta_file_glob)}->{repr(file_glob)})"
                if ":" in file_glob:
                    original_path, new_filename = file_glob.rsplit(":", 1)
                    assert ":" not in original_path, f"Too many ':' in {repr(file_glob)}"
                    assert "/" not in new_filename, f"New filename cannot contain a '/' in {repr(file_glob)}"
                    path = file_mapping_root / original_path
                    arcpath = arc_join(arc_dir, rel_destdir, new_filename)
                    if path.suffix == ".in":
                        data = configure_file(path, context=context)
                        logger.debug("Adding processed %s -> %s", path, arcpath)
                        self.add_file(NodeInArchive.from_data(arcpath=arcpath, data=data, time=time))
                    else:
                        logger.debug("Adding %s -> %s", path, arcpath)
                        self.add_file(NodeInArchive.from_fs(arcpath=arcpath, path=path, time=time))
                else:
                    relative_file_paths = glob.glob(file_glob, root_dir=file_mapping_root)
                    assert relative_file_paths, f"Glob '{file_glob}' does not match any file"
                    self._add_files_recursively(arc_dir=arc_join(arc_dir, rel_destdir), paths=[file_mapping_root / p for p in relative_file_paths], time=time)


class SourceCollector:
    # TreeItem = collections.namedtuple("TreeItem", ("path", "mode", "data", "symtarget", "directory", "time"))
    def __init__(self, root: Path, commit: str, filter: typing.Optional[Callable[[str], bool]], executer: Executer):
        self.root = root
        self.commit = commit
        self.filter = filter
        self.executer = executer

    def get_archive_file_tree(self) -> ArchiveFileTree:
        git_archive_args = ["git", "archive", "--format=tar.gz", self.commit, "-o", "/dev/stdout"]
        logger.info("Executing args=%r", git_archive_args)
        contents_tgz = subprocess.check_output(git_archive_args, cwd=self.root, text=False)
        tar_archive = tarfile.open(fileobj=io.BytesIO(contents_tgz), mode="r:gz")
        filenames = tuple(m.name for m in tar_archive if (m.isfile() or m.issym()))

        file_times = self._get_file_times(paths=filenames)
        git_contents = ArchiveFileTree()
        for ti in tar_archive:
            if self.filter and not self.filter(ti.name):
                continue
            data = None
            symtarget = None
            directory = False
            file_time = None
            if ti.isfile():
                contents_file = tar_archive.extractfile(ti.name)
                data = contents_file.read()
                file_time = file_times[ti.name]
            elif ti.issym():
                symtarget = ti.linkname
                file_time = file_times[ti.name]
            elif ti.isdir():
                directory = True
            else:
                raise ValueError(f"{ti.name}: unknown type")
            node = NodeInArchive(arcpath=ti.name, data=data, mode=ti.mode, symtarget=symtarget, time=file_time, directory=directory)
            git_contents.add_file(node)
        return git_contents

    def _get_file_times(self, paths: tuple[str, ...]) -> dict[str, datetime.datetime]:
        dry_out = textwrap.dedent("""\
            time=2024-03-14T15:40:25-07:00

            M\tCMakeLists.txt
        """)
        git_log_out = self.executer.check_output(["git", "log", "--name-status", '--pretty=time=%cI', self.commit], dry_out=dry_out, cwd=self.root).splitlines(keepends=False)
        current_time = None
        set_paths = set(paths)
        path_times: dict[str, datetime.datetime] = {}
        for line in git_log_out:
            if not line:
                continue
            if line.startswith("time="):
                current_time = safe_isotime_to_datetime(line.removeprefix("time="))
                continue
            mod_type, file_paths = line.split(maxsplit=1)
            assert current_time is not None
            for file_path in file_paths.split("\t"):
                if file_path in set_paths and file_path not in path_times:
                    path_times[file_path] = current_time

        # FIXME: find out why some files are not shown in "git log"
        # assert set(path_times.keys()) == set_paths
        if set(path_times.keys()) != set_paths:
            found_times = set(path_times.keys())
            paths_without_times = set_paths.difference(found_times)
            logger.warning("No times found for these paths: %s", paths_without_times)
            max_time = max(time for time in path_times.values())
            for path in paths_without_times:
                path_times[path] = max_time

        return path_times


class Releaser:
    def __init__(self, release_info: dict, commit: str, revision: str, root: Path, dist_path: Path, section_printer: SectionPrinter, executer: Executer, cmake_generator: str, deps_path: Path, overwrite: bool, github: bool, fast: bool):
        self.release_info = release_info
        self.project = release_info["name"]
        self.version = self.extract_sdl_version(root=root, release_info=release_info)
        self.root = root
        self.commit = commit
        self.revision = revision
        self.dist_path = dist_path
        self.section_printer = section_printer
        self.executer = executer
        self.cmake_generator = cmake_generator
        self.cpu_count = multiprocessing.cpu_count()
        self.deps_path = deps_path
        self.overwrite = overwrite
        self.github = github
        self.fast = fast
        self.arc_time = datetime.datetime.now()

        self.artifacts: dict[str, Path] = {}

    def get_context(self, extra_context: typing.Optional[dict[str, str]]=None) -> dict[str, str]:
        ctx = {
            "PROJECT_NAME": self.project,
            "PROJECT_VERSION": self.version,
            "PROJECT_COMMIT": self.commit,
            "PROJECT_REVISION": self.revision,
            "PROJECT_ROOT": str(self.root),
        }
        if extra_context:
            ctx.update(extra_context)
        return ctx

    @property
    def dry(self) -> bool:
        return self.executer.dry

    def prepare(self):
        logger.debug("Creating dist folder")
        self.dist_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def _path_filter(cls, path: str) -> bool:
        if ".gitmodules" in path:
            return True
        if path.startswith(".git"):
            return False
        return True

    @classmethod
    def _external_repo_path_filter(cls, path: str) -> bool:
        if not cls._path_filter(path):
            return False
        if path.startswith("test/") or path.startswith("tests/"):
            return False
        return True

    def create_source_archives(self) -> None:
        source_collector = SourceCollector(root=self.root, commit=self.commit, executer=self.executer, filter=self._path_filter)
        print(f"Collecting sources of {self.project}...")
        archive_tree = source_collector.get_archive_file_tree()
        latest_mod_time = archive_tree.get_latest_mod_time()
        archive_tree.add_file(NodeInArchive.from_text(arcpath=REVISION_TXT, text=f"{self.revision}\n", time=latest_mod_time))
        archive_tree.add_file(NodeInArchive.from_text(arcpath=f"{GIT_HASH_FILENAME}", text=f"{self.commit}\n", time=latest_mod_time))
        archive_tree.add_file_mapping(arc_dir="", file_mapping=self.release_info["source"].get("files", {}), file_mapping_root=self.root, context=self.get_context(), time=latest_mod_time)

        archive_base = f"{self.project}-{self.version}"
        zip_path = self.dist_path / f"{archive_base}.zip"
        tgz_path = self.dist_path / f"{archive_base}.tar.gz"
        txz_path = self.dist_path / f"{archive_base}.tar.xz"

        logger.info("Creating zip/tgz/txz source archives ...")
        if self.dry:
            zip_path.touch()
            tgz_path.touch()
            txz_path.touch()
        else:
            with Archiver(zip_path=zip_path, tgz_path=tgz_path, txz_path=txz_path) as archiver:
                print(f"Adding source files of {self.project}...")
                archive_tree.add_to_archiver(archive_base=archive_base, archiver=archiver)

                for extra_repo in self.release_info["source"].get("extra-repos", []):
                    extra_repo_root = self.root / extra_repo
                    assert (extra_repo_root / ".git").exists(), f"{extra_repo_root} must be a git repo"
                    extra_repo_commit = self.executer.check_output(["git", "rev-parse", "HEAD"], dry_out=f"gitsha-extra-repo-{extra_repo}", cwd=extra_repo_root).strip()
                    extra_repo_source_collector = SourceCollector(root=extra_repo_root, commit=extra_repo_commit, executer=self.executer, filter=self._external_repo_path_filter)
                    print(f"Collecting sources of {extra_repo} ...")
                    extra_repo_archive_tree = extra_repo_source_collector.get_archive_file_tree()
                    print(f"Adding source files of {extra_repo} ...")
                    extra_repo_archive_tree.add_to_archiver(archive_base=f"{archive_base}/{extra_repo}", archiver=archiver)

            for file in self.release_info["source"]["checks"]:
                assert f"{archive_base}/{file}" in archiver.added_files, f"'{archive_base}/{file}' must exist"

        logger.info("... done")

        self.artifacts["src-zip"] = zip_path
        self.artifacts["src-tar-gz"] = tgz_path
        self.artifacts["src-tar-xz"] = txz_path

        if not self.dry:
            with tgz_path.open("r+b") as f:
                # Zero the embedded timestamp in the gzip'ed tarball
                f.seek(4, 0)
                f.write(b"\x00\x00\x00\x00")

    def create_dmg(self, configuration: str="Release") -> None:
        dmg_in = self.root / self.release_info["dmg"]["path"]
        xcode_project = self.root / self.release_info["dmg"]["project"]
        assert xcode_project.is_dir(), f"{xcode_project} must be a directory"
        assert (xcode_project / "project.pbxproj").is_file, f"{xcode_project} must contain project.pbxproj"
        if not self.fast:
            dmg_in.unlink(missing_ok=True)
        build_xcconfig = self.release_info["dmg"].get("build-xcconfig")
        if build_xcconfig:
            shutil.copy(self.root / build_xcconfig, xcode_project.parent / "build.xcconfig")

        xcode_scheme = self.release_info["dmg"].get("scheme")
        xcode_target = self.release_info["dmg"].get("target")
        assert xcode_scheme or xcode_target, "dmg needs scheme or target"
        assert not (xcode_scheme and xcode_target), "dmg cannot have both scheme and target set"
        if xcode_scheme:
            scheme_or_target = "-scheme"
            target_like = xcode_scheme
        else:
            scheme_or_target = "-target"
            target_like = xcode_target
        self.executer.run(["xcodebuild", "ONLY_ACTIVE_ARCH=NO", "-project", xcode_project, scheme_or_target, target_like, "-configuration", configuration])
        if self.dry:
            dmg_in.parent.mkdir(parents=True, exist_ok=True)
            dmg_in.touch()

        assert dmg_in.is_file(), f"{self.project}.dmg was not created by xcodebuild"

        dmg_out = self.dist_path / f"{self.project}-{self.version}.dmg"
        shutil.copy(dmg_in, dmg_out)
        self.artifacts["dmg"] = dmg_out

    @property
    def git_hash_data(self) -> bytes:
        return f"{self.commit}\n".encode()

    def create_mingw_archives(self) -> None:
        build_type = "Release"
        build_parent_dir = self.root / "build-mingw"
        ARCH_TO_GNU_ARCH = {
            # "arm64": "aarch64",
            "x86": "i686",
            "x64": "x86_64",
        }
        ARCH_TO_TRIPLET = {
            # "arm64": "aarch64-w64-mingw32",
            "x86": "i686-w64-mingw32",
            "x64": "x86_64-w64-mingw32",
        }

        new_env = dict(os.environ)

        cmake_prefix_paths = []
        mingw_deps_path = self.deps_path / "mingw-deps"

        if "dependencies" in self.release_info["mingw"]:
            shutil.rmtree(mingw_deps_path, ignore_errors=True)
            mingw_deps_path.mkdir()

            for triplet in ARCH_TO_TRIPLET.values():
                (mingw_deps_path / triplet).mkdir()

            def extract_filter(member: tarfile.TarInfo, path: str, /):
                if member.name.startswith("SDL"):
                    member.name = "/".join(Path(member.name).parts[1:])
                return member
            for dep in self.release_info.get("dependencies", {}):
                extract_path = mingw_deps_path / f"extract-{dep}"
                extract_path.mkdir()
                with chdir(extract_path):
                    tar_path = self.deps_path / glob.glob(self.release_info["mingw"]["dependencies"][dep]["artifact"], root_dir=self.deps_path)[0]
                    logger.info("Extracting %s to %s", tar_path, mingw_deps_path)
                    assert tar_path.suffix in (".gz", ".xz")
                    with tarfile.open(tar_path, mode=f"r:{tar_path.suffix.strip('.')}") as tarf:
                        tarf.extractall(filter=extract_filter)
                    for arch, triplet in ARCH_TO_TRIPLET.items():
                        install_cmd = self.release_info["mingw"]["dependencies"][dep]["install-command"]
                        extra_configure_data = {
                            "ARCH": ARCH_TO_GNU_ARCH[arch],
                            "TRIPLET": triplet,
                            "PREFIX": str(mingw_deps_path / triplet),
                        }
                        install_cmd = configure_text(install_cmd, context=self.get_context(extra_configure_data))
                        self.executer.run(shlex.split(install_cmd), cwd=str(extract_path))

            dep_binpath = mingw_deps_path / triplet / "bin"
            assert dep_binpath.is_dir(), f"{dep_binpath} for PATH should exist"
            dep_pkgconfig = mingw_deps_path / triplet / "lib/pkgconfig"
            assert dep_pkgconfig.is_dir(), f"{dep_pkgconfig} for PKG_CONFIG_PATH should exist"

            new_env["PATH"] = os.pathsep.join([str(dep_binpath), new_env["PATH"]])
            new_env["PKG_CONFIG_PATH"] = str(dep_pkgconfig)
            cmake_prefix_paths.append(mingw_deps_path)

        new_env["CFLAGS"] = f"-O2 -ffile-prefix-map={self.root}=/src/{self.project}"
        new_env["CXXFLAGS"] = f"-O2 -ffile-prefix-map={self.root}=/src/{self.project}"

        assert any(system in self.release_info["mingw"] for system in ("autotools", "cmake"))
        assert not all(system in self.release_info["mingw"] for system in ("autotools", "cmake"))

        mingw_archs = set()
        arc_root = f"{self.project}-{self.version}"
        archive_file_tree = ArchiveFileTree()

        if "autotools" in self.release_info["mingw"]:
            for arch in self.release_info["mingw"]["autotools"]["archs"]:
                triplet = ARCH_TO_TRIPLET[arch]
                new_env["CC"] = f"{triplet}-gcc"
                new_env["CXX"] = f"{triplet}-g++"
                new_env["RC"] = f"{triplet}-windres"

                assert arch not in mingw_archs
                mingw_archs.add(arch)

                build_path = build_parent_dir / f"build-{triplet}"
                install_path = build_parent_dir / f"install-{triplet}"
                shutil.rmtree(install_path, ignore_errors=True)
                build_path.mkdir(parents=True, exist_ok=True)
                context = self.get_context({
                    "ARCH": arch,
                    "DEP_PREFIX": str(mingw_deps_path / triplet),
                })
                extra_args = configure_text_list(text_list=self.release_info["mingw"]["autotools"]["args"], context=context)

                with self.section_printer.group(f"Configuring MinGW {triplet} (autotools)"):
                    assert "@" not in " ".join(extra_args), f"@ should not be present in extra arguments ({extra_args})"
                    self.executer.run([
                        self.root / "configure",
                        f"--prefix={install_path}",
                        f"--includedir=${{prefix}}/include",
                        f"--libdir=${{prefix}}/lib",
                        f"--bindir=${{prefix}}/bin",
                        f"--exec-prefix=${{prefix}}/bin",
                        f"--host={triplet}",
                        f"--build=x86_64-none-linux-gnu",
                        "CFLAGS=-O2",
                        "CXXFLAGS=-O2",
                        "LDFLAGS=-Wl,-s",
                    ] + extra_args, cwd=build_path, env=new_env)
                with self.section_printer.group(f"Build MinGW {triplet} (autotools)"):
                    self.executer.run(["make", "V=1", f"-j{self.cpu_count}"], cwd=build_path, env=new_env)
                with self.section_printer.group(f"Install MinGW {triplet} (autotools)"):
                    self.executer.run(["make", "install"], cwd=build_path, env=new_env)
                archive_file_tree.add_directory_tree(arc_dir=arc_join(arc_root, triplet), path=install_path, time=self.arc_time)

                print("Recording arch-dependent extra files for MinGW development archive ...")
                extra_context = {
                    "TRIPLET": ARCH_TO_TRIPLET[arch],
                }
                archive_file_tree.add_file_mapping(arc_dir=arc_root, file_mapping=self.release_info["mingw"]["autotools"].get("files", {}), file_mapping_root=self.root, context=self.get_context(extra_context=extra_context), time=self.arc_time)

        if "cmake" in self.release_info["mingw"]:
            assert self.release_info["mingw"]["cmake"]["shared-static"] in ("args", "both")
            for arch in self.release_info["mingw"]["cmake"]["archs"]:
                triplet = ARCH_TO_TRIPLET[arch]
                new_env["CC"] = f"{triplet}-gcc"
                new_env["CXX"] = f"{triplet}-g++"
                new_env["RC"] = f"{triplet}-windres"

                assert arch not in mingw_archs
                mingw_archs.add(arch)

                context = self.get_context({
                    "ARCH": arch,
                    "DEP_PREFIX": str(mingw_deps_path / triplet),
                })
                extra_args = configure_text_list(text_list=self.release_info["mingw"]["cmake"]["args"], context=context)

                build_path = build_parent_dir / f"build-{triplet}"
                install_path = build_parent_dir / f"install-{triplet}"
                shutil.rmtree(install_path, ignore_errors=True)
                build_path.mkdir(parents=True, exist_ok=True)
                if self.release_info["mingw"]["cmake"]["shared-static"] == "args":
                    args_for_shared_static = ([], )
                elif self.release_info["mingw"]["cmake"]["shared-static"] == "both":
                    args_for_shared_static = (["-DBUILD_SHARED_LIBS=ON"], ["-DBUILD_SHARED_LIBS=OFF"])
                for arg_for_shared_static in args_for_shared_static:
                    with self.section_printer.group(f"Configuring MinGW {triplet} (CMake)"):
                        assert "@" not in " ".join(extra_args), f"@ should not be present in extra arguments ({extra_args})"
                        self.executer.run([
                            f"cmake",
                            f"-S", str(self.root), "-B", str(build_path),
                            f"-DCMAKE_BUILD_TYPE={build_type}",
                            f'''-DCMAKE_C_FLAGS="-ffile-prefix-map={self.root}=/src/{self.project}"''',
                            f'''-DCMAKE_CXX_FLAGS="-ffile-prefix-map={self.root}=/src/{self.project}"''',
                            f"-DCMAKE_PREFIX_PATH={mingw_deps_path / triplet}",
                            f"-DCMAKE_INSTALL_PREFIX={install_path}",
                            f"-DCMAKE_INSTALL_INCLUDEDIR=include",
                            f"-DCMAKE_INSTALL_LIBDIR=lib",
                            f"-DCMAKE_INSTALL_BINDIR=bin",
                            f"-DCMAKE_INSTALL_DATAROOTDIR=share",
                            f"-DCMAKE_TOOLCHAIN_FILE={self.root}/build-scripts/cmake-toolchain-mingw64-{ARCH_TO_GNU_ARCH[arch]}.cmake",
                            f"-G{self.cmake_generator}",
                        ] + extra_args + ([] if self.fast else ["--fresh"]) + arg_for_shared_static, cwd=build_path, env=new_env)
                    with self.section_printer.group(f"Build MinGW {triplet} (CMake)"):
                        self.executer.run(["cmake", "--build", str(build_path), "--verbose", "--config", build_type], cwd=build_path, env=new_env)
                    with self.section_printer.group(f"Install MinGW {triplet} (CMake)"):
                        self.executer.run(["cmake", "--install", str(build_path)], cwd=build_path, env=new_env)
                archive_file_tree.add_directory_tree(arc_dir=arc_join(arc_root, triplet), path=install_path, time=self.arc_time)

                print("Recording arch-dependent extra files for MinGW development archive ...")
                extra_context = {
                    "TRIPLET": ARCH_TO_TRIPLET[arch],
                }
                archive_file_tree.add_file_mapping(arc_dir=arc_root, file_mapping=self.release_info["mingw"]["cmake"].get("files", {}), file_mapping_root=self.root, context=self.get_context(extra_context=extra_context), time=self.arc_time)
                print("... done")

        print("Recording extra files for MinGW development archive ...")
        archive_file_tree.add_file_mapping(arc_dir=arc_root, file_mapping=self.release_info["mingw"].get("files", {}), file_mapping_root=self.root, context=self.get_context(), time=self.arc_time)
        print("... done")

        print("Creating zip/tgz/txz development archives ...")
        zip_path = self.dist_path / f"{self.project}-devel-{self.version}-mingw.zip"
        tgz_path = self.dist_path / f"{self.project}-devel-{self.version}-mingw.tar.gz"
        txz_path = self.dist_path / f"{self.project}-devel-{self.version}-mingw.tar.xz"

        with Archiver(zip_path=zip_path, tgz_path=tgz_path, txz_path=txz_path) as archiver:
            archive_file_tree.add_to_archiver(archive_base="", archiver=archiver)
            archiver.add_git_hash(arcdir=arc_root, commit=self.commit, time=self.arc_time)
        print("... done")

        self.artifacts["mingw-devel-zip"] = zip_path
        self.artifacts["mingw-devel-tar-gz"] = tgz_path
        self.artifacts["mingw-devel-tar-xz"] = txz_path

    def _detect_android_api(self, android_home: str) -> typing.Optional[int]:
        platform_dirs = list(Path(p) for p in glob.glob(f"{android_home}/platforms/android-*"))
        re_platform = re.compile("android-([0-9]+)")
        platform_versions = []
        for platform_dir in platform_dirs:
            logger.debug("Found Android Platform SDK: %s", platform_dir)
            if m:= re_platform.match(platform_dir.name):
                platform_versions.append(int(m.group(1)))
        platform_versions.sort()
        logger.info("Available platform versions: %s", platform_versions)
        platform_versions = list(filter(lambda v: v >= self._android_api_minimum, platform_versions))
        logger.info("Valid platform versions (>=%d): %s", self._android_api_minimum, platform_versions)
        if not platform_versions:
            return None
        android_api = platform_versions[0]
        logger.info("Selected API version %d", android_api)
        return android_api

    def _get_prefab_json_text(self) -> str:
        return textwrap.dedent(f"""\
            {{
                "schema_version": 2,
                "name": "{self.project}",
                "version": "{self.version}",
                "dependencies": []
            }}
        """)

    def _get_prefab_module_json_text(self, library_name: typing.Optional[str], export_libraries: list[str]) -> str:
        for lib in export_libraries:
            assert isinstance(lib, str), f"{lib} must be a string"
        module_json_dict = {
            "export_libraries": export_libraries,
        }
        if library_name:
            module_json_dict["library_name"] = f"lib{library_name}"
        return json.dumps(module_json_dict, indent=4)

    @property
    def _android_api_minimum(self):
        return self.release_info["android"]["api-minimum"]

    @property
    def _android_api_target(self):
        return self.release_info["android"]["api-target"]

    @property
    def _android_ndk_minimum(self):
        return self.release_info["android"]["ndk-minimum"]

    def _get_prefab_abi_json_text(self, abi: str, cpp: bool, shared: bool) -> str:
        abi_json_dict = {
            "abi": abi,
            "api": self._android_api_minimum,
            "ndk": self._android_ndk_minimum,
            "stl": "c++_shared" if cpp else "none",
            "static": not shared,
        }
        return json.dumps(abi_json_dict, indent=4)

    def _get_android_manifest_text(self) -> str:
        return textwrap.dedent(f"""\
            <manifest
                xmlns:android="http://schemas.android.com/apk/res/android"
                package="org.libsdl.android.{self.project}" android:versionCode="1"
                android:versionName="1.0">
                <uses-sdk android:minSdkVersion="{self._android_api_minimum}"
                          android:targetSdkVersion="{self._android_api_target}" />
            </manifest>
        """)

    def create_android_archives(self, android_api: int, android_home: Path, android_ndk_home: Path) -> None:
        cmake_toolchain_file = Path(android_ndk_home) / "build/cmake/android.toolchain.cmake"
        if not cmake_toolchain_file.exists():
            logger.error("CMake toolchain file does not exist (%s)", cmake_toolchain_file)
            raise SystemExit(1)
        aar_path =  self.dist_path / f"{self.project}-{self.version}.aar"
        android_abis = self.release_info["android"]["abis"]
        java_jars_added = False
        module_data_added = False
        android_deps_path = self.deps_path / "android-deps"
        shutil.rmtree(android_deps_path, ignore_errors=True)

        for dep, depinfo in self.release_info["android"].get("dependencies", {}).items():
            android_aar = self.deps_path / glob.glob(depinfo["artifact"], root_dir=self.deps_path)[0]
            with self.section_printer.group(f"Extracting Android dependency {dep} ({android_aar.name})"):
                self.executer.run([sys.executable, str(android_aar), "-o", str(android_deps_path)])

        for module_name, module_info in self.release_info["android"]["modules"].items():
            assert "type" in module_info and module_info["type"] in ("interface", "library"), f"module {module_name} must have a valid type"

        archive_file_tree = ArchiveFileTree()

        for android_abi in android_abis:
            with self.section_printer.group(f"Building for Android {android_api} {android_abi}"):
                build_dir = self.root / "build-android" / f"{android_abi}-build"
                install_dir = self.root / "install-android" / f"{android_abi}-install"
                shutil.rmtree(install_dir, ignore_errors=True)
                assert not install_dir.is_dir(), f"{install_dir} should not exist prior to build"
                build_type = "Release"
                cmake_args = [
                    "cmake",
                    "-S", str(self.root),
                    "-B", str(build_dir),
                    f'''-DCMAKE_C_FLAGS="-ffile-prefix-map={self.root}=/src/{self.project}"''',
                    f'''-DCMAKE_CXX_FLAGS="-ffile-prefix-map={self.root}=/src/{self.project}"''',
                    f"-DCMAKE_TOOLCHAIN_FILE={cmake_toolchain_file}",
                    f"-DCMAKE_PREFIX_PATH={str(android_deps_path)}",
                    f"-DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=BOTH",
                    f"-DANDROID_HOME={android_home}",
                    f"-DANDROID_PLATFORM={android_api}",
                    f"-DANDROID_ABI={android_abi}",
                    "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
                    f"-DCMAKE_INSTALL_PREFIX={install_dir}",
                    "-DCMAKE_INSTALL_INCLUDEDIR=include ",
                    "-DCMAKE_INSTALL_LIBDIR=lib",
                    "-DCMAKE_INSTALL_DATAROOTDIR=share",
                    f"-DCMAKE_BUILD_TYPE={build_type}",
                    f"-G{self.cmake_generator}",
                ] + self.release_info["android"]["cmake"]["args"] + ([] if self.fast else ["--fresh"])
                build_args = [
                    "cmake",
                    "--build", str(build_dir),
                    "--verbose",
                    "--config", build_type,
                ]
                install_args = [
                    "cmake",
                    "--install", str(build_dir),
                    "--config", build_type,
                ]
                self.executer.run(cmake_args)
                self.executer.run(build_args)
                self.executer.run(install_args)

                for module_name, module_info in self.release_info["android"]["modules"].items():
                    arcdir_prefab_module = f"prefab/modules/{module_name}"
                    if module_info["type"] == "library":
                        library = install_dir / module_info["library"]
                        assert library.suffix in (".so", ".a")
                        assert library.is_file(), f"CMake should have built library '{library}' for module {module_name}"
                        arcdir_prefab_libs = f"{arcdir_prefab_module}/libs/android.{android_abi}"
                        archive_file_tree.add_file(NodeInArchive.from_fs(arcpath=f"{arcdir_prefab_libs}/{library.name}", path=library, time=self.arc_time))
                        archive_file_tree.add_file(NodeInArchive.from_text(arcpath=f"{arcdir_prefab_libs}/abi.json", text=self._get_prefab_abi_json_text(abi=android_abi, cpp=False, shared=library.suffix == ".so"), time=self.arc_time))

                    if not module_data_added:
                        library_name = None
                        if module_info["type"] == "library":
                            library_name = Path(module_info["library"]).stem.removeprefix("lib")
                        export_libraries = module_info.get("export-libraries", [])
                        archive_file_tree.add_file(NodeInArchive.from_text(arcpath=arc_join(arcdir_prefab_module, "module.json"), text=self._get_prefab_module_json_text(library_name=library_name, export_libraries=export_libraries), time=self.arc_time))
                        arcdir_prefab_include = f"prefab/modules/{module_name}/include"
                        if "includes" in module_info:
                            archive_file_tree.add_file_mapping(arc_dir=arcdir_prefab_include, file_mapping=module_info["includes"], file_mapping_root=install_dir, context=self.get_context(), time=self.arc_time)
                        else:
                            archive_file_tree.add_file(NodeInArchive.from_text(arcpath=arc_join(arcdir_prefab_include, ".keep"), text="\n", time=self.arc_time))
                module_data_added = True

                if not java_jars_added:
                    java_jars_added = True
                    if "jars" in self.release_info["android"]:
                        classes_jar_path = install_dir / configure_text(text=self.release_info["android"]["jars"]["classes"], context=self.get_context())
                        sources_jar_path = install_dir / configure_text(text=self.release_info["android"]["jars"]["sources"], context=self.get_context())
                        doc_jar_path = install_dir / configure_text(text=self.release_info["android"]["jars"]["doc"], context=self.get_context())
                        assert classes_jar_path.is_file(), f"CMake should have compiled the java sources and archived them into a JAR ({classes_jar_path})"
                        assert sources_jar_path.is_file(), f"CMake should have archived the java sources into a JAR ({sources_jar_path})"
                        assert doc_jar_path.is_file(), f"CMake should have archived javadoc into a JAR ({doc_jar_path})"

                        archive_file_tree.add_file(NodeInArchive.from_fs(arcpath="classes.jar", path=classes_jar_path, time=self.arc_time))
                        archive_file_tree.add_file(NodeInArchive.from_fs(arcpath="classes-sources.jar", path=sources_jar_path, time=self.arc_time))
                        archive_file_tree.add_file(NodeInArchive.from_fs(arcpath="classes-doc.jar", path=doc_jar_path, time=self.arc_time))

        assert ("jars" in self.release_info["android"] and java_jars_added) or "jars" not in self.release_info["android"], "Must have archived java JAR archives"

        archive_file_tree.add_file_mapping(arc_dir="", file_mapping=self.release_info["android"].get("files", {}), file_mapping_root=self.root, context=self.get_context(), time=self.arc_time)

        archive_file_tree.add_file(NodeInArchive.from_text(arcpath="prefab/prefab.json", text=self._get_prefab_json_text(), time=self.arc_time))
        archive_file_tree.add_file(NodeInArchive.from_text(arcpath="AndroidManifest.xml", text=self._get_android_manifest_text(), time=self.arc_time))

        with Archiver(zip_path=aar_path) as archiver:
            archive_file_tree.add_to_archiver(archive_base="", archiver=archiver)
            archiver.add_git_hash(arcdir="", commit=self.commit, time=self.arc_time)
        self.artifacts[f"android-aar"] = aar_path

    def download_dependencies(self):
        shutil.rmtree(self.deps_path, ignore_errors=True)
        self.deps_path.mkdir(parents=True)

        if self.github:
            with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                f.write(f"dep-path={self.deps_path.absolute()}\n")

        for dep, depinfo in self.release_info.get("dependencies", {}).items():
            startswith = depinfo["startswith"]
            dep_repo = depinfo["repo"]
            # FIXME: dropped "--exclude-pre-releases"
            dep_string_data = self.executer.check_output(["gh", "-R", dep_repo, "release", "list", "--exclude-drafts", "--json", "name,createdAt,tagName", "--jq", f'[.[]|select(.name|startswith("{startswith}"))]|max_by(.createdAt)']).strip()
            dep_data = json.loads(dep_string_data)
            dep_tag = dep_data["tagName"]
            dep_version = dep_data["name"]
            logger.info("Download dependency %s version %s (tag=%s) ", dep, dep_version, dep_tag)
            self.executer.run(["gh", "-R", dep_repo, "release", "download", dep_tag], cwd=self.deps_path)
            if self.github:
                with open(os.environ["GITHUB_OUTPUT"], "a") as f:
                    f.write(f"dep-{dep.lower()}-version={dep_version}\n")

    def verify_dependencies(self):
        for dep, depinfo in self.release_info.get("dependencies", {}).items():
            if "mingw" in self.release_info:
                mingw_matches = glob.glob(self.release_info["mingw"]["dependencies"][dep]["artifact"], root_dir=self.deps_path)
                assert len(mingw_matches) == 1, f"Exactly one archive matches mingw {dep} dependency: {mingw_matches}"
            if "dmg" in self.release_info:
                dmg_matches = glob.glob(self.release_info["dmg"]["dependencies"][dep]["artifact"], root_dir=self.deps_path)
                assert len(dmg_matches) == 1, f"Exactly one archive matches dmg {dep} dependency: {dmg_matches}"
            if "msvc" in self.release_info:
                msvc_matches = glob.glob(self.release_info["msvc"]["dependencies"][dep]["artifact"], root_dir=self.deps_path)
                assert len(msvc_matches) == 1, f"Exactly one archive matches msvc {dep} dependency: {msvc_matches}"
            if "android" in self.release_info:
                android_matches = glob.glob(self.release_info["android"]["dependencies"][dep]["artifact"], root_dir=self.deps_path)
                assert len(android_matches) == 1, f"Exactly one archive matches msvc {dep} dependency: {msvc_matches}"

    @staticmethod
    def _arch_to_vs_platform(arch: str, configuration: str="Release") -> VsArchPlatformConfig:
        ARCH_TO_VS_PLATFORM = {
            "x86": VsArchPlatformConfig(arch="x86", platform="Win32", configuration=configuration),
            "x64": VsArchPlatformConfig(arch="x64", platform="x64", configuration=configuration),
            "arm64": VsArchPlatformConfig(arch="arm64", platform="ARM64", configuration=configuration),
        }
        return ARCH_TO_VS_PLATFORM[arch]

    def build_msvc(self):
        with self.section_printer.group("Find Visual Studio"):
            vs = VisualStudio(executer=self.executer)
        for arch in self.release_info["msvc"].get("msbuild", {}).get("archs", []):
            self._build_msvc_msbuild(arch_platform=self._arch_to_vs_platform(arch=arch), vs=vs)
        if "cmake" in self.release_info["msvc"]:
            deps_path = self.root / "msvc-deps"
            shutil.rmtree(deps_path, ignore_errors=True)
            dep_roots = []
            for dep, depinfo in self.release_info["msvc"].get("dependencies", {}).items():
                dep_extract_path = deps_path / f"extract-{dep}"
                msvc_zip = self.deps_path / glob.glob(depinfo["artifact"], root_dir=self.deps_path)[0]
                with zipfile.ZipFile(msvc_zip, "r") as zf:
                    zf.extractall(dep_extract_path)
                contents_msvc_zip = glob.glob(str(dep_extract_path / "*"))
                assert len(contents_msvc_zip) == 1, f"There must be exactly one root item in the root directory of {dep}"
                dep_roots.append(contents_msvc_zip[0])

            for arch in self.release_info["msvc"].get("cmake", {}).get("archs", []):
                self._build_msvc_cmake(arch_platform=self._arch_to_vs_platform(arch=arch), dep_roots=dep_roots)
        with self.section_printer.group("Create SDL VC development zip"):
            self._build_msvc_devel()

    def _build_msvc_msbuild(self, arch_platform: VsArchPlatformConfig, vs: VisualStudio):
        platform_context = self.get_context(arch_platform.extra_context())
        for dep, depinfo in self.release_info["msvc"].get("dependencies", {}).items():
            msvc_zip = self.deps_path / glob.glob(depinfo["artifact"], root_dir=self.deps_path)[0]

            src_globs = [configure_text(instr["src"], context=platform_context) for instr in depinfo["copy"]]
            with zipfile.ZipFile(msvc_zip, "r") as zf:
                for member in zf.namelist():
                    member_path = "/".join(Path(member).parts[1:])
                    for src_i, src_glob in enumerate(src_globs):
                        if fnmatch.fnmatch(member_path, src_glob):
                            dst = (self.root / configure_text(depinfo["copy"][src_i]["dst"], context=platform_context)).resolve() / Path(member_path).name
                            zip_data = zf.read(member)
                            if dst.exists():
                                identical = False
                                if dst.is_file():
                                    orig_bytes = dst.read_bytes()
                                    if orig_bytes == zip_data:
                                        identical = True
                                if not identical:
                                    logger.warning("Extracting dependency %s, will cause %s to be overwritten", dep, dst)
                                    if not self.overwrite:
                                        raise RuntimeError("Run with --overwrite to allow overwriting")
                            logger.debug("Extracting %s -> %s", member, dst)

                            dst.parent.mkdir(exist_ok=True, parents=True)
                            dst.write_bytes(zip_data)

        prebuilt_paths = set(self.root / full_prebuilt_path for prebuilt_path in self.release_info["msvc"]["msbuild"].get("prebuilt", []) for full_prebuilt_path in glob.glob(configure_text(prebuilt_path, context=platform_context), root_dir=self.root))
        msbuild_paths = set(self.root / configure_text(f, context=platform_context) for file_mapping in (self.release_info["msvc"]["msbuild"]["files-lib"], self.release_info["msvc"]["msbuild"]["files-devel"]) for files_list in file_mapping.values() for f in files_list)
        assert prebuilt_paths.issubset(msbuild_paths), f"msvc.msbuild.prebuilt must be a subset of (msvc.msbuild.files-lib, msvc.msbuild.files-devel)"
        built_paths = msbuild_paths.difference(prebuilt_paths)
        logger.info("MSbuild builds these files, to be included in the package: %s", built_paths)
        if not self.fast:
            for b in built_paths:
                b.unlink(missing_ok=True)

        rel_projects: list[str] = self.release_info["msvc"]["msbuild"]["projects"]
        projects = list(self.root / p for p in rel_projects)

        directory_build_props_src_relpath = self.release_info["msvc"]["msbuild"].get("directory-build-props")
        for project in projects:
            dir_b_props = project.parent / "Directory.Build.props"
            dir_b_props.unlink(missing_ok = True)
            if directory_build_props_src_relpath:
                src = self.root / directory_build_props_src_relpath
                logger.debug("Copying %s -> %s", src, dir_b_props)
                shutil.copy(src=src, dst=dir_b_props)

        with self.section_printer.group(f"Build {arch_platform.arch} VS binary"):
            vs.build(arch_platform=arch_platform, projects=projects)

        if self.dry:
            for b in built_paths:
                b.parent.mkdir(parents=True, exist_ok=True)
                b.touch()

        for b in built_paths:
            assert b.is_file(), f"{b} has not been created"
            b.parent.mkdir(parents=True, exist_ok=True)
            b.touch()

        zip_path = self.dist_path / f"{self.project}-{self.version}-win32-{arch_platform.arch}.zip"
        zip_path.unlink(missing_ok=True)

        logger.info("Collecting files...")
        archive_file_tree = ArchiveFileTree()
        archive_file_tree.add_file_mapping(arc_dir="", file_mapping=self.release_info["msvc"]["msbuild"]["files-lib"], file_mapping_root=self.root, context=platform_context, time=self.arc_time)
        archive_file_tree.add_file_mapping(arc_dir="", file_mapping=self.release_info["msvc"]["files-lib"], file_mapping_root=self.root, context=platform_context, time=self.arc_time)

        logger.info("Writing to %s", zip_path)
        with Archiver(zip_path=zip_path) as archiver:
            arc_root = f""
            archive_file_tree.add_to_archiver(archive_base=arc_root, archiver=archiver)
            archiver.add_git_hash(arcdir=arc_root, commit=self.commit, time=self.arc_time)
        self.artifacts[f"VC-{arch_platform.arch}"] = zip_path

        for p in built_paths:
            assert p.is_file(), f"{p} should exist"

    def _arch_platform_to_build_path(self, arch_platform: VsArchPlatformConfig) -> Path:
        return self.root / f"build-vs-{arch_platform.arch}"

    def _arch_platform_to_install_path(self, arch_platform: VsArchPlatformConfig) -> Path:
        return self._arch_platform_to_build_path(arch_platform) / "prefix"

    def _build_msvc_cmake(self, arch_platform: VsArchPlatformConfig, dep_roots: list[Path]):
        build_path = self._arch_platform_to_build_path(arch_platform)
        install_path = self._arch_platform_to_install_path(arch_platform)
        platform_context = self.get_context(extra_context=arch_platform.extra_context())

        build_type = "Release"

        built_paths = set(install_path / configure_text(f, context=platform_context) for file_mapping in (self.release_info["msvc"]["cmake"]["files-lib"], self.release_info["msvc"]["cmake"]["files-devel"]) for files_list in file_mapping.values() for f in files_list)
        logger.info("CMake builds these files, to be included in the package: %s", built_paths)
        if not self.fast:
            for b in built_paths:
                b.unlink(missing_ok=True)

        shutil.rmtree(install_path, ignore_errors=True)
        build_path.mkdir(parents=True, exist_ok=True)
        with self.section_printer.group(f"Configure VC CMake project for {arch_platform.arch}"):
            self.executer.run([
                "cmake", "-S", str(self.root), "-B", str(build_path),
                "-A", arch_platform.platform,
                "-DCMAKE_INSTALL_BINDIR=bin",
                "-DCMAKE_INSTALL_DATAROOTDIR=share",
                "-DCMAKE_INSTALL_INCLUDEDIR=include",
                "-DCMAKE_INSTALL_LIBDIR=lib",
                f"-DCMAKE_BUILD_TYPE={build_type}",
                f"-DCMAKE_INSTALL_PREFIX={install_path}",
                # MSVC debug information format flags are selected by an abstraction
                "-DCMAKE_POLICY_DEFAULT_CMP0141=NEW",
                # MSVC debug information format
                "-DCMAKE_MSVC_DEBUG_INFORMATION_FORMAT=ProgramDatabase",
                # Linker flags for executables
                "-DCMAKE_EXE_LINKER_FLAGS=-INCREMENTAL:NO -DEBUG -OPT:REF -OPT:ICF",
                # Linker flag for shared libraries
                "-DCMAKE_SHARED_LINKER_FLAGS=-INCREMENTAL:NO -DEBUG -OPT:REF -OPT:ICF",
                # MSVC runtime library flags are selected by an abstraction
                "-DCMAKE_POLICY_DEFAULT_CMP0091=NEW",
                # Use statically linked runtime (-MT) (ideally, should be "MultiThreaded$<$<CONFIG:Debug>:Debug>")
                "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded",
                f"-DCMAKE_PREFIX_PATH={';'.join(str(s) for s in dep_roots)}",
            ] + self.release_info["msvc"]["cmake"]["args"] + ([] if self.fast else ["--fresh"]))

        with self.section_printer.group(f"Build VC CMake project for {arch_platform.arch}"):
            self.executer.run(["cmake", "--build", str(build_path), "--verbose", "--config", build_type])
        with self.section_printer.group(f"Install VC CMake project for {arch_platform.arch}"):
            self.executer.run(["cmake", "--install", str(build_path), "--config", build_type])

        if self.dry:
            for b in built_paths:
                b.parent.mkdir(parents=True, exist_ok=True)
                b.touch()

        zip_path = self.dist_path / f"{self.project}-{self.version}-win32-{arch_platform.arch}.zip"
        zip_path.unlink(missing_ok=True)

        logger.info("Collecting files...")
        archive_file_tree = ArchiveFileTree()
        archive_file_tree.add_file_mapping(arc_dir="", file_mapping=self.release_info["msvc"]["cmake"]["files-lib"], file_mapping_root=install_path, context=platform_context, time=self.arc_time)
        archive_file_tree.add_file_mapping(arc_dir="", file_mapping=self.release_info["msvc"]["files-lib"], file_mapping_root=self.root, context=self.get_context(), time=self.arc_time)

        logger.info("Creating %s", zip_path)
        with Archiver(zip_path=zip_path) as archiver:
            arc_root = f""
            archive_file_tree.add_to_archiver(archive_base=arc_root, archiver=archiver)
            archiver.add_git_hash(arcdir=arc_root, commit=self.commit, time=self.arc_time)

        for p in built_paths:
            assert p.is_file(), f"{p} should exist"

    def _build_msvc_devel(self) -> None:
        zip_path = self.dist_path / f"{self.project}-devel-{self.version}-VC.zip"
        arc_root = f"{self.project}-{self.version}"

        def copy_files_devel(ctx):
            archive_file_tree.add_file_mapping(arc_dir=arc_root, file_mapping=self.release_info["msvc"]["files-devel"], file_mapping_root=self.root, context=ctx, time=self.arc_time)


        logger.info("Collecting files...")
        archive_file_tree = ArchiveFileTree()
        if "msbuild" in self.release_info["msvc"]:
            for arch in self.release_info["msvc"]["msbuild"]["archs"]:
                arch_platform = self._arch_to_vs_platform(arch=arch)
                platform_context = self.get_context(arch_platform.extra_context())
                archive_file_tree.add_file_mapping(arc_dir=arc_root, file_mapping=self.release_info["msvc"]["msbuild"]["files-devel"], file_mapping_root=self.root, context=platform_context, time=self.arc_time)
                copy_files_devel(ctx=platform_context)
        if "cmake" in self.release_info["msvc"]:
            for arch in self.release_info["msvc"]["cmake"]["archs"]:
                arch_platform = self._arch_to_vs_platform(arch=arch)
                platform_context = self.get_context(arch_platform.extra_context())
                archive_file_tree.add_file_mapping(arc_dir=arc_root, file_mapping=self.release_info["msvc"]["cmake"]["files-devel"], file_mapping_root=self._arch_platform_to_install_path(arch_platform), context=platform_context, time=self.arc_time)
                copy_files_devel(ctx=platform_context)

        with Archiver(zip_path=zip_path) as archiver:
            archive_file_tree.add_to_archiver(archive_base="", archiver=archiver)
            archiver.add_git_hash(arcdir=arc_root, commit=self.commit, time=self.arc_time)
        self.artifacts["VC-devel"] = zip_path

    @classmethod
    def extract_sdl_version(cls, root: Path, release_info: dict) -> str:
        with open(root / release_info["version"]["file"], "r") as f:
            text = f.read()
        major = next(re.finditer(release_info["version"]["re_major"], text, flags=re.M)).group(1)
        minor = next(re.finditer(release_info["version"]["re_minor"], text, flags=re.M)).group(1)
        micro = next(re.finditer(release_info["version"]["re_micro"], text, flags=re.M)).group(1)
        return f"{major}.{minor}.{micro}"


def main(argv=None) -> int:
    if sys.version_info < (3, 11):
        logger.error("This script needs at least python 3.11")
        return 1

    parser = argparse.ArgumentParser(allow_abbrev=False, description="Create SDL release artifacts")
    parser.add_argument("--root", metavar="DIR", type=Path, default=Path(__file__).absolute().parents[1], help="Root of project")
    parser.add_argument("--release-info", metavar="JSON", dest="path_release_info", type=Path, default=Path(__file__).absolute().parent / "release-info.json", help="Path of release-info.json")
    parser.add_argument("--dependency-folder", metavar="FOLDER", dest="deps_path", type=Path, default="deps", help="Directory containing pre-built archives of dependencies (will be removed when downloading archives)")
    parser.add_argument("--out", "-o", metavar="DIR", dest="dist_path", type=Path, default="dist", help="Output directory")
    parser.add_argument("--github", action="store_true", help="Script is running on a GitHub runner")
    parser.add_argument("--commit", default="HEAD", help="Git commit/tag of which a release should be created")
    parser.add_argument("--actions", choices=["download", "source", "android", "mingw", "msvc", "dmg"], required=True, nargs="+", dest="actions", help="What to do?")
    parser.set_defaults(loglevel=logging.INFO)
    parser.add_argument('--vs-year', dest="vs_year", help="Visual Studio year")
    parser.add_argument('--android-api', type=int, dest="android_api", help="Android API version")
    parser.add_argument('--android-home', dest="android_home", default=os.environ.get("ANDROID_HOME"), help="Android Home folder")
    parser.add_argument('--android-ndk-home', dest="android_ndk_home", default=os.environ.get("ANDROID_NDK_HOME"), help="Android NDK Home folder")
    parser.add_argument('--cmake-generator', dest="cmake_generator", default="Ninja", help="CMake Generator")
    parser.add_argument('--debug', action='store_const', const=logging.DEBUG, dest="loglevel", help="Print script debug information")
    parser.add_argument('--dry-run', action='store_true', dest="dry", help="Don't execute anything")
    parser.add_argument('--force', action='store_true', dest="force", help="Ignore a non-clean git tree")
    parser.add_argument('--overwrite', action='store_true', dest="overwrite", help="Allow potentially overwriting other projects")
    parser.add_argument('--fast', action='store_true', dest="fast", help="Don't do a rebuild")

    args = parser.parse_args(argv)
    logging.basicConfig(level=args.loglevel, format='[%(levelname)s] %(message)s')
    args.deps_path = args.deps_path.absolute()
    args.dist_path = args.dist_path.absolute()
    args.root = args.root.absolute()
    args.dist_path = args.dist_path.absolute()
    if args.dry:
        args.dist_path = args.dist_path / "dry"

    if args.github:
        section_printer: SectionPrinter = GitHubSectionPrinter()
    else:
        section_printer = SectionPrinter()

    if args.github and "GITHUB_OUTPUT" not in os.environ:
        os.environ["GITHUB_OUTPUT"] = "/tmp/github_output.txt"

    executer = Executer(root=args.root, dry=args.dry)

    root_git_hash_path = args.root / GIT_HASH_FILENAME
    root_is_maybe_archive = root_git_hash_path.is_file()
    if root_is_maybe_archive:
        logger.warning("%s detected: Building from archive", GIT_HASH_FILENAME)
        archive_commit = root_git_hash_path.read_text().strip()
        if args.commit != archive_commit:
            logger.warning("Commit argument is %s, but archive commit is %s. Using %s.", args.commit, archive_commit, archive_commit)
        args.commit = archive_commit
        revision = (args.root / REVISION_TXT).read_text().strip()
    else:
        args.commit = executer.check_output(["git", "rev-parse", args.commit], dry_out="e5812a9fd2cda317b503325a702ba3c1c37861d9").strip()
        revision = executer.check_output(["git", "describe", "--always", "--tags", "--long", args.commit], dry_out="preview-3.1.3-96-g9512f2144").strip()
        logger.info("Using commit %s", args.commit)

    try:
        with args.path_release_info.open() as f:
            release_info = json.load(f)
    except FileNotFoundError:
        logger.error(f"Could not find {args.path_release_info}")

    releaser = Releaser(
        release_info=release_info,
        commit=args.commit,
        revision=revision,
        root=args.root,
        dist_path=args.dist_path,
        executer=executer,
        section_printer=section_printer,
        cmake_generator=args.cmake_generator,
        deps_path=args.deps_path,
        overwrite=args.overwrite,
        github=args.github,
        fast=args.fast,
    )

    if root_is_maybe_archive:
        logger.warning("Building from archive. Skipping clean git tree check.")
    else:
        porcelain_status = executer.check_output(["git", "status", "--ignored", "--porcelain"], dry_out="\n").strip()
        if porcelain_status:
            print(porcelain_status)
            logger.warning("The tree is dirty! Do not publish any generated artifacts!")
            if not args.force:
                raise Exception("The git repo contains modified and/or non-committed files. Run with --force to ignore.")

    if args.fast:
        logger.warning("Doing fast build! Do not publish generated artifacts!")

    with section_printer.group("Arguments"):
        print(f"project          = {releaser.project}")
        print(f"version          = {releaser.version}")
        print(f"revision         = {revision}")
        print(f"commit           = {args.commit}")
        print(f"out              = {args.dist_path}")
        print(f"actions          = {args.actions}")
        print(f"dry              = {args.dry}")
        print(f"force            = {args.force}")
        print(f"overwrite        = {args.overwrite}")
        print(f"cmake_generator  = {args.cmake_generator}")

    releaser.prepare()

    if "download" in args.actions:
        releaser.download_dependencies()

    if set(args.actions).intersection({"msvc", "mingw", "android"}):
        print("Verifying presence of dependencies (run 'download' action to download) ...")
        releaser.verify_dependencies()
        print("... done")

    if "source" in args.actions:
        if root_is_maybe_archive:
            raise Exception("Cannot build source archive from source archive")
        with section_printer.group("Create source archives"):
            releaser.create_source_archives()

    if "dmg" in args.actions:
        if platform.system() != "Darwin" and not args.dry:
            parser.error("framework artifact(s) can only be built on Darwin")

        releaser.create_dmg()

    if "msvc" in args.actions:
        if platform.system() != "Windows" and not args.dry:
            parser.error("msvc artifact(s) can only be built on Windows")
        releaser.build_msvc()

    if "mingw" in args.actions:
        releaser.create_mingw_archives()

    if "android" in args.actions:
        if args.android_home is None or not Path(args.android_home).is_dir():
            parser.error("Invalid $ANDROID_HOME or --android-home: must be a directory containing the Android SDK")
        if args.android_ndk_home is None or not Path(args.android_ndk_home).is_dir():
            parser.error("Invalid $ANDROID_NDK_HOME or --android_ndk_home: must be a directory containing the Android NDK")
        if args.android_api is None:
            with section_printer.group("Detect Android APIS"):
                args.android_api = releaser._detect_android_api(android_home=args.android_home)
        if args.android_api is None or not (Path(args.android_home) / f"platforms/android-{args.android_api}").is_dir():
            parser.error("Invalid --android-api, and/or could not be detected")
        with section_printer.group("Android arguments"):
            print(f"android_home     = {args.android_home}")
            print(f"android_ndk_home = {args.android_ndk_home}")
            print(f"android_api      = {args.android_api}")
        releaser.create_android_archives(
            android_api=args.android_api,
            android_home=args.android_home,
            android_ndk_home=args.android_ndk_home,
        )
    with section_printer.group("Summary"):
        print(f"artifacts = {releaser.artifacts}")

    if args.github:
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"project={releaser.project}\n")
            f.write(f"version={releaser.version}\n")
            for k, v in releaser.artifacts.items():
                f.write(f"{k}={v.name}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
