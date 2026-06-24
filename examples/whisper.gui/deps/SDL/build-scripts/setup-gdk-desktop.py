#!/usr/bin/env python

import argparse
import functools
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import textwrap
import urllib.request
import zipfile

# Update both variables when updating the GDK
GIT_REF = "June_2024_Update_1"
GDK_EDITION = "240601"  # YYMMUU

logger = logging.getLogger(__name__)

class GdDesktopConfigurator:
    def __init__(self, gdk_path, arch, vs_folder, vs_version=None, vs_toolset=None, temp_folder=None, git_ref=None, gdk_edition=None):
        self.git_ref = git_ref or GIT_REF
        self.gdk_edition = gdk_edition or GDK_EDITION
        self.gdk_path = gdk_path
        self.temp_folder = temp_folder or Path(tempfile.gettempdir())
        self.dl_archive_path = Path(self.temp_folder) / f"{ self.git_ref }.zip"
        self.gdk_extract_path = Path(self.temp_folder) / f"GDK-{ self.git_ref }"
        self.arch = arch
        self.vs_folder = vs_folder
        self._vs_version = vs_version
        self._vs_toolset = vs_toolset

    def download_archive(self) -> None:
        gdk_url = f"https://github.com/microsoft/GDK/archive/refs/tags/{ GIT_REF }.zip"
        logger.info("Downloading %s to %s", gdk_url, self.dl_archive_path)
        urllib.request.urlretrieve(gdk_url, self.dl_archive_path)
        assert self.dl_archive_path.is_file()

    def extract_zip_archive(self) -> None:
        extract_path = self.gdk_extract_path.parent
        assert self.dl_archive_path.is_file()
        logger.info("Extracting %s to %s", self.dl_archive_path, extract_path)
        with zipfile.ZipFile(self.dl_archive_path) as zf:
            zf.extractall(extract_path)
        assert self.gdk_extract_path.is_dir(), f"{self.gdk_extract_path} must exist"

    def extract_development_kit(self) -> None:
        extract_dks_cmd = self.gdk_extract_path / "SetupScripts/ExtractXboxOneDKs.cmd"
        assert extract_dks_cmd.is_file()
        logger.info("Extracting GDK Development Kit: running %s", extract_dks_cmd)
        cmd = ["cmd.exe", "/C", str(extract_dks_cmd), str(self.gdk_extract_path), str(self.gdk_path)]
        logger.debug("Running %r", cmd)
        subprocess.check_call(cmd)

    def detect_vs_version(self) -> str:
        vs_regex = re.compile("VS([0-9]{4})")
        supported_vs_versions = []
        for p in self.gaming_grdk_build_path.iterdir():
            if not p.is_dir():
                continue
            if m := vs_regex.match(p.name):
                supported_vs_versions.append(m.group(1))
        logger.info(f"Supported Visual Studio versions: {supported_vs_versions}")
        vs_versions = set(self.vs_folder.parts).intersection(set(supported_vs_versions))
        if not vs_versions:
            raise RuntimeError("Visual Studio version is incompatible")
        if len(vs_versions) > 1:
            raise RuntimeError(f"Too many compatible VS versions found ({vs_versions})")
        vs_version = vs_versions.pop()
        logger.info(f"Used Visual Studio version: {vs_version}")
        return vs_version

    def detect_vs_toolset(self) -> str:
        toolset_paths = []
        for ts_path in self.gdk_toolset_parent_path.iterdir():
            if not ts_path.is_dir():
                continue
            ms_props = ts_path / "Microsoft.Cpp.props"
            if not ms_props.is_file():
                continue
            toolset_paths.append(ts_path.name)
        logger.info("Detected Visual Studio toolsets: %s", toolset_paths)
        assert toolset_paths, "Have we detected at least one toolset?"

        def toolset_number(toolset: str) -> int:
            if m:= re.match("[^0-9]*([0-9]+).*", toolset):
                return int(m.group(1))
            return -9

        return max(toolset_paths, key=toolset_number)

    @property
    def vs_version(self) -> str:
        if self._vs_version is None:
            self._vs_version = self.detect_vs_version()
        return self._vs_version

    @property
    def vs_toolset(self) -> str:
        if self._vs_toolset is None:
            self._vs_toolset = self.detect_vs_toolset()
        return self._vs_toolset

    @staticmethod
    def copy_files_and_merge_into(srcdir: Path, dstdir: Path) -> None:
        logger.info(f"Copy {srcdir} to {dstdir}")
        for root, _, files in os.walk(srcdir):
            dest_root = dstdir / Path(root).relative_to(srcdir)
            if not dest_root.is_dir():
                dest_root.mkdir()
            for file in files:
                srcfile = Path(root) / file
                dstfile = dest_root / file
                shutil.copy(srcfile, dstfile)

    def copy_msbuild(self) -> None:
        vc_toolset_parent_path = self.vs_folder / "MSBuild/Microsoft/VC"
        if 1:
            logger.info(f"Detected compatible Visual Studio version: {self.vs_version}")
            srcdir = vc_toolset_parent_path
            dstdir = self.gdk_toolset_parent_path
            assert srcdir.is_dir(), "Source directory must exist"
            assert dstdir.is_dir(), "Destination directory must exist"

            self.copy_files_and_merge_into(srcdir=srcdir, dstdir=dstdir)

    @property
    def game_dk_path(self) -> Path:
        return self.gdk_path / "Microsoft GDK"

    @property
    def game_dk_latest_path(self) -> Path:
        return self.game_dk_path / self.gdk_edition

    @property
    def windows_sdk_path(self) -> Path:
        return self.gdk_path / "Windows Kits/10"

    @property
    def gaming_grdk_build_path(self) -> Path:
        return self.game_dk_latest_path / "GRDK"

    @property
    def gdk_toolset_parent_path(self) -> Path:
        return self.gaming_grdk_build_path / f"VS{self.vs_version}/flatDeployment/MSBuild/Microsoft/VC"

    @property
    def env(self) -> dict[str, str]:
        game_dk = self.game_dk_path
        game_dk_latest = self.game_dk_latest_path
        windows_sdk_dir = self.windows_sdk_path
        gaming_grdk_build = self.gaming_grdk_build_path

        return {
            "GRDKEDITION": f"{self.gdk_edition}",
            "GameDK": f"{game_dk}\\",
            "GameDKLatest": f"{ game_dk_latest }\\",
            "WindowsSdkDir": f"{ windows_sdk_dir }\\",
            "GamingGRDKBuild": f"{ gaming_grdk_build }\\",
            "VSInstallDir": f"{ self.vs_folder }\\",
        }

    def create_user_props(self, path: Path) -> None:
        vc_targets_path = self.gaming_grdk_build_path / f"VS{ self.vs_version }/flatDeployment/MSBuild/Microsoft/VC/{ self.vs_toolset }"
        vc_targets_path16 = self.gaming_grdk_build_path / f"VS2019/flatDeployment/MSBuild/Microsoft/VC/{ self.vs_toolset }"
        vc_targets_path17 = self.gaming_grdk_build_path / f"VS2022/flatDeployment/MSBuild/Microsoft/VC/{ self.vs_toolset }"
        additional_include_directories = ";".join(str(p) for p in self.gdk_include_paths)
        additional_library_directories = ";".join(str(p) for p in self.gdk_library_paths)
        durango_xdk_install_path = self.gdk_path / "Microsoft GDK"
        with path.open("w") as f:
            f.write(textwrap.dedent(f"""\
                <?xml version="1.0" encoding="utf-8"?>
                <Project ToolsVersion="4.0" xmlns="http://schemas.microsoft.com/developer/msbuild/2003">
                  <PropertyGroup>
                    <VCTargetsPath>{ vc_targets_path }\\</VCTargetsPath>
                    <VCTargetsPath16>{ vc_targets_path16 }\\</VCTargetsPath16>
                    <VCTargetsPath17>{ vc_targets_path17 }\\</VCTargetsPath17>
                    <BWOI_GDK_Path>{ self.gaming_grdk_build_path }\\</BWOI_GDK_Path>
                    <Platform Condition="'$(Platform)' == ''">Gaming.Desktop.x64</Platform>
                    <Configuration Condition="'$(Configuration)' == ''">Debug</Configuration>
                    <XdkEditionTarget>{ self.gdk_edition }</XdkEditionTarget>
                    <DurangoXdkInstallPath>{ durango_xdk_install_path }</DurangoXdkInstallPath>

                    <DefaultXdkEditionRootVS2019>$(DurangoXdkInstallPath)\\{self.gdk_edition}\\GRDK\\VS2019\\flatDeployment\\MSBuild\\Microsoft\\VC\\{self.vs_toolset}\\Platforms\\$(Platform)\\</DefaultXdkEditionRootVS2019>
                    <XdkEditionRootVS2019>$(DurangoXdkInstallPath)\\{self.gdk_edition}\\GRDK\\VS2019\\flatDeployment\\MSBuild\\Microsoft\\VC\\{self.vs_toolset}\\Platforms\\$(Platform)\\</XdkEditionRootVS2019>
                    <DefaultXdkEditionRootVS2022>$(DurangoXdkInstallPath)\\{self.gdk_edition}\\GRDK\\VS2022\\flatDeployment\\MSBuild\\Microsoft\\VC\\{self.vs_toolset}\\Platforms\\$(Platform)\\</DefaultXdkEditionRootVS2022>
                    <XdkEditionRootVS2022>$(DurangoXdkInstallPath)\\{self.gdk_edition}\\GRDK\\VS2022\\flatDeployment\\MSBuild\\Microsoft\\VC\\{self.vs_toolset}\\Platforms\\$(Platform)\\</XdkEditionRootVS2022>

                    <Deterministic>true</Deterministic>
                    <DisableInstalledVCTargetsUse>true</DisableInstalledVCTargetsUse>
                    <ClearDevCommandPromptEnvVars>true</ClearDevCommandPromptEnvVars>
                  </PropertyGroup>
                  <ItemDefinitionGroup Condition="'$(Platform)' == 'Gaming.Desktop.x64'">
                    <ClCompile>
                      <AdditionalIncludeDirectories>{ additional_include_directories };%(AdditionalIncludeDirectories)</AdditionalIncludeDirectories>
                    </ClCompile>
                    <Link>
                      <AdditionalLibraryDirectories>{ additional_library_directories };%(AdditionalLibraryDirectories)</AdditionalLibraryDirectories>
                    </Link>
                  </ItemDefinitionGroup>
                </Project>
            """))

    @property
    def gdk_include_paths(self) -> list[Path]:
        return [
            self.gaming_grdk_build_path / "gamekit/include",
        ]

    @property
    def gdk_library_paths(self) -> list[Path]:
        return [
            self.gaming_grdk_build_path / f"gamekit/lib/{self.arch}",
        ]

    @property
    def gdk_binary_path(self) -> list[Path]:
        return [
            self.gaming_grdk_build_path / "bin",
            self.game_dk_path / "bin",
        ]

    @property
    def build_env(self) -> dict[str, str]:
        gdk_include = ";".join(str(p) for p in self.gdk_include_paths)
        gdk_lib = ";".join(str(p) for p in self.gdk_library_paths)
        gdk_path = ";".join(str(p) for p in self.gdk_binary_path)
        return {
            "GDK_INCLUDE": gdk_include,
            "GDK_LIB": gdk_lib,
            "GDK_PATH": gdk_path,
        }

    def print_env(self) -> None:
        for k, v in self.env.items():
            print(f"set \"{k}={v}\"")
        print()
        for k, v in self.build_env.items():
            print(f"set \"{k}={v}\"")
        print()
        print(f"set \"PATH=%GDK_PATH%;%PATH%\"")
        print(f"set \"LIB=%GDK_LIB%;%LIB%\"")
        print(f"set \"INCLUDE=%GDK_INCLUDE%;%INCLUDE%\"")


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--arch", choices=["amd64"], default="amd64", help="Architecture")
    parser.add_argument("--download", action="store_true", help="Download GDK")
    parser.add_argument("--extract", action="store_true", help="Extract downloaded GDK")
    parser.add_argument("--copy-msbuild", action="store_true", help="Copy MSBuild files")
    parser.add_argument("--temp-folder", help="Temporary folder where to download and extract GDK")
    parser.add_argument("--gdk-path", required=True, type=Path, help="Folder where to store the GDK")
    parser.add_argument("--ref-edition", type=str, help="Git ref and GDK edition separated by comma")
    parser.add_argument("--vs-folder", required=True, type=Path, help="Installation folder of Visual Studio")
    parser.add_argument("--vs-version", required=False, type=int, help="Visual Studio version")
    parser.add_argument("--vs-toolset", required=False, help="Visual Studio toolset (e.g. v150)")
    parser.add_argument("--props-folder", required=False, type=Path, default=Path(), help="Visual Studio toolset (e.g. v150)")
    parser.add_argument("--no-user-props", required=False, dest="user_props", action="store_false", help="Don't ")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    git_ref = None
    gdk_edition = None
    if args.ref_edition is not None:
        git_ref, gdk_edition = args.ref_edition.split(",", 1)
        try:
            int(gdk_edition)
        except ValueError:
            parser.error("Edition should be an integer (YYMMUU) (Y=year M=month U=update)")

    configurator = GdDesktopConfigurator(
        arch=args.arch,
        git_ref=git_ref,
        gdk_edition=gdk_edition,
        vs_folder=args.vs_folder,
        vs_version=args.vs_version,
        vs_toolset=args.vs_toolset,
        gdk_path=args.gdk_path,
        temp_folder=args.temp_folder,
    )

    if args.download:
        configurator.download_archive()

    if args.extract:
        configurator.extract_zip_archive()

        configurator.extract_development_kit()

    if args.copy_msbuild:
        configurator.copy_msbuild()

    if args.user_props:
        configurator.print_env()
        configurator.create_user_props(args.props_folder / "Directory.Build.props")

if __name__ == "__main__":
    raise SystemExit(main())
