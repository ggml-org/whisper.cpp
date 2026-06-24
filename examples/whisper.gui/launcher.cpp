// A tiny double-clickable launcher for whisper-gui on Windows.
//
// whisper-gui.exe lives under build\bin\<config>\ and must run with the working
// directory at the repo root, so its relative paths (models\..., diarize.py)
// resolve. This launcher walks up from its own location to find the repo root
// (the folder containing examples\whisper.gui\diarize.py), sets the working
// directory there, and starts whisper-gui.exe - so users can just double-click
// it instead of opening a shell in the right folder.

#ifdef _WIN32

#include <windows.h>
#include <string>

static bool exists(const std::wstring & p) {
    return GetFileAttributesW(p.c_str()) != INVALID_FILE_ATTRIBUTES;
}

int WINAPI wWinMain(HINSTANCE, HINSTANCE, PWSTR, int) {
    wchar_t buf[MAX_PATH];
    GetModuleFileNameW(nullptr, buf, MAX_PATH);
    std::wstring dir(buf);
    dir = dir.substr(0, dir.find_last_of(L"\\/"));

    // walk up to the repo root (dir that has examples\whisper.gui\diarize.py)
    std::wstring root = dir;
    for (int i = 0; i < 8; ++i) {
        if (exists(root + L"\\examples\\whisper.gui\\diarize.py")) break;
        const size_t slash = root.find_last_of(L"\\/");
        if (slash == std::wstring::npos) break;
        root = root.substr(0, slash);
    }
    if (!exists(root + L"\\examples\\whisper.gui\\diarize.py")) {
        root = dir; // fall back to the launcher's own folder
    }
    SetCurrentDirectoryW(root.c_str());

    // locate the built GUI (MSVC multi-config puts it under a config subdir)
    const wchar_t * cands[] = {
        L"build\\bin\\Release\\whisper-gui.exe",
        L"build\\bin\\whisper-gui.exe",
        L"build\\bin\\Debug\\whisper-gui.exe",
    };
    std::wstring gui;
    for (const wchar_t * c : cands) {
        if (exists(root + L"\\" + c)) { gui = root + L"\\" + c; break; }
    }
    if (gui.empty()) {
        MessageBoxW(nullptr,
            L"whisper-gui.exe was not found under build\\bin.\n"
            L"Build it first (cmake --build build --config Release --target whisper-gui).",
            L"whisper.cpp", MB_ICONERROR);
        return 1;
    }

    std::wstring cmd = L"\"" + gui + L"\"";
    STARTUPINFOW si{};  si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    if (!CreateProcessW(nullptr, &cmd[0], nullptr, nullptr, FALSE,
                        CREATE_NO_WINDOW, nullptr, root.c_str(), &si, &pi)) {
        MessageBoxW(nullptr, L"Failed to start whisper-gui.exe", L"whisper.cpp", MB_ICONERROR);
        return 1;
    }
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    return 0;
}

#else
int main() { return 0; } // launcher is Windows-only; no-op elsewhere
#endif
