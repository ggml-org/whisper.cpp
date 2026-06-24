// A tiny, dependency-free file browser rendered with Dear ImGui.
//
// Native file dialogs would pull in GTK/portal/zenity (extra deps, and flaky
// under WSLg), so instead this navigates the filesystem with std::filesystem
// inside an ImGui modal. Keeps the GUI self-contained and air-gapped friendly.

#pragma once

#include "imgui.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

namespace fb {

namespace fs = std::filesystem;

class FileBrowser {
public:
    // open the modal, starting from the directory of `start` (or its parent if
    // `start` is a file). Falls back to the current working directory.
    void Open(const std::string & start) {
        std::error_code ec;
        fs::path p(start);
        if (!start.empty() && fs::is_directory(p, ec)) {
            cwd_ = p.string();
        } else if (!start.empty() && p.has_parent_path() && fs::is_directory(p.parent_path(), ec)) {
            cwd_ = p.parent_path().string();
        } else {
            cwd_ = fs::current_path(ec).string();
        }
        selected_.clear();
        request_open_ = true;
    }

    // draw the modal; returns true once the user picks a file (path -> out).
    bool Draw(const char * id, std::string & out) {
        bool chosen = false;

        if (request_open_) {
            ImGui::OpenPopup(id);
            request_open_ = false;
        }

        ImGui::SetNextWindowSize(ImVec2(640, 460), ImGuiCond_FirstUseEver);
        if (!ImGui::BeginPopupModal(id, nullptr)) {
            return false;
        }

        // editable path bar - paste a directory and press Enter to jump there
        char buf[1024];
        std::snprintf(buf, sizeof(buf), "%s", cwd_.c_str());
        ImGui::SetNextItemWidth(-1);
        if (ImGui::InputText("##path", buf, sizeof(buf), ImGuiInputTextFlags_EnterReturnsTrue)) {
            std::error_code ec;
            if (fs::is_directory(buf, ec)) {
                cwd_ = buf;
                selected_.clear();
            }
        }

        ImGui::BeginChild("list", ImVec2(0, -ImGui::GetFrameHeightWithSpacing() * 2.0f), true);

        // parent directory
        if (ImGui::Selectable("../", false)) {
            fs::path parent = fs::path(cwd_).parent_path();
            if (!parent.empty()) {
                cwd_ = parent.string();
                selected_.clear();
            }
        }

        // collect entries (dirs first, then files), tolerating permission errors
        std::vector<fs::path> dirs, files;
        std::error_code ec;
        for (fs::directory_iterator it(cwd_, fs::directory_options::skip_permission_denied, ec), end;
             it != end; it.increment(ec)) {
            if (ec) break;
            std::error_code e2;
            if (it->is_directory(e2)) dirs.push_back(it->path());
            else                      files.push_back(it->path());
        }
        auto by_name = [](const fs::path & a, const fs::path & b) {
            return a.filename().string() < b.filename().string();
        };
        std::sort(dirs.begin(),  dirs.end(),  by_name);
        std::sort(files.begin(), files.end(), by_name);

        std::string next_cwd;
        for (const auto & d : dirs) {
            const std::string label = d.filename().string() + "/";
            if (ImGui::Selectable(label.c_str(), false, ImGuiSelectableFlags_AllowDoubleClick)) {
                if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                    next_cwd = d.string();
                }
            }
        }
        for (const auto & f : files) {
            const bool sel = (selected_ == f.string());
            if (ImGui::Selectable(f.filename().string().c_str(), sel, ImGuiSelectableFlags_AllowDoubleClick)) {
                selected_ = f.string();
                if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                    out = selected_;
                    chosen = true;
                    ImGui::CloseCurrentPopup();
                }
            }
        }
        if (!next_cwd.empty()) {
            cwd_ = next_cwd;
            selected_.clear();
        }

        ImGui::EndChild();

        ImGui::TextDisabled("%s", selected_.empty() ? "(double-click a file, or select and press Open)"
                                                     : selected_.c_str());

        if (selected_.empty()) ImGui::BeginDisabled();
        if (ImGui::Button("Open", ImVec2(100, 0))) {
            out = selected_;
            chosen = true;
            ImGui::CloseCurrentPopup();
        }
        if (selected_.empty()) ImGui::EndDisabled();
        ImGui::SameLine();
        if (ImGui::Button("Cancel", ImVec2(100, 0))) {
            ImGui::CloseCurrentPopup();
        }

        ImGui::EndPopup();
        return chosen;
    }

private:
    std::string cwd_;
    std::string selected_;
    bool        request_open_ = false;
};

} // namespace fb
