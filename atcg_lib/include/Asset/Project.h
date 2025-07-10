#pragma once

#include <Core/Memory.h>

#include <filesystem>

namespace atcg
{

class Project
{
public:
    static atcg::ref_ptr<Project> create(const std::filesystem::path& path);

    static atcg::ref_ptr<Project> load(const std::filesystem::path& path);

    void save();

    static const atcg::ref_ptr<Project>& getActive();

    static void saveActive();

private:
    std::filesystem::path _project_path;

    inline static atcg::ref_ptr<Project> s_active_project = nullptr;
};
}    // namespace atcg