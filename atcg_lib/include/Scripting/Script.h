#pragma once

#include <Core/Memory.h>
#include <Core/Platform.h>

#include <filesystem>

#include <Scene/Entity.h>

namespace atcg
{
class Script
{
public:
    Script(const std::filesystem::path& file_path) : _file_path(file_path) {};

    virtual ~Script() {};

    virtual void init(const atcg::ref_ptr<atcg::Scene>& scene, const atcg::Entity& entity) = 0;

    virtual void onAttach() = 0;

    virtual void onUpdate(const float delta_time) = 0;

    virtual void reload() = 0;

    ATCG_INLINE const std::filesystem::path& getFilePath() const { return _file_path; }

protected:
    std::filesystem::path _file_path;
};

class PythonScript : public Script
{
public:
    PythonScript(const std::filesystem::path& file_path);

    virtual ~PythonScript();

    virtual void init(const atcg::ref_ptr<atcg::Scene>& scene, const atcg::Entity& entity) override;

    virtual void onAttach() override;

    virtual void onUpdate(const float delta_time) override;

    virtual void reload() override;

private:
    class Impl;
    std::unique_ptr<Impl> impl;
};
}    // namespace atcg