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
    Script() = default;

    virtual ~Script() {};

    virtual void init(const std::filesystem::path& file_path) = 0;

    virtual void onAttach(const atcg::ref_ptr<atcg::Scene>& scene, const atcg::Entity& entity) = 0;

    virtual void
    onUpdate(const float delta_time, const atcg::ref_ptr<atcg::Scene>& scene, const atcg::Entity& entity) = 0;

    virtual void reload() = 0;

private:
};

class PythonScript : public Script
{
public:
    PythonScript();

    virtual ~PythonScript();

    virtual void init(const std::filesystem::path& file_path) override;

    virtual void onAttach(const atcg::ref_ptr<atcg::Scene>& scene, const atcg::Entity& entity) override;

    virtual void
    onUpdate(const float delta_time, const atcg::ref_ptr<atcg::Scene>& scene, const atcg::Entity& entity) override;

    virtual void reload() override;

private:
    class Impl;
    std::unique_ptr<Impl> impl;
};
}    // namespace atcg