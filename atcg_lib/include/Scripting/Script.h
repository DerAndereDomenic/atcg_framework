#pragma once

#include <Core/Platform.h>

namespace atcg
{
class Script
{
public:
    Script() = default;

    virtual ~Script() {};

    virtual void init() = 0;

    virtual void onAttach() = 0;

    virtual void onUpdate(const float delta_time) = 0;

    virtual void reload() = 0;

private:
};

class PythonScript : public Script
{
public:
    PythonScript();

    virtual ~PythonScript();

    virtual void init() override;

    virtual void onAttach() override;

    virtual void onUpdate(const float delta_time) override;

    virtual void reload() override;

private:
};
}    // namespace atcg