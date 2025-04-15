#pragma once

#include <Core/Memory.h>
#include <Scripting/Script.h>

namespace atcg
{
class ScriptEngine
{
public:
    ScriptEngine() = default;

    virtual ~ScriptEngine() {};

    virtual void init() = 0;

    virtual void destroy() = 0;
};

class PythonScriptEngine : public ScriptEngine
{
public:
    PythonScriptEngine();

    virtual ~PythonScriptEngine();

    virtual void init() override;

    virtual void destroy() override;

private:
    class Impl;
    std::unique_ptr<Impl> impl;
};
}    // namespace atcg