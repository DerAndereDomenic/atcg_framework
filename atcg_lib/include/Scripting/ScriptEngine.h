#pragma once

#include <Core/Memory.h>
#include <Core/UUID.h>
#include <Scripting/Script.h>

namespace atcg
{
class ScriptEngine
{
public:
    ScriptEngine() = default;

    virtual ~ScriptEngine() {};

    virtual void init() = 0;

    virtual UUID registerScript(const atcg::ref_ptr<atcg::Script>& script) = 0;

    virtual void unregisterScript(const UUID id) = 0;

    virtual atcg::ref_ptr<Script> getScript(const UUID id) = 0;

    virtual void reloadScripts() = 0;

    virtual void destroy() = 0;
};

class PythonScriptEngine : public ScriptEngine
{
public:
    PythonScriptEngine();

    virtual ~PythonScriptEngine();

    virtual void init() override;

    virtual UUID registerScript(const atcg::ref_ptr<atcg::Script>& script) override;

    virtual void unregisterScript(const UUID id) override;

    virtual atcg::ref_ptr<Script> getScript(const UUID id) override;

    virtual void reloadScripts() override;

    virtual void destroy() override;

private:
    class Impl;
    std::unique_ptr<Impl> impl;
};
}    // namespace atcg