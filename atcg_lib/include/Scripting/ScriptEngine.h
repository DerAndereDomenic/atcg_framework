#pragma once

#include <Core/Memory.h>
#include <Scripting/Script.h>

namespace atcg
{
/**
 * @brief The interface for scripting engines
 */
class ScriptEngine
{
public:
    /**
     * @brief Default constructor
     */
    ScriptEngine() = default;

    /**
     * @brief Destructor
     */
    virtual ~ScriptEngine() {};

    /**
     * @brief Initializes the engine
     */
    virtual void init() = 0;

    /**
     * @brief Destroys the engine. All scripts should be destroyed before this is called.
     */
    virtual void destroy() = 0;
};

/**
 * @brief A scripting engine for python scripts
 */
class PythonScriptEngine : public ScriptEngine
{
public:
    /**
     * @brief Constructor
     */
    PythonScriptEngine();

    /**
     * @brief Destructor
     */
    virtual ~PythonScriptEngine();

    /**
     * @brief Initializes the engine
     */
    virtual void init() override;

    /**
     * @brief Destroys the engine. All scripts should be destroyed before this is called.
     */
    virtual void destroy() override;

private:
    class Impl;
    std::unique_ptr<Impl> impl;
};
}    // namespace atcg