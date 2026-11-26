#pragma once

#ifndef PYBIND11_DETAILED_ERROR_MESSAGES
    #define PYBIND11_DETAILED_ERROR_MESSAGES
#endif

#include <pyatcg.h>

#include <functional>
#include <vector>

namespace atcg
{
namespace python
{
namespace py = pybind11;

/**
 * @brief Collects the python bindings of all areas in two phases.
 *
 * pybind11 writes the python names of all types into the signatures and docstrings when a function is defined. Every
 * area therefore declares its classes, enums and submodules immediately (phase 1) and defers the definitions of their
 * members with define() (phase 2). defineAll() runs the deferred definitions after all areas are declared.
 */
class BindingContext
{
public:
    /**
     * @brief Constructor
     *
     * @param module The python module
     */
    explicit BindingContext(const py::module_& module) : _module(module) {}

    /**
     * @brief Get the python module
     *
     * @return The module
     */
    ATCG_INLINE py::module_ module() const { return _module; }

    /**
     * @brief Defer a definition to phase 2. The definitions are run in the order they were registered.
     *
     * @param definition The definition
     */
    template<typename Func>
    ATCG_INLINE void define(Func&& definition)
    {
        _definitions.emplace_back(std::forward<Func>(definition));
    }

    /**
     * @brief Run all deferred definitions and release them
     */
    void defineAll();

private:
    py::module_ _module;
    std::vector<std::function<void()>> _definitions;
};

void bindCore(BindingContext& ctx);
void bindMath(BindingContext& ctx);
void bindAsset(BindingContext& ctx);
void bindDataStructure(BindingContext& ctx);
void bindRenderer(BindingContext& ctx);
void bindUtils(BindingContext& ctx);
void bindMaterial(BindingContext& ctx);
void bindComponents(BindingContext& ctx);
void bindScene(BindingContext& ctx);
void bindGUI(BindingContext& ctx);
void bindNetwork(BindingContext& ctx);
void bindScripting(BindingContext& ctx);
void bindPlugin(BindingContext& ctx);
#ifdef ATCG_CUDA_BACKEND
void bindOptix(BindingContext& ctx);
#endif
#ifndef ATCG_HEADLESS
void bindImGui(BindingContext& ctx);
#endif
}    // namespace python
}    // namespace atcg
