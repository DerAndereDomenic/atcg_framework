#pragma once

#include <Core/Memory.h>
#include <Core/Memory.h>
#include <Core/RaytracingComponent.h>
#include <DataStructure/Dictionary.h>
#include <Medium/PhaseFunctionVPtrTable.cuh>

#ifndef __CUDACC__
    #include <Scene/ComponentGUIHandler.h>
#endif

namespace atcg
{
class ATCG_API PhaseFunction : public RaytracingComponent
{
public:
    PhaseFunction() = default;

    PhaseFunction(const atcg::Dictionary& dict) {}

    virtual ~PhaseFunction() {}

    ATCG_INLINE const PhaseFunctionVPtrTable* getVPtrTable() const { return _vptr_table.get(); }

    /**
     * @brief A callback to display debug information in imgui
     */
    virtual void onImGuiRender() = 0;

protected:
    atcg::dref_ptr<PhaseFunctionVPtrTable> _vptr_table;
};

struct ATCG_API PhaseFunctionComponent
{
    PhaseFunctionComponent() = default;
    PhaseFunctionComponent(const atcg::ref_ptr<PhaseFunction>& phase_function) : phase_function(phase_function) {}

    atcg::ref_ptr<PhaseFunction> phase_function;

    static ATCG_CONSTEXPR ATCG_INLINE const char* toString() { return "PhaseFunctionComponent"; }
};

#ifndef __CUDACC__

namespace GUI
{
template<>
struct is_gui_addable<PhaseFunctionComponent> : std::false_type
{
};

ATCG_DECLARE_COMPONENT_GUI_RENDERER(PhaseFunctionComponent);
}    // namespace GUI

#endif
}    // namespace atcg