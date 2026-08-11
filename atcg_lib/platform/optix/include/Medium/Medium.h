#pragma once

#include <Core/Memory.h>
#include <Core/OptixComponent.h>
#include <DataStructure/Dictionary.h>
#include <Medium/PhaseFunction.h>
#include <Medium/MediumVPtrTable.cuh>

#ifndef __CUDACC__
    #include <Scene/ComponentGUIHandler.h>
#endif

namespace atcg
{
class ATCG_API Medium : public RaytracingComponent
{
public:
    /**
     * @brief Default Constructor
     */
    Medium() = default;

    /**
     * @brief Base Constructor
     *
     * @param dict The dictionary containing the parameter
     */
    Medium(const atcg::Dictionary& dict)
    {
        _phase_function = dict.getValueOr<atcg::ref_ptr<PhaseFunction>>("phase_func", nullptr);
    }

    /**
     * @brief Destructor
     */
    virtual ~Medium() {}

    /**
     * @brief Get the VPtr Table
     *
     * @return THe VPtr Table
     */
    ATCG_INLINE const MediumVPtrTable* getVPtrTable() const { return _vptr_table.get(); }

    /**
     * @brief Get the Phase Function
     *
     * @return The Phase Function
     */
    ATCG_INLINE const atcg::ref_ptr<PhaseFunction>& getPhaseFunction() const { return _phase_function; }

    /**
     * @brief A callback to display debug information in imgui
     */
    virtual void onImGuiRender() = 0;

protected:
    atcg::dref_ptr<MediumVPtrTable> _vptr_table;
    atcg::ref_ptr<PhaseFunction> _phase_function;
};

struct ATCG_API MediumComponent
{
    MediumComponent() = default;
    MediumComponent(const atcg::ref_ptr<Medium>& medium) : medium(medium) {}

    atcg::ref_ptr<Medium> medium;

    static ATCG_CONSTEXPR ATCG_INLINE const char* toString() { return "MediumComponent"; }
};

#ifndef __CUDACC__
namespace GUI
{
template<>
struct is_gui_addable<MediumComponent> : std::false_type
{
};
ATCG_DECLARE_COMPONENT_GUI_RENDERER(MediumComponent);
}    // namespace GUI
#endif
}    // namespace atcg