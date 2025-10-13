#pragma once

#include <Core/OptixComponent.h>
#include <DataStructure/Dictionary.h>
#include <Medium/PhaseFunction.h>
#include <Medium/MediumVPtrTable.cuh>
#include <Core/PipelineInitializer.h>
#include <Scene/ComponentGUIHandler.h>

namespace atcg
{
class Medium : public OptixComponent
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

    ATCG_INLINE atcg::dref_ptr<MediumVPtrTable> getVPtrTableHolder() const { return _vptr_table; }

protected:
    atcg::dref_ptr<MediumVPtrTable> _vptr_table;
    atcg::ref_ptr<PhaseFunction> _phase_function;
};

struct MediumComponent
{
    MediumComponent() = default;
    MediumComponent(const atcg::ref_ptr<Medium>& medium) : medium(medium) {}

    atcg::ref_ptr<Medium> medium;

    static ATCG_CONSTEXPR ATCG_INLINE const char* toString() { return "MediumComponent"; }
};

namespace GUI
{
template<>
struct ComponentGUIRenderer<MediumComponent>
{
    void draw_component(const atcg::ref_ptr<Scene>& scene, Entity entity, MediumComponent& component) const
    {
        component.medium->onImGuiRender();
    }
};
}    // namespace GUI
}    // namespace atcg