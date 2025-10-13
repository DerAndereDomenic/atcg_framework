#pragma once

#include <Core/Platform.h>
#include <DataStructure/Dictionary.h>
#include <Core/OptixComponent.h>
#include <BSDF/BSDFVPtrTable.cuh>
#include <Scene/ComponentGUIHandler.h>

namespace atcg
{
/**
 * @brief A class to model a BSDF
 */
class BSDF : public OptixComponent
{
public:
    /**
     * @brief Constructor
     */
    BSDF() = default;

    /**
     * @brief Construct a BSDF with arbitrary parameters
     *
     * @param dict Dictionary holding the parameters
     */
    BSDF(const atcg::Dictionary& dict) {}

    /**
     * @brief Destructor
     */
    virtual ~BSDF() {}

    /**
     * @brief A callback to display debug information in imgui
     */
    virtual void onImGuiRender() = 0;

    /**
     * @brief Get the VPtrTable
     *
     * @return The VPtrTable
     */
    ATCG_INLINE const BSDFVPtrTable* getVPtrTable() const { return _vptr_table.get(); }

    /**
     * @brief Get the VPtrTable Holder
     *
     * @return The shared ptr that handles the memory of the VPtrTable
     */
    ATCG_INLINE atcg::dref_ptr<BSDFVPtrTable> getVPtrTableHolder() const { return _vptr_table; }

    /**
     * @brief Get the bsdf flags
     *
     * @return The flags
     */
    ATCG_INLINE const BSDFComponentType& flags() const { return _flags; }

protected:
    atcg::dref_ptr<BSDFVPtrTable> _vptr_table;
    BSDFComponentType _flags;
};

struct BSDFComponent
{
    BSDFComponent() = default;
    BSDFComponent(const atcg::ref_ptr<BSDF>& bsdf) : bsdf(bsdf) {}

    atcg::ref_ptr<BSDF> bsdf;

    static ATCG_CONSTEXPR ATCG_INLINE const char* toString() { return "BSDFComponent"; }
};

namespace GUI
{
template<>
struct ComponentGUIRenderer<BSDFComponent>
{
    void draw_component(const atcg::ref_ptr<Scene>& scene, Entity entity, BSDFComponent& component) const
    {
        component.bsdf->onImGuiRender();
    }
};
}    // namespace GUI
}    // namespace atcg