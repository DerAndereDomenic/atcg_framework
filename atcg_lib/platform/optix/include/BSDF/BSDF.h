#pragma once

#include <Core/Platform.h>
#include <DataStructure/Dictionary.h>
#include <Core/OptixComponent.h>
#include <BSDF/BSDFVPtrTable.cuh>

#ifndef __CUDACC__
    #include <Scene/ComponentGUIHandler.h>
#endif

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

#ifndef __CUDACC__
namespace GUI
{
ATCG_DECLARE_COMPONENT_GUI_RENDERER(BSDFComponent);
}    // namespace GUI
#endif
}    // namespace atcg