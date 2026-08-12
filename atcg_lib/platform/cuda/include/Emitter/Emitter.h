#pragma once

#include <Core/API.h>
#include <Core/Platform.h>
#include <Core/RaytracingComponent.h>
#include <Emitter/EmitterVPtrTable.cuh>
#include <DataStructure/Dictionary.h>

#ifndef __CUDACC__
    #include <Scene/ComponentGUIHandler.h>
#endif

namespace atcg
{
/**
 * @brief A class to model an emitter
 */
class ATCG_API Emitter : public RaytracingComponent
{
public:
    /**
     * @brief Constructor
     */
    Emitter() = default;

    /**
     * @brief Create an emitter
     *
     * @param dict The parameters
     */
    Emitter(const atcg::Dictionary& dict) {}

    /**
     * @brief Destructor
     */
    virtual ~Emitter() {}

    /**
     * @brief A callback to display debug information in imgui
     */
    virtual void onImGuiRender() = 0;

    /**
     * @brief Get the VPtrTable
     *
     * @return The VPtrTable
     */
    inline const EmitterVPtrTable* getVPtrTable() const { return _vptr_table.get(); }

    /**
     * @brief Get the bsdf flags
     *
     * @return The flags
     */
    ATCG_INLINE const EmitterFlags& flags() const { return _flags; }

protected:
    atcg::dref_ptr<EmitterVPtrTable> _vptr_table;
    EmitterFlags _flags = EmitterFlags::None;
};

struct ATCG_API EmitterComponent
{
    EmitterComponent() = default;
    EmitterComponent(const atcg::ref_ptr<Emitter>& emitter) : emitter(emitter) {}

    atcg::ref_ptr<Emitter> emitter;

    static ATCG_CONSTEXPR ATCG_INLINE const char* toString() { return "EmitterComponent"; }
};

#ifndef __CUDACC__
namespace GUI
{
template<>
struct is_gui_addable<EmitterComponent> : std::false_type
{
};
ATCG_DECLARE_COMPONENT_GUI_RENDERER(EmitterComponent);
}    // namespace GUI
#endif
}    // namespace atcg