#pragma once

#include <Core/Platform.h>
#include <Core/OptixComponent.h>
#include <Emitter/EmitterVPtrTable.cuh>
#include <DataStructure/Dictionary.h>

namespace atcg
{
/**
 * @brief A class to model an emitter
 */
class Emitter : public OptixComponent
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
     * @brief Get the VPtrTable Holder
     *
     * @return The shared ptr that handles the memory of the VPtrTable
     */
    ATCG_INLINE atcg::dref_ptr<EmitterVPtrTable> getVPtrTableHolder() const { return _vptr_table; }

    /**
     * @brief Get the bsdf flags
     *
     * @return The flags
     */
    ATCG_INLINE const EmitterFlags& flags() const { return _flags; }

protected:
    atcg::dref_ptr<EmitterVPtrTable> _vptr_table;
    EmitterFlags _flags;
};
}    // namespace atcg