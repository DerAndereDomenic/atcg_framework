#pragma once

#include <Core/API.h>
#include <Core/Platform.h>
#include <Core/RaytracingComponent.h>
#include <Emitter/EmitterVPtrTable.cuh>
#include <DataStructure/Dictionary.h>

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
}    // namespace atcg