#pragma once

#include <Core/Memory.h>
#include <Core/OptixComponent.h>
#include <DataStructure/Dictionary.h>
#include <Medium/PhaseFunctionVPtrTable.cuh>

namespace atcg
{
class PhaseFunction : public OptixComponent
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

    ATCG_INLINE atcg::dref_ptr<PhaseFunctionVPtrTable> getVPtrTableHolder() const { return _vptr_table; }

protected:
    atcg::dref_ptr<PhaseFunctionVPtrTable> _vptr_table;
};
}    // namespace atcg