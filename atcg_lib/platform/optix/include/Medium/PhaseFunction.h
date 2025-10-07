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

    virtual void initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                    const atcg::ref_ptr<ShaderBindingTable>& sbt) = 0;

    /**
     * @brief A callback to display debug information in imgui
     */
    virtual void onImGuiRender() = 0;

protected:
    atcg::dref_ptr<PhaseFunctionVPtrTable> _vptr_table;
};
}    // namespace atcg