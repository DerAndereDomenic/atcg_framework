#pragma once

#include <Core/OptixComponent.h>
#include <DataStructure/Dictionary.h>
#include <Medium/PhaseFunction.h>
#include <Medium/MediumVPtrTable.cuh>

namespace atcg
{
class Medium : public OptixComponent
{
public:
    Medium() = default;

    Medium(const atcg::Dictionary& dict) {}

    virtual ~Medium() {}

    ATCG_INLINE const MediumVPtrTable* getVPtrTable() const { return _vptr_table.get(); }

    ATCG_INLINE const atcg::ref_ptr<PhaseFunction>& getPhaseFunction() const { return _phase_function; }

    virtual void initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                    const atcg::ref_ptr<ShaderBindingTable>& sbt) = 0;

private:
    atcg::dref_ptr<MediumVPtrTable> _vptr_table;
    atcg::ref_ptr<PhaseFunction> _phase_function;
};
}    // namespace atcg