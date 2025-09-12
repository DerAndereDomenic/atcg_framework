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
     * @brief Initialize a pipeline.
     * This function should be overwritten by each child class and it should add its functions to the pipeline and the
     * sbt.
     *
     * @param pipeline The pipeline
     * @param sbt The shader binding table
     */
    virtual void initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                    const atcg::ref_ptr<ShaderBindingTable>& sbt) = 0;

protected:
    atcg::dref_ptr<MediumVPtrTable> _vptr_table;
    atcg::ref_ptr<PhaseFunction> _phase_function;
};
}    // namespace atcg