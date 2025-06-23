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
     * @brief Initialize a pipeline.
     * This function should be overwritten by each child class and it should add its functions to the pipeline and the
     * sbt.
     *
     * @param pipeline The pipeline
     * @param sbt The shader binding table
     */
    virtual void initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                    const atcg::ref_ptr<ShaderBindingTable>& sbt) = 0;

    /**
     * @brief Get the VPtrTable
     *
     * @return The VPtrTable
     */
    inline const EmitterVPtrTable* getVPtrTable() const { return _vptr_table.get(); }

protected:
    atcg::dref_ptr<EmitterVPtrTable> _vptr_table;
    EmitterFlags _flags;
};
}    // namespace atcg