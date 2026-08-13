#pragma once

#include <Scene/Components.h>
#include <DataStructure/Dictionary.h>
#include <Emitter/Emitter.h>
#include <Emitter/PointEmitterData.cuh>

namespace atcg
{
/**
 * @brief A class to model a point emitter
 */
class ATCG_API PointEmitter : public Emitter
{
public:
    /**
     * @brief Constructor
     * -"position": glm::vec3
     * -"intensity": float
     * -"color": glm::vec3
     *
     * @param dict The parameters
     */
    PointEmitter(const Dictionary& dict);

    /**
     * @brief Destructor
     */
    virtual ~PointEmitter();

    // TODO
    virtual void updateData() override {}

    /**
     * @brief Initialize the pipeline
     * @param pipeline The raytracing pipeline
     * @param sbt The shader binding table
     */
    virtual void initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                    const atcg::ref_ptr<ShaderBindingTable>& sbt) override;

private:
    atcg::dref_ptr<PointEmitterData> _point_emitter_data;
};

}    // namespace atcg