#pragma once

#include <Scene/Components.h>
#include <Emitter/Emitter.h>
#include <Emitter/PointEmitterData.cuh>

namespace atcg
{
/**
 * @brief A class to model a point emitter
 */
class PointEmitter : public Emitter
{
public:
    /**
     * @brief Constructor
     * 
     * @param position The position
     * @param point_light The point light component
     */
    PointEmitter(const glm::vec3& position, const PointLightComponent& point_light);

    /**
     * @brief Destructor
     */
    ~PointEmitter();

    /**
     * @brief Initialize a pipeline.
     * This function should be overwritten by each child class and it should add its functions to the pipeline and the
     * sbt.
     *
     * @param pipeline The pipeline
     * @param sbt The shader binding table
     */
    virtual void initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                    const atcg::ref_ptr<ShaderBindingTable>& sbt) override;

private:
    atcg::dref_ptr<PointEmitterData> _point_emitter_data;
};
}    // namespace atcg