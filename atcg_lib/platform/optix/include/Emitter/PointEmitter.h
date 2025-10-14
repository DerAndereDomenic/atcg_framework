#pragma once

#include <Scene/Components.h>
#include <DataStructure/Dictionary.h>
#include <Emitter/Emitter.h>
#include <Emitter/PointEmitterData.cuh>
#include <Core/PipelineInitializer.h>

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

    /**
     * @brief A callback to display debug information in imgui
     */
    virtual void onImGuiRender() override {}

    ATCG_INLINE atcg::dref_ptr<PointEmitterData> getDataBuffer() const { return _point_emitter_data; }

private:
    atcg::dref_ptr<PointEmitterData> _point_emitter_data;
};

ATCG_DECLARE_COMPONENT_PIPELINE_INITIALIZER(PointEmitter);
}    // namespace atcg