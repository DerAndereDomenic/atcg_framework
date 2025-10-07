#pragma once

#include <Medium/Medium.h>
#include <Medium/HomogeneousMediumData.cuh>

#include <Scene/ComponentGUIHandler.h>
#include <Scene/ComponentSerializer.h>

namespace atcg
{
class HomogeneousMedium : public Medium
{
public:
    /**
     * @brief Constructor
     * Inputs:
     * - "sigma_s" : vec3
     * - "sigma_a" : vec3
     * - "phase_func" : atcg::ref_ptr<PhaseFunction>
     *
     * @param dict The input parameters
     */
    HomogeneousMedium(const atcg::Dictionary& dict);

    /**
     * @brief Destructor
     */
    virtual ~HomogeneousMedium();

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

    /**
     * @brief A callback to display debug information in imgui
     */
    virtual void onImGuiRender() override {}

private:
    atcg::dref_ptr<HomogeneousMediumData> _data_buffer;
};

struct HomogeneousMediumComponent
{
    HomogeneousMediumComponent() = default;

    glm::vec3 albedo   = glm::vec3(0);
    float density      = 0;
    float g            = 0.0f;
    float Le           = 0.0f;
    glm::vec3 Le_color = glm::vec3(1);

    static ATCG_CONSTEXPR ATCG_INLINE const char* toString() { return "Homogeneous Medium"; }
};

namespace GUI
{
ATCG_DECLARE_COMPONENT_GUI_RENDERER(HomogeneousMediumComponent);
}
namespace Serialization
{
ATCG_DECLARE_COMPONENT_SERIALIZER(HomogeneousMediumComponent);
}

}    // namespace atcg