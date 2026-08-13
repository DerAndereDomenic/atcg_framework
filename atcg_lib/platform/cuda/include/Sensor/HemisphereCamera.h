#pragma once

#include <Sensor/Sensor.h>
#include <Sensor/HemisphereCameraData.cuh>

namespace atcg
{
class ATCG_API HemisphereCamera : public Sensor
{
public:
    /**
     * @brief Constructor from a dictionary.
     * The dictionary expects
     * - position: glm::vec3
     * - normal: glm::vec3
     * - film: atcg::ref_ptr<Film>
     *
     * @param dict The sensor data
     */
    HemisphereCamera(const Dictionary& dict);

    /**
     * @brief Destructor
     */
    virtual ~HemisphereCamera();

    // TODO
    virtual void updateData() override {}

    /**
     * @brief Initialize the pipeline
     *
     * @param pipeline The raytracing pipeline
     * @param sbt The shader binding table
     */
    virtual void initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                    const atcg::ref_ptr<ShaderBindingTable>& sbt) override;

    /**
     * @brief Mark the sensor as dirty (e.g. camera changed)
     */
    virtual void markDirty() override;


private:
    glm::vec3 _position;
    glm::vec3 _normal;
    float _exposure = 1.0f;
    atcg::dref_ptr<HemisphereCameraData> _hemisphere_camera_data;
};

}    // namespace atcg