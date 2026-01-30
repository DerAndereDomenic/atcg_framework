#pragma once

#include <Sensor/Sensor.h>
#include <Sensor/PinholeCameraData.cuh>

namespace atcg
{
class PinholeCamera : public Sensor
{
public:
    /**
     * @brief Constructor from a dictionary.
     * The dictionary expects
     * - camera: atcg::ref_ptr<Camera>
     * - film: atcg::ref_ptr<Film>
     *
     * @param dict The sensor data
     */
    PinholeCamera(const Dictionary& dict);

    /**
     * @brief Destructor
     */
    virtual ~PinholeCamera();

    /**
     * @brief A callback to display debug information in imgui
     */
    virtual void onImGuiRender() override;

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
    atcg::dref_ptr<PinholeCameraData> _pinhole_camera_data;
};

}    // namespace atcg