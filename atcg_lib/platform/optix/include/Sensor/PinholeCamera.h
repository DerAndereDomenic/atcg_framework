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
     * @brief Mark the sensor as dirty (e.g. camera changed)
     */
    virtual void markDirty() override;

    /**
     * @brief Get the data buffer
     *
     * @return The data buffer
     */
    ATCG_INLINE atcg::dref_ptr<PinholeCameraData> getDataBuffer() const { return _pinhole_camera_data; }

private:
    atcg::dref_ptr<PinholeCameraData> _pinhole_camera_data;
};
ATCG_DECLARE_COMPONENT_PIPELINE_INITIALIZER(PinholeCamera);

}    // namespace atcg