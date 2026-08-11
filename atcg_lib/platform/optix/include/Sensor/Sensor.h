#pragma once

#include <Core/API.h>
#include <Core/Platform.h>
#include <Core/OptixComponent.h>
#include <Renderer/Camera.h>
#include <DataStructure/Dictionary.h>
#include <DataStructure/TorchUtils.h>
#include <Sensor/SensorVPtrTable.cuh>
#include <Film/Film.h>

namespace atcg
{
class ATCG_API Sensor : public RaytracingComponent
{
public:
    /**
     * @brief Constructor
     */
    Sensor() = default;

    /**
     * @brief Construct a sensor with arbitrary parameters
     *
     * @param dict Dictionary holding the parameters
     */
    Sensor(const atcg::Dictionary& dict) {}

    /**
     * @brief Destructor
     */
    virtual ~Sensor() {}

    /**
     * @brief A callback to display debug information in imgui
     */
    virtual void onImGuiRender() = 0;

    /**
     * @brief Mark the sensor as dirty (e.g. camera changed)
     */
    virtual void markDirty() = 0;

    /**
     * @brief Get the attached film
     *
     * @return The film
     */
    ATCG_INLINE atcg::ref_ptr<Film> getFilm() const { return _film; }

    /**
     * @brief Get the VPtrTable
     *
     * @return The VPtrTable
     */
    inline const SensorVPtrTable* getVPtrTable() const { return _vptr_table.get(); }

    /**
     * @brief Get the attached camera
     *
     * @return The camera
     */
    ATCG_INLINE atcg::ref_ptr<Camera> getCamera() const { return _camera; }

protected:
    atcg::ref_ptr<Film> _film;
    atcg::dref_ptr<SensorVPtrTable> _vptr_table;
    atcg::ref_ptr<Camera> _camera;
};
}    // namespace atcg