#pragma once

#include <Core/Platform.h>
#include <Shape/Shape.h>
#include <Shape/ShapeInstance.h>
#include <Shape/IAS.h>
#include <Emitter/EnvironmentEmitter.h>

#include <Scene/Scene.h>

namespace atcg
{

class SceneAdapter;

/**
 * @brief A class to model a scene with optix components
 */
class OptixScene : public Scene
{
public:
    /**
     * @brief Default constructor
     */
    OptixScene() : Scene() {};

    /**
     * @brief Get the shapes
     *
     * @return The shapes
     */
    ATCG_INLINE const std::vector<atcg::ref_ptr<ShapeInstance>>& getShapes() const { return _shapes; }

    /**
     * @brief Get the Environment Emitter VPtr
     *
     * @return The VPtr Tables
     */
    ATCG_INLINE const atcg::DeviceBuffer<const EmitterVPtrTable*>& getEmitterVPtrTables() const
    {
        return _emitter_vptr_tables;
    }

    /**
     * @brief Get the environment emitter
     *
     * @return The environment emitter
     */
    ATCG_INLINE const atcg::ref_ptr<EnvironmentEmitter>& getEnvironmentEmitter() const { return _environment_emitter; }

    /**
     * @brief Get the emitter
     *
     * @return The emitter
     */
    ATCG_INLINE const std::vector<atcg::ref_ptr<Emitter>>& getEmitter() const { return _emitter; }

    /**
     * @brief Get the IAS
     *
     * @return The IAS
     */
    ATCG_INLINE const atcg::ref_ptr<InstanceAccelerationStructure>& getIAS() const { return _ias; }

private:
    friend class SceneAdapter;

    std::vector<atcg::ref_ptr<ShapeInstance>> _shapes;

    atcg::DeviceBuffer<const EmitterVPtrTable*> _emitter_vptr_tables;
    atcg::ref_ptr<EnvironmentEmitter> _environment_emitter = nullptr;
    std::vector<atcg::ref_ptr<Emitter>> _emitter;

    atcg::ref_ptr<InstanceAccelerationStructure> _ias;
};
}    // namespace atcg