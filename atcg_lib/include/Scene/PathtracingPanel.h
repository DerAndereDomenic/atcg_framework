#pragma once

#include <Scene/Scene.h>
#include <Renderer/PerspectiveCamera.h>

namespace atcg
{

/**
 * @brief A panel to handle the pathtracing engine
 *
 */
class PathtracingPanel
{
public:
    /**
     * @brief Default constructor
     */
    PathtracingPanel() = default;

    /**
     * @brief Constructor
     *
     * @param scene The scene
     */
    PathtracingPanel(const atcg::ref_ptr<Scene>& scene);

    /**
     * @brief Render the panel
     * TODO: Keep camera as parameter here?
     * @param camera The camera
     */
    void renderPanel(const atcg::ref_ptr<PerspectiveCamera>& camera);

private:
    atcg::ref_ptr<Scene> _scene;
    bool _samples_mode      = true;
    float _rendering_time   = 10.0f;
    int _num_samples        = 1024;
    int _width              = 0;
    int _height             = 0;
    bool _use_viewport_size = true;
};
}    // namespace atcg