#pragma once

#include <Core/Layer.h>

#include <vector>

namespace atcg
{
/**
 * @brief This class handles the layers of the application
 */
class LayerStack
{
public:
    /**
     * @brief Construct a new Layer Stack object
     */
    LayerStack() = default;

    /**
     * @brief Destroy the Layer Stack object
     */
    ~LayerStack();

    /**
     * @brief Push a new layer to the layer stack
     *
     * @param layer The layer
     */
    void pushLayer(Layer* layer);

    /**
     * @brief Push an overlay to the stack.
     * Overlays always are on top of the whole stack
     *
     * @param layer The layer
     */
    void pushOverlay(Layer* layer);

    /**
     * @brief Remove a layer from the stack
     *
     * @param layer The layer
     */
    void popLayer(Layer* layer);

    /**
     * @brief Pop an overlay
     *
     * @param layer The overlay layer
     */
    void popOverlay(Layer* layer);

    // Iterators for easy access
    std::vector<Layer*>::iterator begin() { return _layers.begin(); }
    std::vector<Layer*>::iterator end() { return _layers.end(); }
    std::vector<Layer*>::reverse_iterator rbegin() { return _layers.rbegin(); }
    std::vector<Layer*>::reverse_iterator rend() { return _layers.rend(); }

    std::vector<Layer*>::const_iterator begin() const { return _layers.begin(); }
    std::vector<Layer*>::const_iterator end() const { return _layers.end(); }
    std::vector<Layer*>::const_reverse_iterator rbegin() const { return _layers.rbegin(); }
    std::vector<Layer*>::const_reverse_iterator rend() const { return _layers.rend(); }

private:
    std::vector<Layer*> _layers;
    unsigned int _layer_insert_index = 0;
};
}    // namespace atcg