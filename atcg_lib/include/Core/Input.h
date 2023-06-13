#pragma once

#include <Core/glm.h>

namespace atcg
{
/**
 * @brief A class used for event polling
 */
class Input
{
public:
    /**
     * @brief Check if a key was pressed
     *
     * @param key The keycode
     * @return true If the key was pressed
     * @return false If the key was not pressed
     */
    static bool isKeyPressed(const int32_t& key);

    /**
     * @brief Check if a mouse button is pressed
     *
     * @param button The button to check
     * @return true If the button is pressed
     * @return false If the button is not pressed
     */
    static bool isMouseButtonPressed(const int32_t& button);

    /**
     * @brief Get the Mouse Position object
     *
     * @return glm::vec2 The mouse position
     */
    static glm::vec2 getMousePosition();
};
}    // namespace atcg