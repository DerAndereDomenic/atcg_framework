#pragma once

#include <Core/glm.h>
#include <Scene/ComponentGUIHandler.h>

namespace atcg
{
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

}    // namespace atcg