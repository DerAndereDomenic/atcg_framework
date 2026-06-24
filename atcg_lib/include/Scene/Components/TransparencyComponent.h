#pragma once

#include <Scene/ComponentSerializer.h>
#include <Scene/ComponentGUIHandler.h>

namespace atcg
{
struct ATCG_API TransparencyComponent
{
    TransparencyComponent() = default;

    TransparencyComponent(bool transparent) : transparent(transparent) {}

    bool transparent = true;

    static ATCG_CONSTEXPR ATCG_INLINE const char* toString() { return "Transparency"; }
};

namespace Serialization
{
ATCG_DECLARE_COMPONENT_SERIALIZER(TransparencyComponent);
}

namespace GUI
{
ATCG_DECLARE_COMPONENT_GUI_RENDERER(TransparencyComponent);
}
}    // namespace atcg