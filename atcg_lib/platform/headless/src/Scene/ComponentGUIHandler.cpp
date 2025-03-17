#include <Scene/ComponentGUIHandler.h>

namespace atcg
{
template<typename T>
void ComponentGUIHandler::draw_component(Entity entity, T& component)
{
}

template<>
void ComponentGUIHandler::draw_component<TransformComponent>(Entity entity, TransformComponent& transform)
{
}

template<>
void ComponentGUIHandler::draw_component<CameraComponent>(Entity entity, CameraComponent& camera_component)
{
}

template<>
void ComponentGUIHandler::draw_component<GeometryComponent>(Entity entity, GeometryComponent& component)
{
}

template<>
void ComponentGUIHandler::draw_component<MeshRenderComponent>(Entity entity, MeshRenderComponent& component)
{
}

template<>
void ComponentGUIHandler::draw_component<PointRenderComponent>(Entity entity, PointRenderComponent& component)
{
}

template<>
void ComponentGUIHandler::draw_component<PointSphereRenderComponent>(Entity entity,
                                                                     PointSphereRenderComponent& component)
{
}

template<>
void ComponentGUIHandler::draw_component<EdgeRenderComponent>(Entity entity, EdgeRenderComponent& component)
{
}

template<>
void ComponentGUIHandler::draw_component<EdgeCylinderRenderComponent>(Entity entity,
                                                                      EdgeCylinderRenderComponent& component)
{
}

template<>
void ComponentGUIHandler::draw_component<InstanceRenderComponent>(Entity entity, InstanceRenderComponent& component)
{
}

template<>
void ComponentGUIHandler::draw_component<PointLightComponent>(Entity entity, PointLightComponent& component)
{
}

bool ComponentGUIHandler::displayMaterial(const std::string& key, Material& material) {}
}    // namespace atcg