#pragma once


#include <Core/UUID.h>
#include <Core/Memory.h>

#include <Renderer/Camera.h>
#include <Renderer/Texture.h>
#include <Renderer/RenderGraph.h>
#include <DataStructure/Dictionary.h>
#include <DataStructure/Image.h>
#include <DataStructure/Skybox.h>

#include <memory>
#include <entt.hpp>
namespace atcg
{
class Entity;

/**
 * @brief A class to model a scene.
 */
class Scene
{
public:
    /**
     * @brief Constructor
     */
    Scene();

    /**
     * @brief Destructor
     */
    ~Scene();

    /**
     * @brief Create a new entity that is part of this scene.
     *
     * @param name The name of the entity
     *
     * @return The entity
     */
    Entity createEntity(const std::string& name = "Entity");

    /**
     * @brief Create a new entity that is part of this scene using a handle.
     * The handle is a hint that is used if no other entity with this handle already exists
     *
     * @param handle The entity handle
     * @param uuid The uuid
     * @param name The name of the entity
     *
     * @return The entity
     */
    Entity createEntity(const entt::entity handle, UUID uuid, const std::string& name);

    /**
     * @brief Get an entity by its UUID.
     * If no entity exists with that ID, an invalid entity is returned
     *
     * @param id The UUID of the entity
     * @return The entity
     */
    Entity getEntityByID(UUID id) const;

    /**
     * @brief Get entities with a given name.
     *
     * @param name The name to search
     *
     * @return Entitites with the given name
     */
    std::vector<Entity> getEntitiesByName(const std::string& name);

    /**
     * @brief Get a view of all the entities with the requested components
     *
     * @tparam Components The Components that each entity should have
     *
     * @return A view onto the entities
     */
    template<typename... Components>
    auto getAllEntitiesWith() const
    {
        return _registry.view<Components...>();
    }

    /**
     * @brief Removes an entity from the scene
     *
     * @param id The id of the entity
     */
    void removeEntity(UUID id);

    /**
     * @brief Removes an entity from the scene
     *
     * @param entity The entity to remove
     */
    void removeEntity(Entity entity);

    /**
     * @brief Remove all entities
     */
    void removeAllEntites();

    /**
     * @brief Register the main camera of the scene
     *
     * @param camera The camera to register
     */
    void setCamera(const atcg::ref_ptr<atcg::Camera>& camera);

    /**
     * @brief Get the main camera of the scene
     *
     * @return The registered camera. This is nullptr if no camera was set before
     */
    atcg::ref_ptr<atcg::Camera> getCamera() const;

    /**
     * @brief Remove the registered main camera
     */
    void removeCamera();

    /**
     * @brief Render the scene
     * The only manditory entry in the context is the "camera" key which should hold an atcg::ref_ptr<Camera>. If this
     * is not given and the scene has a camera via setCamera, this camera is used instead. Otherwise, this is a NOP. The
     * Dictionary will be changed by this method, in particular it will add the "scene" key with a pointer to this and
     * "has_skybox" if the scene is equipped with a skybox.
     *
     * @param context The context (mutable, will be changed)
     */
    void draw(Dictionary& context);

    /**
     * @brief Render the scene
     * A shortcut if the only thing inside the context is the camera.
     *
     * @param camera The camera
     */
    void draw(const atcg::ref_ptr<Camera>& camera);

    /**
     * @brief Set a skybox
     *
     * @param skybox An equirectangular representation of the skybox
     */
    void setSkybox(const atcg::ref_ptr<Image>& skybox);

    /**
     * @brief Set a skybox
     *
     * @param skybox An equirectangular representation of the skybox
     */
    void setSkybox(const atcg::ref_ptr<Texture2D>& skybox);

    /**
     * @brief If a skybox is set.
     *
     * @return True if there is a skybox set.
     */
    bool hasSkybox() const;

    /**
     * @brief Remove the skybox
     */
    void removeSkybox();

    /**
     * @brief Return the equirectangular skybox texture
     *
     * @return A pointer to the texture (only is valid if hasSkybox() == true)
     */
    atcg::ref_ptr<Texture2D> getSkyboxTexture() const;

    /**
     * @brief Get the cube map of the skybox
     *
     * @return The skybox cubemap
     */
    atcg::ref_ptr<TextureCube> getSkyboxCubemap() const;

    /**
     * @brief Get the skybox
     *
     * @return The skybox
     */
    atcg::ref_ptr<Skybox> getSkybox() const;

    /**
     * @brief Set a new render graph.
     * The graph needs to be compiled
     *
     * @param graph The new graph
     */
    void setRenderGraph(const atcg::ref_ptr<RenderGraph>& graph);

    /**
     * @brief Get the rendergraph
     *
     * @return The render graph
     */
    atcg::ref_ptr<RenderGraph> getRenderGraph() const;

private:
    void _updateEntityID(atcg::Entity entity, const UUID old_id, const UUID new_id);
    void _updateEntityName(atcg::Entity entity, const std::string& old_name, const std::string& new_name);

private:
    friend class Entity;
    entt::registry _registry;

    class Impl;
    std::unique_ptr<Impl> impl;
};
}    // namespace atcg