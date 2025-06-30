#pragma once

#include <Core/Platform.h>
#include <Core/Memory.h>
#include <Core/SystemRegistry.h>
#include <Core/Assert.h>
#include <Scene/Scene.h>
#include <Scene/Entity.h>
#include <Scene/Components.h>

#include <stack>

namespace atcg
{

/**
 * @brief This class models a revision of a scene element
 */
class Revision
{
public:
    /**
     * @brief Constructor
     *
     * @param scene The scene
     * @param entity The entity
     */
    Revision(const atcg::ref_ptr<atcg::Scene>& scene, atcg::Entity entity)
        : _scene(scene),
          _entity_handle(entity.entity_handle())
    {
    }

    /**
     * @brief Callback that is called on a rollback
     */
    virtual void rollback() = 0;

    /**
     * @brief Callback that is called on a redo
     */
    virtual void apply() = 0;

    /**
     * @brief Function that is called at the start of a capture. This should store all relevant information before any
     * changes to the scene elements were made.
     */
    virtual void record_start_state() = 0;

    /**
     * @brief Function that is called at the end of a capture. This should store all relevant information after the
     * scene element was changed.
     */
    virtual void record_end_state() = 0;

protected:
    atcg::ref_ptr<Scene> _scene;
    uint32_t _entity_handle;
};

/**
 * @brief A class to model a revision system. This can be used to take snapshots of a scene and rollback to previous
 * versions. The default Revision System (atcg::RevisionStack) tracks the standard components of the engine (see
 * Components.h). It is used to implement Redo and Undo functionality in the editor.
 */
class RevisionSystem
{
public:
    /**
     * @brief Default constructor
     */
    RevisionSystem() = default;

    /**
     * @brief Default destructor
     */
    ~RevisionSystem() = default;

    /**
     * @brief Push a new revision
     *
     * @param revision
     */
    ATCG_INLINE void pushRevision(const atcg::ref_ptr<Revision>& revision)
    {
        _rollback_stack.push(revision);
        _apply_stack = std::stack<atcg::ref_ptr<Revision>>();    // Clear the current stack
        ++_total_revisions;
    }

    /**
     * @brief Applies a revision that was undone previously.
     */
    ATCG_INLINE void apply()
    {
        if(_apply_stack.empty()) return;

        auto revision = _apply_stack.top();
        _apply_stack.pop();

        revision->apply();
        _rollback_stack.push(std::move(revision));
    }

    /**
     * @brief Rolls back to the previous revision
     */
    ATCG_INLINE void rollback()
    {
        if(_rollback_stack.empty()) return;

        auto revision = _rollback_stack.top();
        _rollback_stack.pop();

        revision->rollback();
        _apply_stack.push(std::move(revision));
    }

    /**
     * @brief Start recording a revision.
     * Each revision that should be handled by the system has to be recorded. There can always be only one recording at
     * a time. The recording has to be stopped using RevisionSystem::endRecording before a new recording can be started.
     *
     * @tparam RevisionType The type of the revision that should be recorded
     * @param scene The scene
     * @param entity The entity
     */
    template<typename RevisionType>
    ATCG_INLINE void startRecording(const atcg::ref_ptr<atcg::Scene>& scene, atcg::Entity entity)
    {
        ATCG_ASSERT(_current_revision == nullptr, "Can only have one revision at a time");

        _current_revision = atcg::make_ref<RevisionType>(scene, entity);
        _current_revision->record_start_state();
    }

    /**
     * @brief End a recording.
     * This function should only be called after a recording was started using RevisionSystem::endRecording.
     */
    ATCG_INLINE void endRecording()
    {
        ATCG_ASSERT(_current_revision != nullptr, "Can only have one revision at a time");
        _current_revision->record_end_state();

        pushRevision(_current_revision);

        _current_revision = nullptr;
    }

    /**
     * @brief Get the current number of possible undos
     *
     * @return The number of undos
     */
    ATCG_INLINE uint32_t numUndos() const { return _rollback_stack.size(); }

    /**
     * @brief Get the current number of possible redos
     *
     * @return The number of redos
     */
    ATCG_INLINE uint32_t numRedos() const { return _apply_stack.size(); }

    /**
     * @brief If there is a recording currently taking place
     *
     * @return True if a recording was started and is not yet finished
     */
    ATCG_INLINE bool isRecording() const { return _current_revision != nullptr; }

    /**
     * @brief The total number of revisions that were recorded by the system.
     * This is not reset if the cache was cleared using RevisionSystem::clearCache.
     *
     * @return The number of revisions
     */
    ATCG_INLINE uint32_t totalRevisions() const { return _total_revisions; }

    /**
     * @brief Clears the cache of revisions.
     * This should only be called if there is no active recording
     */
    ATCG_INLINE void clearChache()
    {
        ATCG_ASSERT(_current_revision == nullptr, "Can't clear cache while recording");

        _rollback_stack = std::stack<atcg::ref_ptr<Revision>>();
        _apply_stack    = std::stack<atcg::ref_ptr<Revision>>();
    }

private:
    std::stack<atcg::ref_ptr<Revision>> _rollback_stack;
    std::stack<atcg::ref_ptr<Revision>> _apply_stack;
    atcg::ref_ptr<Revision> _current_revision = nullptr;
    uint32_t _total_revisions                 = 0;
};

/**
 * @brief Revision if an entity was added to the scene
 */
class EntityAddedRevision : public Revision
{
public:
    /**
     * @brief Constructor
     *
     * @param scene The scene
     * @param entity The entity
     */
    EntityAddedRevision(const atcg::ref_ptr<atcg::Scene>& scene, atcg::Entity entity) : Revision(scene, entity) {}

    /**
     * @brief Callback that is called on a redo
     */
    virtual void apply() override { auto entity = _scene->createEntity((entt::entity)_entity_handle, _uuid, _name); }

    /**
     * @brief Callback that is called on a rollback
     */
    virtual void rollback() override { _scene->removeEntity(atcg::Entity((entt::entity)_entity_handle, _scene.get())); }

    /**
     * @brief Function that is called at the start of a capture. This should store all relevant information before any
     * changes to the scene elements were made.
     */
    virtual void record_start_state() override
    {
        atcg::Entity entity((entt::entity)_entity_handle, _scene.get());
        _uuid = entity.getComponent<IDComponent>().ID();
        _name = entity.getComponent<NameComponent>().name();
    }

    /**
     * @brief Function that is called at the end of a capture. This should store all relevant information after the
     * scene element was changed.
     */
    virtual void record_end_state() override {}

private:
    UUID _uuid;
    std::string _name;
};

/**
 * @brief Revision if an entity was removed from the scene
 */
class EntityRemovedRevision : public Revision
{
public:
    /**
     * @brief Constructor
     *
     * @param scene The scene
     * @param entity The entity
     */
    EntityRemovedRevision(const atcg::ref_ptr<atcg::Scene>& scene, atcg::Entity entity) : Revision(scene, entity) {}

    /**
     * @brief Callback that is called on a redo
     */
    virtual void apply() override
    {
        atcg::Entity entity((entt::entity)_entity_handle, _scene.get());
        _scene->removeEntity(entity);
    }

    /**
     * @brief Callback that is called on a rollback
     */
    virtual void rollback() override
    {
        auto entity = _scene->createEntity((entt::entity)_entity_handle, _uuid, _name);
        restoreComponents<TransformComponent,
                          CameraComponent,
                          GeometryComponent,
                          AccelerationStructureComponent,
                          MeshRenderComponent,
                          PointRenderComponent,
                          PointSphereRenderComponent,
                          EdgeRenderComponent,
                          EdgeCylinderRenderComponent,
                          InstanceRenderComponent,
                          CustomRenderComponent,
                          PointLightComponent,
                          MeshLightComponent,
                          ScriptComponent>(entity);
    }

    /**
     * @brief Function that is called at the start of a capture. This should store all relevant information before any
     * changes to the scene elements were made.
     */
    virtual void record_start_state() override
    {
        atcg::Entity entity((entt::entity)_entity_handle, _scene.get());
        _uuid = entity.getComponent<IDComponent>().ID();
        _name = entity.getComponent<NameComponent>().name();

        storeComponents<TransformComponent,
                        CameraComponent,
                        GeometryComponent,
                        AccelerationStructureComponent,
                        MeshRenderComponent,
                        PointRenderComponent,
                        PointSphereRenderComponent,
                        EdgeRenderComponent,
                        EdgeCylinderRenderComponent,
                        InstanceRenderComponent,
                        CustomRenderComponent,
                        PointLightComponent,
                        MeshLightComponent,
                        ScriptComponent>(entity);
    }

    /**
     * @brief Function that is called at the end of a capture. This should store all relevant information after the
     * scene element was changed.
     */
    virtual void record_end_state() override {}

private:
    template<typename... Components>
    void storeComponents(atcg::Entity entity)
    {
        // Capture all components
        (
            [&]
            {
                if(entity.hasAnyComponent<Components>())
                {
                    _components[entt::type_hash<Components>::value()] =
                        std::make_shared<Components>(entity.getComponent<Components>());
                }
            }(),
            ...);
    }

    template<typename... Components>
    void restoreComponents(atcg::Entity entity)
    {
        // Restore components
        for(auto& [id, component]: _components)
        {
            (
                [&]
                {
                    if(id == entt::type_hash<Components>::value())
                    {
                        entity.addOrReplaceComponent<Components>(*std::static_pointer_cast<Components>(component));
                    }
                }(),
                ...);
        }
    }

private:
    UUID _uuid;
    std::string _name;
    std::unordered_map<entt::id_type, std::shared_ptr<void>> _components;
};

/**
 * @brief Revision if a component was added to an entity
 * @tparam Component The type of the component that was added
 */
template<typename Component>
class ComponentAddedRevision : public Revision
{
public:
    /**
     * @brief Constructor
     *
     * @param scene The scene
     * @param entity The entity
     */
    ComponentAddedRevision(const atcg::ref_ptr<atcg::Scene>& scene, atcg::Entity entity) : Revision(scene, entity) {}

    /**
     * @brief Callback that is called on a redo
     */
    virtual void apply() override
    {
        atcg::Entity entity((entt::entity)_entity_handle, _scene.get());
        entity.addComponent<Component>(_new_component);
    }

    /**
     * @brief Callback that is called on a rollback
     */
    virtual void rollback() override
    {
        atcg::Entity entity((entt::entity)_entity_handle, _scene.get());
        entity.removeComponent<Component>();
    }

    /**
     * @brief Function that is called at the start of a capture. This should store all relevant information before any
     * changes to the scene elements were made.
     */
    virtual void record_start_state() override {}

    /**
     * @brief Function that is called at the end of a capture. This should store all relevant information after the
     * scene element was changed.
     */
    virtual void record_end_state() override
    {
        atcg::Entity entity((entt::entity)_entity_handle, _scene.get());
        _new_component = entity.getComponent<Component>();
    }

private:
    Component _new_component;
};

/**
 * @brief Revision if a component was removed from the entity
 * @tparam Component The type of the component that was removed
 */
template<typename Component>
class ComponentRemovedRevision : public Revision
{
public:
    /**
     * @brief Constructor
     *
     * @param scene The scene
     * @param entity The entity
     */
    ComponentRemovedRevision(const atcg::ref_ptr<atcg::Scene>& scene, atcg::Entity entity) : Revision(scene, entity) {}

    /**
     * @brief Callback that is called on a redo
     */
    virtual void apply() override
    {
        atcg::Entity entity((entt::entity)_entity_handle, _scene.get());
        entity.removeComponent<Component>();
    }

    /**
     * @brief Callback that is called on a rollback
     */
    virtual void rollback() override
    {
        atcg::Entity entity((entt::entity)_entity_handle, _scene.get());
        entity.addOrReplaceComponent<Component>(_old_component);
    }

    /**
     * @brief Function that is called at the start of a capture. This should store all relevant information before any
     * changes to the scene elements were made.
     */
    virtual void record_start_state() override
    {
        atcg::Entity entity((entt::entity)_entity_handle, _scene.get());
        _old_component = entity.getComponent<Component>();
    }

    /**
     * @brief Function that is called at the end of a capture. This should store all relevant information after the
     * scene element was changed.
     */
    virtual void record_end_state() override {}

private:
    Component _old_component;
};

/**
 * @brief Revision if a component was updated
 * @tparam Component The type of the component that was edited
 */
template<typename Component>
class ComponentEditedRevision : public Revision
{
public:
    /**
     * @brief Constructor
     *
     * @param scene The scene
     * @param entity The entity
     */
    ComponentEditedRevision(const atcg::ref_ptr<atcg::Scene>& scene, atcg::Entity entity) : Revision(scene, entity) {}

    /**
     * @brief Callback that is called on a redo
     */
    virtual void apply() override
    {
        atcg::Entity entity((entt::entity)_entity_handle, _scene.get());
        entity.replaceComponent<Component>(_new_component);
    }
    /**
     * @brief Callback that is called on a rollback
     */
    virtual void rollback() override
    {
        atcg::Entity entity((entt::entity)_entity_handle, _scene.get());
        if(!entity.hasComponent<Component>()) return;
        entity.replaceComponent<Component>(_old_component);
    }

    /**
     * @brief Function that is called at the start of a capture. This should store all relevant information before any
     * changes to the scene elements were made.
     */
    virtual void record_start_state() override
    {
        atcg::Entity entity((entt::entity)_entity_handle, _scene.get());
        if(!entity.hasComponent<Component>()) return;
        _old_component = entity.getComponent<Component>();
    }

    /**
     * @brief Function that is called at the end of a capture. This should store all relevant information after the
     * scene element was changed.
     */
    virtual void record_end_state() override
    {
        atcg::Entity entity((entt::entity)_entity_handle, _scene.get());
        if(!entity.hasComponent<Component>()) return;
        _new_component = entity.getComponent<Component>();
    }

private:
    Component _old_component;
    Component _new_component;    // Not needed right now?
};

/**
 * @brief A tuple of two revisions.
 * This can be used if multiple components update in one revision step (for example if transform and geometry change by
 * normalization).
 *
 * @tparam RevisionType1 The revision type of the first revision
 * @tparam RevisionType2 The revision type of the second revision
 */
template<typename RevisionType1, typename RevisionType2>
class UnionRevision : public Revision
{
public:
    /**
     * @brief Constructor
     *
     * @param scene The scene
     * @param entity The entity
     */
    UnionRevision(const atcg::ref_ptr<atcg::Scene>& scene, atcg::Entity entity) : Revision(scene, entity)
    {
        _revision1 = atcg::make_ref<RevisionType1>(scene, entity);
        _revision2 = atcg::make_ref<RevisionType2>(scene, entity);
    }

    /**
     * @brief Callback that is called on a redo
     */
    virtual void apply() override
    {
        _revision1->apply();
        _revision2->apply();
    }
    /**
     * @brief Callback that is called on a rollback
     */
    virtual void rollback() override
    {
        _revision1->rollback();
        _revision2->rollback();
    }

    /**
     * @brief Function that is called at the start of a capture. This should store all relevant information before any
     * changes to the scene elements were made.
     */
    virtual void record_start_state() override
    {
        _revision1->record_start_state();
        _revision2->record_start_state();
    }

    /**
     * @brief Function that is called at the end of a capture. This should store all relevant information after the
     * scene element was changed.
     */
    virtual void record_end_state() override
    {
        _revision1->record_end_state();
        _revision2->record_end_state();
    }

private:
    atcg::ref_ptr<RevisionType1> _revision1;
    atcg::ref_ptr<RevisionType2> _revision2;
};

namespace RevisionStack
{
/**
 * @brief Start recording a revision.
 * Each revision that should be handled by the system has to be recorded. There can always be only one recording at
 * a time. The recording has to be stopped using RevisionSystem::endRecording before a new recording can be started.
 *
 * @tparam RevisionType The type of the revision that should be recorded
 * @param scene The scene
 * @param entity The entity
 */
template<typename RevisionType>
ATCG_INLINE void startRecording(const atcg::ref_ptr<atcg::Scene>& scene, atcg::Entity entity)
{
    RevisionSystem* system = atcg::SystemRegistry::instance()->getSystem<atcg::RevisionSystem>();
    system->startRecording<RevisionType>(scene, entity);
}

/**
 * @brief End a recording.
 * This function should only be called after a recording was started using RevisionSystem::endRecording.
 */
ATCG_INLINE void endRecording()
{
    RevisionSystem* system = atcg::SystemRegistry::instance()->getSystem<atcg::RevisionSystem>();
    system->endRecording();
}

/**
 * @brief Applies a revision that was undone previously.
 */
ATCG_INLINE void apply()
{
    RevisionSystem* system = atcg::SystemRegistry::instance()->getSystem<atcg::RevisionSystem>();
    system->apply();
}

/**
 * @brief Rolls back to the previous revision
 */
ATCG_INLINE void rollback()
{
    RevisionSystem* system = atcg::SystemRegistry::instance()->getSystem<atcg::RevisionSystem>();
    system->rollback();
}

/**
 * @brief Get the current number of possible undos
 *
 * @return The number of undos
 */
ATCG_INLINE uint32_t numUndos()
{
    RevisionSystem* system = atcg::SystemRegistry::instance()->getSystem<atcg::RevisionSystem>();
    return system->numUndos();
}

/**
 * @brief Get the current number of possible redos
 *
 * @return The number of redos
 */
ATCG_INLINE uint32_t numRedos()
{
    RevisionSystem* system = atcg::SystemRegistry::instance()->getSystem<atcg::RevisionSystem>();
    return system->numRedos();
}

/**
 * @brief If there is a recording currently taking place
 *
 * @return True if a recording was started and is not yet finished
 */
ATCG_INLINE bool isRecording()
{
    RevisionSystem* system = atcg::SystemRegistry::instance()->getSystem<atcg::RevisionSystem>();
    return system->isRecording();
}

/**
 * @brief The total number of revisions that were recorded by the system.
 * This is not reset if the cache was cleared using RevisionSystem::clearCache.
 *
 * @return The number of revisions
 */
ATCG_INLINE uint32_t totalRevisions()
{
    RevisionSystem* system = atcg::SystemRegistry::instance()->getSystem<atcg::RevisionSystem>();
    return system->totalRevisions();
}

/**
 * @brief Clears the cache of revisions.
 * This should only be called if there is no active recording
 */
ATCG_INLINE void clearChache()
{
    RevisionSystem* system = atcg::SystemRegistry::instance()->getSystem<atcg::RevisionSystem>();
    system->clearChache();
}
}    // namespace RevisionStack

}    // namespace atcg