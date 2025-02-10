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

class Revision
{
public:
    Revision(const atcg::ref_ptr<atcg::Scene>& scene, atcg::Entity entity)
        : _scene(scene),
          _entity_handle(entity.entity_handle())
    {
    }

    virtual void rollback() = 0;

    virtual void apply() = 0;

    virtual void record_start_state() = 0;

    virtual void record_end_state() = 0;

protected:
    atcg::ref_ptr<Scene> _scene;
    uint32_t _entity_handle;
};

class RevisionSystem
{
public:
    RevisionSystem() = default;

    ~RevisionSystem() = default;

    ATCG_INLINE void pushRevision(const atcg::ref_ptr<Revision>& revision)
    {
        _rollback_stack.push(revision);
        _apply_stack = std::stack<atcg::ref_ptr<Revision>>();    // Clear the current stack
        ++_total_revisions;
    }

    ATCG_INLINE void apply()
    {
        if(_apply_stack.empty()) return;

        auto revision = _apply_stack.top();
        _apply_stack.pop();

        revision->apply();
        _rollback_stack.push(std::move(revision));
    }

    ATCG_INLINE void rollback()
    {
        if(_rollback_stack.empty()) return;

        auto revision = _rollback_stack.top();
        _rollback_stack.pop();

        revision->rollback();
        _apply_stack.push(std::move(revision));
    }

    template<typename RevisionType>
    ATCG_INLINE void startRecording(const atcg::ref_ptr<atcg::Scene>& scene, atcg::Entity entity)
    {
        ATCG_ASSERT(_current_revision == nullptr, "Can only have one revision at a time");

        _current_revision = atcg::make_ref<RevisionType>(scene, entity);
        _current_revision->record_start_state();
    }

    ATCG_INLINE void endRecording()
    {
        ATCG_ASSERT(_current_revision != nullptr, "Can only have one revision at a time");
        _current_revision->record_end_state();

        pushRevision(_current_revision);

        _current_revision = nullptr;
    }

    ATCG_INLINE uint32_t numUndos() const { return _rollback_stack.size(); }

    ATCG_INLINE uint32_t numRedos() const { return _apply_stack.size(); }

    ATCG_INLINE bool isRecording() const { return _current_revision != nullptr; }

    ATCG_INLINE uint32_t totalRevisions() const { return _total_revisions; }

    ATCG_INLINE void clearChache()
    {
        ATCG_ASSERT(_current_revision != nullptr, "Can't clear cache while recording");

        _rollback_stack = std::stack<atcg::ref_ptr<Revision>>();
        _apply_stack    = std::stack<atcg::ref_ptr<Revision>>();
    }

private:
    std::stack<atcg::ref_ptr<Revision>> _rollback_stack;
    std::stack<atcg::ref_ptr<Revision>> _apply_stack;
    atcg::ref_ptr<Revision> _current_revision = nullptr;
    uint32_t _total_revisions                 = 0;
};

class EntityAddedRevision : public Revision
{
public:
    EntityAddedRevision(const atcg::ref_ptr<atcg::Scene>& scene, atcg::Entity entity) : Revision(scene, entity) {}

    virtual void apply() override { auto entity = _scene->createEntity((entt::entity)_entity_handle, _uuid, _name); }

    virtual void rollback() override { _scene->removeEntity(atcg::Entity((entt::entity)_entity_handle, _scene.get())); }

    virtual void record_start_state() override
    {
        atcg::Entity entity((entt::entity)_entity_handle, _scene.get());
        _uuid = entity.getComponent<IDComponent>().ID;
        _name = entity.getComponent<NameComponent>().name;
    }

    virtual void record_end_state() override {}

private:
    UUID _uuid;
    std::string _name;
};

class EntityRemovedRevision : public Revision
{
public:
    EntityRemovedRevision(const atcg::ref_ptr<atcg::Scene>& scene, atcg::Entity entity) : Revision(scene, entity) {}

    virtual void apply() override
    {
        atcg::Entity entity((entt::entity)_entity_handle, _scene.get());
        _scene->removeEntity(entity);
    }

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
                          PointLightComponent>(entity);
    }

    virtual void record_start_state() override
    {
        atcg::Entity entity((entt::entity)_entity_handle, _scene.get());
        _uuid = entity.getComponent<IDComponent>().ID;
        _name = entity.getComponent<NameComponent>().name;

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
                        PointLightComponent>(entity);
    }

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

template<typename Component>
class ComponentAddedRevision : public Revision
{
public:
    ComponentAddedRevision(const atcg::ref_ptr<atcg::Scene>& scene, atcg::Entity entity) : Revision(scene, entity) {}

    virtual void apply() override
    {
        atcg::Entity entity((entt::entity)_entity_handle, _scene.get());
        entity.addComponent<Component>(_new_component);
    }

    virtual void rollback() override
    {
        atcg::Entity entity((entt::entity)_entity_handle, _scene.get());
        entity.removeComponent<Component>();
    }

    virtual void record_start_state() override {}

    virtual void record_end_state() override
    {
        atcg::Entity entity((entt::entity)_entity_handle, _scene.get());
        _new_component = entity.getComponent<Component>();
    }

private:
    Component _new_component;
};

template<typename Component>
class ComponentRemovedRevision : public Revision
{
public:
    ComponentRemovedRevision(const atcg::ref_ptr<atcg::Scene>& scene, atcg::Entity entity) : Revision(scene, entity) {}

    virtual void apply() override
    {
        atcg::Entity entity((entt::entity)_entity_handle, _scene.get());
        entity.removeComponent<Component>();
    }

    virtual void rollback() override
    {
        atcg::Entity entity((entt::entity)_entity_handle, _scene.get());
        entity.addOrReplaceComponent<Component>(_old_component);
    }

    virtual void record_start_state() override
    {
        atcg::Entity entity((entt::entity)_entity_handle, _scene.get());
        _old_component = entity.getComponent<Component>();
    }

    virtual void record_end_state() override {}

private:
    Component _old_component;
};

template<typename Component>
class ComponentEditedRevision : public Revision
{
public:
    ComponentEditedRevision(const atcg::ref_ptr<atcg::Scene>& scene, atcg::Entity entity) : Revision(scene, entity) {}

    virtual void apply() override
    {
        atcg::Entity entity((entt::entity)_entity_handle, _scene.get());
        entity.replaceComponent<Component>(_new_component);
    }

    virtual void rollback() override
    {
        atcg::Entity entity((entt::entity)_entity_handle, _scene.get());
        if(!entity.hasComponent<Component>()) return;
        entity.replaceComponent<Component>(_old_component);
    }

    virtual void record_start_state() override
    {
        atcg::Entity entity((entt::entity)_entity_handle, _scene.get());
        if(!entity.hasComponent<Component>()) return;
        _old_component = entity.getComponent<Component>();
    }

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

template<typename RevisionType1, typename RevisionType2>
class UnionRevision : public Revision
{
public:
    UnionRevision(const atcg::ref_ptr<atcg::Scene>& scene, atcg::Entity entity) : Revision(scene, entity)
    {
        _revision1 = atcg::make_ref<RevisionType1>(scene, entity);
        _revision2 = atcg::make_ref<RevisionType2>(scene, entity);
    }

    virtual void apply() override
    {
        _revision1->apply();
        _revision2->apply();
    }

    virtual void rollback() override
    {
        _revision1->rollback();
        _revision2->rollback();
    }

    virtual void record_start_state() override
    {
        _revision1->record_start_state();
        _revision2->record_start_state();
    }

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
template<typename RevisionType>
ATCG_INLINE void startRecording(const atcg::ref_ptr<atcg::Scene>& scene, atcg::Entity entity)
{
    RevisionSystem* system = atcg::SystemRegistry::instance()->getSystem<atcg::RevisionSystem>();
    system->startRecording<RevisionType>(scene, entity);
}

ATCG_INLINE void endRecording()
{
    RevisionSystem* system = atcg::SystemRegistry::instance()->getSystem<atcg::RevisionSystem>();
    system->endRecording();
}

ATCG_INLINE void rollback()
{
    RevisionSystem* system = atcg::SystemRegistry::instance()->getSystem<atcg::RevisionSystem>();
    system->rollback();
}

ATCG_INLINE uint32_t numUndos()
{
    RevisionSystem* system = atcg::SystemRegistry::instance()->getSystem<atcg::RevisionSystem>();
    return system->numUndos();
}

ATCG_INLINE uint32_t numRedos()
{
    RevisionSystem* system = atcg::SystemRegistry::instance()->getSystem<atcg::RevisionSystem>();
    return system->numRedos();
}

ATCG_INLINE bool isRecording()
{
    RevisionSystem* system = atcg::SystemRegistry::instance()->getSystem<atcg::RevisionSystem>();
    return system->isRecording();
}

ATCG_INLINE uint32_t totalRevisions()
{
    RevisionSystem* system = atcg::SystemRegistry::instance()->getSystem<atcg::RevisionSystem>();
    return system->totalRevisions();
}

ATCG_INLINE void clearChache()
{
    RevisionSystem* system = atcg::SystemRegistry::instance()->getSystem<atcg::RevisionSystem>();
    system->clearChache();
}
}    // namespace RevisionStack

}    // namespace atcg