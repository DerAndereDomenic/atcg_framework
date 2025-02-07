#pragma once

#include <Core/Platform.h>
#include <Core/Memory.h>
#include <Core/SystemRegistry.h>
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

    void pushRevision(const atcg::ref_ptr<Revision>& revision)
    {
        _revisions.push(revision);
        ATCG_TRACE("New revision added");
    }

    void rollback()
    {
        if(_revisions.empty()) return;

        auto revision = _revisions.top();
        _revisions.pop();
        revision->rollback();

        ATCG_TRACE("Rollback");
    }

private:
    std::stack<atcg::ref_ptr<Revision>> _revisions;
};

template<typename RevisionType>
class RevisionRecorder
{
public:
    [[nodiscard]] RevisionRecorder(RevisionSystem* revision_system,
                                   const atcg::ref_ptr<atcg::Scene>& scene,
                                   atcg::Entity entity)
        : _revision_system(revision_system),
          _scene(scene),
          _entity(entity),
          _revision(atcg::make_ref<RevisionType>(scene, entity))
    {
        _revision->record_start_state();
    }

    ~RevisionRecorder()
    {
        _revision->record_end_state();
        _revision_system->pushRevision(_revision);
    }

private:
    RevisionSystem* _revision_system;
    atcg::ref_ptr<Scene> _scene;
    atcg::Entity _entity;
    atcg::ref_ptr<RevisionType> _revision;
};

// class EntityAddedRevision : public Revision
// {
// public:
//     EntityAddedRevision(const atcg::ref_ptr<atcg::Scene>& scene, atcg::Entity entity) : Revision(scene, entity) {}

//     virtual void rollback() override;

//     virtual void record_start_state() override;

//     virtual void record_end_state() override;
// };

// class EntityRemovedRevision : public Revision
// {
// public:
//     EntityRemovedRevision(const atcg::ref_ptr<atcg::Scene>& scene, atcg::Entity entity) : Revision(scene, entity) {}

//     virtual void rollback() override;

//     virtual void record_start_state() override;

//     virtual void record_end_state() override;
// };

template<typename Component>
class ComponentAddedRevision : public Revision
{
public:
    ComponentAddedRevision(const atcg::ref_ptr<atcg::Scene>& scene, atcg::Entity entity) : Revision(scene, entity) {}

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
ATCG_INLINE [[nodiscard]] RevisionRecorder<RevisionType> recordRevision(const atcg::ref_ptr<atcg::Scene>& scene,
                                                                        atcg::Entity entity)
{
    RevisionSystem* system = atcg::SystemRegistry::instance()->getSystem<atcg::RevisionSystem>();
    return RevisionRecorder<RevisionType>(system, scene, entity);
}

ATCG_INLINE void rollback()
{
    RevisionSystem* system = atcg::SystemRegistry::instance()->getSystem<atcg::RevisionSystem>();
    system->rollback();
}
}    // namespace RevisionStack

}    // namespace atcg