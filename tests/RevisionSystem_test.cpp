#include <gtest/gtest.h>
#include <Scene/RevisionStack.h>

TEST(RevisionStackTest, defaultStack)
{
    atcg::RevisionSystem revision_stack;

    EXPECT_EQ(revision_stack.isRecording(), false);
    EXPECT_EQ(revision_stack.numRedos(), 0);
    EXPECT_EQ(revision_stack.numUndos(), 0);
    EXPECT_EQ(revision_stack.totalRevisions(), 0);
}

TEST(RevisionStackTest, emptyRollbacks)
{
    atcg::RevisionSystem revision_stack;

    revision_stack.rollback();
    revision_stack.rollback();
    revision_stack.rollback();

    EXPECT_EQ(revision_stack.isRecording(), false);
    EXPECT_EQ(revision_stack.numRedos(), 0);
    EXPECT_EQ(revision_stack.numUndos(), 0);
    EXPECT_EQ(revision_stack.totalRevisions(), 0);
}

TEST(RevisionStackTest, emptyApplies)
{
    atcg::RevisionSystem revision_stack;

    revision_stack.apply();
    revision_stack.apply();
    revision_stack.apply();

    EXPECT_EQ(revision_stack.isRecording(), false);
    EXPECT_EQ(revision_stack.numRedos(), 0);
    EXPECT_EQ(revision_stack.numUndos(), 0);
    EXPECT_EQ(revision_stack.totalRevisions(), 0);
}

TEST(RevisionStackTest, emptyAppliesRollbacks)
{
    atcg::RevisionSystem revision_stack;

    revision_stack.apply();
    revision_stack.rollback();
    revision_stack.apply();
    revision_stack.apply();
    revision_stack.rollback();
    revision_stack.rollback();

    EXPECT_EQ(revision_stack.isRecording(), false);
    EXPECT_EQ(revision_stack.numRedos(), 0);
    EXPECT_EQ(revision_stack.numUndos(), 0);
    EXPECT_EQ(revision_stack.totalRevisions(), 0);
}

TEST(RevisionStackTest, entityAdded)
{
    atcg::RevisionSystem revision_stack;
    atcg::ref_ptr<atcg::Scene> scene = atcg::make_ref<atcg::Scene>();

    auto entity = scene->createEntity("Test");
    revision_stack.startRecording<atcg::EntityAddedRevision>(scene, entity);
    EXPECT_EQ(revision_stack.isRecording(), true);
    revision_stack.endRecording();

    EXPECT_EQ(revision_stack.isRecording(), false);
    EXPECT_EQ(revision_stack.numRedos(), 0);
    EXPECT_EQ(revision_stack.numUndos(), 1);
    EXPECT_EQ(revision_stack.totalRevisions(), 1);
}

TEST(RevisionStackTest, entityAddedRollback)
{
    atcg::RevisionSystem revision_stack;
    atcg::ref_ptr<atcg::Scene> scene = atcg::make_ref<atcg::Scene>();

    auto entity = scene->createEntity("Test");
    revision_stack.startRecording<atcg::EntityAddedRevision>(scene, entity);
    EXPECT_EQ(revision_stack.isRecording(), true);
    revision_stack.endRecording();

    EXPECT_EQ(revision_stack.isRecording(), false);
    EXPECT_EQ(revision_stack.numRedos(), 0);
    EXPECT_EQ(revision_stack.numUndos(), 1);
    EXPECT_EQ(revision_stack.totalRevisions(), 1);

    revision_stack.rollback();

    EXPECT_EQ(scene->getAllEntitiesWith<atcg::IDComponent>().size(), 0);
    EXPECT_EQ(revision_stack.isRecording(), false);
    EXPECT_EQ(revision_stack.numRedos(), 1);
    EXPECT_EQ(revision_stack.numUndos(), 0);
    EXPECT_EQ(revision_stack.totalRevisions(), 1);
}

TEST(RevisionStackTest, entityAddedRollbackApply)
{
    atcg::RevisionSystem revision_stack;
    atcg::ref_ptr<atcg::Scene> scene = atcg::make_ref<atcg::Scene>();

    auto entity = scene->createEntity("Test");
    revision_stack.startRecording<atcg::EntityAddedRevision>(scene, entity);
    EXPECT_EQ(revision_stack.isRecording(), true);
    revision_stack.endRecording();

    EXPECT_EQ(revision_stack.isRecording(), false);
    EXPECT_EQ(revision_stack.numRedos(), 0);
    EXPECT_EQ(revision_stack.numUndos(), 1);
    EXPECT_EQ(revision_stack.totalRevisions(), 1);

    revision_stack.rollback();

    EXPECT_EQ(scene->getAllEntitiesWith<atcg::IDComponent>().size(), 0);
    EXPECT_EQ(revision_stack.isRecording(), false);
    EXPECT_EQ(revision_stack.numRedos(), 1);
    EXPECT_EQ(revision_stack.numUndos(), 0);
    EXPECT_EQ(revision_stack.totalRevisions(), 1);
    EXPECT_EQ(entity.operator bool(), false);

    revision_stack.apply();

    EXPECT_EQ(scene->getAllEntitiesWith<atcg::IDComponent>().size(), 1);
    EXPECT_EQ(revision_stack.isRecording(), false);
    EXPECT_EQ(revision_stack.numRedos(), 0);
    EXPECT_EQ(revision_stack.numUndos(), 1);
    EXPECT_EQ(revision_stack.totalRevisions(), 1);
    EXPECT_EQ(entity.operator bool(), true);
}

TEST(RevisionStackTest, entityRemoved)
{
    atcg::RevisionSystem revision_stack;
    atcg::ref_ptr<atcg::Scene> scene = atcg::make_ref<atcg::Scene>();

    auto entity = scene->createEntity("Test");
    revision_stack.startRecording<atcg::EntityRemovedRevision>(scene, entity);
    EXPECT_EQ(revision_stack.isRecording(), true);
    scene->removeEntity(entity);
    revision_stack.endRecording();

    EXPECT_EQ(revision_stack.isRecording(), false);
    EXPECT_EQ(revision_stack.numRedos(), 0);
    EXPECT_EQ(revision_stack.numUndos(), 1);
    EXPECT_EQ(revision_stack.totalRevisions(), 1);
}

TEST(RevisionStackTest, entityRemovedRollback)
{
    atcg::RevisionSystem revision_stack;
    atcg::ref_ptr<atcg::Scene> scene = atcg::make_ref<atcg::Scene>();

    auto entity = scene->createEntity("Test");
    revision_stack.startRecording<atcg::EntityRemovedRevision>(scene, entity);
    EXPECT_EQ(revision_stack.isRecording(), true);
    scene->removeEntity(entity);
    revision_stack.endRecording();

    EXPECT_EQ(revision_stack.isRecording(), false);
    EXPECT_EQ(revision_stack.numRedos(), 0);
    EXPECT_EQ(revision_stack.numUndos(), 1);
    EXPECT_EQ(revision_stack.totalRevisions(), 1);

    revision_stack.rollback();

    EXPECT_EQ(scene->getAllEntitiesWith<atcg::IDComponent>().size(), 1);
    EXPECT_EQ(revision_stack.isRecording(), false);
    EXPECT_EQ(revision_stack.numRedos(), 1);
    EXPECT_EQ(revision_stack.numUndos(), 0);
    EXPECT_EQ(revision_stack.totalRevisions(), 1);
}

TEST(RevisionStackTest, entityRemovedRollbackApply)
{
    atcg::RevisionSystem revision_stack;
    atcg::ref_ptr<atcg::Scene> scene = atcg::make_ref<atcg::Scene>();

    auto entity = scene->createEntity("Test");
    revision_stack.startRecording<atcg::EntityRemovedRevision>(scene, entity);
    EXPECT_EQ(revision_stack.isRecording(), true);
    scene->removeEntity(entity);
    revision_stack.endRecording();

    EXPECT_EQ(revision_stack.isRecording(), false);
    EXPECT_EQ(revision_stack.numRedos(), 0);
    EXPECT_EQ(revision_stack.numUndos(), 1);
    EXPECT_EQ(revision_stack.totalRevisions(), 1);

    revision_stack.rollback();

    EXPECT_EQ(scene->getAllEntitiesWith<atcg::IDComponent>().size(), 1);
    EXPECT_EQ(revision_stack.isRecording(), false);
    EXPECT_EQ(revision_stack.numRedos(), 1);
    EXPECT_EQ(revision_stack.numUndos(), 0);
    EXPECT_EQ(revision_stack.totalRevisions(), 1);
    EXPECT_EQ(entity.operator bool(), true);

    revision_stack.apply();

    EXPECT_EQ(scene->getAllEntitiesWith<atcg::IDComponent>().size(), 0);
    EXPECT_EQ(revision_stack.isRecording(), false);
    EXPECT_EQ(revision_stack.numRedos(), 0);
    EXPECT_EQ(revision_stack.numUndos(), 1);
    EXPECT_EQ(revision_stack.totalRevisions(), 1);
    EXPECT_EQ(entity.operator bool(), false);
}

TEST(RevisionStackTest, entityRemovedRollbackApplyRestoreComponents)
{
    atcg::RevisionSystem revision_stack;
    atcg::ref_ptr<atcg::Scene> scene = atcg::make_ref<atcg::Scene>();

    auto entity = scene->createEntity("Test");
    entity.addComponent<atcg::TransformComponent>(glm::vec3(1, 2, 3));
    entity.addComponent<atcg::MeshRenderComponent>();
    entity.addComponent<atcg::GeometryComponent>();

    revision_stack.startRecording<atcg::EntityRemovedRevision>(scene, entity);
    EXPECT_EQ(revision_stack.isRecording(), true);
    scene->removeEntity(entity);
    revision_stack.endRecording();

    EXPECT_EQ(revision_stack.isRecording(), false);
    EXPECT_EQ(revision_stack.numRedos(), 0);
    EXPECT_EQ(revision_stack.numUndos(), 1);
    EXPECT_EQ(revision_stack.totalRevisions(), 1);

    revision_stack.rollback();

    EXPECT_EQ(scene->getAllEntitiesWith<atcg::IDComponent>().size(), 1);
    EXPECT_EQ(entity.hasComponent<atcg::TransformComponent>(), true);
    EXPECT_EQ(entity.hasComponent<atcg::MeshRenderComponent>(), true);
    EXPECT_EQ(entity.hasComponent<atcg::GeometryComponent>(), true);
    EXPECT_EQ(entity.getComponent<atcg::TransformComponent>().getPosition().x, 1.0f);
    EXPECT_EQ(revision_stack.isRecording(), false);
    EXPECT_EQ(revision_stack.numRedos(), 1);
    EXPECT_EQ(revision_stack.numUndos(), 0);
    EXPECT_EQ(revision_stack.totalRevisions(), 1);
    EXPECT_EQ(entity.operator bool(), true);

    revision_stack.apply();

    EXPECT_EQ(scene->getAllEntitiesWith<atcg::IDComponent>().size(), 0);
    EXPECT_EQ(revision_stack.isRecording(), false);
    EXPECT_EQ(revision_stack.numRedos(), 0);
    EXPECT_EQ(revision_stack.numUndos(), 1);
    EXPECT_EQ(revision_stack.totalRevisions(), 1);
    EXPECT_EQ(entity.operator bool(), false);
}

TEST(RevisionStackTest, componentAdded)
{
    atcg::RevisionSystem revision_stack;
    atcg::ref_ptr<atcg::Scene> scene = atcg::make_ref<atcg::Scene>();

    auto entity = scene->createEntity("Test");
    revision_stack.startRecording<atcg::ComponentAddedRevision<atcg::TransformComponent>>(scene, entity);
    EXPECT_EQ(revision_stack.isRecording(), true);
    entity.addComponent<atcg::TransformComponent>(glm::vec3(42.0f));
    revision_stack.endRecording();

    EXPECT_EQ(entity.hasComponent<atcg::TransformComponent>(), true);
    EXPECT_EQ(revision_stack.isRecording(), false);
    EXPECT_EQ(revision_stack.numRedos(), 0);
    EXPECT_EQ(revision_stack.numUndos(), 1);
    EXPECT_EQ(revision_stack.totalRevisions(), 1);
}

TEST(RevisionStackTest, componentAddedRollback)
{
    atcg::RevisionSystem revision_stack;
    atcg::ref_ptr<atcg::Scene> scene = atcg::make_ref<atcg::Scene>();

    auto entity = scene->createEntity("Test");
    revision_stack.startRecording<atcg::ComponentAddedRevision<atcg::TransformComponent>>(scene, entity);
    EXPECT_EQ(revision_stack.isRecording(), true);
    entity.addComponent<atcg::TransformComponent>(glm::vec3(42.0f));
    revision_stack.endRecording();

    EXPECT_EQ(revision_stack.isRecording(), false);
    EXPECT_EQ(revision_stack.numRedos(), 0);
    EXPECT_EQ(revision_stack.numUndos(), 1);
    EXPECT_EQ(revision_stack.totalRevisions(), 1);

    revision_stack.rollback();

    EXPECT_EQ(entity.hasComponent<atcg::TransformComponent>(), false);
    EXPECT_EQ(revision_stack.isRecording(), false);
    EXPECT_EQ(revision_stack.numRedos(), 1);
    EXPECT_EQ(revision_stack.numUndos(), 0);
    EXPECT_EQ(revision_stack.totalRevisions(), 1);
}

TEST(RevisionStackTest, componentAddedRollbackApply)
{
    atcg::RevisionSystem revision_stack;
    atcg::ref_ptr<atcg::Scene> scene = atcg::make_ref<atcg::Scene>();

    auto entity = scene->createEntity("Test");
    revision_stack.startRecording<atcg::ComponentAddedRevision<atcg::TransformComponent>>(scene, entity);
    EXPECT_EQ(revision_stack.isRecording(), true);
    entity.addComponent<atcg::TransformComponent>(glm::vec3(42.0f));
    revision_stack.endRecording();

    EXPECT_EQ(revision_stack.isRecording(), false);
    EXPECT_EQ(entity.hasComponent<atcg::TransformComponent>(), true);
    EXPECT_EQ(revision_stack.numRedos(), 0);
    EXPECT_EQ(revision_stack.numUndos(), 1);
    EXPECT_EQ(revision_stack.totalRevisions(), 1);

    revision_stack.rollback();

    EXPECT_EQ(entity.hasComponent<atcg::TransformComponent>(), false);
    EXPECT_EQ(revision_stack.isRecording(), false);
    EXPECT_EQ(revision_stack.numRedos(), 1);
    EXPECT_EQ(revision_stack.numUndos(), 0);
    EXPECT_EQ(revision_stack.totalRevisions(), 1);

    revision_stack.apply();

    EXPECT_EQ(entity.hasComponent<atcg::TransformComponent>(), true);
    EXPECT_EQ(entity.getComponent<atcg::TransformComponent>().getPosition().x, 42.0f);
    EXPECT_EQ(revision_stack.isRecording(), false);
    EXPECT_EQ(revision_stack.numRedos(), 0);
    EXPECT_EQ(revision_stack.numUndos(), 1);
    EXPECT_EQ(revision_stack.totalRevisions(), 1);
}

TEST(RevisionStackTest, componentRemoved)
{
    atcg::RevisionSystem revision_stack;
    atcg::ref_ptr<atcg::Scene> scene = atcg::make_ref<atcg::Scene>();

    auto entity = scene->createEntity("Test");
    entity.addComponent<atcg::TransformComponent>(glm::vec3(42.0f));
    revision_stack.startRecording<atcg::ComponentRemovedRevision<atcg::TransformComponent>>(scene, entity);
    EXPECT_EQ(revision_stack.isRecording(), true);
    entity.removeComponent<atcg::TransformComponent>();
    revision_stack.endRecording();

    EXPECT_EQ(entity.hasComponent<atcg::TransformComponent>(), false);
    EXPECT_EQ(revision_stack.isRecording(), false);
    EXPECT_EQ(revision_stack.numRedos(), 0);
    EXPECT_EQ(revision_stack.numUndos(), 1);
    EXPECT_EQ(revision_stack.totalRevisions(), 1);
}

TEST(RevisionStackTest, componentRemovedRollback)
{
    atcg::RevisionSystem revision_stack;
    atcg::ref_ptr<atcg::Scene> scene = atcg::make_ref<atcg::Scene>();

    auto entity = scene->createEntity("Test");
    entity.addComponent<atcg::TransformComponent>(glm::vec3(42.0f));
    revision_stack.startRecording<atcg::ComponentRemovedRevision<atcg::TransformComponent>>(scene, entity);
    EXPECT_EQ(revision_stack.isRecording(), true);
    entity.removeComponent<atcg::TransformComponent>();
    revision_stack.endRecording();

    EXPECT_EQ(entity.hasComponent<atcg::TransformComponent>(), false);
    EXPECT_EQ(revision_stack.isRecording(), false);
    EXPECT_EQ(revision_stack.numRedos(), 0);
    EXPECT_EQ(revision_stack.numUndos(), 1);
    EXPECT_EQ(revision_stack.totalRevisions(), 1);

    revision_stack.rollback();

    EXPECT_EQ(entity.hasComponent<atcg::TransformComponent>(), true);
    EXPECT_EQ(entity.getComponent<atcg::TransformComponent>().getPosition().x, 42.0f);
    EXPECT_EQ(revision_stack.isRecording(), false);
    EXPECT_EQ(revision_stack.numRedos(), 1);
    EXPECT_EQ(revision_stack.numUndos(), 0);
    EXPECT_EQ(revision_stack.totalRevisions(), 1);
}

TEST(RevisionStackTest, componentRemovedRollbackApply)
{
    atcg::RevisionSystem revision_stack;
    atcg::ref_ptr<atcg::Scene> scene = atcg::make_ref<atcg::Scene>();

    auto entity = scene->createEntity("Test");
    entity.addComponent<atcg::TransformComponent>(glm::vec3(42.0f));
    revision_stack.startRecording<atcg::ComponentRemovedRevision<atcg::TransformComponent>>(scene, entity);
    EXPECT_EQ(revision_stack.isRecording(), true);
    entity.removeComponent<atcg::TransformComponent>();
    revision_stack.endRecording();

    EXPECT_EQ(entity.hasComponent<atcg::TransformComponent>(), false);
    EXPECT_EQ(revision_stack.isRecording(), false);
    EXPECT_EQ(revision_stack.numRedos(), 0);
    EXPECT_EQ(revision_stack.numUndos(), 1);
    EXPECT_EQ(revision_stack.totalRevisions(), 1);

    revision_stack.rollback();

    EXPECT_EQ(entity.hasComponent<atcg::TransformComponent>(), true);
    EXPECT_EQ(entity.getComponent<atcg::TransformComponent>().getPosition().x, 42.0f);
    EXPECT_EQ(revision_stack.isRecording(), false);
    EXPECT_EQ(revision_stack.numRedos(), 1);
    EXPECT_EQ(revision_stack.numUndos(), 0);
    EXPECT_EQ(revision_stack.totalRevisions(), 1);

    revision_stack.apply();

    EXPECT_EQ(entity.hasComponent<atcg::TransformComponent>(), false);
    EXPECT_EQ(revision_stack.isRecording(), false);
    EXPECT_EQ(revision_stack.numRedos(), 0);
    EXPECT_EQ(revision_stack.numUndos(), 1);
    EXPECT_EQ(revision_stack.totalRevisions(), 1);
}

TEST(RevisionStackTest, componentEdited)
{
    atcg::RevisionSystem revision_stack;
    atcg::ref_ptr<atcg::Scene> scene = atcg::make_ref<atcg::Scene>();

    auto entity = scene->createEntity("Test");
    entity.addComponent<atcg::TransformComponent>(glm::vec3(42.0f));
    revision_stack.startRecording<atcg::ComponentEditedRevision<atcg::TransformComponent>>(scene, entity);
    EXPECT_EQ(revision_stack.isRecording(), true);
    entity.getComponent<atcg::TransformComponent>().setPosition(glm::vec3(787.0f));
    revision_stack.endRecording();

    EXPECT_EQ(entity.hasComponent<atcg::TransformComponent>(), true);
    EXPECT_EQ(revision_stack.isRecording(), false);
    EXPECT_EQ(revision_stack.numRedos(), 0);
    EXPECT_EQ(revision_stack.numUndos(), 1);
    EXPECT_EQ(revision_stack.totalRevisions(), 1);
}

TEST(RevisionStackTest, componentEditedRollback)
{
    atcg::RevisionSystem revision_stack;
    atcg::ref_ptr<atcg::Scene> scene = atcg::make_ref<atcg::Scene>();

    auto entity = scene->createEntity("Test");
    entity.addComponent<atcg::TransformComponent>(glm::vec3(42.0f));
    revision_stack.startRecording<atcg::ComponentEditedRevision<atcg::TransformComponent>>(scene, entity);
    EXPECT_EQ(revision_stack.isRecording(), true);
    entity.getComponent<atcg::TransformComponent>().setPosition(glm::vec3(787.0f));
    revision_stack.endRecording();

    EXPECT_EQ(entity.hasComponent<atcg::TransformComponent>(), true);
    EXPECT_EQ(revision_stack.isRecording(), false);
    EXPECT_EQ(revision_stack.numRedos(), 0);
    EXPECT_EQ(revision_stack.numUndos(), 1);
    EXPECT_EQ(revision_stack.totalRevisions(), 1);

    revision_stack.rollback();

    EXPECT_EQ(entity.hasComponent<atcg::TransformComponent>(), true);
    EXPECT_EQ(entity.getComponent<atcg::TransformComponent>().getPosition().x, 42.0f);
    EXPECT_EQ(revision_stack.isRecording(), false);
    EXPECT_EQ(revision_stack.numRedos(), 1);
    EXPECT_EQ(revision_stack.numUndos(), 0);
    EXPECT_EQ(revision_stack.totalRevisions(), 1);
}

TEST(RevisionStackTest, componentEditedRollbackApply)
{
    atcg::RevisionSystem revision_stack;
    atcg::ref_ptr<atcg::Scene> scene = atcg::make_ref<atcg::Scene>();

    auto entity = scene->createEntity("Test");
    entity.addComponent<atcg::TransformComponent>(glm::vec3(42.0f));
    revision_stack.startRecording<atcg::ComponentEditedRevision<atcg::TransformComponent>>(scene, entity);
    EXPECT_EQ(revision_stack.isRecording(), true);
    entity.getComponent<atcg::TransformComponent>().setPosition(glm::vec3(787.0f));
    revision_stack.endRecording();

    EXPECT_EQ(entity.hasComponent<atcg::TransformComponent>(), true);
    EXPECT_EQ(revision_stack.isRecording(), false);
    EXPECT_EQ(revision_stack.numRedos(), 0);
    EXPECT_EQ(revision_stack.numUndos(), 1);
    EXPECT_EQ(revision_stack.totalRevisions(), 1);

    revision_stack.rollback();

    EXPECT_EQ(entity.hasComponent<atcg::TransformComponent>(), true);
    EXPECT_EQ(entity.getComponent<atcg::TransformComponent>().getPosition().x, 42.0f);
    EXPECT_EQ(revision_stack.isRecording(), false);
    EXPECT_EQ(revision_stack.numRedos(), 1);
    EXPECT_EQ(revision_stack.numUndos(), 0);
    EXPECT_EQ(revision_stack.totalRevisions(), 1);

    revision_stack.apply();

    EXPECT_EQ(entity.hasComponent<atcg::TransformComponent>(), true);
    EXPECT_EQ(entity.getComponent<atcg::TransformComponent>().getPosition().x, 787.0f);
    EXPECT_EQ(revision_stack.isRecording(), false);
    EXPECT_EQ(revision_stack.numRedos(), 0);
    EXPECT_EQ(revision_stack.numUndos(), 1);
    EXPECT_EQ(revision_stack.totalRevisions(), 1);
}