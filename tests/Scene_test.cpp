#include <gtest/gtest.h>
#include <Scene/Scene.h>
#include <Scene/Entity.h>
#include <Scene/Components.h>

TEST(SceneTest, emptyScene)
{
    auto scene = atcg::make_ref<atcg::Scene>();
}

TEST(SceneTest, createEntity)
{
    auto scene  = atcg::make_ref<atcg::Scene>();
    auto entity = scene->createEntity();
}

TEST(SceneTest, findEntityByID)
{
    auto scene  = atcg::make_ref<atcg::Scene>();
    auto entity = scene->createEntity("Test Entity");

    auto& id         = entity.getComponent<atcg::IDComponent>();
    auto entity_copy = scene->getEntityByID(id.ID);

    auto& other_id   = entity_copy.getComponent<atcg::IDComponent>();
    auto& other_name = entity_copy.getComponent<atcg::NameComponent>();

    EXPECT_EQ(other_id.ID, id.ID);
    EXPECT_EQ(other_name.name, "Test Entity");
}

TEST(SceneTest, findEntityByIDMultipleEntities)
{
    auto scene = atcg::make_ref<atcg::Scene>();
    scene->createEntity("Test Entity 1");
    scene->createEntity("Test Entity 2");
    scene->createEntity("Test Entity 3");
    scene->createEntity("Test Entity 4");
    auto entity = scene->createEntity("Test Entity");

    auto& id         = entity.getComponent<atcg::IDComponent>();
    auto entity_copy = scene->getEntityByID(id.ID);

    auto& other_id   = entity_copy.getComponent<atcg::IDComponent>();
    auto& other_name = entity_copy.getComponent<atcg::NameComponent>();

    EXPECT_EQ(other_id.ID, id.ID);
    EXPECT_EQ(other_name.name, "Test Entity");
}

TEST(SceneTest, findEntityByIDMultipleEntitiesSameName)
{
    auto scene = atcg::make_ref<atcg::Scene>();
    scene->createEntity("Test Entity");
    scene->createEntity("Test Entity");
    scene->createEntity("Test Entity");
    scene->createEntity("Test Entity");
    auto entity = scene->createEntity("Test Entity");

    auto& id         = entity.getComponent<atcg::IDComponent>();
    auto entity_copy = scene->getEntityByID(id.ID);

    auto& other_id   = entity_copy.getComponent<atcg::IDComponent>();
    auto& other_name = entity_copy.getComponent<atcg::NameComponent>();

    EXPECT_EQ(other_id.ID, id.ID);
    EXPECT_EQ(other_name.name, "Test Entity");
}

TEST(SceneTest, findEntityByIDNotValid)
{
    auto scene  = atcg::make_ref<atcg::Scene>();
    auto entity = scene->createEntity("Test Entity");

    auto& id         = entity.getComponent<atcg::IDComponent>();
    auto entity_copy = scene->getEntityByID(1337);

    EXPECT_EQ(entity_copy, false);
}

TEST(SceneTest, findEntitiesByName)
{
    auto scene  = atcg::make_ref<atcg::Scene>();
    auto entity = scene->createEntity("Test Entity");
    auto& id    = entity.getComponent<atcg::IDComponent>();

    auto entities = scene->getEntitiesByName("Test Entity");

    auto& other_id   = entities.front().getComponent<atcg::IDComponent>();
    auto& other_name = entities.front().getComponent<atcg::NameComponent>();

    EXPECT_EQ(other_id.ID, id.ID);
    EXPECT_EQ(other_name.name, "Test Entity");
    EXPECT_EQ(entities.size(), 1);
}

TEST(SceneTest, findEntitiesByNameMultipleEntities)
{
    auto scene = atcg::make_ref<atcg::Scene>();
    scene->createEntity("Test Entity 1");
    scene->createEntity("Test Entity 2");
    scene->createEntity("Test Entity 3");
    scene->createEntity("Test Entity 4");
    auto entity = scene->createEntity("Test Entity");

    auto& id      = entity.getComponent<atcg::IDComponent>();
    auto entities = scene->getEntitiesByName("Test Entity");

    auto& other_id   = entities.front().getComponent<atcg::IDComponent>();
    auto& other_name = entities.front().getComponent<atcg::NameComponent>();

    EXPECT_EQ(other_id.ID, id.ID);
    EXPECT_EQ(other_name.name, "Test Entity");
    EXPECT_EQ(entities.size(), 1);
}

TEST(SceneTest, findEntitiesByNameMultipleEntitiesSameName)
{
    auto scene = atcg::make_ref<atcg::Scene>();
    scene->createEntity("Test Entity");
    scene->createEntity("Test Entity");
    scene->createEntity("Test Entity");
    scene->createEntity("Test Entity3");
    auto entity = scene->createEntity("Test Entity");

    auto entities = scene->getEntitiesByName("Test Entity");

    for(int i = 0; i < entities.size(); ++i)
    {
        auto& other_name = entities[i].getComponent<atcg::NameComponent>();
        EXPECT_EQ(other_name.name, "Test Entity");
    }

    EXPECT_EQ(entities.size(), 4);
}

TEST(SceneTest, findEntitiesByNameNotValid)
{
    auto scene  = atcg::make_ref<atcg::Scene>();
    auto entity = scene->createEntity("Test Entity");

    auto& id      = entity.getComponent<atcg::IDComponent>();
    auto entities = scene->getEntitiesByName("Hello");

    EXPECT_EQ(entities.size(), 0);
}

TEST(SceneTest, clearScene)
{
    auto scene = atcg::make_ref<atcg::Scene>();
    scene->createEntity("Test Entity");
    scene->createEntity("Test Entity");
    scene->createEntity("Test Entity");
    scene->createEntity("Test Entity");
    auto entity = scene->createEntity("Test Entity");

    scene->removeAllEntites();

    auto entities = scene->getEntitiesByName("Test Entity");

    EXPECT_EQ(entities.size(), 0);
}

TEST(SceneTest, findEntitiesByComponent)
{
    auto scene = atcg::make_ref<atcg::Scene>();

    for(int i = 0; i < 10; ++i)
    {
        auto entity = scene->createEntity("Transform");
        entity.addComponent<atcg::TransformComponent>();
    }

    auto view = scene->getAllEntitiesWith<atcg::TransformComponent>();

    int num_entities = 0;
    for(auto e: view)
    {
        atcg::Entity entity(e, scene.get());

        auto& name = entity.getComponent<atcg::NameComponent>();
        EXPECT_EQ(name.name, "Transform");
        EXPECT_TRUE(entity.hasComponent<atcg::TransformComponent>());

        ++num_entities;
    }

    EXPECT_EQ(num_entities, 10);
}