#include <Scene/Clipboard.h>

#include <Asset/AssetManagerSystem.h>
#include <DataStructure/Graph.h>
#include <Renderer/Material.h>
#include <Renderer/Shader.h>
#include <Renderer/Texture.h>
#include <Scene/ComponentRegistry.h>
#include <Scene/Components.h>
#include <Scene/RevisionStack.h>
#include <Scene/Scene.h>
#include <Scripting/Script.h>

namespace atcg
{

namespace detail
{
/**
 * @brief Create an independent copy of an asset. The copy gets a new handle and is not registered.
 *
 * @param asset The asset
 * @param type The type of the asset
 *
 * @return The copy or nullptr if this asset type can't be copied
 */
atcg::ref_ptr<Asset> cloneAsset(const atcg::ref_ptr<Asset>& asset, AssetType type)
{
    if(!asset) return nullptr;

    // TODO: Add inherited copy function to all assets?
    atcg::ref_ptr<Asset> copy = nullptr;
    switch(type)
    {
        case AssetType::Material:
        {
            copy = std::static_pointer_cast<Material>(asset)->clone();
        }
        break;
        case AssetType::Graph:
        {
            copy = std::static_pointer_cast<Graph>(asset)->copy();
        }
        break;
        case AssetType::Texture2D:
        {
            copy = std::dynamic_pointer_cast<Texture2D>(std::static_pointer_cast<Texture2D>(asset)->clone());
        }
        break;
        case AssetType::Texture3D:
        {
            copy = std::dynamic_pointer_cast<Texture3D>(std::static_pointer_cast<Texture3D>(asset)->clone());
        }
        break;
        case AssetType::Shader:
        {
            // A new shader for the same source files
            auto shader = std::static_pointer_cast<Shader>(asset);
            if(shader->isComputeShader())
            {
                copy = atcg::make_ref<Shader>(shader->getComputePath());
            }
            else if(shader->hasGeometryShader())
            {
                copy = atcg::make_ref<Shader>(shader->getVertexPath(),
                                              shader->getFragmentPath(),
                                              shader->getGeometryPath());
            }
            else
            {
                copy = atcg::make_ref<Shader>(shader->getVertexPath(), shader->getFragmentPath());
            }
        }
        break;
        case AssetType::Script:
        {
            // A new script for the same source file
            copy = atcg::make_ref<PythonScript>(std::static_pointer_cast<Script>(asset)->getFilePath());
        }
        break;
        default:
            break;
    }

    // The copy must not replace the original in the asset registry
    if(copy) copy->handle = AssetHandle();

    return copy;
}

/**
 * @brief Snapshot of a copied entity
 */
struct EntitySnapshot
{
    std::string name;
    std::unordered_map<entt::id_type, std::shared_ptr<void>> components;
};
}    // namespace detail

class ClipboardSystem::Impl
{
public:
    std::vector<detail::EntitySnapshot> entities;

    // The copied asset. If the asset was not loaded (only meta data), only the meta data is copied.
    bool has_asset             = false;
    atcg::ref_ptr<Asset> asset = nullptr;
    AssetMetaData asset_data;
};

ClipboardSystem::ClipboardSystem()
{
    impl = std::make_unique<Impl>();
}

ClipboardSystem::~ClipboardSystem() {}

void ClipboardSystem::copyEntities(const std::vector<Entity>& entities)
{
    std::vector<detail::EntitySnapshot> snapshots;
    for(Entity entity: entities)
    {
        if(!entity || entity.hasComponent<EditorCameraComponent>()) continue;

        detail::EntitySnapshot snapshot;
        snapshot.name = entity.getComponent<NameComponent>().name();
        ComponentRegistry::copyAllComponents(entity, snapshot.components);
        snapshots.push_back(std::move(snapshot));
    }

    if(snapshots.empty()) return;

    clear();
    impl->entities = std::move(snapshots);
}

std::vector<Entity> ClipboardSystem::pasteEntities(const atcg::ref_ptr<Scene>& scene)
{
    if(!scene || impl->entities.empty()) return {};

    if(RevisionStack::isRecording())
    {
        ATCG_WARN("Cannot paste entities while another revision is recorded");
        return {};
    }

    // The entities have to exist when the recording starts, the components are recorded at the end
    std::vector<Entity> pasted;
    pasted.reserve(impl->entities.size());
    for(const auto& snapshot: impl->entities)
    {
        pasted.push_back(scene->createEntity(snapshot.name + " (Copy)"));
    }

    RevisionStack::startRecording<MultiEntityRevision<EntityAddedRevision>>(scene, pasted);
    for(size_t i = 0; i < pasted.size(); ++i)
    {
        // Pasting creates new copies, so the clipboard can be pasted multiple times
        ComponentRegistry::pasteAllComponents(scene, pasted[i], impl->entities[i].components);
    }
    RevisionStack::endRecording();

    // Script components expect the whole entity to be created before the behavior is created, so we create the
    // behaviors after the recording
    for(auto entity: pasted)
    {
        if(!entity) continue;
        if(entity.hasComponent<ScriptComponent>())
        {
            auto& script = entity.getComponent<ScriptComponent>();
            if(script.script())
            {
                auto behavior = script.behavior(scene, entity, true);
                if(behavior) behavior->onAttach();
            }
        }
    }

    for(size_t i = 0; i < pasted.size(); ++i)
    {
        scene->selectEntity(pasted[i], i > 0);
    }

    return pasted;
}

void ClipboardSystem::copyAsset(AssetHandle handle)
{
    if(!AssetManager::isAssetHandleValid(handle)) return;

    const AssetMetaData data = AssetManager::getMetaData(handle);
    if(data.type == AssetType::Scene || data.type == AssetType::None)
    {
        ATCG_WARN("Assets of this type can't be copied: {}", data.name);
        return;
    }

    // Assets that were not created yet (e.g. a new, empty graph) only consist of their meta data
    auto asset = AssetManager::getAsset(handle);
    auto copy  = detail::cloneAsset(asset, data.type);
    if(asset && !copy)
    {
        ATCG_WARN("Could not copy asset {}", data.name);
        return;
    }

    clear();
    impl->has_asset  = true;
    impl->asset      = copy;
    impl->asset_data = data;
}

AssetHandle ClipboardSystem::pasteAsset()
{
    if(!impl->has_asset) return 0;

    if(RevisionStack::isRecording())
    {
        ATCG_WARN("Cannot paste an asset while another revision is recorded");
        return 0;
    }

    const std::string name = impl->asset_data.name + " (Copy)";

    AssetHandle handle = 0;
    if(impl->asset)
    {
        // Pasting creates a new copy, so the clipboard can be pasted multiple times
        auto copy = detail::cloneAsset(impl->asset, impl->asset_data.type);
        if(!copy) return 0;
        handle = AssetManager::registerAsset(copy, name, impl->asset_data.show_in_editor, impl->asset_data.serialize);
    }
    else
    {
        AssetMetaData data = impl->asset_data;
        data.name          = name;
        handle             = AssetManager::registerAsset(data);
    }

    if(!AssetManager::isAssetHandleValid(handle)) return 0;

    RevisionStack::startRecording<AssetAddedRevision>(handle);
    RevisionStack::endRecording();

    return handle;
}

bool ClipboardSystem::hasEntities() const
{
    return !impl->entities.empty();
}

bool ClipboardSystem::hasAsset() const
{
    return impl->has_asset;
}

void ClipboardSystem::clear()
{
    impl->entities.clear();
    impl->has_asset  = false;
    impl->asset      = nullptr;
    impl->asset_data = AssetMetaData();
}

}    // namespace atcg
