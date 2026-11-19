#pragma once

#include <Core/API.h>
#include <Core/Memory.h>
#include <Core/SystemRegistry.h>
#include <Asset/Asset.h>
#include <Scene/Entity.h>

#include <vector>

namespace atcg
{
class Scene;

/**
 * @brief An in-app clipboard for copy & paste of entities and assets.
 *
 * Copying takes a snapshot (independent copies of the components or of the asset), so the clipboard content stays
 * valid if the source is changed or deleted. Pasting creates new copies of the snapshot, so it can be pasted multiple
 * times. The clipboard holds either entities or an asset; the last copy wins.
 */
class ATCG_API ClipboardSystem
{
public:
    /**
     * @brief Constructor
     */
    ClipboardSystem();

    /**
     * @brief Destructor
     */
    ~ClipboardSystem();

    /**
     * @brief Copy entities to the clipboard. Invalid entities and the editor camera are skipped.
     *
     * @param entities The entities
     */
    void copyEntities(const std::vector<Entity>& entities);

    /**
     * @brief Paste the copied entities into a scene.
     * The new entities are named "<name> (Copy)", keep the components (and therefore the transform) of the copied
     * entities and become the new selection. The paste is recorded as a single revision.
     *
     * @param scene The scene
     *
     * @return The new entities
     */
    std::vector<Entity> pasteEntities(const atcg::ref_ptr<Scene>& scene);

    /**
     * @brief Copy an asset to the clipboard. Scenes can't be copied.
     *
     * @param handle The asset handle
     */
    void copyAsset(AssetHandle handle);

    /**
     * @brief Paste the copied asset. A new asset named "<name> (Copy)" is registered and recorded as a revision.
     *
     * @return The handle of the new asset or 0 if nothing was pasted
     */
    AssetHandle pasteAsset();

    /**
     * @brief Check if the clipboard holds entities
     *
     * @return True if entities were copied
     */
    bool hasEntities() const;

    /**
     * @brief Check if the clipboard holds an asset
     *
     * @return True if an asset was copied
     */
    bool hasAsset() const;

    /**
     * @brief Clear the clipboard
     */
    void clear();

private:
    class Impl;
    std::unique_ptr<Impl> impl;
};

namespace Clipboard
{
/**
 * @brief Copy entities to the clipboard. Invalid entities and the editor camera are skipped.
 *
 * @param entities The entities
 */
ATCG_INLINE void copyEntities(const std::vector<Entity>& entities)
{
    SystemRegistry::instance()->getSystem<ClipboardSystem>()->copyEntities(entities);
}

/**
 * @brief Paste the copied entities into a scene (see ClipboardSystem::pasteEntities)
 *
 * @param scene The scene
 *
 * @return The new entities
 */
ATCG_INLINE std::vector<Entity> pasteEntities(const atcg::ref_ptr<Scene>& scene)
{
    return SystemRegistry::instance()->getSystem<ClipboardSystem>()->pasteEntities(scene);
}

/**
 * @brief Copy an asset to the clipboard. Scenes can't be copied.
 *
 * @param handle The asset handle
 */
ATCG_INLINE void copyAsset(AssetHandle handle)
{
    SystemRegistry::instance()->getSystem<ClipboardSystem>()->copyAsset(handle);
}

/**
 * @brief Paste the copied asset (see ClipboardSystem::pasteAsset)
 *
 * @return The handle of the new asset or 0 if nothing was pasted
 */
ATCG_INLINE AssetHandle pasteAsset()
{
    return SystemRegistry::instance()->getSystem<ClipboardSystem>()->pasteAsset();
}

/**
 * @brief Check if the clipboard holds entities
 *
 * @return True if entities were copied
 */
ATCG_INLINE bool hasEntities()
{
    return SystemRegistry::instance()->getSystem<ClipboardSystem>()->hasEntities();
}

/**
 * @brief Check if the clipboard holds an asset
 *
 * @return True if an asset was copied
 */
ATCG_INLINE bool hasAsset()
{
    return SystemRegistry::instance()->getSystem<ClipboardSystem>()->hasAsset();
}

/**
 * @brief Clear the clipboard
 */
ATCG_INLINE void clear()
{
    SystemRegistry::instance()->getSystem<ClipboardSystem>()->clear();
}
}    // namespace Clipboard
}    // namespace atcg
