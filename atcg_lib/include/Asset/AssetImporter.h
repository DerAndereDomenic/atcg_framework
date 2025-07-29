#pragma once

#include <Asset/Asset.h>

namespace atcg
{
// Based on Hazel Engine (https://github.com/TheCherno/Hazel)
// Modified by Domenic Zingsheim in 2025
namespace AssetImporter
{

/**
 * @brief Import an asset.
 * The path specified is only the high level folder where the assets are located. The complete asset path is constructed
 * based on the asset's type and name.
 *
 *
 * @param path The path
 * @param handle The handle
 * @param metadata The meta data
 *
 * @return The deserialized asset
 */
atcg::ref_ptr<Asset> importAsset(const std::filesystem::path& path, AssetHandle handle, const AssetMetaData& metadata);
};    // namespace AssetImporter
}    // namespace atcg