#pragma once

#include <Core/API.h>
#include <Asset/Asset.h>

namespace atcg
{
namespace AssetExporter
{
/**
 * @brief Export an asset to the specified file path.
 * The given path is only the main path (folder) where the assets should be serialized. Further subfolders may be
 * created based on the asset type.
 *
 * @param path The path
 * @param asset The asset
 * @param data The asset meta data
 */
ATCG_API void
exportAsset(const std::filesystem::path& path, const atcg::ref_ptr<Asset>& asset, const AssetMetaData& data);
};    // namespace AssetExporter
}    // namespace atcg