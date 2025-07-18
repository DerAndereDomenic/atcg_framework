#pragma once

#include <Asset/Asset.h>

namespace atcg
{
// Based on Hazel Engine (https://github.com/TheCherno/Hazel)
// Modified by Domenic Zingsheim in 2025
class AssetImporter
{
public:
    static atcg::ref_ptr<Asset> importAsset(const std::filesystem::path& path, const AssetMetaData& metadata);
};
}    // namespace atcg