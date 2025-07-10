#pragma once

#include <Asset/Asset.h>

namespace atcg
{
class AssetExporter
{
public:
    static void
    exportAsset(const std::filesystem::path& path, const atcg::ref_ptr<Asset>& asset, const AssetMetaData& data);
};
}    // namespace atcg