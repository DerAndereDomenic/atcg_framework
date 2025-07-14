#include <Asset/AssetImporter.h>

#include <DataStructure/Image.h>
#include <Renderer/Texture.h>

namespace atcg
{
atcg::ref_ptr<Asset> AssetImporter::importAsset(const AssetMetaData& metadata)
{
    atcg::ref_ptr<Asset> asset = nullptr;
    switch(metadata.type)
    {
        case AssetType::Texture2D:
        {
            // TODO
        }
        break;
        case AssetType::Material:
        {
            // TODO
        }
        break;
        case AssetType::Graph:
        {
            // TODO
        }
        break;
    }

    return asset;
}
}    // namespace atcg