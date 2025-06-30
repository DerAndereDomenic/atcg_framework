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
            auto img = atcg::IO::imread(metadata.file_path.string());
            asset    = atcg::Texture2D::create(img);
        }
        break;
    }

    return asset;
}
}    // namespace atcg