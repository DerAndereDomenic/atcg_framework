#pragma once

#include <Core/Memory.h>
#include <Medium/Medium.h>
#include <Medium/HeterogeneousMediumData.cuh>
#include <DataStructure/Dictionary.h>
#include <Renderer/Texture.h>
#include <Asset/AssetManagerSystem.h>

#include <Scene/ComponentGUIHandler.h>
#include <Scene/ComponentSerializer.h>

namespace atcg
{
class HeterogeneousMedium : public Medium
{
public:
    HeterogeneousMedium(const Dictionary& dict);

    virtual ~HeterogeneousMedium();
    /**
     * @brief A callback to display debug information in imgui
     */
    virtual void onImGuiRender() override {}

    /**
     * @brief Initialize the pipeline
     *
     * @param pipeline The raytracing pipeline
     * @param sbt The shader binding table
     */
    virtual void initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                    const atcg::ref_ptr<ShaderBindingTable>& sbt);

private:
    atcg::ref_ptr<Texture3D> _density_texture;
    atcg::ref_ptr<Texture3D> _albedo_texture;
    atcg::ref_ptr<Texture3D> _emission_texture;

    atcg::dref_ptr<HeterogeneousMediumData> _data_buffer;
};

}    // namespace atcg