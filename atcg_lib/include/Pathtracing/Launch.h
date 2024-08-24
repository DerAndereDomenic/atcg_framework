#pragma once

#include <Pathtracing/PathtracingPlatform.h>
#include <DataStructure/TorchUtils.h>
#include <torch/types.h>

#include <thread>
#include <mutex>
#include <execution>

namespace atcg
{
ATCG_INLINE void launch(const ATCGPipeline pipeline,
                        void* stream,
                        void* params,
                        const uint32_t params_size,
                        ATCGShaderBindingTable sbt,
                        const uint32_t width,
                        const uint32_t height,
                        const uint32_t depth)
{
    uint32_t _raygen_idx = 0;
    void* dst_params     = (sbt->sbt_entries_raygen.first->module->get_params());

    std::memcpy(dst_params, params, params_size);

    auto horizontalScanLine = torch::arange((int)width, atcg::TensorOptions::int32HostOptions());
    auto verticalScanLine   = torch::arange((int)height, atcg::TensorOptions::int32HostOptions());
    std::for_each(std::execution::par,
                  verticalScanLine.data_ptr<int32_t>(),
                  verticalScanLine.data_ptr<int32_t>() + verticalScanLine.numel(),
                  [&](int32_t y)
                  {
                      std::for_each(std::execution::par,
                                    horizontalScanLine.data_ptr<int32_t>(),
                                    horizontalScanLine.data_ptr<int32_t>() + horizontalScanLine.numel(),
                                    [&](int32_t x)
                                    {
                                        auto raygen_entry = sbt->sbt_entries_raygen;
                                        raygen_entry.first->module->set_sbt(sbt);
                                        raygen_entry.first->module->set_pixel_index(x, y);
                                        ((void (*)())(raygen_entry.first->function.second))();
                                    });
                  });
}
}    // namespace atcg