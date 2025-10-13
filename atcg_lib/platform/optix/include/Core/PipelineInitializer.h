#pragma once

#include <Core/RaytracingPipeline.h>
#include <Core/ShaderBindingTable.h>

namespace atcg
{

template<typename T>
struct PipelineInitializerBase
{
    PipelineInitializerBase(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                            const atcg::ref_ptr<ShaderBindingTable>& sbt)
        : pipeline(pipeline),
          sbt(sbt)
    {
    }

    virtual void apply(const atcg::ref_ptr<T>& component) const {}

    void ensureInitialized(const atcg::ref_ptr<T>& component) const
    {
        if(!component->isInitialized()) apply(component, pipeline, sbt);
    }

    atcg::ref_ptr<RayTracingPipeline> pipeline;
    atcg::ref_ptr<ShaderBindingTable> sbt;
};

template<typename T>
struct PipelineInitializer : public PipelineInitializerBase<T>
{
    PipelineInitializer(const atcg::ref_ptr<RayTracingPipeline>& pipeline, const atcg::ref_ptr<ShaderBindingTable>& sbt)
        : PipelineInitializerBase<T>(pipeline, sbt)
    {
    }
};

#define ATCG_DECLARE_COMPONENT_PIPELINE_INITIALIZER(ComponentType)                                                     \
    template<>                                                                                                         \
    struct PipelineInitializer<ComponentType> : public PipelineInitializerBase<ComponentType>                          \
    {                                                                                                                  \
        PipelineInitializer(const atcg::ref_ptr<RayTracingPipeline>& pipeline,                                         \
                            const atcg::ref_ptr<ShaderBindingTable>& sbt)                                              \
            : PipelineInitializerBase<ComponentType>(pipeline, sbt)                                                    \
        {                                                                                                              \
        }                                                                                                              \
                                                                                                                       \
        void apply(const atcg::ref_ptr<ComponentType>& component) const override;                                      \
    }
}    // namespace atcg