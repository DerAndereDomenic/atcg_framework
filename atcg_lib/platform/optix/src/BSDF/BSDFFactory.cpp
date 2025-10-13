#include <BSDF/BSDFFactory.h>

namespace atcg
{

class BSDFFactory_T
{
public:
    static BSDFFactory_T* getInstance()
    {
        if(!_instance) _instance = std::make_unique<BSDFFactory_T>();
        return _instance.get();
    }

    void registerBuilder(MaterialType type, BSDFBuilder builder) { _registry[type] = std::move(builder); }

    atcg::ref_ptr<BSDF> create(MaterialType type,
                               const Dictionary& dict,
                               const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                               const atcg::ref_ptr<ShaderBindingTable>& sbt)
    {
        auto it = _registry.find(type);
        if(it == _registry.end())
        {
            throw std::runtime_error("Unknown MaterialType");
        }

        return it->second(dict, pipeline, sbt);
    }

private:
    std::unordered_map<MaterialType, BSDFBuilder> _registry;
    static std::unique_ptr<BSDFFactory_T> _instance;
};

std::unique_ptr<BSDFFactory_T> BSDFFactory_T::_instance;

void BSDFFactory::registerBSDF(MaterialType type, BSDFBuilder builder)
{
    auto instance = BSDFFactory_T::getInstance();
    instance->registerBuilder(type, builder);
}

atcg::ref_ptr<BSDF> BSDFFactory::createBSDF(MaterialType type,
                                            const Dictionary& dict,
                                            const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                            const atcg::ref_ptr<ShaderBindingTable>& sbt)
{
    auto instance = BSDFFactory_T::getInstance();
    return instance->create(type, dict, pipeline, sbt);
}
}    // namespace atcg