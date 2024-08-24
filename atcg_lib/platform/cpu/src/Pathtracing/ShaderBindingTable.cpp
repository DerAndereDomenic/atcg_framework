#include <Pathtracing/ShaderBindingTable.h>

namespace atcg
{
class ShaderBindingTable::Impl
{
public:
    Impl() = default;

    Impl(ATCGDeviceContext context);

    ~Impl();

    ATCGDeviceContext context;

    std::vector<atcg::ref_ptr<ATCGShaderBindingTable_t>> sbt;

    std::vector<std::pair<ATCGProgramGroup, std::vector<uint8_t>>> sbt_entries_raygen;
    std::vector<std::pair<ATCGProgramGroup, std::vector<uint8_t>>> sbt_entries_miss;
    std::vector<std::pair<ATCGProgramGroup, std::vector<uint8_t>>> sbt_entries_hitgroup;
    std::vector<std::pair<ATCGProgramGroup, std::vector<uint8_t>>> sbt_entries_callable;
};

ShaderBindingTable::Impl::Impl(ATCGDeviceContext context)
{
    this->context = context;
}

ShaderBindingTable::Impl::~Impl() {}

ShaderBindingTable::ShaderBindingTable(ATCGDeviceContext context)
{
    impl = std::make_unique<Impl>(context);
}

ShaderBindingTable::~ShaderBindingTable() {}

uint32_t ShaderBindingTable::addRaygenEntry(ATCGProgramGroup prog_group, std::vector<uint8_t> sbt_record_custom_data)
{
    uint32_t index = impl->sbt_entries_raygen.size();
    impl->sbt_entries_raygen.push_back(std::make_pair(prog_group, sbt_record_custom_data));
    return index;
}

uint32_t ShaderBindingTable::addMissEntry(ATCGProgramGroup prog_group, std::vector<uint8_t> sbt_record_custom_data)
{
    uint32_t index = impl->sbt_entries_miss.size();
    impl->sbt_entries_miss.push_back(std::make_pair(prog_group, sbt_record_custom_data));
    return index;
}

uint32_t ShaderBindingTable::addHitEntry(ATCGProgramGroup prog_group, std::vector<uint8_t> sbt_record_custom_data)
{
    uint32_t index = impl->sbt_entries_hitgroup.size();
    impl->sbt_entries_hitgroup.push_back(std::make_pair(prog_group, sbt_record_custom_data));
    return index;
}

uint32_t ShaderBindingTable::addCallableEntry(ATCGProgramGroup prog_group, std::vector<uint8_t> sbt_record_custom_data)
{
    uint32_t index = impl->sbt_entries_callable.size();
    impl->sbt_entries_callable.push_back(std::make_pair(prog_group, sbt_record_custom_data));
    return index;
}

void ShaderBindingTable::createSBT()
{
    impl->sbt.resize(impl->sbt_entries_raygen.size());

    for(int i = 0; i < impl->sbt_entries_raygen.size(); ++i)
    {
        auto current_sbt                  = atcg::make_ref<ATCGShaderBindingTable_t>();
        current_sbt->sbt_entries_raygen   = impl->sbt_entries_raygen[i];
        current_sbt->sbt_entries_callable = impl->sbt_entries_callable;
        current_sbt->sbt_entries_miss     = impl->sbt_entries_miss;
        current_sbt->sbt_entries_hitgroup = impl->sbt_entries_hitgroup;

        impl->sbt[i] = current_sbt;
    }
}

const ATCGShaderBindingTable ShaderBindingTable::getSBT(uint32_t raygen_index) const
{
    return impl->sbt[raygen_index].get();
}

}    // namespace atcg