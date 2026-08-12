#include <Core/ShaderBindingTable.h>

#include <Core/Optix.h>

#include <optix_stubs.h>

namespace atcg
{
class ShaderBindingTable::Impl
{
public:
    Impl();

    ~Impl();
};

ShaderBindingTable::Impl::Impl() {}

ShaderBindingTable::Impl::~Impl() {}

ShaderBindingTable::ShaderBindingTable()
{
    impl = std::make_unique<Impl>();
}

ShaderBindingTable::~ShaderBindingTable() {}

uint32_t ShaderBindingTable::addRaygenEntry(OptixProgramGroup prog_group, std::vector<uint8_t> sbt_record_custom_data)
{
    return 0;
}

uint32_t ShaderBindingTable::addMissEntry(OptixProgramGroup prog_group, std::vector<uint8_t> sbt_record_custom_data)
{
    return 0;
}

uint32_t ShaderBindingTable::addHitEntry(OptixProgramGroup prog_group, std::vector<uint8_t> sbt_record_custom_data)
{
    return 0;
}

uint32_t ShaderBindingTable::addCallableEntry(OptixProgramGroup prog_group, std::vector<uint8_t> sbt_record_custom_data)
{
    return 0;
}

void ShaderBindingTable::createSBT() {}

const OptixShaderBindingTable *ShaderBindingTable::getSBT(uint32_t raygen_index) const
{
    return nullptr;
}

}    // namespace atcg