#include "ShaderBindingTable.h"

#include "Common.h"

#include <optix_stubs.h>

namespace atcg
{
template<typename T>
inline T ceil_to_multiple(T value, T mod)
{
    return value - T(1) + mod - ((value - T(1)) % mod);
}

class ShaderBindingTable::Impl
{
public:
    Impl() = default;

    Impl(OptixDeviceContext context);

    ~Impl();

    OptixDeviceContext context;

    size_t max_raygen_data_size   = 0;
    size_t max_miss_data_size     = 0;
    size_t max_hitgroup_data_size = 0;
    size_t max_callable_data_size = 0;

    std::vector<std::pair<OptixProgramGroup, std::vector<uint8_t>>> sbt_entries_raygen;
    std::vector<std::pair<OptixProgramGroup, std::vector<uint8_t>>> sbt_entries_miss;
    std::vector<std::pair<OptixProgramGroup, std::vector<uint8_t>>> sbt_entries_hitgroup;
    std::vector<std::pair<OptixProgramGroup, std::vector<uint8_t>>> sbt_entries_callable;

    // typedef SBTRecord<void*> RecordType;

    atcg::DeviceBuffer<uint8_t> sbt_buffer;

    std::vector<OptixShaderBindingTable> sbt;
};

ShaderBindingTable::Impl::Impl(OptixDeviceContext context)
{
    this->context = context;
}

ShaderBindingTable::Impl::~Impl() {}

ShaderBindingTable::ShaderBindingTable(OptixDeviceContext context)
{
    impl = std::make_unique<Impl>(context);
}

ShaderBindingTable::~ShaderBindingTable() {}

uint32_t ShaderBindingTable::addRaygenEntry(OptixProgramGroup prog_group, std::vector<uint8_t> sbt_record_custom_data)
{
    uint32_t raygen_sbt_index  = impl->sbt_entries_raygen.size();
    impl->max_raygen_data_size = std::max(impl->max_raygen_data_size, sbt_record_custom_data.size());
    impl->sbt_entries_raygen.push_back({prog_group, std::move(sbt_record_custom_data)});
    return raygen_sbt_index;
}

uint32_t ShaderBindingTable::addMissEntry(OptixProgramGroup prog_group, std::vector<uint8_t> sbt_record_custom_data)
{
    uint32_t miss_sbt_index  = impl->sbt_entries_miss.size();
    impl->max_miss_data_size = std::max(impl->max_miss_data_size, sbt_record_custom_data.size());
    impl->sbt_entries_miss.push_back({prog_group, std::move(sbt_record_custom_data)});
    return miss_sbt_index;
}

uint32_t ShaderBindingTable::addHitEntry(OptixProgramGroup prog_group, std::vector<uint8_t> sbt_record_custom_data)
{
    uint32_t hitgroup_sbt_index  = impl->sbt_entries_hitgroup.size();
    impl->max_hitgroup_data_size = std::max(impl->max_hitgroup_data_size, sbt_record_custom_data.size());
    impl->sbt_entries_hitgroup.push_back({prog_group, std::move(sbt_record_custom_data)});
    return hitgroup_sbt_index;
}

uint32_t ShaderBindingTable::addCallableEntry(OptixProgramGroup prog_group, std::vector<uint8_t> sbt_record_custom_data)
{
    uint32_t callable_sbt_index  = impl->sbt_entries_callable.size();
    impl->max_callable_data_size = std::max(impl->max_callable_data_size, sbt_record_custom_data.size());
    impl->sbt_entries_callable.push_back({prog_group, std::move(sbt_record_custom_data)});
    return callable_sbt_index;
}

void ShaderBindingTable::createSBT()
{
    std::vector<uint8_t> records_data;

    auto add_records = [&records_data](size_t record_stride, auto &sbt_entries)
    {
        // All entries of the same kind must occupy the same amount of storage
        std::vector<uint8_t> temp_record_data(record_stride);
        // Pointer to the header in temporary data
        uint8_t *temp_record_header_ptr = temp_record_data.data();
        // Pointer to the data segment in temporary data
        uint8_t *temp_record_data_ptr = temp_record_data.data() + OPTIX_SBT_RECORD_HEADER_SIZE;
        for(const auto [prog_group, data]: sbt_entries)
        {
            // Write record header
            OPTIX_CHECK(optixSbtRecordPackHeader(prog_group, temp_record_header_ptr));
            // Write record data
            std::copy(data.begin(), data.end(), temp_record_data_ptr);

            // Append new record to records_data
            records_data.insert(records_data.end(), temp_record_data.begin(), temp_record_data.end());
        }
    };

    // TODO align the strides!!!

    // Compute the stride of each entry type in the SBT
    auto raygen_stride =
        ceil_to_multiple<size_t>(OPTIX_SBT_RECORD_HEADER_SIZE + impl->max_raygen_data_size, OPTIX_SBT_RECORD_ALIGNMENT);
    auto miss_stride =
        ceil_to_multiple<size_t>(OPTIX_SBT_RECORD_HEADER_SIZE + impl->max_miss_data_size, OPTIX_SBT_RECORD_ALIGNMENT);
    auto hitgroup_stride = ceil_to_multiple<size_t>(OPTIX_SBT_RECORD_HEADER_SIZE + impl->max_hitgroup_data_size,
                                                    OPTIX_SBT_RECORD_ALIGNMENT);
    auto callable_stride = ceil_to_multiple<size_t>(OPTIX_SBT_RECORD_HEADER_SIZE + impl->max_callable_data_size,
                                                    OPTIX_SBT_RECORD_ALIGNMENT);

    // Compute the offset for the start of each entry type in the SBT based on the preceeding entries in our records
    // data
    auto raygen_offset   = 0ull;
    auto miss_offset     = raygen_offset + raygen_stride * impl->sbt_entries_raygen.size();
    auto hitgroup_offset = miss_offset + miss_stride * impl->sbt_entries_miss.size();
    auto callable_offset = hitgroup_offset + hitgroup_stride * impl->sbt_entries_hitgroup.size();
    auto sbt_size        = callable_offset + callable_stride * impl->sbt_entries_callable.size();

    // Reserve enough memory to avoid intermediate allocations
    records_data.reserve(sbt_size);

    // Fill records_data with content!
    add_records(raygen_stride, impl->sbt_entries_raygen);
    add_records(miss_stride, impl->sbt_entries_miss);
    add_records(hitgroup_stride, impl->sbt_entries_hitgroup);
    add_records(callable_stride, impl->sbt_entries_callable);

    if(records_data.size() != sbt_size) throw std::runtime_error("Error encountered while computing SBT data!");

    // Upload records_data to GPU
    impl->sbt_buffer.create(records_data.size());
    impl->sbt_buffer.upload(records_data.data());

    // Allocate a separate shader binding table structure for every raygen program.
    impl->sbt.resize(impl->sbt_entries_raygen.size());

    // For all raygen programs...
    for(uint32_t raygen_index = 0; raygen_index < impl->sbt.size(); ++raygen_index)
    {
        // Alias for the sbt we are currently filling...
        auto &sbt = impl->sbt[raygen_index];

        sbt.raygenRecord = (CUdeviceptr)(impl->sbt_buffer.get() + (raygen_offset + raygen_index * raygen_stride));

        if(!impl->sbt_entries_miss.empty())
        {
            sbt.missRecordBase          = (CUdeviceptr)(impl->sbt_buffer.get() + (miss_offset));
            sbt.missRecordStrideInBytes = miss_stride;
            sbt.missRecordCount         = impl->sbt_entries_miss.size();
        }
        else
        {
            sbt.missRecordBase          = 0;
            sbt.missRecordStrideInBytes = 0;
            sbt.missRecordCount         = 0;
        }

        if(!impl->sbt_entries_hitgroup.empty())
        {
            sbt.hitgroupRecordBase          = (CUdeviceptr)(impl->sbt_buffer.get() + (hitgroup_offset));
            sbt.hitgroupRecordStrideInBytes = hitgroup_stride;
            sbt.hitgroupRecordCount         = impl->sbt_entries_hitgroup.size();
        }
        else
        {
            sbt.hitgroupRecordBase          = 0;
            sbt.hitgroupRecordStrideInBytes = 0;
            sbt.hitgroupRecordCount         = 0;
        }

        if(!impl->sbt_entries_callable.empty())
        {
            sbt.callablesRecordBase          = (CUdeviceptr)(impl->sbt_buffer.get() + (callable_offset));
            sbt.callablesRecordStrideInBytes = callable_stride;
            sbt.callablesRecordCount         = impl->sbt_entries_callable.size();
        }
        else
        {
            sbt.callablesRecordBase          = 0;
            sbt.callablesRecordStrideInBytes = 0;
            sbt.callablesRecordCount         = 0;
        }
    }
}

const OptixShaderBindingTable *ShaderBindingTable::getSBT(uint32_t raygen_index) const
{
    return raygen_index < impl->sbt.size() ? &impl->sbt[raygen_index] : nullptr;
}

}    // namespace atcg