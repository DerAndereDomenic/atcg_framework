#pragma once

#ifdef ATCG_GPU_PATHTRACING
namespace atcg
{
    #include <optix.h>

typedef OptixDeviceContext ATCGDeviceContext;
typedef OptixProgramGroup ATCGProgramGroup;
typedef OptixPipeline ATCGPipeline;
typedef OptixShaderBindingTable ATCGShaderbindingTable;
}    // namespace atcg
#else

    // TODO:
    #define NOMINMAX
    #include <Windows.h>
    #include <vector>
    #include <iostream>
    #include <Core/glm.h>

namespace atcg
{

struct ATCGDeviceContext_t
{
};
typedef ATCGDeviceContext_t* ATCGDeviceContext;

struct ATCGPipeline_t
{
};
typedef ATCGPipeline_t* ATCGPipeline;

class ATCGModule_t;
struct ATCGProgramGroup_t
{
    ATCGModule_t* module;
    std::pair<std::string, void*> function;
};
typedef ATCGProgramGroup_t* ATCGProgramGroup;

struct ATCGShaderBindingTable_t
{
    std::pair<ATCGProgramGroup, std::vector<uint8_t> > sbt_entries_raygen;
    std::vector<std::pair<ATCGProgramGroup, std::vector<uint8_t> > > sbt_entries_miss;
    std::vector<std::pair<ATCGProgramGroup, std::vector<uint8_t> > > sbt_entries_hitgroup;
    std::vector<std::pair<ATCGProgramGroup, std::vector<uint8_t> > > sbt_entries_callable;
};
typedef ATCGShaderBindingTable_t* ATCGShaderBindingTable;

class ATCGModule_t
{
public:
    ATCGModule_t(const char* filename)
    {
        _handle = (HMODULE*)malloc(sizeof(HMODULE));

        *_handle = LoadLibraryA(filename);

        if(*_handle == NULL)
        {
            std::cerr << "Could not load dll\n";
        }

        _set_memory_for_function = (void (*)(const char*, uint8_t*))GetProcAddress(*_handle, "set_memory_for_function");
        _set_sbt                 = (void (*)(ATCGShaderBindingTable))GetProcAddress(*_handle, "set_sbt");
        _set_pixel_index     = (void (*)(const uint32_t, const uint32_t))GetProcAddress(*_handle, "set_pixel_index");
        _set_payload_pointer = (void (*)(void*))GetProcAddress(*_handle, "set_payload_pointer");

        _set_ray_origin     = (void (*)(const glm::vec3&))GetProcAddress(*_handle, "set_ray_origin");
        _set_ray_direction  = (void (*)(const glm::vec3&))GetProcAddress(*_handle, "set_ray_direction");
        _set_ray_tmin       = (void (*)(const float))GetProcAddress(*_handle, "set_ray_tmin");
        _set_ray_tmax       = (void (*)(const float))GetProcAddress(*_handle, "set_ray_tmax");
        _set_primitive_idx  = (void (*)(const int))GetProcAddress(*_handle, "set_primitive_idx");
        _set_local_to_world = (void (*)(const glm::mat4&))GetProcAddress(*_handle, "set_local_to_world");
        _set_barys          = (void (*)(const glm::vec2&))GetProcAddress(*_handle, "set_barys");
        _params             = GetProcAddress(*_handle, "params");
    }

    ~ATCGModule_t()
    {
        FreeLibrary(*_handle);
        free(_handle);
    }

    void set_memory_for_function(const char* name, uint8_t* data) { _set_memory_for_function(name, data); }

    void set_sbt(ATCGShaderBindingTable sbt) { _set_sbt(sbt); }

    void set_pixel_index(const uint32_t x, const uint32_t y) { _set_pixel_index(x, y); }

    void set_payload_pointer(void* ptr) { _set_payload_pointer(ptr); }

    void* get_params() const { return _params; }

    void set_ray_origin(const glm::vec3& origin) { _set_ray_origin(origin); }

    void set_ray_direction(const glm::vec3& direction) { _set_ray_direction(direction); }

    void set_ray_tmin(const float tmin) { _set_ray_tmin(tmin); }

    void set_ray_tmax(const float tmax) { _set_ray_tmax(tmax); }

    void set_primitive_idx(const int idx) { _set_primitive_idx(idx); }

    void set_local_to_world(const glm::mat4& transform) { _set_local_to_world(transform); }

    void set_barys(const glm::vec2& barys) { _set_barys(barys); }

    HMODULE handle() const { return *_handle; }

private:
    HMODULE* _handle;

    void (*_set_memory_for_function)(const char*, uint8_t*);
    void (*_set_sbt)(ATCGShaderBindingTable);
    void (*_set_pixel_index)(const uint32_t, const uint32_t);
    void (*_set_payload_pointer)(void*);
    void (*_set_ray_origin)(const glm::vec3&);
    void (*_set_ray_direction)(const glm::vec3&);
    void (*_set_ray_tmin)(const float);
    void (*_set_ray_tmax)(const float);
    void (*_set_primitive_idx)(const int);
    void (*_set_local_to_world)(const glm::mat4&);
    void (*_set_barys)(const glm::vec2&);
    void* _params;
};
typedef ATCGModule_t* ATCGModule;

}    // namespace atcg
#endif
