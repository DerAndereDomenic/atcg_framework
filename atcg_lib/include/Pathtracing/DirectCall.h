#pragma once

#ifdef ATCG_RT_MODULE
    #include <Pathtracing/PathtracingPlatform.h>
    #include <Pathtracing/ShaderBindingTableDefinition.h>

namespace atcg
{
template<typename ReturnT, typename... Args>
ReturnT directCall(uint32_t sbt_index, Args... args)
{
    using funcT        = ReturnT (*)(Args...);
    auto [group, data] = g_sbt->sbt_entries_callable[sbt_index];
    group->module->set_sbt(g_sbt);
    group->module->set_memory_for_function(std::get<0>(group->function).c_str(), data.data());
    return reinterpret_cast<funcT>(std::get<1>(group->function))(args...);
}

}    // namespace atcg
#endif
