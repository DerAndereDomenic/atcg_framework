#pragma once

#include <optix.h>

namespace atcg
{
class ShaderBindingTable
{
public:
    /**
     * @brief Create a shader binding table
     *
     * @param context The Optix context
     */
    ShaderBindingTable(OptixDeviceContext context);

    /**
     * @brief Destructor
     */
    ~ShaderBindingTable();

    /**
     * @brief Add a raygen entry without any data
     *
     * @param prog_group The program group associated with the shader
     *
     * @return The index of the entry into the SBT
     */
    inline uint32_t addRaygenEntry(OptixProgramGroup prog_group)
    {
        return addRaygenEntry(prog_group, std::vector<uint8_t>());
    }

    /**
     * @brief Add a miss entry without any data
     *
     * @param prog_group The program group associated with the shader
     *
     * @return The index of the entry into the SBT
     */
    inline uint32_t addMissEntry(OptixProgramGroup prog_group)
    {
        return addMissEntry(prog_group, std::vector<uint8_t>());
    }

    /**
     * @brief Add a hit entry without any data
     *
     * @param prog_group The program group associated with the shader
     *
     * @return The index of the entry into the SBT
     */
    inline uint32_t addHitEntry(OptixProgramGroup prog_group)
    {
        return addHitEntry(prog_group, std::vector<uint8_t>());
    }

    /**
     * @brief Add a callable entry without any data
     *
     * @param prog_group The program group associated with the shader
     *
     * @return The index of the entry into the SBT
     */
    inline uint32_t addCallableEntry(OptixProgramGroup prog_group)
    {
        return addCallableEntry(prog_group, std::vector<uint8_t>());
    }

    /**
     * @brief Add a raygen entry with custom data that is copied into the data segment of the shader entry
     *
     * @tparam T Type of the custom data
     * @param prog_group The program group associated with the shader
     * @param sbt_record_data The data
     *
     * @return The index of the entry into the SBT
     */
    template<typename T>
    inline uint32_t addRaygenEntry(OptixProgramGroup prog_group, const T& sbt_record_data)
    {
        return addRaygenEntry(prog_group,
                              std::vector<uint8_t>(reinterpret_cast<const uint8_t*>(&sbt_record_data),
                                                   reinterpret_cast<const uint8_t*>(&sbt_record_data + 1)));
    }

    /**
     * @brief Add a miss entry with custom data that is copied into the data segment of the shader entry
     *
     * @tparam T Type of the custom data
     * @param prog_group The program group associated with the shader
     * @param sbt_record_data The data
     *
     * @return The index of the entry into the SBT
     */
    template<typename T>
    inline uint32_t addMissEntry(OptixProgramGroup prog_group, const T& sbt_record_data)
    {
        return addMissEntry(prog_group,
                            std::vector<uint8_t>(reinterpret_cast<const uint8_t*>(&sbt_record_data),
                                                 reinterpret_cast<const uint8_t*>(&sbt_record_data + 1)));
    }

    /**
     * @brief Add a hit entry with custom data that is copied into the data segment of the shader entry
     *
     * @tparam T Type of the custom data
     * @param prog_group The program group associated with the shader
     * @param sbt_record_data The data
     *
     * @return The index of the entry into the SBT
     */
    template<typename T>
    inline uint32_t addHitEntry(OptixProgramGroup prog_group, const T& sbt_record_data)
    {
        return addHitEntry(prog_group,
                           std::vector<uint8_t>(reinterpret_cast<const uint8_t*>(&sbt_record_data),
                                                reinterpret_cast<const uint8_t*>(&sbt_record_data + 1)));
    }

    /**
     * @brief Add a callable entry with custom data that is copied into the data segment of the shader entry
     *
     * @tparam T Type of the custom data
     * @param prog_group The program group associated with the shader
     * @param sbt_record_data The data
     *
     * @return The index of the entry into the SBT
     */
    template<typename T>
    inline uint32_t addCallableEntry(OptixProgramGroup prog_group, const T& sbt_record_data)
    {
        return addCallableEntry(prog_group,
                                std::vector<uint8_t>(reinterpret_cast<const uint8_t*>(&sbt_record_data),
                                                     reinterpret_cast<const uint8_t*>(&sbt_record_data + 1)));
    }

    /**
     * @brief Add a raygen entry with custom data that is copied into the data segment of the shader entry
     *
     * @param prog_group The program group associated with the shader
     * @param sbt_record_custom_data The data stored in a vector
     *
     * @return The index of the entry into the SBT
     */
    uint32_t addRaygenEntry(OptixProgramGroup prog_group, std::vector<uint8_t> sbt_record_custom_data);

    /**
     * @brief Add a raygen entry with custom data that is copied into the data segment of the shader entry
     *
     * @param prog_group The program group associated with the shader
     * @param sbt_record_custom_data The data stored in a vector
     *
     * @return The index of the entry into the SBT
     */
    uint32_t addMissEntry(OptixProgramGroup prog_group, std::vector<uint8_t> sbt_record_custom_data);

    /**
     * @brief Add a raygen entry with custom data that is copied into the data segment of the shader entry
     *
     * @param prog_group The program group associated with the shader
     * @param sbt_record_custom_data The data stored in a vector
     *
     * @return The index of the entry into the SBT
     */
    uint32_t addHitEntry(OptixProgramGroup prog_group, std::vector<uint8_t> sbt_record_custom_data);

    /**
     * @brief Add a raygen entry with custom data that is copied into the data segment of the shader entry
     *
     * @param prog_group The program group associated with the shader
     * @param sbt_record_custom_data The data stored in a vector
     *
     * @return The index of the entry into the SBT
     */
    uint32_t addCallableEntry(OptixProgramGroup prog_group, std::vector<uint8_t> sbt_record_custom_data);

    /**
     * @brief Create the shader binding table
     */
    void createSBT();

    /**
     * @brief Get the shader binding table relative to a raygen index
     *
     * @param raygen_index The index of the main entry point
     *
     * @return The Shader Binding Table data
     */
    const OptixShaderBindingTable* getSBT(uint32_t raygen_index = 0) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl;
};
}    // namespace atcg