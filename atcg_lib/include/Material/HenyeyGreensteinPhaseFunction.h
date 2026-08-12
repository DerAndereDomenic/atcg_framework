#pragma once

#include <Material/PhaseFunction.h>
#include <Material/PhaseFunctionRegistry.h>

namespace atcg
{
class ATCG_API HenyeyGreensteinPhaseFunction : public PhaseFunction
{
public:
    HenyeyGreensteinPhaseFunction(const atcg::Dictionary& dict);

    virtual void uploadPhaseFunction(RendererSystem* renderer, const atcg::ref_ptr<Shader>& shader) override;

    ATCG_INLINE void setG(const float g) { _g = g; }

    ATCG_INLINE float g() const { return _g; }

    virtual atcg::ref_ptr<PhaseFunction> clone() const override;

    static void registerPhaseFunction(PhaseFunctionRegistry::Registry* registry);

protected:
    float _g = 0.0f;
};

template<>
struct ATCG_API PhaseFunctionSerializer<HenyeyGreensteinPhaseFunction>
{
    static void serialize(const atcg::ref_ptr<HenyeyGreensteinPhaseFunction>& phase_function,
                          const std::filesystem::path& path);

    static atcg::ref_ptr<HenyeyGreensteinPhaseFunction> deserialize(const std::filesystem::path& path,
                                                                    const nlohmann::json& phase_function_node);
};

template<>
struct ATCG_API PhaseFunctionGUIRenderer<HenyeyGreensteinPhaseFunction>
{
    static bool renderGUI(const atcg::ref_ptr<HenyeyGreensteinPhaseFunction>& phase_function,
                          const std::string& key,
                          bool& deactivated);
};

}    // namespace atcg