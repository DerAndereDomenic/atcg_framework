#pragma once

#include <Asset/Asset.h>
#include <Core/API.h>
#include <Core/RaytracingComponent.h>
#include <DataStructure/Dictionary.h>
#include <Material/PhaseFunctionVPtrTable.h>
#include <Material/PhaseFlags.h>
#include <Renderer/Renderer.h>

namespace atcg
{
class ATCG_API PhaseFunction : public Asset, public RaytracingComponent
{
public:
    PhaseFunction(const std::string& type, const atcg::Dictionary& dict);

    virtual ~PhaseFunction() {}

    virtual void uploadPhaseFunction(RendererSystem* renderer, const atcg::ref_ptr<Shader>& shader) = 0;

    ATCG_INLINE static AssetType getStaticType() { return AssetType::PhaseFunction; }

    ATCG_INLINE virtual AssetType getType() const override { return getStaticType(); }

    ATCG_INLINE const std::string& getPhaseFunctionType() const { return _phase_function_type; }

    virtual atcg::ref_ptr<PhaseFunction> clone() const = 0;

    ATCG_INLINE const PhaseFlag& flags() const { return _flags; }

    ATCG_INLINE const PhaseFunctionVPtrTable* getVPtrTable() const { return _phase_function_vptr_table.get(); }

protected:
    std::string _phase_function_type;
    bool _uploaded   = false;
    PhaseFlag _flags = PhaseFlag::None;

    atcg::dref_ptr<PhaseFunctionVPtrTable> _phase_function_vptr_table;
};

template<typename T>
struct ATCG_API PhaseFunctionSerializer
{
    static void serialize(const atcg::ref_ptr<T>& phase_function, const std::filesystem::path& path) {}

    static atcg::ref_ptr<T> deserialize(const std::filesystem::path& path, const nlohmann::json& phase_function_node)
    {
        return nullptr;
    }
};

template<typename T>
struct ATCG_API PhaseFunctionGUIRenderer
{
    static bool renderGUI(const atcg::ref_ptr<T>& phase_function, const std::string& key, bool& deactivated)
    {
        return false;
    }
};

}    // namespace atcg