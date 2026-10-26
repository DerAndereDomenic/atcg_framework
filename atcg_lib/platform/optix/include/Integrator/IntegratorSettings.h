#pragma once

#include <DataStructure/Dictionary.h>
#include <json.hpp>

namespace atcg
{

struct IntegratorSettingsBase
{
    virtual std::string getType() const = 0;

    virtual bool renderSettings() = 0;

    virtual atcg::Dictionary toDictionary() const = 0;

    virtual nlohmann::json toJson() const = 0;

    virtual void fromJson(const nlohmann::json& j) = 0;
};

template<typename T>
struct IntegratorSettings : public IntegratorSettingsBase
{
    std::string getType() const override { return T::getType(); }

    bool renderSettings() override { return false; }

    atcg::Dictionary toDictionary() const override
    {
        atcg::Dictionary dict;
        return dict;
    }

    nlohmann::json toJson() const override { return nlohmann::json(); }

    void fromJson(const nlohmann::json& j) override {}
};
}    // namespace atcg