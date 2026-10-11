#pragma once

#include <DataStructure/Dictionary.h>

namespace atcg
{

struct IntegratorSettingsBase
{
    virtual void renderSettings() = 0;

    virtual atcg::Dictionary toDictionary() const = 0;
};

template<typename T>
struct IntegratorSettings : public IntegratorSettingsBase
{
    void renderSettings() override {}

    atcg::Dictionary toDictionary() const override
    {
        atcg::Dictionary dict;
        return dict;
    }
};
}    // namespace atcg