#pragma once

#include <DataStructure/Dictionary.h>

namespace atcg
{

struct IntegratorSettingsBase
{
    virtual bool renderSettings() = 0;

    virtual atcg::Dictionary toDictionary() const = 0;
};

template<typename T>
struct IntegratorSettings : public IntegratorSettingsBase
{
    bool renderSettings() override { return false; }

    atcg::Dictionary toDictionary() const override
    {
        atcg::Dictionary dict;
        return dict;
    }
};
}    // namespace atcg