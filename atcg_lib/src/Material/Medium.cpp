#include <Material/Medium.h>

namespace atcg
{
Medium::Medium(const std::string& type, const atcg::Dictionary& dict) : _medium_type(type) {}
}    // namespace atcg