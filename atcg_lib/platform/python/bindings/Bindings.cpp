#include "Bindings.h"

namespace atcg
{
namespace python
{
void BindingContext::defineAll()
{
    for(auto& definition: _definitions)
    {
        definition();
    }

    // Release the captured python handles during module initialization
    _definitions.clear();
}
}    // namespace python
}    // namespace atcg
