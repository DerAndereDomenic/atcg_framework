#include <Registration/Registration.h>

namespace atcg
{
    Registration::Registration(const std::shared_ptr<PointCloud>& source, const std::shared_ptr<PointCloud>& target)
        :X(source->asMatrix()), Y(target->asMatrix())
    {
        N = static_cast<uint32_t>(X.rows());
        M = static_cast<uint32_t>(Y.rows());
    }
}