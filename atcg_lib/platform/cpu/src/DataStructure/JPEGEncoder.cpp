#include <DataStructure/JPEGEncoder.h>

#include <cstdlib>

namespace atcg
{

class JPEGEncoder::Impl
{
public:
    Impl();

    ~Impl();
};

JPEGEncoder::Impl::Impl() {}

JPEGEncoder::Impl::~Impl() {}

JPEGEncoder::JPEGEncoder(JPEGBackend backend)
{
    impl = std::make_unique<Impl>();
}

JPEGEncoder::~JPEGEncoder() {}

torch::Tensor JPEGEncoder::compress(const torch::Tensor& img)
{
    ATCG_ERROR("JPEGEncoder not implemented for CPU backend!");
    return {};
}
}    // namespace atcg