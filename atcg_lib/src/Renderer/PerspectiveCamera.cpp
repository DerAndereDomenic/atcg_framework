#include <Renderer/PerspectiveCamera.h>

namespace atcg
{
PerspectiveCamera::PerspectiveCamera(const CameraExtrinsics& extrinsics, const CameraIntrinsics& intrinsics)
{
    _extrinsics = extrinsics;
    _intrinsics = intrinsics;
}

atcg::ref_ptr<Camera> PerspectiveCamera::copy() const
{
    atcg::ref_ptr<PerspectiveCamera> camera = atcg::make_ref<PerspectiveCamera>(_extrinsics, _intrinsics);

    return camera;
}

}    // namespace atcg