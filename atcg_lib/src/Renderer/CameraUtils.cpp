#include <Renderer/CameraUtils.h>

namespace atcg
{

CameraExtrinsics::CameraExtrinsics(const glm::vec3& position, const glm::vec3& target)
    : _position(position),
      _target(target)
{
    recalculateView();
}

CameraExtrinsics::CameraExtrinsics(const glm::mat4& extrinsics_matrix) : _extrinsics_matrix(extrinsics_matrix)
{
    glm::mat4 inv_view = glm::inverse(_extrinsics_matrix);
    _position          = glm::vec3(inv_view[3]);
    _target            = _position - glm::vec3(inv_view[2]);
}

void CameraExtrinsics::setPosition(const glm::vec3& position)
{
    _position = position;
    recalculateView();
}

void CameraExtrinsics::setTarget(const glm::vec3& target)
{
    _target = target;
    recalculateView();
}

void CameraExtrinsics::setExtrinsicMatrix(const glm::mat4& extrinsics)
{
    _extrinsics_matrix = extrinsics;

    glm::mat4 inv_view = glm::inverse(_extrinsics_matrix);
    _position          = glm::vec3(inv_view[3]);
    _target            = _position - glm::vec3(inv_view[2]);
}

void CameraExtrinsics::recalculateView()
{
    _extrinsics_matrix = glm::lookAt(_position, _target, glm::vec3(0, 1, 0));
}

CameraIntrinsics::CameraIntrinsics(const float aspect_ratio,
                                   const float fov_y,
                                   const float znear,
                                   const float zfar,
                                   const glm::vec2& optical_center)
    : _aspect_ratio(aspect_ratio),
      _fov_y(fov_y),
      _near(znear),
      _far(zfar),
      _optical_center(optical_center)
{
    recalculateProjection();
}

CameraIntrinsics::CameraIntrinsics(const glm::mat4& projection) : _projection(projection)
{
    _aspect_ratio   = _projection[1][1] / _projection[0][0];
    _fov_y          = glm::degrees(2.0f * glm::atan(1.0f / _projection[1][1]));
    _optical_center = glm::vec2(_projection[2][0], _projection[2][1]);
    _near           = _projection[3][2] / (_projection[2][2] - 1.0f);
    _far            = _projection[3][2] / (_projection[2][2] + 1.0f);
}

void CameraIntrinsics::setAspectRatio(const float aspect_ratio)
{
    _aspect_ratio = aspect_ratio;
    recalculateProjection();
}

void CameraIntrinsics::setFOV(const float fov_y)
{
    _fov_y = fov_y;
    recalculateProjection();
}

void CameraIntrinsics::setNear(const float znear)
{
    _near = znear;
    recalculateProjection();
}

void CameraIntrinsics::setFar(const float zfar)
{
    _far = zfar;
    recalculateProjection();
}

void CameraIntrinsics::setOpticalCenter(const glm::vec2& optical_center)
{
    _optical_center = optical_center;
    recalculateProjection();
}

void CameraIntrinsics::setProjection(const glm::mat4& projection)
{
    _projection = projection;

    _aspect_ratio   = _projection[1][1] / _projection[0][0];
    _fov_y          = glm::degrees(2.0f * glm::atan(1.0f / _projection[1][1]));
    _optical_center = glm::vec2(_projection[2][0], _projection[2][1]);
    _near           = _projection[3][2] / (_projection[2][2] - 1.0f);
    _far            = _projection[3][2] / (_projection[2][2] + 1.0f);
}

void CameraIntrinsics::recalculateProjection()
{
    _projection       = glm::perspective(glm::radians(_fov_y), _aspect_ratio, _near, _far);
    _projection[2][0] = _optical_center.x;
    _projection[2][1] = _optical_center.y;
}

glm::mat3 CameraUtils::convert_to_opencv(const glm::mat4& projection, uint32_t width, const uint32_t height)
{
    float fx = projection[0][0] * (float)width / 2.0f;
    float fy = projection[1][1] * (float)height / 2.0f;
    float cx = (1.0f - projection[2][0]) / 2.0f * (float)width;
    float cy = (1.0f + projection[2][1]) / 2.0f * (float)height;

    glm::mat3 K(1);

    K[0][0] = fx;
    K[1][1] = fy;
    K[0][2] = cx;
    K[1][2] = cy;

    return K;
}

glm::mat3
CameraUtils::convert_to_opencv(const CameraIntrinsics& intrinsics, const uint32_t width, const uint32_t height)
{
    return convert_to_opencv(intrinsics.projection(), width, height);
}

CameraIntrinsics CameraUtils::convert_from_opencv(const float fx,
                                                  const float fy,
                                                  const float cx,
                                                  const float cy,
                                                  float n,
                                                  float f,
                                                  const uint32_t width,
                                                  const uint32_t height)
{
    glm::mat4 intrinsic_matrix(0);
    intrinsic_matrix[0][0] = 2.0f * fx / (float)width;
    intrinsic_matrix[1][1] = 2.0f * fy / (float)height;
    intrinsic_matrix[2][0] = ((float)width - 2.0f * cx) / ((float)width);
    intrinsic_matrix[2][1] = (-(float)height + 2.0f * cy) / ((float)height);
    intrinsic_matrix[2][2] = -(f + n) / (f - n);
    intrinsic_matrix[3][2] = -(2.0f * f * n) / (f - n);
    intrinsic_matrix[2][3] = -1.0f;

    return CameraIntrinsics(intrinsic_matrix);
}

CameraIntrinsics
CameraUtils::convert_from_opencv(const glm::mat3& K, float n, float f, const uint32_t width, const uint32_t height)
{
    float fx = K[0][0];
    float fy = K[1][1];
    float cx = K[0][2];
    float cy = K[1][2];

    return convert_from_opencv(fx, fy, cx, cy, n, f, width, height);
}
}    // namespace atcg