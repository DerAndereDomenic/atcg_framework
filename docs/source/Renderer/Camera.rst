Camera
=======

.. doxygenclass:: atcg::CameraExtrinsics
   :members:
   :undoc-members:
.. doxygenclass:: atcg::CameraIntrinsics
   :members:
   :undoc-members:
.. doxygenfunction:: atcg::CameraUtils::convert_to_opencv(const glm::mat4& projection, const uint32_t width, const uint32_t height)
   :project: ATCGLIB
.. doxygenfunction:: atcg::CameraUtils::convert_to_opencv(const CameraIntrinsics& intrinsics, const uint32_t width, const uint32_t height)
   :project: ATCGLIB
.. doxygenfunction:: atcg::CameraUtils::convert_from_opencv(const float fx, const float fy, const float cx, const float cy,const float n,const float f,const uint32_t width,const uint32_t height)
   :project: ATCGLIB
.. doxygenfunction:: atcg::CameraUtils::convert_from_opencv(const glm::mat3& K, const float n, const float f, const uint32_t width, const uint32_t height)
   :project: ATCGLIB
.. doxygenclass:: atcg::Camera
   :members:
   :undoc-members:
.. doxygenclass:: atcg::OrthographicCamera
   :members:
   :undoc-members:
.. doxygenclass:: atcg::PerspectiveCamera
   :members:
   :undoc-members:
.. doxygenclass:: atcg::CameraController
   :members:
   :undoc-members:
.. doxygenclass:: atcg::FocusedController
   :members:
   :undoc-members:
.. doxygenclass:: atcg::FirstPersonController
   :members:
   :undoc-members:
.. doxygenclass:: atcg::VRController
   :members:
   :undoc-members: