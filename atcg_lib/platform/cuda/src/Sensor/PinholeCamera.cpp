#include <Sensor/PinholeCamera.h>
#include <Renderer/PerspectiveCamera.h>

namespace atcg
{
PinholeCamera::PinholeCamera(const Dictionary& dict)
{
    _camera = dict.getValue<atcg::ref_ptr<Camera>>("camera");
    _film   = dict.getValue<atcg::ref_ptr<Film>>("film");
}

PinholeCamera::~PinholeCamera() {}

void PinholeCamera::onImGuiRender() {}

void PinholeCamera::markDirty()
{
    PinholeCameraData pinhole_camera_data;

    atcg::ref_ptr<PerspectiveCamera> camera = std::dynamic_pointer_cast<PerspectiveCamera>(_camera);

    glm::mat4 inv_camera_view          = glm::inverse(camera->getView());
    pinhole_camera_data.cam_eye        = inv_camera_view[3];
    pinhole_camera_data.U              = glm::normalize(inv_camera_view[0]);
    pinhole_camera_data.V              = glm::normalize(inv_camera_view[1]);
    pinhole_camera_data.W              = -glm::normalize(inv_camera_view[2]);
    pinhole_camera_data.aspect_ratio   = camera->getAspectRatio();
    pinhole_camera_data.fov_y          = camera->getFOV();
    pinhole_camera_data.optical_center = camera->getIntrinsics().opticalCenter();
    pinhole_camera_data.film           = _film->getVPtrTable();
    pinhole_camera_data.exposure       = camera->getIntrinsics().getExposure();

    _pinhole_camera_data.upload(&pinhole_camera_data);
}

void PinholeCamera::initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                       const atcg::ref_ptr<ShaderBindingTable>& sbt)
{
    if(!_film) return;
    _film->ensureInitialized(pipeline, sbt);
    markDirty();

    const std::string ptx_sensor_filename = "./bin/PinholeCamera_ptx.ptx";
    auto generate_ray_prog_group =
        pipeline->addCallableShader({ptx_sensor_filename, "__direct_callable__generate_ray_pinhole"});
    auto add_sample_prog_group =
        pipeline->addCallableShader({ptx_sensor_filename, "__direct_callable__add_sample_pinhole"});
    uint32_t generate_ray_idx = sbt->addCallableEntry(generate_ray_prog_group, _pinhole_camera_data.get());
    uint32_t add_sample_idx   = sbt->addCallableEntry(add_sample_prog_group, _pinhole_camera_data.get());

    SensorVPtrTable vptr_table;
    vptr_table.generateRayCallIndex = generate_ray_idx;
    vptr_table.addSampleCallIndex   = add_sample_idx;

    _vptr_table.upload(&vptr_table);
    markInitialized();
}
}    // namespace atcg