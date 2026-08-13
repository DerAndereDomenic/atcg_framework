#include <Sensor/HemisphereCamera.h>
#include <Renderer/PerspectiveCamera.h>

namespace atcg
{
HemisphereCamera::HemisphereCamera(const Dictionary& dict)
{
    _position = dict.getValue<glm::vec3>("position");
    _normal   = dict.getValue<glm::vec3>("normal");
    _exposure = dict.getValueOr<float>("exposure", 1.0f);
    _film     = dict.getValue<atcg::ref_ptr<Film>>("film");
}

HemisphereCamera::~HemisphereCamera() {}

void HemisphereCamera::markDirty()
{
    HemisphereCameraData hemisphere_camera_data;

    hemisphere_camera_data.cam_eye  = _position;
    hemisphere_camera_data.normal   = _normal;
    hemisphere_camera_data.exposure = _exposure;
    hemisphere_camera_data.film     = _film->getVPtrTable();

    _hemisphere_camera_data.upload(&hemisphere_camera_data);
}

void HemisphereCamera::initializePipeline(const atcg::ref_ptr<RayTracingPipeline>& pipeline,
                                          const atcg::ref_ptr<ShaderBindingTable>& sbt)
{
    if(!_film) return;
    _film->ensureInitialized(pipeline, sbt);
    markDirty();

    const std::string ptx_sensor_filename = "./bin/HemisphereCamera_ptx.ptx";
    auto generate_ray_prog_group =
        pipeline->addCallableShader({ptx_sensor_filename, "__direct_callable__generate_ray_hemisphere"});
    auto add_sample_prog_group =
        pipeline->addCallableShader({ptx_sensor_filename, "__direct_callable__add_sample_hemisphere"});
    uint32_t generate_ray_idx = sbt->addCallableEntry(generate_ray_prog_group, _hemisphere_camera_data.get());
    uint32_t add_sample_idx   = sbt->addCallableEntry(add_sample_prog_group, _hemisphere_camera_data.get());

    SensorVPtrTable vptr_table;
    vptr_table.generateRayCallIndex = generate_ray_idx;
    vptr_table.addSampleCallIndex   = add_sample_idx;

    _vptr_table.upload(&vptr_table);
    markInitialized();
}
}    // namespace atcg