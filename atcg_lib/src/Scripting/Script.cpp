#include <Scripting/Script.h>

#include <pyatcg.h>

#include <pybind11/embed.h>

namespace atcg
{

class PythonScript::Impl
{
public:
    Impl();
    ~Impl();

    py::module_ script;

    atcg::ref_ptr<atcg::Scene> scene;
    atcg::Entity entity;
};

PythonScript::Impl::Impl() {}

PythonScript::Impl::~Impl() {}

PythonScript::PythonScript(const std::filesystem::path& file_path) : Script(file_path)
{
    impl = std::make_unique<Impl>();
}

PythonScript::~PythonScript() {}

void PythonScript::init(const atcg::ref_ptr<atcg::Scene>& scene, const atcg::Entity& entity)
{
    std::filesystem::path script_dir = _file_path.parent_path();
    std::string module_name          = _file_path.stem().string();

    py::module_ sys = py::module_::import("sys");
    sys.attr("path").attr("insert")(0, script_dir.string());

    impl->script = py::module_::import(module_name.c_str());

    impl->scene  = scene;
    impl->entity = entity;
}

void PythonScript::onAttach()
{
    impl->script.attr("onAttach")(py::cast(impl->scene), py::cast(impl->entity));
}

void PythonScript::onUpdate(const float delta_time)
{
    impl->script.attr("onUpdate")(delta_time, py::cast(impl->scene), py::cast(impl->entity));
}

void PythonScript::reload()
{
    impl->script.reload();
}

}    // namespace atcg