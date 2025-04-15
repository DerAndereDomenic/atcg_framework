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
};

PythonScript::Impl::Impl() {}

PythonScript::Impl::~Impl() {}

PythonScript::PythonScript()
{
    impl = std::make_unique<Impl>();
}

PythonScript::~PythonScript() {}

void PythonScript::init(const std::filesystem::path& file_path)
{
    std::filesystem::path script_dir = file_path.parent_path();
    std::string module_name          = file_path.stem().string();

    py::module_ sys = py::module_::import("sys");
    sys.attr("path").attr("insert")(0, script_dir.string());

    impl->script = py::module_::import(module_name.c_str());
}

void PythonScript::onAttach(const atcg::ref_ptr<atcg::Scene>& scene, const atcg::Entity& entity)
{
    impl->script.attr("onAttach")(py::cast(scene), py::cast(entity));
}

void PythonScript::onUpdate(const float delta_time, const atcg::ref_ptr<atcg::Scene>& scene, const atcg::Entity& entity)
{
    impl->script.attr("onUpdate")(delta_time, py::cast(scene), py::cast(entity));
}

void PythonScript::reload()
{
    impl->script.reload();
}

}    // namespace atcg