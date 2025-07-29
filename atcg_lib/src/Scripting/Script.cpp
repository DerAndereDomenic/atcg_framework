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

PythonScript::PythonScript(const std::filesystem::path& file_path) : Script(file_path)
{
    impl = std::make_unique<Impl>();
}

PythonScript::~PythonScript() {}

void PythonScript::init()
{
    std::filesystem::path script_dir = _file_path.parent_path();
    std::string module_name          = _file_path.stem().string();

    py::module_ sys = py::module_::import("sys");
    sys.attr("path").attr("insert")(0, script_dir.string());


    try
    {
        impl->script = py::module_::import(module_name.c_str());
    }
    catch(const py::error_already_set& e)
    {
        ATCG_ERROR(e.what());
    }
    catch(const std::exception& e)
    {
        ATCG_ERROR(e.what());
    }

    ATCG_INFO("Initialized Script {}", module_name);
}

void PythonScript::onAttach(const atcg::ref_ptr<Scene>& scene, Entity entity)
{
    try
    {
        impl->script.attr("onAttach")(py::cast(scene), py::cast(entity));
    }
    catch(const py::error_already_set& e)
    {
        ATCG_ERROR(e.what());
    }
    catch(const std::exception& e)
    {
        ATCG_ERROR(e.what());
    }
}

void PythonScript::onUpdate(const atcg::ref_ptr<Scene>& scene, Entity entity, const float delta_time)
{
    try
    {
        impl->script.attr("onUpdate")(delta_time, py::cast(scene), py::cast(entity));
    }
    catch(const py::error_already_set& e)
    {
        ATCG_ERROR(e.what());
    }
    catch(const std::exception& e)
    {
        ATCG_ERROR(e.what());
    }
}

void PythonScript::onEvent(const atcg::ref_ptr<Scene>& scene, Entity entity, atcg::Event* event)
{
    try
    {
        impl->script.attr("onEvent")(py::cast(event), py::cast(scene), py::cast(entity));
    }
    catch(const py::error_already_set& e)
    {
        ATCG_ERROR(e.what());
    }
    catch(const std::exception& e)
    {
        ATCG_ERROR(e.what());
    }
}

void PythonScript::onDetach(const atcg::ref_ptr<Scene>& scene, Entity entity)
{
    try
    {
        impl->script.attr("onDetach")(py::cast(scene), py::cast(entity));
    }
    catch(const py::error_already_set& e)
    {
        ATCG_ERROR(e.what());
    }
    catch(const std::exception& e)
    {
        ATCG_ERROR(e.what());
    }
}

void PythonScript::reload()
{
    try
    {
        impl->script.reload();
    }
    catch(const py::error_already_set& e)
    {
        ATCG_ERROR(e.what());
    }
    catch(const std::exception& e)
    {
        ATCG_ERROR(e.what());
    }
}

void Scripting::handleScriptReloads(const atcg::ref_ptr<atcg::Scene>& scene)
{
    auto view = scene->getAllEntitiesWith<atcg::ScriptComponent>();

    for(auto e: view)
    {
        atcg::Entity entity(e, scene.get());

        auto& script = entity.getComponent<atcg::ScriptComponent>();

        if(script.script() != nullptr)
        {
            script.script()->onDetach(scene, entity);
            script.script()->reload();
            script.script()->onAttach(scene, entity);
        }
    }

    ATCG_INFO("Reloaded Scripts");
}

void Scripting::handleScriptEvents(const atcg::ref_ptr<atcg::Scene>& scene, atcg::Event* event)
{
    auto view = scene->getAllEntitiesWith<atcg::ScriptComponent>();

    for(auto e: view)
    {
        atcg::Entity entity(e, scene.get());

        auto& script = entity.getComponent<atcg::ScriptComponent>();

        if(script.script() != nullptr)
        {
            script.script()->onEvent(scene, entity, event);
        }
    }
}

void Scripting::handleScriptUpdates(const atcg::ref_ptr<atcg::Scene>& scene, const float dt)
{
    auto view = scene->getAllEntitiesWith<atcg::ScriptComponent>();

    for(auto e: view)
    {
        atcg::Entity entity(e, scene.get());

        auto& script = entity.getComponent<atcg::ScriptComponent>();

        if(script.script() != nullptr)
        {
            script.script()->onUpdate(scene, entity, dt);
        }
    }
}

}    // namespace atcg