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

atcg::ref_ptr<Behavior> PythonScript::createBehavior(const atcg::ref_ptr<Scene>& scene, atcg::Entity entity)
{
    std::string module_name = _file_path.stem().string();

    try
    {
        auto behavior_class = impl->script.attr(module_name.c_str());
        auto instance       = behavior_class();

        instance.attr("entity") = py::cast(entity);
        instance.attr("scene")  = py::cast(scene);

        atcg::ref_ptr<PythonBehavior> py_behavior = instance.cast<atcg::ref_ptr<PythonBehavior>>();
        py_behavior->setSelf(instance);

        return py_behavior;
    }
    catch(const std::exception& e)
    {
        ATCG_ERROR(e.what());
        return nullptr;
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

        if(!script.script())
        {
            continue;
        }

        auto behavior = script.behavior(scene, entity);

        if(behavior) behavior->onDetach();
        script.script()->reload();
        behavior = script.behavior(scene, entity, true);
        if(behavior) behavior->onAttach();
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

        if(!script.script())
        {
            continue;
        }

        auto behavior = script.behavior(scene, entity);

        if(behavior) behavior->onEvent(event);
    }
}

void Scripting::handleScriptUpdates(const atcg::ref_ptr<atcg::Scene>& scene, const float dt)
{
    auto view = scene->getAllEntitiesWith<atcg::ScriptComponent>();

    for(auto e: view)
    {
        atcg::Entity entity(e, scene.get());

        auto& script = entity.getComponent<atcg::ScriptComponent>();

        if(!script.script())
        {
            continue;
        }

        auto behavior = script.behavior(scene, entity);

        if(behavior) behavior->onUpdate(dt);
    }
}

}    // namespace atcg