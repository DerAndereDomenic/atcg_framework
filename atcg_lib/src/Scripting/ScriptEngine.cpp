#include <Scripting/ScriptEngine.h>

#include <unordered_map>

#include <pyatcg.h>

#include <pybind11/embed.h>

namespace py = pybind11;

extern "C" PyObject* PyInit_pyatcg();

namespace atcg
{

class PythonScriptEngine::Impl
{
public:
    Impl();

    ~Impl();

    std::unordered_map<atcg::UUID, atcg::ref_ptr<Script>> _scripts;
    py::module_ pyatcg;

    bool initialized = false;
};

PythonScriptEngine::Impl::Impl() {}

PythonScriptEngine::Impl::~Impl() {}

PythonScriptEngine::PythonScriptEngine()
{
    impl = std::make_unique<Impl>();
}

PythonScriptEngine::~PythonScriptEngine() {}

void PythonScriptEngine::init()
{
    if(!impl->initialized)
    {
        if(PyImport_AppendInittab("pyatcg", &PyInit_pyatcg) == -1)
        {
            ATCG_ERROR("Failed to add engine module to interpreter.\n");
            return;
        }
        Py_Initialize();

        try
        {
            impl->pyatcg = py::module_::import("pyatcg");
        }
        catch(const py::error_already_set& e)
        {
            ATCG_ERROR(e.what());
        }
        catch(const std::exception& e)
        {
            ATCG_ERROR(e.what());
        }

        impl->initialized = true;
    }
}

UUID PythonScriptEngine::registerScript(const atcg::ref_ptr<atcg::Script>& script)
{
    UUID uuid;

    impl->_scripts.insert(std::make_pair(uuid, script));

    return uuid;
}

void PythonScriptEngine::unregisterScript(const UUID id)
{
    impl->_scripts.erase(id);
}

atcg::ref_ptr<Script> PythonScriptEngine::getScript(const UUID id)
{
    return impl->_scripts[id];
}

void PythonScriptEngine::reloadScripts()
{
    for(auto& script: impl->_scripts)
    {
        script.second->reload();
    }
}

void PythonScriptEngine::destroy()
{
    if(impl->initialized)
    {
        impl->pyatcg.release();
        Py_Finalize();
        impl->initialized = false;
    }
}

}    // namespace atcg