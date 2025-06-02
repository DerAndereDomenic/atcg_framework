#include <Scripting/ScriptEngine.h>

#include <pyatcg.h>

#include <pybind11/embed.h>

namespace py = pybind11;

#define _ATCG_CONCAT(x, y) x##y
#define ATCG_CONCAT(x, y)  _ATCG_CONCAT(x, y)

extern "C" PyObject* ATCG_CONCAT(PyInit_, TORCH_EXTENSION_NAME)();

namespace atcg
{

class PythonScriptEngine::Impl
{
public:
    Impl();

    ~Impl();

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
#ifdef ATCG_INIT_PYTHON
        if(PyImport_AppendInittab("pyatcg", &ATCG_CONCAT(PyInit_, TORCH_EXTENSION_NAME)) == -1)
        {
            ATCG_ERROR("Failed to add engine module to interpreter.\n");
            return;
        }
        Py_Initialize();
#endif

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

        ATCG_INFO("Initialized Scripting engine");
    }
}

void PythonScriptEngine::destroy()
{
    if(impl->initialized)
    {
        ATCG_INFO("Shutting down Scripting engine...");
        impl->pyatcg.release();
#ifdef ATCG_INIT_PYTHON
        Py_Finalize();
#endif
        impl->initialized = false;
    }
}

}    // namespace atcg