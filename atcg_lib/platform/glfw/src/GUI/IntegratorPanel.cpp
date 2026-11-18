#include <GUI/IntegratorPanel.h>

#ifdef ATCG_CUDA_BACKEND

    #include <Integrator/IntegratorRegistry.h>

    #include <imgui.h>

namespace atcg
{
namespace GUI
{
void IntegratorPanel::renderPanel(bool& show_window)
{
    if(!show_window || !_controller)
    {
        return;
    }

    ImGui::Begin("Integrator##atcg_lib", &show_window);

    if(_controller->isRunning())
    {
        if(ImGui::Button("Stop"))
        {
            _controller->stop();
        }
    }
    else
    {
        if(ImGui::Button("Start"))
        {
            _controller->start();
        }
    }

    const std::vector<std::string>& integrator_names = IntegratorRegistry::getRegistry()->getRegisteredTypes();
    std::string current_type                         = _controller->getIntegratorType();

    if(ImGui::BeginCombo("Integrator", current_type.c_str()))
    {
        for(const auto& name: integrator_names)
        {
            const bool is_selected = (name == current_type);
            if(ImGui::Selectable(name.c_str(), is_selected))
            {
                _controller->setIntegratorType(name);
            }

            // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
            if(is_selected) ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }

    auto settings = _controller->getSettings();
    if(settings && settings->renderSettings())
    {
        _controller->requestRebuild();
    }

    ImGui::End();

    // The integrator draws its own debug windows
    auto integrator = _controller->getIntegrator();
    if(_controller->isRunning() && integrator)
    {
        integrator->onImGuiRender();
    }
}
}    // namespace GUI
}    // namespace atcg

#endif
