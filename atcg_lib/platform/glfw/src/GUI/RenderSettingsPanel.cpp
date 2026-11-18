#include <GUI/RenderSettingsPanel.h>

#include <Core/Application.h>
#include <Scene/SceneRenderer.h>

#include <imgui.h>

namespace atcg
{
namespace GUI
{
namespace detail
{
template<size_t N>
void renderValueCombo(const char* label, const uint32_t (&values)[N], uint32_t current, void (*setter)(uint32_t))
{
    std::string preview = std::to_string(current);
    if(ImGui::BeginCombo(label, preview.c_str()))
    {
        for(uint32_t value: values)
        {
            const bool is_selected = (value == current);
            if(ImGui::Selectable(std::to_string(value).c_str(), is_selected))
            {
                setter(value);
            }

            // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
            if(is_selected) ImGui::SetItemDefaultFocus();
        }
        ImGui::EndCombo();
    }
}
}    // namespace detail

void RenderSettingsPanel::renderPanel(bool& show_window)
{
    if(!show_window)
    {
        return;
    }

    ImGui::Begin("Render Settings##atcg_lib", &show_window);

    const auto& window = atcg::Application::get()->getWindow();
    bool vsync         = window->isVSync();
    if(ImGui::Checkbox("VSync", &vsync))
    {
        window->toggleVSync(vsync);
    }

    constexpr uint32_t msaa_samples[]       = {1, 2, 4, 8, 16, 32};
    constexpr uint32_t shadow_resolutions[] = {256, 512, 1024, 2048, 4096};

    detail::renderValueCombo("MSAA Samples",
                             msaa_samples,
                             SceneRenderer::getNumberMSAASamples(),
                             &SceneRenderer::setNumberMSAASamples);
    detail::renderValueCombo("Shadow Map Resolution",
                             shadow_resolutions,
                             SceneRenderer::getShadowPassResolution(),
                             &SceneRenderer::setShadowPassResolution);

    ImGui::End();
}
}    // namespace GUI
}    // namespace atcg
