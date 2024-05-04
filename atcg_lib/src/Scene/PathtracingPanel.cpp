#include <Scene/PathtracingPanel.h>

#include <Core/Application.h>
#include <Pathtracing/Pathtracer.h>
#include <Pathtracing/RaytracingShaderManager.h>

#include <imgui.h>

namespace atcg
{
PathtracingPanel::PathtracingPanel(const atcg::ref_ptr<Scene>& scene) : _scene(scene) {}

void PathtracingPanel::renderPanel(const atcg::ref_ptr<PerspectiveCamera>& camera)
{
    ImGui::Begin("PT Panel");

    if(atcg::Pathtracer::isFinished())
    {
        ImGui::DragInt("Number Samples", &_num_samples, 1.0f, 0, INT_MAX);
        ImGui::Checkbox("Viewport size", &_use_viewport_size);

        auto imgui_layer = atcg::Application::get()->getImGuiLayer();
        if(!_use_viewport_size)
        {
            ImGui::DragInt("Width", &_width, 1.0f, 1, 4096);
            ImGui::DragInt("Height", &_height, 1.0f, 1, 4096);
        }
        else
        {
            auto size = imgui_layer->getViewportSize();
            _width    = size.x;
            _height   = size.y;
        }

        if(ImGui::Button("Start"))
        {
            imgui_layer->setPathtracingFocus();
            atcg::Pathtracer::draw(_scene,
                                   camera,
                                   atcg::RaytracingShaderManager::getShader("Pathtracing"),
                                   _width,
                                   _height,
                                   _num_samples);
        }
    }
    else
    {
        float progress = (float)atcg::Pathtracer::getFrameIndex() / (float)_num_samples;
        ImGui::ProgressBar(progress);
        // ImGui::SameLine();
        ImGui::Text((std::to_string(atcg::Pathtracer::getFrameIndex()) + "/" + std::to_string(_num_samples)).c_str());

        if(ImGui::Button("Stop"))
        {
            atcg::Pathtracer::stop();
        }
    }


    ImGui::End();
}
}    // namespace atcg