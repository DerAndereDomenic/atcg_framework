#include <Core/Assert.h>
#include <Core/Application.h>
#include <Material/PhaseFunction.h>
#include <Renderer/Renderer.h>
#include <Renderer/Shader.h>

namespace atcg
{
PhaseFunction::PhaseFunction(const std::string& type, const atcg::Dictionary& dict) : _phase_function_type(type) {}

}    // namespace atcg