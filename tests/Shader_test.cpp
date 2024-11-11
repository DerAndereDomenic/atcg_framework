#include <gtest/gtest.h>
#include <Core/Path.h>
#include <Renderer/Shader.h>
#include <Renderer/ShaderManager.h>


TEST(ShaderTest, standardShader)
{
    auto shader = atcg::make_ref<atcg::Shader>((atcg::shader_directory() / "base.vs").string(),
                                               (atcg::shader_directory() / "base.fs").string());

    EXPECT_EQ(shader->getVertexPath(), (atcg::shader_directory() / "base.vs").string());
    EXPECT_EQ(shader->getFragmentPath(), (atcg::shader_directory() / "base.fs").string());
    EXPECT_EQ(shader->isComputeShader(), false);
    EXPECT_EQ(shader->hasGeometryShader(), false);
}

TEST(ShaderTest, geometryShader)
{
    auto shader = atcg::make_ref<atcg::Shader>((atcg::shader_directory() / "edge.vs").string(),
                                               (atcg::shader_directory() / "edge.fs").string(),
                                               (atcg::shader_directory() / "edge.gs").string());

    EXPECT_EQ(shader->getVertexPath(), (atcg::shader_directory() / "edge.vs").string());
    EXPECT_EQ(shader->getFragmentPath(), (atcg::shader_directory() / "edge.fs").string());
    EXPECT_EQ(shader->getGeometryPath(), (atcg::shader_directory() / "edge.gs").string());
    EXPECT_EQ(shader->isComputeShader(), false);
    EXPECT_EQ(shader->hasGeometryShader(), true);
}

TEST(ShaderTest, computeShader)
{
    auto shader = atcg::make_ref<atcg::Shader>("src/Mandelbulb/Mandelbulb.glsl");

    EXPECT_EQ(shader->getComputePath(), "src/Mandelbulb/Mandelbulb.glsl");
    EXPECT_EQ(shader->isComputeShader(), true);
    EXPECT_EQ(shader->hasGeometryShader(), false);
}

TEST(ShaderTest, recompileStandardShader)
{
    auto shader = atcg::make_ref<atcg::Shader>((atcg::shader_directory() / "base.vs").string(),
                                               (atcg::shader_directory() / "base.fs").string());
    shader->recompile((atcg::shader_directory() / "base.vs").string(), (atcg::shader_directory() / "base.fs").string());

    EXPECT_EQ(shader->getVertexPath(), (atcg::shader_directory() / "base.vs").string());
    EXPECT_EQ(shader->getFragmentPath(), (atcg::shader_directory() / "base.fs").string());
    EXPECT_EQ(shader->isComputeShader(), false);
    EXPECT_EQ(shader->hasGeometryShader(), false);
}

TEST(ShaderTest, recompileGeometryShader)
{
    auto shader = atcg::make_ref<atcg::Shader>((atcg::shader_directory() / "edge.vs").string(),
                                               (atcg::shader_directory() / "edge.fs").string(),
                                               (atcg::shader_directory() / "edge.gs").string());
    shader->recompile((atcg::shader_directory() / "edge.vs").string(),
                      (atcg::shader_directory() / "edge.fs").string(),
                      (atcg::shader_directory() / "edge.gs").string());

    EXPECT_EQ(shader->getVertexPath(), (atcg::shader_directory() / "edge.vs").string());
    EXPECT_EQ(shader->getFragmentPath(), (atcg::shader_directory() / "edge.fs").string());
    EXPECT_EQ(shader->getGeometryPath(), (atcg::shader_directory() / "edge.gs").string());
    EXPECT_EQ(shader->isComputeShader(), false);
    EXPECT_EQ(shader->hasGeometryShader(), true);
}

TEST(ShaderTest, recompileComputeShader)
{
    auto shader = atcg::make_ref<atcg::Shader>("src/Mandelbulb/Mandelbulb.glsl");
    shader->recompile("src/Mandelbulb/Mandelbulb.glsl");

    EXPECT_EQ(shader->getComputePath(), "src/Mandelbulb/Mandelbulb.glsl");
    EXPECT_EQ(shader->isComputeShader(), true);
    EXPECT_EQ(shader->hasGeometryShader(), false);
}

TEST(ShaderTest, recompileStandardToGeometryShader)
{
    auto shader = atcg::make_ref<atcg::Shader>((atcg::shader_directory() / "base.vs").string(),
                                               (atcg::shader_directory() / "base.fs").string());
    shader->recompile((atcg::shader_directory() / "edge.vs").string(),
                      (atcg::shader_directory() / "edge.fs").string(),
                      (atcg::shader_directory() / "edge.gs").string());

    EXPECT_EQ(shader->getVertexPath(), (atcg::shader_directory() / "edge.vs").string());
    EXPECT_EQ(shader->getFragmentPath(), (atcg::shader_directory() / "edge.fs").string());
    EXPECT_EQ(shader->getGeometryPath(), (atcg::shader_directory() / "edge.gs").string());
    EXPECT_EQ(shader->isComputeShader(), false);
    EXPECT_EQ(shader->hasGeometryShader(), true);
}

TEST(ShaderTest, recompileGeometryToStandardShader)
{
    auto shader = atcg::make_ref<atcg::Shader>((atcg::shader_directory() / "edge.vs").string(),
                                               (atcg::shader_directory() / "edge.fs").string(),
                                               (atcg::shader_directory() / "edge.gs").string());
    shader->recompile((atcg::shader_directory() / "base.vs").string(), (atcg::shader_directory() / "base.fs").string());

    EXPECT_EQ(shader->getVertexPath(), (atcg::shader_directory() / "base.vs").string());
    EXPECT_EQ(shader->getFragmentPath(), (atcg::shader_directory() / "base.fs").string());
    EXPECT_EQ(shader->isComputeShader(), false);
    EXPECT_EQ(shader->hasGeometryShader(), false);
}

TEST(ShaderTest, recompileComputeToStandardShader)
{
    auto shader = atcg::make_ref<atcg::Shader>("src/Mandelbulb/Mandelbulb.glsl");
    shader->recompile((atcg::shader_directory() / "base.vs").string(), (atcg::shader_directory() / "base.fs").string());

    EXPECT_EQ(shader->getVertexPath(), (atcg::shader_directory() / "base.vs").string());
    EXPECT_EQ(shader->getFragmentPath(), (atcg::shader_directory() / "base.fs").string());
    EXPECT_EQ(shader->isComputeShader(), false);
    EXPECT_EQ(shader->hasGeometryShader(), false);
}

TEST(ShaderTest, recompileStandardToComputeShader)
{
    auto shader = atcg::make_ref<atcg::Shader>((atcg::shader_directory() / "base.vs").string(),
                                               (atcg::shader_directory() / "base.fs").string());
    shader->recompile("src/Mandelbulb/Mandelbulb.glsl");

    EXPECT_EQ(shader->getComputePath(), "src/Mandelbulb/Mandelbulb.glsl");
    EXPECT_EQ(shader->isComputeShader(), true);
    EXPECT_EQ(shader->hasGeometryShader(), false);
}

TEST(ShaderManagerTest, addShader)
{
    auto shader  = atcg::make_ref<atcg::Shader>((atcg::shader_directory() / "base.vs").string(),
                                               (atcg::shader_directory() / "base.fs").string());
    auto manager = atcg::make_ref<atcg::ShaderManagerSystem>();

    manager->addShader("base", shader);

    EXPECT_EQ(manager->getShader("base") == shader, true);
    EXPECT_EQ(manager->hasShader("base"), true);
}

TEST(ShaderManagerTest, addShaderRecompile)
{
    auto shader = atcg::make_ref<atcg::Shader>((atcg::shader_directory() / "base.vs").string(),
                                               (atcg::shader_directory() / "base.fs").string());
    shader->recompile((atcg::shader_directory() / "base.vs").string(), (atcg::shader_directory() / "base.fs").string());
    auto manager = atcg::make_ref<atcg::ShaderManagerSystem>();

    manager->addShader("base", shader);

    EXPECT_EQ(manager->getShader("base") == shader, true);
    EXPECT_EQ(manager->hasShader("base"), true);
}

TEST(ShaderManagerTest, invalidShader)
{
    auto manager = atcg::make_ref<atcg::ShaderManagerSystem>();

    EXPECT_EQ(manager->hasShader("base"), false);
}

TEST(ShaderManagerTest, addShaderFromName)
{
    auto manager = atcg::make_ref<atcg::ShaderManagerSystem>();

    manager->addShaderFromName("base");

    EXPECT_EQ(manager->hasShader("base"), true);
    EXPECT_EQ(manager->getShader("base")->hasGeometryShader(), false);
    EXPECT_EQ(manager->getShader("base")->isComputeShader(), false);
}

TEST(ShaderManagerTest, addShaderFromNameGeometry)
{
    auto manager = atcg::make_ref<atcg::ShaderManagerSystem>();

    manager->addShaderFromName("edge");

    EXPECT_EQ(manager->hasShader("edge"), true);
    EXPECT_EQ(manager->getShader("edge")->hasGeometryShader(), true);
    EXPECT_EQ(manager->getShader("edge")->isComputeShader(), false);
}

TEST(ShaderManagerTest, addComputeShader)
{
    auto manager = atcg::make_ref<atcg::ShaderManagerSystem>();
    manager->setShaderPath("src/Mandelbulb");

    manager->addComputeShaderFromName("Mandelbulb");

    EXPECT_EQ(manager->hasShader("Mandelbulb"), true);
    EXPECT_EQ(manager->getShader("Mandelbulb")->hasGeometryShader(), false);
    EXPECT_EQ(manager->getShader("Mandelbulb")->isComputeShader(), true);
    EXPECT_EQ(manager->getShaderPath(), "src/Mandelbulb");
}