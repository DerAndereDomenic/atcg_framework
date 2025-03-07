#include <gtest/gtest.h>
#include <Core/Path.h>
#include <Renderer/Framebuffer.h>
#include <Renderer/Renderer.h>
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

TEST(ShaderTest, uniformInt)
{
    atcg::ref_ptr<atcg::Graph> quad;
    std::vector<atcg::Vertex> vertices = {atcg::Vertex(glm::vec3(-1, -1, 0)),
                                          atcg::Vertex(glm::vec3(1, -1, 0)),
                                          atcg::Vertex(glm::vec3(1, 1, 0)),
                                          atcg::Vertex(glm::vec3(-1, 1, 0))};

    std::vector<glm::u32vec3> edges = {glm::u32vec3(0, 1, 2), glm::u32vec3(0, 2, 3)};

    quad = atcg::Graph::createTriangleMesh(vertices, edges);

    atcg::ref_ptr<atcg::Shader> shader =
        atcg::make_ref<atcg::Shader>("tests/data/shader/test_vertex.vs", "tests/data/shader/test_int.fs");

    atcg::Framebuffer fbo(1, 1);
    fbo.attachColor();
    fbo.attachDepth();

    fbo.use();

    atcg::Renderer::setViewport(0, 0, 1, 1);
    atcg::Renderer::setClearColor(glm::vec4(0, 0, 0, 1));

    {
        shader->setInt("unif", 0);
        atcg::Renderer::clear();
        atcg::Renderer::draw(quad, nullptr, glm::mat4(1), glm::vec3(1), shader);
        auto data = fbo.getColorAttachement(0)->getData(atcg::CPU);

        EXPECT_NEAR(data[0][0][0].item<float>(), 0.0f, 1e-5f);
    }

    {
        shader->setInt("unif", 128);
        atcg::Renderer::clear();
        atcg::Renderer::draw(quad, nullptr, glm::mat4(1), glm::vec3(1), shader);
        auto data = fbo.getColorAttachement(0)->getData(atcg::CPU);

        EXPECT_NEAR(data[0][0][0].item<float>(), 128.0f, 1e-5f);
    }

    {
        shader->setInt("unif", 255);
        atcg::Renderer::clear();
        atcg::Renderer::draw(quad, nullptr, glm::mat4(1), glm::vec3(1), shader);
        auto data = fbo.getColorAttachement(0)->getData(atcg::CPU);

        EXPECT_NEAR(data[0][0][0].item<float>(), 255.0f, 1e-5f);
    }

    atcg::Renderer::useScreenBuffer();
    atcg::Renderer::setDefaultViewport();
}

TEST(ShaderTest, uniformFloat)
{
    atcg::ref_ptr<atcg::Graph> quad;
    std::vector<atcg::Vertex> vertices = {atcg::Vertex(glm::vec3(-1, -1, 0)),
                                          atcg::Vertex(glm::vec3(1, -1, 0)),
                                          atcg::Vertex(glm::vec3(1, 1, 0)),
                                          atcg::Vertex(glm::vec3(-1, 1, 0))};

    std::vector<glm::u32vec3> edges = {glm::u32vec3(0, 1, 2), glm::u32vec3(0, 2, 3)};

    quad = atcg::Graph::createTriangleMesh(vertices, edges);

    atcg::ref_ptr<atcg::Shader> shader =
        atcg::make_ref<atcg::Shader>("tests/data/shader/test_vertex.vs", "tests/data/shader/test_float.fs");

    atcg::Framebuffer fbo(1, 1);
    fbo.attachColor();
    fbo.attachDepth();

    fbo.use();

    atcg::Renderer::setViewport(0, 0, 1, 1);
    atcg::Renderer::setClearColor(glm::vec4(0, 0, 0, 1));

    {
        shader->setFloat("unif", 0);
        atcg::Renderer::clear();
        atcg::Renderer::draw(quad, nullptr, glm::mat4(1), glm::vec3(1), shader);
        auto data = fbo.getColorAttachement(0)->getData(atcg::CPU);

        EXPECT_NEAR(data[0][0][0].item<float>(), 0.0f, 1e-5f);
    }

    {
        shader->setFloat("unif", 0.5f);
        atcg::Renderer::clear();
        atcg::Renderer::draw(quad, nullptr, glm::mat4(1), glm::vec3(1), shader);
        auto data = fbo.getColorAttachement(0)->getData(atcg::CPU);

        EXPECT_NEAR(data[0][0][0].item<float>(), 127.0f, 1e-5f);
    }

    {
        shader->setFloat("unif", 1.0f);
        atcg::Renderer::clear();
        atcg::Renderer::draw(quad, nullptr, glm::mat4(1), glm::vec3(1), shader);
        auto data = fbo.getColorAttachement(0)->getData(atcg::CPU);

        EXPECT_NEAR(data[0][0][0].item<float>(), 255.0f, 1e-5f);
    }

    atcg::Renderer::useScreenBuffer();
    atcg::Renderer::setDefaultViewport();
}

TEST(ShaderTest, uniformVec2)
{
    atcg::ref_ptr<atcg::Graph> quad;
    std::vector<atcg::Vertex> vertices = {atcg::Vertex(glm::vec3(-1, -1, 0)),
                                          atcg::Vertex(glm::vec3(1, -1, 0)),
                                          atcg::Vertex(glm::vec3(1, 1, 0)),
                                          atcg::Vertex(glm::vec3(-1, 1, 0))};

    std::vector<glm::u32vec3> edges = {glm::u32vec3(0, 1, 2), glm::u32vec3(0, 2, 3)};

    quad = atcg::Graph::createTriangleMesh(vertices, edges);

    atcg::ref_ptr<atcg::Shader> shader =
        atcg::make_ref<atcg::Shader>("tests/data/shader/test_vertex.vs", "tests/data/shader/test_vec2.fs");

    atcg::Framebuffer fbo(1, 1);
    fbo.attachColor();
    fbo.attachDepth();

    fbo.use();

    atcg::Renderer::setViewport(0, 0, 1, 1);
    atcg::Renderer::setClearColor(glm::vec4(0, 0, 0, 1));

    {
        shader->setVec2("unif", glm::vec2(0, 0));
        atcg::Renderer::clear();
        atcg::Renderer::draw(quad, nullptr, glm::mat4(1), glm::vec3(1), shader);
        auto data = fbo.getColorAttachement(0)->getData(atcg::CPU);

        EXPECT_NEAR(data[0][0][0].item<float>(), 0.0f, 1e-5f);
        EXPECT_NEAR(data[0][0][1].item<float>(), 0.0f, 1e-5f);
    }

    {
        shader->setVec2("unif", glm::vec2(0.5, 0));
        atcg::Renderer::clear();
        atcg::Renderer::draw(quad, nullptr, glm::mat4(1), glm::vec3(1), shader);
        auto data = fbo.getColorAttachement(0)->getData(atcg::CPU);

        EXPECT_NEAR(data[0][0][0].item<float>(), 127.0f, 1e-5f);
        EXPECT_NEAR(data[0][0][1].item<float>(), 0.0f, 1e-5f);
    }

    {
        shader->setVec2("unif", glm::vec2(0, 0.5));
        atcg::Renderer::clear();
        atcg::Renderer::draw(quad, nullptr, glm::mat4(1), glm::vec3(1), shader);
        auto data = fbo.getColorAttachement(0)->getData(atcg::CPU);

        EXPECT_NEAR(data[0][0][0].item<float>(), 0.0f, 1e-5f);
        EXPECT_NEAR(data[0][0][1].item<float>(), 127.0f, 1e-5f);
    }

    {
        shader->setVec2("unif", glm::vec2(1.0, 1.0));
        atcg::Renderer::clear();
        atcg::Renderer::draw(quad, nullptr, glm::mat4(1), glm::vec3(1), shader);
        auto data = fbo.getColorAttachement(0)->getData(atcg::CPU);

        EXPECT_NEAR(data[0][0][0].item<float>(), 255.0f, 1e-5f);
        EXPECT_NEAR(data[0][0][1].item<float>(), 255.0f, 1e-5f);
    }

    atcg::Renderer::useScreenBuffer();
    atcg::Renderer::setDefaultViewport();
}

TEST(ShaderTest, uniformVec3)
{
    atcg::ref_ptr<atcg::Graph> quad;
    std::vector<atcg::Vertex> vertices = {atcg::Vertex(glm::vec3(-1, -1, 0)),
                                          atcg::Vertex(glm::vec3(1, -1, 0)),
                                          atcg::Vertex(glm::vec3(1, 1, 0)),
                                          atcg::Vertex(glm::vec3(-1, 1, 0))};

    std::vector<glm::u32vec3> edges = {glm::u32vec3(0, 1, 2), glm::u32vec3(0, 2, 3)};

    quad = atcg::Graph::createTriangleMesh(vertices, edges);

    atcg::ref_ptr<atcg::Shader> shader =
        atcg::make_ref<atcg::Shader>("tests/data/shader/test_vertex.vs", "tests/data/shader/test_vec3.fs");

    atcg::Framebuffer fbo(1, 1);
    fbo.attachColor();
    fbo.attachDepth();

    fbo.use();

    atcg::Renderer::setViewport(0, 0, 1, 1);
    atcg::Renderer::setClearColor(glm::vec4(0, 0, 0, 1));

    {
        shader->setVec3("unif", glm::vec3(0, 0, 0));
        atcg::Renderer::clear();
        atcg::Renderer::draw(quad, nullptr, glm::mat4(1), glm::vec3(1), shader);
        auto data = fbo.getColorAttachement(0)->getData(atcg::CPU);

        EXPECT_NEAR(data[0][0][0].item<float>(), 0.0f, 1e-5f);
        EXPECT_NEAR(data[0][0][1].item<float>(), 0.0f, 1e-5f);
        EXPECT_NEAR(data[0][0][2].item<float>(), 0.0f, 1e-5f);
    }

    {
        shader->setVec3("unif", glm::vec3(0, 0.5, 0));
        atcg::Renderer::clear();
        atcg::Renderer::draw(quad, nullptr, glm::mat4(1), glm::vec3(1), shader);
        auto data = fbo.getColorAttachement(0)->getData(atcg::CPU);

        EXPECT_NEAR(data[0][0][0].item<float>(), 0.0f, 1e-5f);
        EXPECT_NEAR(data[0][0][1].item<float>(), 127.0f, 1e-5f);
        EXPECT_NEAR(data[0][0][2].item<float>(), 0.0f, 1e-5f);
    }

    {
        shader->setVec3("unif", glm::vec3(0, 0.5, 0.5));
        atcg::Renderer::clear();
        atcg::Renderer::draw(quad, nullptr, glm::mat4(1), glm::vec3(1), shader);
        auto data = fbo.getColorAttachement(0)->getData(atcg::CPU);

        EXPECT_NEAR(data[0][0][0].item<float>(), 0.0f, 1e-5f);
        EXPECT_NEAR(data[0][0][1].item<float>(), 127.0f, 1e-5f);
        EXPECT_NEAR(data[0][0][2].item<float>(), 127.0f, 1e-5f);
    }

    {
        shader->setVec3("unif", glm::vec3(1, 1, 1));
        atcg::Renderer::clear();
        atcg::Renderer::draw(quad, nullptr, glm::mat4(1), glm::vec3(1), shader);
        auto data = fbo.getColorAttachement(0)->getData(atcg::CPU);

        EXPECT_NEAR(data[0][0][0].item<float>(), 255.0f, 1e-5f);
        EXPECT_NEAR(data[0][0][1].item<float>(), 255.0f, 1e-5f);
        EXPECT_NEAR(data[0][0][2].item<float>(), 255.0f, 1e-5f);
    }

    atcg::Renderer::useScreenBuffer();
    atcg::Renderer::setDefaultViewport();
}

TEST(ShaderTest, uniformVec4)
{
    atcg::ref_ptr<atcg::Graph> quad;
    std::vector<atcg::Vertex> vertices = {atcg::Vertex(glm::vec3(-1, -1, 0)),
                                          atcg::Vertex(glm::vec3(1, -1, 0)),
                                          atcg::Vertex(glm::vec3(1, 1, 0)),
                                          atcg::Vertex(glm::vec3(-1, 1, 0))};

    std::vector<glm::u32vec3> edges = {glm::u32vec3(0, 1, 2), glm::u32vec3(0, 2, 3)};

    quad = atcg::Graph::createTriangleMesh(vertices, edges);

    atcg::ref_ptr<atcg::Shader> shader =
        atcg::make_ref<atcg::Shader>("tests/data/shader/test_vertex.vs", "tests/data/shader/test_vec4.fs");

    atcg::Framebuffer fbo(1, 1);
    fbo.attachColor();
    fbo.attachDepth();

    fbo.use();

    atcg::Renderer::setViewport(0, 0, 1, 1);
    atcg::Renderer::setClearColor(glm::vec4(0, 0, 0, 0));

    {
        shader->setVec4("unif", glm::vec4(0, 0, 0, 0));
        atcg::Renderer::clear();
        atcg::Renderer::draw(quad, nullptr, glm::mat4(1), glm::vec3(1), shader);
        auto data = fbo.getColorAttachement(0)->getData(atcg::CPU);

        EXPECT_NEAR(data[0][0][0].item<float>(), 0.0f, 1e-5f);
        EXPECT_NEAR(data[0][0][1].item<float>(), 0.0f, 1e-5f);
        EXPECT_NEAR(data[0][0][2].item<float>(), 0.0f, 1e-5f);
        EXPECT_NEAR(data[0][0][3].item<float>(), 0.0f, 1e-5f);
    }

    {
        shader->setVec4("unif", glm::vec4(0, 0.25, 0.5, 1.0));
        atcg::Renderer::clear();
        atcg::Renderer::draw(quad, nullptr, glm::mat4(1), glm::vec3(1), shader);
        auto data = fbo.getColorAttachement(0)->getData(atcg::CPU);

        EXPECT_NEAR(data[0][0][0].item<float>(), 0.0f, 1e-5f);
        EXPECT_NEAR(data[0][0][1].item<float>(), 64.0f, 1e-5f);
        EXPECT_NEAR(data[0][0][2].item<float>(), 127.0f, 1e-5f);
        EXPECT_NEAR(data[0][0][3].item<float>(), 255.0f, 1e-5f);
    }

    {
        shader->setVec4("unif",
                        glm::vec4(1.0, 0, 0.5, 0.25));    // Expected output: 0.25, 0, 0.125, 0.0625 due to blending
        atcg::Renderer::clear();
        atcg::Renderer::draw(quad, nullptr, glm::mat4(1), glm::vec3(1), shader);
        auto data = fbo.getColorAttachement(0)->getData(atcg::CPU);

        EXPECT_NEAR(data[0][0][0].item<float>(), 64.0f, 1e-5f);
        EXPECT_NEAR(data[0][0][1].item<float>(), 0.0f, 1e-5f);
        EXPECT_NEAR(data[0][0][2].item<float>(), 32.0f, 1e-5f);
        EXPECT_NEAR(data[0][0][3].item<float>(), 16.0f, 1e-5f);
    }

    {
        shader->setVec4("unif", glm::vec4(1, 1, 1, 1));
        atcg::Renderer::clear();
        atcg::Renderer::draw(quad, nullptr, glm::mat4(1), glm::vec3(1), shader);
        auto data = fbo.getColorAttachement(0)->getData(atcg::CPU);

        EXPECT_NEAR(data[0][0][0].item<float>(), 255.0f, 1e-5f);
        EXPECT_NEAR(data[0][0][1].item<float>(), 255.0f, 1e-5f);
        EXPECT_NEAR(data[0][0][2].item<float>(), 255.0f, 1e-5f);
        EXPECT_NEAR(data[0][0][3].item<float>(), 255.0f, 1e-5f);
    }

    atcg::Renderer::useScreenBuffer();
    atcg::Renderer::setDefaultViewport();
}

TEST(ShaderTest, uniformMat4)
{
    atcg::ref_ptr<atcg::Graph> quad;
    std::vector<atcg::Vertex> vertices = {atcg::Vertex(glm::vec3(-1, -1, 0)),
                                          atcg::Vertex(glm::vec3(1, -1, 0)),
                                          atcg::Vertex(glm::vec3(1, 1, 0)),
                                          atcg::Vertex(glm::vec3(-1, 1, 0))};

    std::vector<glm::u32vec3> edges = {glm::u32vec3(0, 1, 2), glm::u32vec3(0, 2, 3)};

    quad = atcg::Graph::createTriangleMesh(vertices, edges);

    atcg::ref_ptr<atcg::Shader> shader =
        atcg::make_ref<atcg::Shader>("tests/data/shader/test_vertex.vs", "tests/data/shader/test_mat4.fs");

    atcg::Framebuffer fbo(1, 1);
    fbo.attachColor();
    fbo.attachDepth();

    fbo.use();

    atcg::Renderer::setViewport(0, 0, 1, 1);
    atcg::Renderer::setClearColor(glm::vec4(0, 0, 0, 0));

    {
        glm::mat4 m(1);
        m[0] = glm::vec4(0.25f, 0.25f, 0.25f, 0.25f);
        m[1] = glm::vec4(0.5f, 0.1f, 0.4f, 0.0f);
        m[2] = glm::vec4(0.2f, 0.3f, 0.25f, 0.25f);

        shader->setMat4("unif", glm::transpose(m));
        atcg::Renderer::clear();
        atcg::Renderer::draw(quad, nullptr, glm::mat4(1), glm::vec3(1), shader);
        auto data = fbo.getColorAttachement(0)->getData(atcg::CPU);

        EXPECT_NEAR(data[0][0][0].item<float>(), 255.0f, 1e-5f);
        EXPECT_NEAR(data[0][0][1].item<float>(), 255.0f, 1e-5f);
        EXPECT_NEAR(data[0][0][2].item<float>(), 255.0f, 1e-5f);
        EXPECT_NEAR(data[0][0][3].item<float>(), 255.0f, 1e-5f);
    }

    {
        glm::mat4 m(1);
        m[0] = glm::vec4(0.25f, 0.25f, 0.25f, 0.25f);
        m[1] = glm::vec4(0.5f, 0.1f, 0.4f, 0.0f);
        m[2] = glm::vec4(0.2f, 0.3f, 0.25f, 0.25f);

        shader->setMat4("unif", m);
        atcg::Renderer::clear();
        atcg::Renderer::draw(quad, nullptr, glm::mat4(1), glm::vec3(1), shader);
        auto data = fbo.getColorAttachement(0)->getData(atcg::CPU);

        EXPECT_NEAR(data[0][0][0].item<float>(), 242.0f, 1e-5f);
        EXPECT_NEAR(data[0][0][1].item<float>(), 166.0f, 1e-5f);
        EXPECT_NEAR(data[0][0][2].item<float>(), 229.0f, 1e-5f);
        EXPECT_NEAR(data[0][0][3].item<float>(), 255.0f, 1e-5f);
    }

    atcg::Renderer::useScreenBuffer();
    atcg::Renderer::setDefaultViewport();
}

TEST(ShaderTest, uniformSubroutineUni)
{
    atcg::ref_ptr<atcg::Graph> quad;
    std::vector<atcg::Vertex> vertices = {atcg::Vertex(glm::vec3(-1, -1, 0)),
                                          atcg::Vertex(glm::vec3(1, -1, 0)),
                                          atcg::Vertex(glm::vec3(1, 1, 0)),
                                          atcg::Vertex(glm::vec3(-1, 1, 0))};

    std::vector<glm::u32vec3> edges = {glm::u32vec3(0, 1, 2), glm::u32vec3(0, 2, 3)};

    quad = atcg::Graph::createTriangleMesh(vertices, edges);

    atcg::ref_ptr<atcg::Shader> shader = atcg::make_ref<atcg::Shader>("tests/data/shader/test_subroutine_uni.vs",
                                                                      "tests/data/shader/test_subroutine_uni.fs",
                                                                      "tests/data/shader/test_subroutine_uni.gs");

    atcg::Framebuffer fbo(1, 1);
    atcg::TextureSpecification spec;
    spec.width  = 1;
    spec.height = 1;
    spec.format = atcg::TextureFormat::RGBAFLOAT;
    fbo.attachTexture(atcg::Texture2D::create(spec));
    fbo.attachDepth();

    fbo.use();

    atcg::Renderer::setViewport(0, 0, 1, 1);
    atcg::Renderer::setClearColor(glm::vec4(0, 0, 0, 1));

    {
        shader->selectSubroutine("color", "blue");
        shader->selectSubroutine("multiply", "multiply_half");

        atcg::Renderer::clear();
        atcg::Renderer::draw(quad, nullptr, glm::mat4(1), glm::vec3(1), shader);
        auto data = fbo.getColorAttachement(0)->getData(atcg::CPU);

        EXPECT_NEAR(data[0][0][0].item<float>(), 0.0f, 1e-5f);
        EXPECT_NEAR(data[0][0][1].item<float>(), 0.0f, 1e-5f);
        EXPECT_NEAR(data[0][0][2].item<float>(), 1.5f, 1e-5f);
    }

    {
        shader->selectSubroutine("color", "red");
        shader->selectSubroutine("multiply", "multiply_quarter");

        atcg::Renderer::clear();
        atcg::Renderer::draw(quad, nullptr, glm::mat4(1), glm::vec3(1), shader);
        auto data = fbo.getColorAttachement(0)->getData(atcg::CPU);

        EXPECT_NEAR(data[0][0][0].item<float>(), 1.25f, 1e-5f);
        EXPECT_NEAR(data[0][0][1].item<float>(), 0.0f, 1e-5f);
        EXPECT_NEAR(data[0][0][2].item<float>(), 0.0f, 1e-5f);
    }

    atcg::Renderer::useScreenBuffer();
    atcg::Renderer::setDefaultViewport();
}

TEST(ShaderTest, uniformSubroutine)
{
    atcg::ref_ptr<atcg::Graph> quad;
    std::vector<atcg::Vertex> vertices = {atcg::Vertex(glm::vec3(-1, -1, 0)),
                                          atcg::Vertex(glm::vec3(1, -1, 0)),
                                          atcg::Vertex(glm::vec3(1, 1, 0)),
                                          atcg::Vertex(glm::vec3(-1, 1, 0))};

    std::vector<glm::u32vec3> edges = {glm::u32vec3(0, 1, 2), glm::u32vec3(0, 2, 3)};

    quad = atcg::Graph::createTriangleMesh(vertices, edges);

    atcg::ref_ptr<atcg::Shader> shader = atcg::make_ref<atcg::Shader>("tests/data/shader/test_subroutine.vs",
                                                                      "tests/data/shader/test_subroutine.fs",
                                                                      "tests/data/shader/test_subroutine.gs");

    atcg::Framebuffer fbo(1, 1);
    atcg::TextureSpecification spec;
    spec.width  = 1;
    spec.height = 1;
    spec.format = atcg::TextureFormat::RGBAFLOAT;
    fbo.attachTexture(atcg::Texture2D::create(spec));
    fbo.attachDepth();

    fbo.use();

    atcg::Renderer::setViewport(0, 0, 1, 1);
    atcg::Renderer::setClearColor(glm::vec4(0, 0, 0, 1));

    {
        shader->selectSubroutine("color_vs", "red");
        shader->selectSubroutine("color_fs", "red");
        shader->selectSubroutine("color_gs", "red");
        shader->selectSubroutine("multiply", "multiply_half");

        atcg::Renderer::clear();
        atcg::Renderer::draw(quad, nullptr, glm::mat4(1), glm::vec3(1), shader);
        auto data = fbo.getColorAttachement(0)->getData(atcg::CPU);

        EXPECT_NEAR(data[0][0][0].item<float>(), 1.5f, 1e-5f);
        EXPECT_NEAR(data[0][0][1].item<float>(), 0.0f, 1e-5f);
        EXPECT_NEAR(data[0][0][2].item<float>(), 0.0f, 1e-5f);
    }

    {
        shader->selectSubroutine("color_vs", "red");
        shader->selectSubroutine("color_fs", "white");
        shader->selectSubroutine("color_gs", "green");
        shader->selectSubroutine("multiply", "multiply_half");

        atcg::Renderer::clear();
        atcg::Renderer::draw(quad, nullptr, glm::mat4(1), glm::vec3(1), shader);
        auto data = fbo.getColorAttachement(0)->getData(atcg::CPU);

        EXPECT_NEAR(data[0][0][0].item<float>(), 0.5f, 1e-5f);
        EXPECT_NEAR(data[0][0][1].item<float>(), 1.0f, 1e-5f);
        EXPECT_NEAR(data[0][0][2].item<float>(), 0.0f, 1e-5f);
    }

    {
        shader->selectSubroutine("color_vs", "green");
        shader->selectSubroutine("color_fs", "white");
        shader->selectSubroutine("color_gs", "blue");
        shader->selectSubroutine("multiply", "multiply_quarter");

        atcg::Renderer::clear();
        atcg::Renderer::draw(quad, nullptr, glm::mat4(1), glm::vec3(1), shader);
        auto data = fbo.getColorAttachement(0)->getData(atcg::CPU);

        EXPECT_NEAR(data[0][0][0].item<float>(), 0.0f, 1e-5f);
        EXPECT_NEAR(data[0][0][1].item<float>(), 0.25f, 1e-5f);
        EXPECT_NEAR(data[0][0][2].item<float>(), 1.0f, 1e-5f);
    }

    atcg::Renderer::useScreenBuffer();
    atcg::Renderer::setDefaultViewport();
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