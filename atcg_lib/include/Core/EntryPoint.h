#pragma once

#include <ATCG.h>

extern atcg::Application* atcg::createApplication();

int main(int argc, char** argv)
{
    atcg::Application* app = atcg::createApplication();
    app->run();

    delete app;

    return 0;
}