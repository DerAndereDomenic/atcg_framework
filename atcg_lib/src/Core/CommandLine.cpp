#include <Core/CommandLine.h>

namespace atcg
{
static std::vector<std::string> g_cmd_arguments;

void registerCommandLineArguments(int argc, char** argv)
{
    for(int i = 0; i < argc; ++i)
    {
        g_cmd_arguments.emplace_back(argv[i]);
    }
}

const std::vector<std::string>& getCommandLineArguments()
{
    return g_cmd_arguments;
}
}    // namespace atcg