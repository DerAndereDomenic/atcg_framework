#pragma once

#include <Core/API.h>

namespace atcg
{
/**
 * @brief This function stores the cmd arguments into a global varibale that can be retrieved by getCommandLineArguments
 *
 * @param argc The number of arguments
 * @param argv The list of arguments as char*
 */
ATCG_API void registerCommandLineArguments(int argc, char** argv);

/**
 * @brief Get the command line arguments of the application
 *
 * @return The command line arguments
 */
ATCG_API const std::vector<std::string>& getCommandLineArguments();
}    // namespace atcg