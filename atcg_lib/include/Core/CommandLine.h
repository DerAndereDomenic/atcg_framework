#pragma once

namespace atcg
{
/**
 * @brief This function stores the cmd arguments into a global varibale that can be retrieved by getCommandLineArguments
 *
 * @param argc The number of arguments
 * @param argv The list of arguments as char*
 */
void registerCommandLineArguments(int argc, char** argv);

/**
 * @brief Get the command line arguments of the application
 *
 * @return The command line arguments
 */
const std::vector<std::string>& getCommandLineArguments();
}    // namespace atcg