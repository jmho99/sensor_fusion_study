#ifndef SERVE_LOAD_HPP
#define SERVE_LOAD_HPP

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

namespace jmh_utils
{
    int extractNumber(const std::string filename);

    bool sorting_lowest_number(const std::string &a, const std::string &b);

    std::vector<std::string> loadFiles(const std::string &extension, const std::string &directory);
}
#endif