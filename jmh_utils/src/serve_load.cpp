#include "jmh_utils/serve_load.hpp"

#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

namespace jmh_utils
{
    int extractNumber(const std::string filename)
    {
        std::string number;
        for (char c : filename)
        {
            if (std::isdigit(static_cast<unsigned char>(c)))
            {
                number += c;
            }
        }
        return std::stoi(number);
    }

    bool sortingLowestNumber(const std::string &a, const std::string &b)
    {
        std::array<std::string, 2> digits{};
        std::array<std::string, 2> filename = {a, b};
        for (int i = 0; i < filename.size(); i++)
        {
            digits[i] = jmh_utils::extractNumber(filename[i]);
        }
        return digits[0] < digits[1];
    }

    std::vector<std::string> loadFiles(const std::string &extension, const std::string &directory)
    {
        std::vector<std::string> files;
        for (const auto &entry : std::filesystem::directory_iterator(directory))
        {
            if (entry.is_regular_file() && entry.path().extension() == extension)
            {
                files.push_back(entry.path().string());
            }
        }

        std::sort(files.begin(), files.end(), sortingLowestNumber);

        return files;
    }

}