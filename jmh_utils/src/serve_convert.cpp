#include "jmh_utils/serve_convert.hpp"

#include <vector>
#include <opencv2/opencv.hpp>

namespace jmh_utils
{
    cv::Mat vector2Mat(const std::vector<std::vector<double>> &input)
    {
        std::cout << "convert vector<vector> to cv::Mat" << std::endl;
        std::size_t columns = input.front().size();
        std::size_t rows = input.size();

        cv::Mat output(rows, columns, CV_64F);
        for (size_t i = 0; i < rows; i++)
        {
            for (size_t j = 0; j < columns; j++)
            {
                output.at<double>(i, j) = input[i][j];
            }
        }
        return output;
    }

    cv::Mat vector2Mat(const std::vector<double> &input)
    {
        std::cout << "convert vector to cv::Mat" << std::endl;
        std::size_t nums = input.size();

        cv::Mat output(nums, 1, CV_64F);

        for (size_t j = 0; j < nums; j++)
        {
            output.at<double>(j, 0) = input[j];
        }

        return output;
    }
}