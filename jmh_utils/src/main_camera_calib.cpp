#include "jmh_utils/main_camera_calib.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

namespace jmh_utils
{
    struct boardParameter
    {
        int columns, rows, frame_width, frame_height;
        float square_size;
    };

    struct resultIntrinsic
    {
        double rms;
        cv::Mat intrinsic_mat, distortion_coeffs;
    };

    jmh_utils::resultIntrinsic runCalibrate(const jmh_utils::boardParameter &params,
                                            std::vector<cv::String> all_images)
    {
        cv::Size pattern_size(params.columns, params.rows);
        float square_size = params.square_size;
        cv::Size frame_size(params.frame_width, params.frame_height);

        std::vector<cv::Point3f> pattern_object_points;

        for (int i = 0; i < pattern_size.height; ++i)
            for (int j = 0; j < pattern_size.width; ++j)
                pattern_object_points.emplace_back(j * square_size, i * square_size, 0.0f);

        std::vector<std::vector<cv::Point2f>> all_image_corners;
        std::vector<std::vector<cv::Point3f>> all_object_points;

        for (size_t frame_index = 0; frame_index < all_images.size(); ++frame_index)
        {
            const auto &file = all_images[frame_index];
            cv::Mat frame_image = cv::imread(file);
            if (frame_image.empty())
                continue;

            std::vector<cv::Point2f> image_corners;
            bool found = cv::findChessboardCorners(frame_image, pattern_size, image_corners,
                                                   cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);

            if (found)
            {
                cv::Mat gray;
                cv::cvtColor(frame_image, gray, cv::COLOR_BGR2GRAY);
                cv::cornerSubPix(gray, image_corners, cv::Size(5, 5), cv::Size(-1, -1),
                                 cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001));

                all_image_corners.push_back(image_corners);
                all_object_points.push_back(pattern_object_points);
            }
        }

        jmh_utils::resultIntrinsic r;

        std::vector<cv::Mat> rvecs, tvecs;

        r.rms = cv::calibrateCamera(all_image_corners, all_object_points, frame_size,
                                    r.intrinsic_mat, r.distortion_coeffs, rvecs, tvecs);

        return r;
    }

}