#include "jmh_utils/main_camera_calib.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include "jmh_utils/serve_load.hpp"

namespace jmh_utils
{
    jmh_utils::ResultIntrinsic runCalibrate(const jmh_utils::BoardParameter &params,
                                            std::vector<std::string> all_images)
    {
        jmh_utils::FindCorners corners = jmh_utils::findCorners(params, all_images);

        cv::Size frame_size(params.frame_width, params.frame_height);
        std::vector<cv::Mat> rvecs, tvecs;

        jmh_utils::ResultIntrinsic result;
        result.rms = cv::calibrateCamera(corners.object_points, corners.image_corners, frame_size,
                                    result.intrinsic_mat, result.distortion_coeffs, rvecs, tvecs);
        result.visualize_corners = corners.visual_corners;
        result.successed_index = corners.successed_index;

        return result;
    }

    jmh_utils::ResultUndistort runUndistorted(const int frame_width, const int &frame_height,
                                              const cv::Mat &intinsic, const cv::Mat &distortion,
                                              std::vector<std::string> all_images)
    {
        if (intinsic.empty() || distortion.empty())
        {
            std::cout << "No Intrinsic matrix or distortion coefficients!!!" << std::endl;
            return {};
        }

        cv::Mat map1, map2;
        cv::Size image_size(frame_width, frame_height);

        cv::initUndistortRectifyMap(intinsic, distortion, cv::Mat(),
                                    cv::getOptimalNewCameraMatrix(intinsic, distortion, image_size, 1, image_size),
                                    image_size, CV_32FC1, map1, map2);

        std::vector<cv::Mat> all_undistort;
        std::vector<int> undistorted_frame_index;

        for (size_t i = 0; i < all_images.size(); ++i)
        {
            const auto &file = all_images[i];
            const auto &file_num = jmh_utils::extractNumber(file);
            cv::Mat img = cv::imread(file);
            if (img.empty())
            {
                std::cout << "No [ " << file_num << " ] image!!! Check folder." << std::endl;
                continue;
            }

            cv::Mat undistorted_img;
            cv::remap(img, undistorted_img, map1, map2, cv::INTER_LINEAR);

            std::cout << "Successed [ " << file_num << " ] undistorted image." << std::endl;

            all_undistort.push_back(undistorted_img);
            undistorted_frame_index.push_back(file_num);
        }

        jmh_utils::ResultUndistort result;
        result.visualize_undistort = all_undistort;
        result.undistroted_index = undistorted_frame_index;

        return result;
    }

    jmh_utils::ResultRmse runRMSE(const jmh_utils::BoardParameter &params,
                                  const cv::Mat &intinsic, const cv::Mat &distortion,
                                  std::vector<std::string> all_images)
    {
        jmh_utils::FindCorners corners = jmh_utils::findCorners(params, all_images);

        std::vector<double> per_image_rmse;
        double total_sqerr = 0.0;
        size_t total_points = 0;

        for (size_t frame_index = 0; frame_index < corners.successed_index.size(); ++frame_index)
        {
            const int frame_num = corners.successed_index[frame_index];
            cv::Mat rvec, tvec;
            cv::solvePnP(corners.object_points[frame_index], corners.image_corners[frame_index],
                         intinsic, distortion, rvec, tvec, cv::SOLVEPNP_ITERATIVE);
            if (rvec.empty())
            {
                std::cout << "Failed solvePnP: [ " << frame_index << " ] frame";
                continue;
            }

            std::vector<cv::Point2f> image_point_2f;

            cv::projectPoints(corners.object_points[frame_index], rvec, tvec,
                              intinsic, distortion, image_point_2f);

            double err = cv::norm(corners.image_corners[frame_index], image_point_2f, cv::NORM_L2);
            size_t point_number = corners.object_points[frame_index].size();
            double rmse = std::sqrt((err * err) / point_number);
            per_image_rmse.push_back(rmse);

            std::cout << "[ " << frame_num << " ] frame RMSE is [ " << rmse << " ]" << std::endl;

            total_sqerr += (err * err);
            total_points += point_number;
        }
        double overall_rmse = std::sqrt(total_sqerr / total_points);
        std::cout << "Overall RMSE is [ " << overall_rmse << " ]" << std::endl;

        jmh_utils::ResultRmse result;
        result.error_each_frame = per_image_rmse;
        result.rmse_overall = overall_rmse;

        return result;
    }

    static jmh_utils::FindCorners findCorners(const jmh_utils::BoardParameter &params,
                                              std::vector<std::string> all_images)
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
        std::vector<cv::Mat> all_corners_visual;
        std::vector<int> successed_frame_index;
        successed_frame_index.clear();

        for (size_t frame_index = 0; frame_index < all_images.size(); ++frame_index)
        {
            const auto &file = all_images[frame_index];
            const auto &file_num = jmh_utils::extractNumber(file);

            cv::Mat frame_image = cv::imread(file);
            if (frame_image.empty())
            {
                std::cout << "No [ " << file_num << " ] image!!! Check folder." << std::endl;
                continue;
            }

            std::vector<cv::Point2f> image_corners;
            bool found = cv::findChessboardCorners(frame_image, pattern_size, image_corners,
                                                   cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);

            std::cout << "Find [ " << file_num << " ] image corners." << std::endl;

            if (!found)
            {
                std::cout << "Can't find corneres!!! Check [ " << file_num << " ] image." << std::endl;
                continue;
            }

            cv::Mat gray;
            cv::cvtColor(frame_image, gray, cv::COLOR_BGR2GRAY);
            cv::cornerSubPix(gray, image_corners, cv::Size(5, 5), cv::Size(-1, -1),
                             cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001));

            all_image_corners.push_back(image_corners);
            all_object_points.push_back(pattern_object_points);

            cv::Mat corners_visual = frame_image.clone();
            cv::drawChessboardCorners(corners_visual, pattern_size, image_corners, found);

            all_corners_visual.push_back(corners_visual);
            successed_frame_index.push_back(file_num);
        }

        jmh_utils::FindCorners result;
        result.image_corners = all_image_corners;
        result.object_points = all_object_points;
        result.visual_corners = all_corners_visual;
        result.successed_index = successed_frame_index;

        return result;
    }

}