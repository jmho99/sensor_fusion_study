#include "jmh_utils/main_camera_calib.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include "jmh_utils/serve_load.hpp"

namespace jmh_utils
{
    jmh_utils::ResultIntrinsic runCalibrate(const jmh_utils::BoardParameter &params,
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

            std::cout << "Successed [ " << file_num << " ] image calibrate." << std::endl;

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

        jmh_utils::ResultIntrinsic r;

        std::vector<cv::Mat> rvecs, tvecs;

        r.rms = cv::calibrateCamera(all_object_points, all_image_corners, frame_size,
                                    r.intrinsic_mat, r.distortion_coeffs, rvecs, tvecs);
        r.visualize_corners = all_corners_visual;
        r.successed_index = successed_frame_index;

        return r;
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

    void runRMSE(const jmh_utils::BoardParameter &params, const cv::Mat &intinsic, const cv::Mat &distortion, std::vector<std::string> all_images)
    {
        // def computeRmse(image_paths,camera_matrix, dist_coeffs,cols, rows, square_size) :

        cv::Size pattern_size(params.columns, params.rows);
        float square_size = params.square_size;
        cv::Size frame_size(params.frame_width, params.frame_height);

        std::vector<cv::Point3f> pattern_object_points;

        for (int i = 0; i < pattern_size.height; ++i)
            for (int j = 0; j < pattern_size.width; ++j)
                pattern_object_points.emplace_back(j * square_size, i * square_size, 0.0f);

    std::vector<double> per_image_rmse;
    float total_sqerr = 0.0;
    int total_points = 0;

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

            std::cout << "Successed [ " << file_num << " ] image calibrate." << std::endl;

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

        for (size_t frame_index = 0; frame_index < successed_frame_index.size(); ++frame_index)
        {
            #외부 파라미터 추정(rvec, tvec)
        ok, rvec, tvec = cv2.solvePnP(
            objp, corners, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            print(f"Failed solvePnP: {img_path}")
            continue

            proj, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
        proj = proj.reshape(-1, 2)

        err = cv2.norm(corners, proj, cv2.NORM_L2)
        npts = len(objp)
        rmse = np.sqrt((err**2) / npts)
        per_image_rmse.append((str(img_path), rmse))

        total_sqerr += (err**2)
        total_points += npts
        }
    overall_rmse = np.sqrt(total_sqerr / total_points) if total_points > 0 else np.nan
        
    return overall_rmse, per_image_rmse
    }

}