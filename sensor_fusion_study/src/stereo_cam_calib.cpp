#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <filesystem>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

using std::placeholders::_1;

namespace fs = std::filesystem;

class StereoCamCalibNode : public rclcpp::Node
{
public:
    StereoCamCalibNode() : Node("stereo_cam_calib")
    {
        left_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/left/image_raw", 10, std::bind(&StereoCamCalibNode::leftCallback, this, _1));
        right_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/right/image_raw", 10, std::bind(&StereoCamCalibNode::rightCallback, this, _1));

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(30),
            std::bind(&StereoCamCalibNode::timerCallback, this));

        std::string where_ = "company";
        readWritePath(where_);
        initializedParameters();
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr left_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr right_sub_;
    cv::Mat left_frame_, right_frame_;

    std::string save_absolute_path_;
    std::string save_origin_path_;
    std::string save_calib_path_;
    std::string one_cam_result_path_;

    std::vector<std::vector<cv::Point2f>> left_img_points_, right_img_points_;
    std::vector<std::vector<cv::Point3f>> obj_points_;
    cv::Size board_size_;
    cv::Size img_size_;
    float square_size_;
    int cols_, rows_;
    int frame_width_, frame_height_;
    cv::Mat intrinsic_matrix_, distortion_coeffs_;
    int collected_ = 0;
    int count_ = 0;
    double rms_;

    rclcpp::TimerBase::SharedPtr timer_;
    cv::Mat last_image_;

    void leftCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        left_frame_ = cv_bridge::toCvCopy(msg, "bgr8")->image;
    }

    void rightCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        right_frame_ = cv_bridge::toCvCopy(msg, "bgr8")->image;
    }

    void timerCallback()
    {

        // 빈 화면이라도 띄우도록 할 수 있음
        cv::Mat dummy = cv::Mat::zeros(480, 640, CV_8UC3);
        cv::putText(dummy, "No camera image", cv::Point(50, 240),
                    cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
        cv::imshow("Camera Image", dummy);

        inputKeyboard(last_image_);
    }

    void initializedParameters()
    {
        std::string where = "company";
        readWritePath(where);

        cv::FileStorage fs(one_cam_result_path_ + "/one_cam_calib_result.yaml", cv::FileStorage::READ);
        if (!fs.isOpened())
        {
            RCLCPP_WARN(this->get_logger(), "Failed open one_cam_calib_result.yaml file!");
            return;
        }
        else
        {

            fs["checkerboard_cols"] >> cols_;
            fs["checkerboard_rows"] >> rows_;
            fs["square_size"] >> square_size_;
            fs["frame_width"] >> frame_width_;
            fs["frame_height"] >> frame_height_;
            fs["intrinsic_matrix"] >> intrinsic_matrix_;
            fs["distortion_coefficients"] >> distortion_coeffs_;
            fs.release();
            board_size_ = cv::Size(cols_, rows_);
            img_size_ = cv::Size(frame_width_, frame_height_);
        }
    }

    void readWritePath(std::string where)
    {
        std::string change_path;
        if (where == "company")
        {
            change_path = "/antlab/sensor_fusion_study_ws";
        }
        else if (where == "home")
        {
            change_path = "/icrs/sensor_fusion_study_ws";
        }

        save_absolute_path_ = "/home" + change_path + "/src/sensor_fusion_study/stereo_cam_calib/";
        save_origin_path_ = save_absolute_path_ + "origin_images/";
        save_calib_path_ = save_absolute_path_ + "calib_images/";
        one_cam_result_path_ = "/home" + change_path + "/src/sensor_fusion_study/one_cam_calib";
        fs::create_directories(save_origin_path_);
        fs::create_directories(save_calib_path_);
    }

    void inputKeyboard(const cv::Mat &frame)
    {
        int key = cv::waitKey(1);
        if (key == 's')
        {
            saveFrame();
        }
        else if (key == 'c')
        {
            calibrateEachCamera();
        }
        else if (key == 'e')
        {
        }
    }

    void saveFrame()
    {
        if (left_frame_.empty() || right_frame_.empty())
            return;

        // [ADD] Save origin images with numbering
        std::string filename_left = save_origin_path_ + "img_" + std::to_string(count_) + "_left_origin.png";
        std::string filename_right = save_origin_path_ + "img_" + std::to_string(count_) + "_right_origin.png";
        cv::imwrite(filename_left, left_frame_);
        cv::imwrite(filename_right, right_frame_);

        std::vector<cv::Point2f> corners_left, corners_right;
        bool found_left = cv::findChessboardCorners(left_frame_, board_size_, corners_left, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
        bool found_right = cv::findChessboardCorners(right_frame_, board_size_, corners_right, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);

        if (found_left && found_right)
        {
            cv::drawChessboardCorners(left_frame_, board_size_, corners_left, found_left);
            cv::drawChessboardCorners(right_frame_, board_size_, corners_right, found_right);

            std::vector<cv::Point3f> objp;
            for (int i = 0; i < board_size_.height; ++i)
                for (int j = 0; j < board_size_.width; ++j)
                    objp.emplace_back(j * square_size_, i * square_size_, 0);

            obj_points_.push_back(objp);
            left_img_points_.push_back(corners_left);
            right_img_points_.push_back(corners_right);

            RCLCPP_INFO(this->get_logger(), "📸 캘리브레이션 이미지 %d 장 수집됨", ++collected_);

            collected_++;
            count_++;
        }
    }

    void calibrateEachCamera()
    {
        int i = 0;
        cv::Mat left_intrinsic_matrix_, right_intrinsic_matrix_;
        cv::Mat left_dist_coeffs_, right_dist_coeffs_;
        while (true)
        {
            std::string fname_left = save_origin_path_ + "img_" + std::to_string(i) + "_left_origin.png";
            std::string fname_right = save_origin_path_ + "img_" + std::to_string(i) + "_right_origin.png";

            if (!fs::exists(fname_left) || !fs::exists(fname_right))
            {
                RCLCPP_INFO(this->get_logger(), "✅ 이미지 로드 완료. %d 쌍의 이미지가 존재.", i);
                break;
            }

            cv::Mat orig_left = cv::imread(fname_left, cv::IMREAD_COLOR);
            cv::Mat orig_right = cv::imread(fname_right, cv::IMREAD_COLOR);

            if (orig_left.empty() || orig_right.empty())
            {
                RCLCPP_WARN(this->get_logger(), "Failed to load image pair index %d.", i);
                i++;
                continue;
            }

            std::vector<cv::Point2f> corners_left, corners_right;
            bool found_left = cv::findChessboardCorners(orig_left, board_size_, corners_left,
                                                        cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
            bool found_right = cv::findChessboardCorners(orig_right, board_size_, corners_right,
                                                         cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);

            if (found_left && found_right)
            {
                cv::Mat gray;
                cv::cvtColor(orig_left, gray, cv::COLOR_BGR2GRAY);
                cv::cvtColor(orig_right, gray, cv::COLOR_BGR2GRAY);
                cv::cornerSubPix(gray, corners_left, cv::Size(5, 5), cv::Size(-1, -1),
                                 cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001));
                cv::cornerSubPix(gray, corners_right, cv::Size(5, 5), cv::Size(-1, -1),
                                 cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001));

                std::vector<cv::Point3f> objp;
                for (int y = 0; y < board_size_.height; ++y)
                    for (int x = 0; x < board_size_.width; ++x)
                        objp.emplace_back(x * square_size_, y * square_size_, 0.0);

                obj_points_.push_back(objp);
                left_img_points_.push_back(corners_left);
                right_img_points_.push_back(corners_right);

                std::vector<cv::Mat> left_rvecs_, left_tvecs_;
                std::vector<cv::Mat> right_rvecs_, right_tvecs_;

                cv::calibrateCamera(obj_points_, left_img_points_, cv::Size(frame_width_, frame_height_),
                                    left_intrinsic_matrix_, left_dist_coeffs_, left_rvecs_, left_tvecs_);
                cv::calibrateCamera(obj_points_, right_img_points_, cv::Size(frame_width_, frame_height_),
                                    right_intrinsic_matrix_, right_dist_coeffs_, right_rvecs_, right_tvecs_);

                RCLCPP_INFO(this->get_logger(), "✔︎ index %d chessboard corners detected and saved.", i);
            }
            else
            {
                RCLCPP_WARN(this->get_logger(), "Chessboard not found in pair index %d", i);
            }

            i++;
        }

        collected_ = static_cast<int>(left_img_points_.size());

        if (collected_ > 0)
        {
            calibrateStereo(left_intrinsic_matrix_, right_intrinsic_matrix_, left_dist_coeffs_, right_dist_coeffs_);
        }
        else
        {
            RCLCPP_ERROR(this->get_logger(), "No valid image pairs found for calibration!");
        }
    }

    void calibrateStereo(cv::Mat &left_intrinsic_matrix_, cv::Mat &right_intrinsic_matrix_,
                         cv::Mat &left_dist_coeffs_, cv::Mat &right_dist_coeffs_)
    {
        cv::Mat R, T, E, F;
        cv::Mat K1, D1, K2, D2;
        K1 = left_intrinsic_matrix_;
        K2 = right_intrinsic_matrix_;
        D1 = left_dist_coeffs_;
        D2 = right_dist_coeffs_;

        rms_ = cv::stereoCalibrate(
            obj_points_, left_img_points_, right_img_points_,
            K1, D1, K2, D2, img_size_, R, T, E, F,
            cv::CALIB_FIX_INTRINSIC,
            cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1e-5));

        RCLCPP_INFO(this->get_logger(), "RMS error: %.4f", rms_);
        RCLCPP_INFO(this->get_logger(), "✅ 스테레오 캘리브레이션 완료");

        // 파일 저장도 가능
        cv::FileStorage fs(save_absolute_path_ + "stereo_cam_calib.yaml", cv::FileStorage::WRITE);
        RCLCPP_INFO(this->get_logger(), "K1 matrix:\n%s", matToString(K1).c_str());
        fs << "checkerboard_cols" << cols_;
        fs << "checkerboard_rows" << rows_;
        fs << "square_size" << square_size_;
        fs << "frame_width" << frame_width_;
        fs << "frame_height" << frame_height_;
        fs << "left_intrinsic_matrix" << K1;
        fs << "left_distortion_coefficients" << D1;
        fs << "right_intrinsic_matrix" << K2;
        fs << "right_distortion_coefficients" << D2;
        fs << "rotation" << R;
        fs << "translation" << T;
        fs << "essential_matrix" << E;
        fs << "fundamental_matrix" << F;
        fs << "RMS error" << rms_;
        fs.release();
        fs.release();

        cv::Mat R1, R2, P1, P2, Q;

        cv::Mat map1x, map1y, map2x, map2y;

        cv::stereoRectify(K1, D1, K2, D2, img_size_, R, T, R1, R2, P1, P2, Q);
        cv::initUndistortRectifyMap(K1, D1, R1, P1, img_size_, CV_32FC1, map1x, map1y);
        cv::initUndistortRectifyMap(K2, D2, R2, P2, img_size_, CV_32FC1, map2x, map2y);

        // [ADD] Rectify and save all collected images
        for (int i = 0; i < collected_; i++)
        {
            // load original images
            std::string fname_left = save_origin_path_ + "img_" + std::to_string(i) + "_left_origin.png";
            std::string fname_right = save_origin_path_ + "img_" + std::to_string(i) + "_right_origin.png";

            cv::Mat orig_left = cv::imread(fname_left, cv::IMREAD_COLOR);
            cv::Mat orig_right = cv::imread(fname_right, cv::IMREAD_COLOR);

            if (orig_left.empty() || orig_right.empty())
            {
                RCLCPP_WARN(this->get_logger(), "Failed to load original images for rectification!");
                continue;
            }

            cv::Mat rect_left, rect_right;
            cv::remap(orig_left, rect_left, map1x, map1y, cv::INTER_LINEAR);
            cv::remap(orig_right, rect_right, map2x, map2y, cv::INTER_LINEAR);

            std::vector<cv::Point2f> corners_left, corners_right;
            cv::findChessboardCorners(rect_left, board_size_, corners_left);
            cv::findChessboardCorners(rect_right, board_size_, corners_right);

            cv::Mat left_gray, right_gray;
            cv::cvtColor(rect_left, left_gray, cv::COLOR_BGR2GRAY);
            cv::cvtColor(rect_right, right_gray, cv::COLOR_BGR2GRAY);
            cv::cornerSubPix(left_gray, corners_left, cv::Size(5, 5), cv::Size(-1, -1),
                             cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001));
            cv::cornerSubPix(right_gray, corners_right, cv::Size(5, 5), cv::Size(-1, -1),
                             cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001));

            std::vector<cv::Point3f> objp;
            for (int y = 0; y < board_size_.height; ++y)
                for (int x = 0; x < board_size_.width; ++x)
                    objp.emplace_back(x * square_size_, y * square_size_, 0.0);

            obj_points_.push_back(objp);
            left_img_points_.push_back(corners_left);
            right_img_points_.push_back(corners_right);

            int corner_idx = 30; // 아무 index 골라서 (0 ~ (cols*rows - 1))
            auto pt_left = corners_left[corner_idx];
            auto pt_right = corners_right[corner_idx];

            auto stereoBM = cv::StereoBM::create(64, 11);
            cv::Mat disparity, disparity_normalized;
            stereoBM->compute(left_gray, right_gray, disparity);

            disparity.convertTo(disparity, CV_32F, 1.0 / 16.0); // StereoBM의 결과는 fixed-point로 나오므로
                                                                // Disparity → Depth
            cv::Mat depth;
            cv::reprojectImageTo3D(disparity, depth, Q);

            double y_diff = std::abs(pt_left.y - pt_right.y);
            std::cout << "Y좌표 차이 = " << y_diff << " pixels" << std::endl;

            if (y_diff < 2.0)
            {
                std::cout << "✅ Rectification OK (epipolar lines 수평)" << std::endl;
            }
            else
            {
                std::cout << "⚠️ Rectification 미흡 - epipolar line이 완전 수평이 아님" << std::endl;
            }

            // 코너 좌표 그리기
            cv::circle(rect_left, pt_left, 3, cv::Scalar(0, 255, 0), -1);
            cv::circle(rect_right, pt_right, 3, cv::Scalar(0, 255, 0), -1);

            float d = disparity.at<float>(pt_left.y, pt_left.x);
            std::cout << "Disparity at (" << pt_left.x << "," << pt_left.y << ") = " << d << std::endl;

            float z_left = depth.at<cv::Vec3f>(
                static_cast<int>(pt_left.y),
                static_cast<int>(pt_left.x))[2];

            std::ostringstream oss_left;
            oss_left << "( " << std::fixed << std::setprecision(2) << pt_left.x
                     << " , " << std::fixed << std::setprecision(2) << pt_left.y
                     << " , " << std::fixed << std::setprecision(1) << z_left << " )";

            cv::putText(rect_left, oss_left.str(),
                        cv::Point(pt_left.x, pt_left.y - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

            std::ostringstream oss_right;
            oss_right << "( " << std::fixed << std::setprecision(2) << pt_right.x
                      << " , " << std::fixed << std::setprecision(2) << pt_right.y
                      << " , " << std::fixed << std::setprecision(1) << z_left << " )";

            cv::putText(rect_right, oss_right.str(),
                        cv::Point(pt_right.x, pt_right.y - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

            // 선택한 y 라인 그리기
            int y_line_left = static_cast<int>(pt_left.y);
            int y_line_right = static_cast<int>(pt_right.y);
            cv::line(rect_left, cv::Point(0, y_line_left), cv::Point(rect_left.cols, y_line_left), cv::Scalar(0, 0, 255), 1);
            cv::line(rect_right, cv::Point(0, y_line_right), cv::Point(rect_right.cols, y_line_right), cv::Scalar(0, 255, 0), 1);

            std::string save_left = save_calib_path_ + "img_" + std::to_string(i) + "_left_calib.png";
            std::string save_right = save_calib_path_ + "img_" + std::to_string(i) + "_right_calib.png";

            cv::imwrite(save_left, rect_left);
            cv::imwrite(save_right, rect_right);
        }

        RCLCPP_INFO(this->get_logger(), "✅ 모든 rectified 이미지 저장 완료");

        rclcpp::shutdown(); // 작업 완료 후 노드 종료
    }

    std::string matToString(const cv::Mat &mat)
    {
        std::ostringstream oss;
        oss << mat;
        return oss.str();
    }

    void visualizeEpipolarLines(const cv::Mat &F)
    {
        for (int i = 0; i < collected_; i++)
        {
            std::string fname_left = save_origin_path_ + "img_" + std::to_string(i) + "_left_origin.png";
            std::string fname_right = save_origin_path_ + "img_" + std::to_string(i) + "_right_origin.png";

            cv::Mat img_left = cv::imread(fname_left);
            cv::Mat img_right = cv::imread(fname_right);

            if (img_left.empty() || img_right.empty())
                continue;

            // left → right epipolar lines
            std::vector<cv::Vec3f> lines_right;
            cv::computeCorrespondEpilines(left_img_points_[i], 1, F, lines_right);

            for (size_t j = 0; j < lines_right.size(); j++)
            {
                const auto &line = lines_right[j];
                double a = line[0], b = line[1], c = line[2];

                if (std::abs(b) < 1e-6)
                    continue; // avoid div by zero

                cv::Point pt1(0, static_cast<int>(-c / b));
                cv::Point pt2(img_right.cols, static_cast<int>((-c - a * img_right.cols) / b));

                cv::line(img_right, pt1, pt2, cv::Scalar(0, 255, 0), 1);
            }

            // right → left epipolar lines
            std::vector<cv::Vec3f> lines_left;
            cv::computeCorrespondEpilines(right_img_points_[i], 2, F, lines_left);

            for (size_t j = 0; j < lines_left.size(); j++)
            {
                const auto &line = lines_left[j];
                double a = line[0], b = line[1], c = line[2];

                if (std::abs(b) < 1e-6)
                    continue;

                cv::Point pt1(0, static_cast<int>(-c / b));
                cv::Point pt2(img_left.cols, static_cast<int>((-c - a * img_left.cols) / b));

                cv::line(img_left, pt1, pt2, cv::Scalar(255, 0, 0), 1);
            }

            cv::imshow("Left Epipolar Lines", img_left);
            cv::imshow("Right Epipolar Lines", img_right);
            cv::waitKey(0);
        }
    }
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<StereoCamCalibNode>());
    rclcpp::shutdown();
    return 0;
}
