#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <filesystem>

using std::placeholders::_1;

namespace fs = std::filesystem;

class StereoCalibNode : public rclcpp::Node
{
public:
    StereoCalibNode() : Node("stereo_calib_node")
    {
        left_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/left/image_raw", 10, std::bind(&StereoCalibNode::leftCallback, this, _1));
        right_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/right/image_raw", 10, std::bind(&StereoCalibNode::rightCallback, this, _1));

            std::string where_ = "company";

        if (where_ == "company")
        {
            save_origin_path_ = "/home/antlab/sensor_fusion_study_ws/src/sensor_fusion_study/origin_images/";
            save_calib_path_ = "/home/antlab/sensor_fusion_study_ws/src/sensor_fusion_study/calib_images/";
            save_calib_result_ = "/home/antlab/sensor_fusion_study_ws/src/sensor_fusion_study/";
        }
        else if (where_ == "home")
        {
            save_origin_path_ = "/home/icrs/sensor_fusion_study_ws/src/sensor_fusion_study/origin_images/";
            save_calib_path_ = "/home/icrs/sensor_fusion_study_ws/src/sensor_fusion_study/calib_images/";
            save_calib_result_ = "/home/icrs/sensor_fusion_study_ws/src/sensor_fusion_study/";
        }
        fs::create_directories(save_origin_path_);
        fs::create_directories(save_calib_path_);
        fs::create_directories(save_calib_result_);
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr left_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr right_sub_;

    std::string save_origin_path_;
    std::string save_calib_path_;
    std::string save_calib_result_;

    std::vector<std::vector<cv::Point2f>> left_img_points_, right_img_points_;
    std::vector<std::vector<cv::Point3f>> obj_points_;
    cv::Size board_size_ = cv::Size(10, 7);
    int collected_ = 0;

    void leftCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        left_frame_ = cv_bridge::toCvCopy(msg, "bgr8")->image;
        processFrame();
    }

    void rightCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        right_frame_ = cv_bridge::toCvCopy(msg, "bgr8")->image;
        processFrame();
    }

    void processFrame()
    {
        if (left_frame_.empty() || right_frame_.empty())
            return;

        std::vector<cv::Point2f> corners_left, corners_right;
        bool found_left = cv::findChessboardCorners(left_frame_, board_size_, corners_left);
        bool found_right = cv::findChessboardCorners(right_frame_, board_size_, corners_right);

        if (found_left && found_right)
        {
            cv::drawChessboardCorners(left_frame_, board_size_, corners_left, found_left);
            cv::drawChessboardCorners(right_frame_, board_size_, corners_right, found_right);

            std::vector<cv::Point3f> objp;
            for (int i = 0; i < board_size_.height; ++i)
                for (int j = 0; j < board_size_.width; ++j)
                    objp.emplace_back(j, i, 0);

            obj_points_.push_back(objp);
            left_img_points_.push_back(corners_left);
            right_img_points_.push_back(corners_right);

            RCLCPP_INFO(this->get_logger(), "📸 캘리브레이션 이미지 %d 장 수집됨", ++collected_);

            if (collected_ >= 20)
            {
                calibrateStereo();
            }
        }
    }

    void calibrateStereo()
    {
        cv::Mat R, T, E, F;
        cv::FileStorage fss(save_calib_result_ + "calibration_result.yaml", cv::FileStorage::READ);
        cv::Mat K1, D1, K2, D2;
        fss["camera_matrix"] >> K1;
        fss["distortion_coefficients"] >> D1;
        fss["camera_matrix"] >> K2;
        fss["distortion_coefficients"] >> D2;

        cv::Size img_size = left_frame_.size();

        cv::stereoCalibrate(
            obj_points_, left_img_points_, right_img_points_,
            K1, D1, K2, D2, img_size, R, T, E, F,
            cv::CALIB_FIX_INTRINSIC,
            cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1e-5));

        RCLCPP_INFO(this->get_logger(), "✅ 스테레오 캘리브레이션 완료");

        // 파일 저장도 가능
        cv::FileStorage fs(save_calib_result_ + "stereo_calib.yaml", cv::FileStorage::WRITE);
        RCLCPP_INFO(this->get_logger(), "K1 matrix:\n%s", matToString(K1).c_str());
        fs << "K1" << K1 << "D1" << D1 << "K2" << K2 << "D2" << D2 << "R" << R << "T" << T << "E" << E << "F" << F;
        fs.release();

        cv::Mat R1, R2, P1, P2, Q;

        cv::Mat map1x, map1y, map2x, map2y;

        cv::stereoRectify(K1, D1, K2, D2, img_size, R, T, R1, R2, P1, P2, Q);
        cv::initUndistortRectifyMap(K1, D1, R1, P1, img_size, CV_32FC1, map1x, map1y);
        cv::initUndistortRectifyMap(K2, D2, R2, P2, img_size, CV_32FC1, map2x, map2y);

        cv::Mat rectified_left, rectified_right;
        cv::remap(left_frame_, rectified_left, map1x, map1y, cv::INTER_LINEAR);
        cv::remap(right_frame_, rectified_right, map2x, map2y, cv::INTER_LINEAR);

        cv::imwrite(save_calib_result_ + "rectified_left.png", rectified_left);
        cv::imwrite(save_calib_result_ + "rectified_right.png", rectified_right);

        rclcpp::shutdown(); // 작업 완료 후 노드 종료
    }

    std::string matToString(const cv::Mat &mat)
    {
        std::ostringstream oss;
        oss << mat;
        return oss.str();
    }

    cv::Mat left_frame_, right_frame_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<StereoCalibNode>());
    rclcpp::shutdown();
    return 0;
}