#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <filesystem>

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
          std::bind(&StereoCamCalibNode::timer_callback, this));
    
        std::string where_ = "company";
        read_write_path(where_);
        initializedParameters();
          }

private:
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr left_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr right_sub_;

    std::string save_absolute_path_;
    std::string save_origin_path_;
    std::string save_calib_path_;
    std::string one_cam_result_path_;

    std::vector<std::vector<cv::Point2f>> left_img_points_, right_img_points_;
    std::vector<std::vector<cv::Point3f>> obj_points_;
    cv::Size board_size_;
    float square_size_;
    int cols_, rows_;
    cv::Mat intrinsic_matrix_, distortion_coeffs_;
    int collected_ = 0;
    int count_ = 0;
    
    rclcpp::TimerBase::SharedPtr timer_;
  cv::Mat last_image_;

    void timer_callback()
  {
    
      // 빈 화면이라도 띄우도록 할 수 있음
      cv::Mat dummy = cv::Mat::zeros(480, 640, CV_8UC3);
      cv::putText(dummy, "No camera image", cv::Point(50, 240),
                  cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
      cv::imshow("Camera Image", dummy);
    
    input_keyboard(last_image_);
  }
 
  
    void initializedParameters()
    {
        std::string where = "company";
        read_write_path(where);

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
            fs["intrinsic_matrix"] >> intrinsic_matrix_;
            fs["distortion_coefficients"] >> distortion_coeffs_;
            fs.release();
            board_size_ = cv::Size(cols_, rows_);
        }
    }

    void read_write_path(std::string where)
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

    void leftCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        left_frame_ = cv_bridge::toCvCopy(msg, "bgr8")->image;
        
    }

    void rightCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        right_frame_ = cv_bridge::toCvCopy(msg, "bgr8")->image;
        
    }

    void processFrame()
    {
        if (left_frame_.empty() || right_frame_.empty())
            return;
            

        // [ADD] Save origin images with numbering
            std::string filename_left = save_origin_path_ + "img_" + std::to_string(count_) + "_left_origin.png";
            std::string filename_right = save_origin_path_ + "img_" + std::to_string(count_) + "_right_origin.png";
            cv::imwrite(filename_left, left_frame_);
            cv::imwrite(filename_right, right_frame_);
            
        std::vector<cv::Point2f> corners_left, corners_right;
        bool found_left = cv::findChessboardCorners(left_frame_, board_size_, corners_left,cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
        bool found_right = cv::findChessboardCorners(right_frame_, board_size_, corners_right,cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);

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
            
            collected_++;
            count_++;

        }
    }
    
    void input_keyboard(const cv::Mat &frame)
  {
    int key = cv::waitKey(1);
    if (key == 's')
    {
      processFrame();
    }
    else if (key == 'c')
    {

    }
    else if (key == 'e')
    {

    }
  }

    void calibrateStereo()
    {
        cv::Mat R, T, E, F;
        cv::Mat K1, D1, K2, D2;
        K1 = intrinsic_matrix_;
        K2 = K1;
        D1 = distortion_coeffs_;
        D2 = D1;

        cv::Size img_size = left_frame_.size();

        cv::stereoCalibrate(
            obj_points_, left_img_points_, right_img_points_,
            K1, D1, K2, D2, img_size, R, T, E, F,
            cv::CALIB_FIX_INTRINSIC,
            cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 100, 1e-5));

        RCLCPP_INFO(this->get_logger(), "✅ 스테레오 캘리브레이션 완료");

        // 파일 저장도 가능
        cv::FileStorage fs(save_absolute_path_ + "stereo_cam_calib.yaml", cv::FileStorage::WRITE);
        RCLCPP_INFO(this->get_logger(), "K1 matrix:\n%s", matToString(K1).c_str());
        fs << "K1" << K1 << "D1" << D1 << "K2" << K2 << "D2" << D2 << "R" << R << "T" << T << "E" << E << "F" << F;
        fs << "checkerboard_cols" << cols_;
        fs << "checkerboard_rows" << rows_;
        fs << "square_size" << square_size_;
        fs << "left_intrinsic_matrix" << K1;
        fs << "left_distortion_coefficients" << D1;
        fs << "right_intrinsic_matrix" << K2;
        fs << "right_distortion_coefficients" << D2;
        fs << "rotation" << R;
        fs << "translation" << T;
        fs << "essential_matrix" << E;
        fs << "fundamental_matrix" << F;
        fs.release();
        fs.release();

        cv::Mat R1, R2, P1, P2, Q;

        cv::Mat map1x, map1y, map2x, map2y;

        cv::stereoRectify(K1, D1, K2, D2, img_size, R, T, R1, R2, P1, P2, Q);
        cv::initUndistortRectifyMap(K1, D1, R1, P1, img_size, CV_32FC1, map1x, map1y);
        cv::initUndistortRectifyMap(K2, D2, R2, P2, img_size, CV_32FC1, map2x, map2y);

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

    cv::Mat left_frame_, right_frame_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<StereoCamCalibNode>());
    rclcpp::shutdown();
    return 0;
}
