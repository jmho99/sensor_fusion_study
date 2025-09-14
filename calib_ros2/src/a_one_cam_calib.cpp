#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "std_msgs/msg/header.hpp"
#include <chrono>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <filesystem>

#include "jmh_utils/jmh_utils.hpp"

namespace fs = std::filesystem;

class OneCamCalibNode : public rclcpp::Node
{
public:
  OneCamCalibNode() : Node("a_one_cam_calib"), frame_counter_(0)
  {
    declare_parameter("select_connect", "ethernet");
    declare_parameter("device_path", "/dev/video1");
    declare_parameter("checkerboard_cols", 5);
    declare_parameter("checkerboard_rows", 7);
    declare_parameter("square_size", 0.1);
    declare_parameter("frame_width", 1920);
    declare_parameter("frame_height", 1200);

    get_parameter("select_connect", select_connect_);
    get_parameter("device_path", device_path_);
    get_parameter("checkerboard_cols", cols_);
    get_parameter("checkerboard_rows", rows_);
    get_parameter("square_size", square_size_);
    get_parameter("frame_width", frame_width_);
    get_parameter("frame_height", frame_height_);

    RCLCPP_INFO(this->get_logger(), "Open camera using %s", select_connect_.c_str());
    RCLCPP_INFO(this->get_logger(), "checkerboard %d x %d", cols_, rows_);
    RCLCPP_INFO(this->get_logger(), "checkerboard_size %f%s", square_size_, " m");
    RCLCPP_INFO(this->get_logger(), "frame_size %d x %d", frame_width_, frame_height_);

    if (select_connect_ == "usb")
    {
      connectUsbCamera();
    }
    else if (select_connect_ == "ethernet")
    {
      auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).reliable().durability_volatile();
      subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
          "/flir_camera/image_raw", qos,
          std::bind(&OneCamCalibNode::imageCallback, this, std::placeholders::_1));
      RCLCPP_INFO(this->get_logger(), "Open camera using ETHERNET");
    }

    keyboard_timer_ = this->create_wall_timer(
          std::chrono::milliseconds(30),
          std::bind(&OneCamCalibNode::keyboardCallback, this));


    readWritePath();
  }

private:
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
  std::string origin_path_;
  std::string calib_path_;
  std::string one_cam_path_;
  cv::Mat current_frame_;
  int frame_counter_;

  std::string select_connect_;
  std::string device_path_;
  int cols_;
  int rows_;
  float square_size_;
  int frame_width_;
  int frame_height_;
  double rms_;

  std::vector<std::vector<cv::Point2f>> img_points_;
  std::vector<std::vector<cv::Point3f>> obj_points_;
  std::vector<cv::Mat> rvecs_, tvecs_;
  cv::Mat intrinsic_matrix_, dist_coeffs_;
  std::vector<cv::String> image_files_;

  std::vector<int> successful_indices_;
  rclcpp::TimerBase::SharedPtr keyboard_timer_;
  cv::Mat last_image_;

  void readWritePath()
  {
    std::string home_dir = std::getenv("HOME");
    std::string calibration_path = home_dir + "/sensor_fusion_study_ws/src/sensor_fusion_study/calib_data";

    one_cam_path_ = calibration_path + "/a_one_cam_calib/";
    origin_path_ = one_cam_path_ + "origin_images/";
    calib_path_ = one_cam_path_ + "calib_images/";
    fs::create_directories(origin_path_);
    fs::create_directories(calib_path_);
  }

  void connectUsbCamera()
  {
    cv::VideoCapture cap;
    cap.open(device_path_, cv::CAP_V4L2);
    if (!cap.isOpened())
    {
      RCLCPP_INFO(this->get_logger(), "ERROR_open");
      return;
    }
    RCLCPP_INFO(this->get_logger(), "Open camera using USB");

    cap.set(cv::CAP_PROP_FRAME_WIDTH, frame_width_);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, frame_height_);
    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));

    if (!cap.read(current_frame_) || current_frame_.empty())
    {
      RCLCPP_INFO(this->get_logger(), "ERROR_frame");
      return;
    }
    while (true)
    {
      cap >> current_frame_;
      cv::imshow("MJPEG CAM", current_frame_);
    }
  }

  void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    try
    {
      current_frame_ = cv_bridge::toCvCopy(msg, "bgr8")->image;
      cv::namedWindow("FLIR View", cv::WINDOW_NORMAL);
      cv::resizeWindow("FLIR View", 640, 480);
      cv::imshow("FLIR View", current_frame_);
    }
    catch (cv_bridge::Exception &e)
    {
      RCLCPP_ERROR(this->get_logger(), "cv_bridge error: %s", e.what());
    }
  }

  void keyboardCallback()
  {
    if (jmh_utils::keyboardAvailable())
    {
      std::string input;
      std::getline(std::cin, input);
      
      if (input == "s")
      {
        jmh_utils::saveImageFile("png", origin_path_, frame_counter_, current_frame_);
        frame_counter_++;
      }
      else if (input == "c")
      {
        runCalibrateFromFolder();
      }
      else if (input == "e")
      {
        reporjectionError(obj_points_, img_points_, rvecs_, tvecs_, intrinsic_matrix_, dist_coeffs_, successful_indices_);
      }

      else if (input == "u")
      {
        saveUndistortedImages();
      }
    }
  }

  int extractNumber(const std::string &filename)
  {
    std::string number;
    for (char c : filename)
    {
      if (std::isdigit(c))
      {
        number += c;
      }
    }
    return number.empty() ? -1 : std::stoi(number);
  }

  void runCalibrateFromFolder()
  {
    RCLCPP_INFO(this->get_logger(), "Start calibration...");

    cv::glob(origin_path_ + "*.png", image_files_);
    if (image_files_.size() < 10)
    {
      RCLCPP_WARN(this->get_logger(), "Not enough image (%lu)", image_files_.size());
      return;
    }
    std::sort(image_files_.begin(), image_files_.end(),
              [this](const std::string &a, const std::string &b)
              {
                return extractNumber(a) < extractNumber(b);
              });

    cv::Size pattern_size(cols_, rows_);
    float square_size = square_size_;
    std::vector<cv::Point3f> objp;

    for (int i = 0; i < pattern_size.height; ++i)
      for (int j = 0; j < pattern_size.width; ++j)
        objp.emplace_back(j * square_size, i * square_size, 0.0f);

    successful_indices_.clear();

    for (size_t idx = 0; idx < image_files_.size(); ++idx)
    {
      const auto &file = image_files_[idx];
      cv::Mat img = cv::imread(file);
      if (img.empty())
        continue;

      std::vector<cv::Point2f> corners;
      bool found = cv::findChessboardCorners(img, pattern_size, corners,
                                             cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);

      if (found)
      {
        cv::Mat gray;
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        cv::cornerSubPix(gray, corners, cv::Size(5, 5), cv::Size(-1, -1),
                         cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001));

        img_points_.push_back(corners);
        obj_points_.push_back(objp);

        rms_ = cv::calibrateCamera(obj_points_, img_points_, cv::Size(frame_width_, frame_height_),
                                   intrinsic_matrix_, dist_coeffs_, rvecs_, tvecs_);

        RCLCPP_INFO(this->get_logger(), "RMS error: %.4f", rms_);
        cv::FileStorage fs(one_cam_path_ + "a_one_cam_calib_result.yaml", cv::FileStorage::WRITE);
        fs << "checkerboard_cols" << cols_;
        fs << "checkerboard_rows" << rows_;
        fs << "square_size" << square_size_;
        fs << "frame_width" << frame_width_;
        fs << "frame_height" << frame_height_;
        fs << "intrinsic_matrix" << intrinsic_matrix_;
        fs << "distortion_coefficients" << dist_coeffs_;
        fs << "rotation" << rvecs_;
        fs << "translation" << tvecs_;
        fs << "RMS error" << rms_;
        fs.release();
        RCLCPP_INFO(this->get_logger(), "Succeeded result.yaml saving");

        // ✅ 여기부터 시각화 코드 최소화
        cv::Mat vis = img.clone();

        // corners 만 표시 (체스보드 코너)
        cv::drawChessboardCorners(vis, pattern_size, corners, found);

        std::string save_name = calib_path_ + "img_" + std::to_string(idx) + "_calib.png";
        cv::imwrite(save_name, vis);
        RCLCPP_INFO(this->get_logger(), "Save calibration image: ", std::to_string(idx).c_str());
        successful_indices_.push_back(idx);
      }
      else
      {
        // 코너 검출 실패 시 원본 이미지만 저장
        cv::Mat vis = img.clone();
        std::string failed_save_name = calib_path_ + "img_" + std::to_string(idx) + "_failed.png";
        cv::imwrite(failed_save_name, vis);
        RCLCPP_INFO(this->get_logger(), "Save failed image: %s", std::to_string(idx).c_str());
      }

      if (img_points_.empty())
      {
        RCLCPP_ERROR(this->get_logger(), "Failed calibration.");
        return;
      }
    }
  }

  void reporjectionError(const std::vector<std::vector<cv::Point3f>> &obj_points_,
                         const std::vector<std::vector<cv::Point2f>> &img_points_,
                         const std::vector<cv::Mat> &rvecs_,
                         const std::vector<cv::Mat> &tvecs_,
                         const cv::Mat &intrinsic_matrix_,
                         const cv::Mat &dist_coeffs_,
                         const std::vector<int> &successful_indices_)
  {

    for (size_t i = 0; i < successful_indices_.size(); ++i)
    {
      int idx = successful_indices_[i];

      // 원본 이미지 경로
      std::string origin_file = origin_path_ + "img_" + std::to_string(idx) + ".png";
      cv::Mat img = cv::imread(origin_file);
      if (img.empty())
      {
        RCLCPP_WARN(rclcpp::get_logger("reporjectionError"),
                    "Image load failed: %s", std::to_string(idx).c_str());
        continue;
      }
      /*
            cv::Mat optimal_intrinsic = cv::getOptimalNewCameraMatrix(intrinsic_matrix_, dist_coeffs_, cv::Size(frame_width_, frame_height_),
                                                                      1, cv::Size(frame_width_, frame_height_));

            cv::Mat undistort_img;
            img = cv::undistort(img, undistort_img, intrinsic_matrix_, dist_coeffs_, optimal_intrinsic);*/

      std::vector<cv::Point2f> projected_points;
      cv::projectPoints(obj_points_[i], rvecs_[i], tvecs_[i],
                        intrinsic_matrix_, dist_coeffs_, projected_points);

      cv::Mat vis = img.clone();

      for (size_t j = 0; j < img_points_[i].size(); ++j)
      {
        cv::Point2f actual = img_points_[i][j];
        cv::Point2f reprojected = projected_points[j];

        cv::circle(vis, actual, 1, cv::Scalar(0, 255, 0), -1);
        cv::circle(vis, reprojected, 1, cv::Scalar(0, 0, 255), -1);
      }

      cv::Mat act = cv::Mat(img_points_[i]);
      cv::Mat reproj = cv::Mat(projected_points);
      float mean_error = cv::norm(act, reproj, cv::NORM_L2);

      RCLCPP_INFO(this->get_logger(), "norm error: %f", mean_error / img_points_[i].size());

      std::string save_name = calib_path_ + "img_" + std::to_string(idx) + "_error.png";
      cv::imwrite(save_name, vis);
      RCLCPP_INFO(this->get_logger(), "Save error visualization: %s", std::to_string(idx).c_str());

      cv::Point3f p = obj_points_[i][69];
      cv::Point2f c = img_points_[i][69];
    }
  }

  void saveUndistortedImages()
  {
    RCLCPP_INFO(this->get_logger(), "Start saving undistorted images...");

    if (intrinsic_matrix_.empty() || dist_coeffs_.empty())
    {
      RCLCPP_ERROR(this->get_logger(), "Intrinsic matrix or distortion coefficients are not available. Please run calibration ('c') first.");
      return;
    }

    cv::Mat map1, map2;
    cv::Size image_size(frame_width_, frame_height_);

    // 왜곡 보정 맵 생성
    cv::initUndistortRectifyMap(intrinsic_matrix_, dist_coeffs_, cv::Mat(),
                                cv::getOptimalNewCameraMatrix(intrinsic_matrix_, dist_coeffs_, image_size, 1, image_size),
                                image_size, CV_32FC1, map1, map2);

    std::string undistorted_path = one_cam_path_ + "undistorted_images/";
    fs::create_directories(undistorted_path);

    // 캘리브레이션에 성공한 이미지들에 대해서만 왜곡 보정 적용
    for (size_t i = 0; i < successful_indices_.size(); ++i)
    {
      int idx = successful_indices_[i];
      std::string origin_file = origin_path_ + "img_" + std::to_string(idx) + ".png";
      cv::Mat img = cv::imread(origin_file);
      if (img.empty())
      {
        RCLCPP_WARN(this->get_logger(), "Image load failed: %s", std::to_string(idx).c_str());
        continue;
      }

      cv::Mat undistorted_img;
      cv::remap(img, undistorted_img, map1, map2, cv::INTER_LINEAR);

      std::string save_name = undistorted_path + "img_" + std::to_string(idx) + "_undistorted.png";
      cv::imwrite(save_name, undistorted_img);
      RCLCPP_INFO(this->get_logger(), "Saved undistorted image: %s", std::to_string(idx).c_str());
    }
    RCLCPP_INFO(this->get_logger(), "Completed saving undistorted images.");
  }
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<OneCamCalibNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}