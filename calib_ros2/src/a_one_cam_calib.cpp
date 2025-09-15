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
  int cols_, rows_, frame_width_, frame_height_;
  float square_size_;
  double rms_;

  std::vector<std::vector<cv::Point2f>> img_points_;
  std::vector<std::vector<cv::Point3f>> obj_points_;
  std::vector<cv::Mat> rvecs_, tvecs_;
  cv::Mat intrinsic_matrix_, dist_coeffs_;
  std::vector<std::string> image_files_;

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

  void runCalibrateFromFolder()
  {
    RCLCPP_INFO(this->get_logger(), "Start calibration...");

    image_files_ = jmh_utils::loadFiles(".png", origin_path_);

    jmh_utils::BoardParameter params;
    params.columns = cols_;
    params.rows = rows_;
    params.square_size = square_size_;
    params.frame_width = frame_width_;
    params.frame_height = frame_height_;

    jmh_utils::ResultIntrinsic result;
    result = jmh_utils::runCalibrate(params, image_files_);

    intrinsic_matrix_ = result.intrinsic_mat;
    dist_coeffs_ = result.distortion_coeffs;

    RCLCPP_INFO(this->get_logger(), "RMS error: %.4f", result.rms);
    cv::FileStorage fs(one_cam_path_ + "a_one_cam_calib_result.yaml", cv::FileStorage::WRITE);
    fs << "checkerboard_cols" << params.columns;
    fs << "checkerboard_rows" << params.rows;
    fs << "square_size" << params.square_size;
    fs << "frame_width" << params.frame_width;
    fs << "frame_height" << params.frame_height;
    fs << "intrinsic_matrix" << result.intrinsic_mat;
    fs << "distortion_coefficients" << result.distortion_coeffs;
    fs << "RMS error" << result.rms;
    fs.release();
    RCLCPP_INFO(this->get_logger(), "Succeeded result.yaml saving");

    for (int idx = 0; idx < result.visualize_corners.size(); idx++)
    {
      const auto &file_num = result.successed_index[idx];
      std::string save_name = calib_path_ + "img_" + std::to_string(file_num) + "_calib.png";
      cv::imwrite(save_name, result.visualize_corners[idx]);
      RCLCPP_INFO(this->get_logger(), "Save calibration image: %d", file_num);
    }
    /*
    // 코너 검출 실패 시 원본 이미지만 저장
    cv::Mat vis = img.clone();
    std::string failed_save_name = calib_path_ + "img_" + std::to_string(idx) + "_failed.png";
    cv::imwrite(failed_save_name, vis);
    RCLCPP_INFO(this->get_logger(), "Save failed image: %s", std::to_string(idx).c_str());
    */
  }

  void saveUndistortedImages()
  {
    RCLCPP_INFO(this->get_logger(), "Start saving undistorted images...");

    if (intrinsic_matrix_.empty() || dist_coeffs_.empty())
    {
      RCLCPP_INFO(this->get_logger(), "Find intrinsic parameter...");
      cv::FileStorage fs(one_cam_path_ + "a_one_cam_calib_result.yaml", cv::FileStorage::READ);
      fs ["frame_width"] >> frame_width_;
      fs ["frame_height"] >> frame_height_;
      fs["intrinsic_matrix"] >> intrinsic_matrix_;
      fs["distortion_coefficients"] >> dist_coeffs_;
      fs.release();

      image_files_ = jmh_utils::loadFiles(".png", origin_path_);
    }

    jmh_utils::ResultUndistort result_undistort = jmh_utils::runUndistorted(frame_width_, frame_height_,
                                                                            intrinsic_matrix_, dist_coeffs_,
                                                                            image_files_);

    std::string undistorted_path = one_cam_path_ + "undistorted_images/";
    fs::create_directories(undistorted_path);
    for (int i = 0; i < result_undistort.visualize_undistort.size(); i++)
    {
      const auto &file_num = result_undistort.undistroted_index[i];
      std::string save_name = undistorted_path + "img_" + std::to_string(file_num) + "_undistorted.png";
      cv::imwrite(save_name, result_undistort.visualize_undistort[i]);
      RCLCPP_INFO(this->get_logger(), "Save undistorted image: %d", file_num);
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
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<OneCamCalibNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}