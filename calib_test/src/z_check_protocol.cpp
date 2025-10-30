// dump_image_bytes.cpp
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <chrono>

using std::placeholders::_1;

std::string hex_preview(const std::vector<uint8_t>& v, size_t n = 64) {
  auto block = [](const uint8_t* p, size_t len){
    std::ostringstream oss;
    for (size_t i=0;i<len;i++){
      oss << std::uppercase << std::ios::binary << std::setw(2) << std::setfill('0')
          << static_cast<int>(p[i]) << (i+1<len ? ' ' : '\0');
    }
    return oss.str();
  };
  std::ostringstream out;
  size_t head = std::min(n, v.size());
  out << "HEAD(" << head << "B): " << block(v.data(), head);
  if (v.size() > n) {
    size_t tail = n;
    out << "\nTAIL(" << tail << "B): " << block(v.data()+v.size()-tail, tail);
  }
  return out.str();
}

class OneShotDump : public rclcpp::Node {
public:
  OneShotDump(const std::string& topic, size_t preview, const std::string& out)
  : Node("image_bytes_dump_cpp"), preview_(preview), out_(out) {
    sub_ = create_subscription<sensor_msgs::msg::Image>(
      topic, rclcpp::SensorDataQoS(),
      std::bind(&OneShotDump::cb, this, _1));
  }
private:
  void cb(const sensor_msgs::msg::Image::SharedPtr msg) {
    if (done_) return;
    done_ = true;

    const auto& data = msg->data;
    auto now = std::chrono::system_clock::now();
    auto t  = std::chrono::system_clock::to_time_t(now);
    std::stringstream base;
    if (!out_.empty()) base << out_;
    else               base << "img_" << t;

    std::string bin_path  = base.str() + ".bin";
    std::string meta_path = base.str() + ".meta.txt";

    // 바이트 그대로 저장
    std::ofstream f(bin_path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(data.data()), data.size());
    f.close();

    // 메타
    std::ofstream m(meta_path);
    m << "encoding=" << msg->encoding << "\n"
      << "width="    << msg->width    << "\n"
      << "height="   << msg->height   << "\n"
      << "step="     << msg->step     << "\n"
      << "data_size="<< data.size()   << "\n";
    m.close();

    RCLCPP_INFO(get_logger(), "enc=%s size=%ux%u step=%u bytes=%zu",
                msg->encoding.c_str(), msg->width, msg->height, msg->step, data.size());
    RCLCPP_INFO(get_logger(), "\n%s", hex_preview(data, preview_).c_str());

    rclcpp::shutdown();
  }
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
  bool done_{false};
  size_t preview_;
  std::string out_;
};

int main(int argc, char** argv){
  rclcpp::init(argc, argv);
  std::string topic = "/flir_camera/image_raw";
  size_t preview = 64;
  std::string out = "flir_raw_cpp";

  // 간단 파라미터
  for (int i=1;i<argc;i++){
    std::string a = argv[i];
    if (a=="--topic" && i+1<argc) topic = argv[++i];
    else if (a=="--preview" && i+1<argc) preview = std::stoul(argv[++i]);
    else if (a=="--out" && i+1<argc) out = argv[++i];
  }

  auto node = std::make_shared<OneShotDump>(topic, preview, out);
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
