## 사용 센서 빌드 정리
  * __Ouster__
    rosdep install --from-paths ./src/ouster-ros/ -y --ignore-src
    colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release --packages-select ouster_ros ouster_sensor_msgs
    ouster page : <https://github.com/ouster-lidar/ouster-ros/tree/humble-devel>

  * __Flir__
    rosdep install --from-paths ./src/flir_camera_driver/ --ignore-src
    colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_EXPORT_COMPILE_COMMANDS=ON --packages-select spinnaker_camera_driver spinnaker_synchronized_camera_driver flir_camera_msgs flir_camera_description
    flir page : <https://github.com/ros-drivers/flir_camera_driver?tab=readme-ov-file>

  * __Microstrain__
    rosdep install --from-paths ./src/microstrain_inertial/ -i -r -y
    colcon build --packages-select microstrain_inertial_driver microstrain_inertial_msgs microstrain_inretial_examples microstrain_inertial_rqt microstrain_inertial_description
    microstrain page : <https://github.com/LORD-MicroStrain/microstrain_inertial>

  * __Axis__
    colcon build --symlink-install --packages-select camera_info_manager_py ptz_action_server_msgs axis_camera axis_description axis_msgs
    axis page : <https://github.com/ros-drivers/axis_camera/tree/humble-devel>
