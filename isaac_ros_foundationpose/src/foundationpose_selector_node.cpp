////////// Add FSM flag for pose estimation and tracking (exact sync for fpt)
// SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <rclcpp/time.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>

#include "message_filters/subscriber.h"
#include "message_filters/synchronizer.h"
#include "message_filters/sync_policies/exact_time.h"

#include "isaac_ros_tensor_list_interfaces/msg/tensor_list.hpp"

#include "isaac_ros_managed_nitros/managed_nitros_publisher.hpp"
#include "isaac_ros_managed_nitros/managed_nitros_subscriber.hpp"
#include "isaac_ros_managed_nitros/managed_nitros_message_filters_subscriber.hpp"

#include "isaac_ros_nitros_image_type/nitros_image.hpp"
#include "isaac_ros_nitros_image_type/nitros_image_view.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list_view.hpp"

#include <std_msgs/msg/bool.hpp>
#include "std_msgs/msg/int8.hpp"
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>



namespace nvidia
{
namespace isaac_ros
{
namespace foundationpose
{

/*
ROS 2 node that select the next action - tracking or pose estimation.
State flow: kPoseEstimation -> kWaitingReset -> kTracking
*/

class Selector : public rclcpp::Node
{
public:
  explicit Selector(const rclcpp::NodeOptions & options)
  : Node("selector", options), 
    pose_estimation_or_tracking_flag_(0)
  {
    // Create publishers for pose estimation
    pose_estimation_image_pub_ = std::make_shared<nvidia::isaac_ros::nitros::ManagedNitrosPublisher<
          nvidia::isaac_ros::nitros::NitrosImage>>(
      this, "pose_estimation/image",
      nvidia::isaac_ros::nitros::nitros_image_rgb8_t::supported_type_name);
    pose_estimation_depth_pub_ = std::make_shared<nvidia::isaac_ros::nitros::ManagedNitrosPublisher<
          nvidia::isaac_ros::nitros::NitrosImage>>(
      this, "pose_estimation/depth_image",
      nvidia::isaac_ros::nitros::nitros_image_32FC1_t::supported_type_name);
    pose_estimation_segmenation_pub_ = std::make_shared<
      nvidia::isaac_ros::nitros::ManagedNitrosPublisher<nvidia::isaac_ros::nitros::NitrosImage>>(
      this, "pose_estimation/segmentation",
      nvidia::isaac_ros::nitros::nitros_image_mono8_t::supported_type_name);
    pose_estimation_camera_pub_ = this->create_publisher<
      sensor_msgs::msg::CameraInfo>("pose_estimation/camera_info", 1);

    // Create publishers for tracking
    tracking_image_pub_ = std::make_shared<nvidia::isaac_ros::nitros::ManagedNitrosPublisher<
          nvidia::isaac_ros::nitros::NitrosImage>>(
      this, "tracking/image",
      nvidia::isaac_ros::nitros::nitros_image_rgb8_t::supported_type_name);
    tracking_depth_pub_ = std::make_shared<nvidia::isaac_ros::nitros::ManagedNitrosPublisher<
          nvidia::isaac_ros::nitros::NitrosImage>>(
      this, "tracking/depth_image",
      nvidia::isaac_ros::nitros::nitros_image_32FC1_t::supported_type_name);
    tracking_pose_pub_ = this->create_publisher<
      isaac_ros_tensor_list_interfaces::msg::TensorList>("tracking/pose_input", 1);
    tracking_camera_pub_ = this->create_publisher<
      sensor_msgs::msg::CameraInfo>("tracking/camera_info", 1);

    // TF broadcaster
    tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(this);

    // Exact Sync
    using namespace std::placeholders;
    exact_sync_ = std::make_shared<ExactSync>(
      ExactPolicy(20), rgb_image_sub_, depth_image_sub_, segmentation_sub_,
      camera_info_sub_);
    exact_sync_->registerCallback(
      std::bind(&Selector::selectionCallback, this, _1, _2, _3, _4));

    segmentation_sub_.subscribe(this, "segmentation");
    rgb_image_sub_.subscribe(this, "image");
    depth_image_sub_.subscribe(this, "depth_image");
    camera_info_sub_.subscribe(this, "camera_info");

    // Create subscriber for pose input
    tracking_output_sub_ =
      this->create_subscription<isaac_ros_tensor_list_interfaces::msg::TensorList>(
      "tracking/pose_matrix_output", 1, std::bind(&Selector::poseForwardCallback, this, _1));
    pose_estimation_output_sub_ =
      this->create_subscription<isaac_ros_tensor_list_interfaces::msg::TensorList>(
      "pose_estimation/pose_matrix_output", 1, std::bind(&Selector::poseResetCallback, this, _1));

    // Subscriber for FSM flag
    selector_state_sub_ = this->create_subscription<std_msgs::msg::Int8>(
      "/app/fp_select", 1, std::bind(&Selector::selectorStateCallback, this, _1));
  }

  // void PublishTF(const isaac_ros_tensor_list_interfaces::msg::TensorList::ConstSharedPtr & pose_msg, const std::string & parent_frame_id, const std::string & child_frame_id)
  // {
  //     // Check if the pose_msg has tensors
  //     if (pose_msg->tensors.empty()) {
  //         RCLCPP_WARN(this->get_logger(), "Received empty TensorList message, cannot broadcast transform.");
  //         return;
  //     }

  //     // Assuming the pose is stored in the first tensor
  //     const auto & tensor = pose_msg->tensors[0];

  //     // Check the shape of the tensor (should be 4x4 for a transformation matrix)
  //     if (tensor.shape.size() != 2 || tensor.shape[0] != 4 || tensor.shape[1] != 4) {
  //         RCLCPP_WARN(this->get_logger(), "Tensor does not have the correct shape for a transformation matrix.");
  //         return;
  //     }

  //     // Extract the transformation matrix
  //     const auto & data = tensor.data;

  //     if (data.size() != 16) {
  //         RCLCPP_WARN(this->get_logger(), "Tensor data size is not 16, cannot interpret as 4x4 matrix.");
  //         return;
  //     }

  //     // Create a TransformStamped message
  //     geometry_msgs::msg::TransformStamped transformStamped;
  //     transformStamped.header.stamp = this->now();
  //     transformStamped.header.frame_id = parent_frame_id;
  //     transformStamped.child_frame_id = child_frame_id;

  //     // Fill in the translation and rotation
  //     // Assuming row-major order in data vector
  //     tf2::Matrix3x3 rotation_matrix(
  //         data[0], data[1], data[2],
  //         data[4], data[5], data[6],
  //         data[8], data[9], data[10]);

  //     tf2::Vector3 translation(
  //         data[3],
  //         data[7],
  //         data[11]);

  //     tf2::Quaternion q;
  //     rotation_matrix.getRotation(q);

  //     transformStamped.transform.translation.x = translation.x();
  //     transformStamped.transform.translation.y = translation.y();
  //     transformStamped.transform.translation.z = translation.z();

  //     transformStamped.transform.rotation.w = q.w();
  //     transformStamped.transform.rotation.x = q.x();
  //     transformStamped.transform.rotation.y = q.y();
  //     transformStamped.transform.rotation.z = q.z();

  //     // Broadcast the transform
  //     tf_broadcaster_->sendTransform(transformStamped);
  // }

  void PublishTF(
      const isaac_ros_tensor_list_interfaces::msg::TensorList::ConstSharedPtr & pose_msg,
      const std::string & parent_frame_id,
      const std::string & child_frame_id)
  {
      // Check if the pose_msg has tensors
      if (pose_msg->tensors.empty()) {
          RCLCPP_WARN(this->get_logger(), "Received empty TensorList message, cannot broadcast transform.");
          return;
      }

      // Assuming the pose is stored in the first tensor
      const auto & tensor = pose_msg->tensors[0];

      // Check the shape of the tensor (should be 4x4 for a transformation matrix)
      if (tensor.shape.rank != 2 || tensor.shape.dims.size() < 2 ||
          tensor.shape.dims[0] != 4 || tensor.shape.dims[1] != 4) {
          RCLCPP_WARN(this->get_logger(), "Tensor does not have the correct shape for a transformation matrix.");
          return;
      }

      // Check the data type (assuming DT_FLOAT is represented by 1)
      const int DT_FLOAT = 1;
      if (tensor.data_type != DT_FLOAT) {
          RCLCPP_WARN(this->get_logger(), "Tensor data type is not float.");
          return;
      }

      // Extract the transformation matrix
      const auto & data = tensor.data;

      if (data.size() != 16 * sizeof(float)) {
          RCLCPP_WARN(this->get_logger(), "Tensor data size is not 16 floats, cannot interpret as 4x4 matrix.");
          return;
      }

      // Reinterpret data as float array
      const float* matrix_data = reinterpret_cast<const float*>(data.data());

      // Create a TransformStamped message
      geometry_msgs::msg::TransformStamped transformStamped;
      transformStamped.header.stamp = this->now();
      transformStamped.header.frame_id = parent_frame_id;
      transformStamped.child_frame_id = child_frame_id;

      // Fill in the translation and rotation
      tf2::Matrix3x3 rotation_matrix(
          matrix_data[0], matrix_data[1], matrix_data[2],
          matrix_data[4], matrix_data[5], matrix_data[6],
          matrix_data[8], matrix_data[9], matrix_data[10]);

      tf2::Vector3 translation(
          matrix_data[3],
          matrix_data[7],
          matrix_data[11]);

      tf2::Quaternion q;
      rotation_matrix.getRotation(q);

      transformStamped.transform.translation.x = translation.x();
      transformStamped.transform.translation.y = translation.y();
      transformStamped.transform.translation.z = translation.z();

      transformStamped.transform.rotation.x = q.x();
      transformStamped.transform.rotation.y = q.y();
      transformStamped.transform.rotation.z = q.z();
      transformStamped.transform.rotation.w = q.w();

      // Broadcast the transform
      tf_broadcaster_->sendTransform(transformStamped);
  }



  void selectionCallback(
    const nvidia::isaac_ros::nitros::NitrosImage::ConstSharedPtr & image_msg,
    const nvidia::isaac_ros::nitros::NitrosImage::ConstSharedPtr & depth_msg,
    const nvidia::isaac_ros::nitros::NitrosImage::ConstSharedPtr & segmentation_msg,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr & camera_info_msg)
  {
    std::unique_lock<std::mutex> lock(mutex_);
    // Trigger next action
    if (state_ == State::kPoseEstimation) {
        // Publish messages to pose estimation
        pose_estimation_image_pub_->publish(*image_msg);
        pose_estimation_camera_pub_->publish(*camera_info_msg);
        pose_estimation_depth_pub_->publish(*depth_msg);
        pose_estimation_segmenation_pub_->publish(*segmentation_msg);
        state_ = State::kWaitingReset;
    } else if (state_ == State::kTracking) {
        // Publish messages to tracking
        tracking_image_pub_->publish(*image_msg);
        tracking_camera_pub_->publish(*camera_info_msg);
        tracking_depth_pub_->publish(*depth_msg);
        if (tracking_pose_msg_) {
            tracking_pose_pub_->publish(*tracking_pose_msg_);
        } else {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "tracking_pose_msg_ is null, skipping publish.");
        }
    }
  }


  void poseForwardCallback(
    const isaac_ros_tensor_list_interfaces::msg::TensorList::ConstSharedPtr & tracking_output_msg)
  {
    std::unique_lock<std::mutex> lock(pose_mutex_);
    // Discard the stale pose messages from tracking to avoid drift
    if (state_ == State::kTracking) {
      tracking_pose_msg_ = tracking_output_msg;
      // // Broadcast TF
      // PublishTF(tracking_pose_msg_, "d435_color_optical_frame", "fpt");
    }
  }

  void poseResetCallback(
    const isaac_ros_tensor_list_interfaces::msg::TensorList::ConstSharedPtr
    & pose_estimation_output_msg)
  {
    std::unique_lock<std::mutex> lock(pose_mutex_);
    tracking_pose_msg_ = pose_estimation_output_msg;
    state_ = kTracking;
    // // Broadcast TF
    // PublishTF(tracking_pose_msg_, "d435_color_optical_frame", "fpe");
  }

  
  void selectorStateCallback(const std_msgs::msg::Int8::SharedPtr msg)
  {
    pose_estimation_or_tracking_flag_ = msg->data;

    if (pose_estimation_or_tracking_flag_ == 0) {
      state_ = State::kPoseEstimation;
      RCLCPP_INFO(this->get_logger(), "[Pose estimation]");
    }
    else if (pose_estimation_or_tracking_flag_ == 1) {
      state_ = State::kTracking;
      RCLCPP_INFO(this->get_logger(), "[Pose tracking]");
    }
  }

private:
  // Publishers for pose estimation
  std::shared_ptr<nvidia::isaac_ros::nitros::ManagedNitrosPublisher<
      nvidia::isaac_ros::nitros::NitrosImage>> pose_estimation_image_pub_;
  std::shared_ptr<nvidia::isaac_ros::nitros::ManagedNitrosPublisher<
      nvidia::isaac_ros::nitros::NitrosImage>> pose_estimation_depth_pub_;
  std::shared_ptr<nvidia::isaac_ros::nitros::ManagedNitrosPublisher<
      nvidia::isaac_ros::nitros::NitrosImage>> pose_estimation_segmenation_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr pose_estimation_camera_pub_;

  // Publishers for tracking
  std::shared_ptr<nvidia::isaac_ros::nitros::ManagedNitrosPublisher<
      nvidia::isaac_ros::nitros::NitrosImage>> tracking_image_pub_;
  std::shared_ptr<nvidia::isaac_ros::nitros::ManagedNitrosPublisher<
      nvidia::isaac_ros::nitros::NitrosImage>> tracking_depth_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr tracking_camera_pub_;
  rclcpp::Publisher<isaac_ros_tensor_list_interfaces::msg::TensorList>::SharedPtr
    tracking_pose_pub_;

  // Subscribers
  nvidia::isaac_ros::nitros::message_filters::Subscriber<
    nvidia::isaac_ros::nitros::NitrosImageView> rgb_image_sub_;
  nvidia::isaac_ros::nitros::message_filters::Subscriber<
    nvidia::isaac_ros::nitros::NitrosImageView> depth_image_sub_;
  nvidia::isaac_ros::nitros::message_filters::Subscriber<
    nvidia::isaac_ros::nitros::NitrosImageView> segmentation_sub_;
  message_filters::Subscriber<sensor_msgs::msg::CameraInfo> camera_info_sub_;

  rclcpp::Subscription<isaac_ros_tensor_list_interfaces::msg::TensorList>::SharedPtr
    tracking_output_sub_;
  rclcpp::Subscription<isaac_ros_tensor_list_interfaces::msg::TensorList>::SharedPtr
    pose_estimation_output_sub_;
  rclcpp::Subscription<std_msgs::msg::Int8>::SharedPtr 
    selector_state_sub_;

  // TF broadcaster
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  void PublishTF(const isaac_ros_tensor_list_interfaces::msg::TensorList::ConstSharedPtr & pose_msg, const std::string & child_frame_id);

  enum State
  {
    kTracking,
    kPoseEstimation,
    kWaitingReset
  };

  // State
  State state_ = State::kPoseEstimation;

  // Exact message sync policy
  using ExactPolicy = message_filters::sync_policies::ExactTime<
    nvidia::isaac_ros::nitros::NitrosImage,
    nvidia::isaac_ros::nitros::NitrosImage,
    nvidia::isaac_ros::nitros::NitrosImage,
    sensor_msgs::msg::CameraInfo>;
  using ExactSync = message_filters::Synchronizer<ExactPolicy>;
  std::shared_ptr<ExactSync> exact_sync_;

  std::mutex mutex_;
  std::mutex pose_mutex_;
  isaac_ros_tensor_list_interfaces::msg::TensorList::ConstSharedPtr tracking_pose_msg_;

  int pose_estimation_or_tracking_flag_;
};

}  // namespace foundationpose
}  // namespace isaac_ros
}  // namespace nvidia

// Register the component with the ROS system to create a shared library
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::foundationpose::Selector)