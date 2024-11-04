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

#include <filesystem>

#include "ament_index_cpp/get_package_share_directory.hpp"
#include "detection3_d_array_message/detection3_d_array_message.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "std_msgs/msg/string.hpp"  // Include for String message

#include "isaac_ros_nitros_camera_info_type/nitros_camera_info.hpp"
#include "isaac_ros_nitros_detection3_d_array_type/nitros_detection3_d_array.hpp"
#include "isaac_ros_nitros_image_type/nitros_image.hpp"
#include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list.hpp"

#include "isaac_ros_foundationpose/foundationpose_node.hpp"

namespace
{
  // Define the subscriber and control variable in an anonymous namespace
  // rclcpp::Subscription<std_msgs::msg::String>::SharedPtr estimate_pose_subscription_;
  bool estimate_pose_or_not = true;  // Default to false

  // // Define the callback function for the subscriber
  // void EstimatePoseCallback(const std_msgs::msg::String::SharedPtr msg)
  // {
  //   if (msg->data == "button_registration") {
  //     estimate_pose_or_not = true;
  //   }
  // }
} // namespace

namespace nvidia
{
namespace isaac_ros
{
namespace foundationpose
{

using nvidia::gxf::optimizer::GraphIOGroupSupportedDataTypesInfoList;

constexpr char INPUT_DEPTH_COMPONENT_KEY[] = "sync/depth_receiver";
constexpr char INPUT_DEPTH_TENSOR_FORMAT[] = "nitros_image_32FC1";
constexpr char INPUT_DEPTH_TOPIC_NAME[] = "pose_estimation/depth_image";

constexpr char INPUT_RGB_IMAGE_COMPONENT_KEY[] = "sync/rgb_image_receiver";
constexpr char INPUT_RGB_IMAGE_TENSOR_FORMAT[] = "nitros_image_rgb8";
constexpr char INPUT_RGB_IMAGE_TOPIC_NAME[] = "pose_estimation/image";

constexpr char INPUT_CAMERA_INFO_COMPONENT_KEY[] = "sync/camera_model_receiver";
constexpr char INPUT_CAMERA_INFO_FORMAT[] = "nitros_camera_info";
constexpr char INPUT_CAMERA_INFO_TOPIC_NAME[] = "pose_estimation/camera_info";

constexpr char INPUT_SEGMENTATION_COMPONENT_KEY[] = "sync/mask_receiver";
constexpr char INPUT_SEGMENTATION_FORMAT[] = "nitros_image_mono8";
constexpr char INPUT_SEGMENTATION_TOPIC_NAME[] = "pose_estimation/segmentation";

constexpr char OUTPUT_MATRIX_COMPONENT_KEY[] = "pose_matrix_sink/sink";
constexpr char OUTPUT_MATRIX_FORMAT[] = "nitros_tensor_list_nchw";
constexpr char OUTPUT_MATRIX_TOPIC_NAME[] = "pose_estimation/pose_matrix_output";

constexpr char OUTPUT_POSE_COMPONENT_KEY[] = "pose_sink/sink";
constexpr char OUTPUT_POSE_FORMAT[] = "nitros_detection3_d_array";
constexpr char OUTPUT_POSE_TOPIC_NAME[] = "pose_estimation/output";

constexpr char APP_YAML_FILENAME[] = "config/nitros_foundationpose_node.yaml";
constexpr char PACKAGE_NAME[] = "isaac_ros_foundationpose";

const std::vector<std::pair<std::string, std::string>> EXTENSIONS = {
  {"isaac_ros_gxf", "gxf/lib/std/libgxf_std.so"},
  {"isaac_ros_gxf", "gxf/lib/multimedia/libgxf_multimedia.so"},
  {"isaac_ros_gxf", "gxf/lib/serialization/libgxf_serialization.so"},
  {"isaac_ros_gxf", "gxf/lib/cuda/libgxf_cuda.so"},
  {"gxf_isaac_depth_image_proc", "gxf/lib/libgxf_isaac_depth_image_proc.so"},
  {"gxf_isaac_sgm", "gxf/lib/libgxf_isaac_sgm.so"},
  {"gxf_isaac_messages", "gxf/lib/libgxf_isaac_messages.so"},
  {"gxf_isaac_ros_messages", "gxf/lib/libgxf_isaac_ros_messages.so"},
  {"gxf_isaac_tensor_rt", "gxf/lib/libgxf_isaac_tensor_rt.so"},
  {"gxf_isaac_foundationpose", "gxf/lib/libgxf_isaac_foundationpose.so"},
};

const std::vector<std::string> PRESET_EXTENSION_SPEC_NAMES = {
  "isaac_ros_foundationpose",
};
const std::vector<std::string> EXTENSION_SPEC_FILENAMES = {};
const std::vector<std::string> GENERATOR_RULE_FILENAMES = {"config/namespace_injector_rule.yaml"};
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
const nitros::NitrosPublisherSubscriberConfigMap CONFIG_MAP = {
  {INPUT_DEPTH_COMPONENT_KEY,
    {
      .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
      .qos = rclcpp::QoS(10),
      .compatible_data_format = INPUT_DEPTH_TENSOR_FORMAT,
      .topic_name = INPUT_DEPTH_TOPIC_NAME,
    }},
  {INPUT_RGB_IMAGE_COMPONENT_KEY,
    {
      .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
      .qos = rclcpp::QoS(10),
      .compatible_data_format = INPUT_RGB_IMAGE_TENSOR_FORMAT,
      .topic_name = INPUT_RGB_IMAGE_TOPIC_NAME,
    }},
  {INPUT_CAMERA_INFO_COMPONENT_KEY,
    {
      .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
      .qos = rclcpp::QoS(10),
      .compatible_data_format = INPUT_CAMERA_INFO_FORMAT,
      .topic_name = INPUT_CAMERA_INFO_TOPIC_NAME,
    }},
  {INPUT_SEGMENTATION_COMPONENT_KEY,
    {
      .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
      .qos = rclcpp::QoS(10),
      .compatible_data_format = INPUT_SEGMENTATION_FORMAT,
      .topic_name = INPUT_SEGMENTATION_TOPIC_NAME,
    }},
  {OUTPUT_POSE_COMPONENT_KEY,
    {
      .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
      .qos = rclcpp::QoS(10),
      .compatible_data_format = OUTPUT_POSE_FORMAT,
      .topic_name = OUTPUT_POSE_TOPIC_NAME,
      .frame_id_source_key = INPUT_RGB_IMAGE_COMPONENT_KEY,
    }},
  {OUTPUT_MATRIX_COMPONENT_KEY,
    {
      .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
      .qos = rclcpp::QoS(10),
      .compatible_data_format = OUTPUT_MATRIX_FORMAT,
      .topic_name = OUTPUT_MATRIX_TOPIC_NAME,
      .frame_id_source_key = INPUT_RGB_IMAGE_COMPONENT_KEY,
    }},
};
#pragma GCC diagnostic pop

FoundationPoseNode::FoundationPoseNode(rclcpp::NodeOptions options)
: nitros::NitrosNode(
    options, APP_YAML_FILENAME, CONFIG_MAP, PRESET_EXTENSION_SPEC_NAMES, EXTENSION_SPEC_FILENAMES,
    GENERATOR_RULE_FILENAMES, EXTENSIONS, PACKAGE_NAME),
  configuration_file_(
    declare_parameter<std::string>("configuration_file", "foundationpose_model_config.yaml")),
  mesh_file_path_(declare_parameter<std::string>("mesh_file_path", "textured_simple.obj")),
  texture_path_(declare_parameter<std::string>("texture_path", "texture_map.png")),
  min_depth_(declare_parameter<float>("min_depth", 0.1)),
  max_depth_(declare_parameter<float>("max_depth", 4.0)),
  refine_iterations_(declare_parameter<int>("refine_iterations", 1)),
  refine_model_file_path_(
    declare_parameter<std::string>("refine_model_file_path", "/tmp/refine_model.onnx")),
  refine_engine_file_path_(
    declare_parameter<std::string>("refine_engine_file_path", "/tmp/refine_trt_engine.plan")),
  score_model_file_path_(
    declare_parameter<std::string>("score_model_file_path", "/tmp/score_model.onnx")),
  score_engine_file_path_(
    declare_parameter<std::string>("score_engine_file_path", "/tmp/score_trt_engine.plan")),

  refine_input_tensor_names_(
    declare_parameter<StringList>("refine_input_tensor_names", StringList())),
  refine_input_binding_names_(
    declare_parameter<StringList>("refine_input_binding_names", StringList())),
  score_input_tensor_names_(
    declare_parameter<StringList>("score_input_tensor_names", StringList())),
  score_input_binding_names_(
    declare_parameter<StringList>("score_input_binding_names", StringList())),

  refine_output_tensor_names_(
    declare_parameter<StringList>("refine_output_tensor_names", StringList())),
  refine_output_binding_names_(
    declare_parameter<StringList>("refine_output_binding_names", StringList())),
  score_output_tensor_names_(
    declare_parameter<StringList>("score_output_tensor_names", StringList())),
  score_output_binding_names_(
    declare_parameter<StringList>("score_output_binding_names", StringList())),

  // tf_frame_name_(declare_parameter<std::string>("tf_frame_name", "fp_object"))
  tf_frame_name_(declare_parameter<std::string>("tf_frame_name", "fpe"))
{
  RCLCPP_DEBUG(get_logger(), "[FoundationPoseNode] Constructor");

  // Initialize subscriber for the "estimate_pose" topic
  // estimate_pose_subscription_ = this->create_subscription<std_msgs::msg::String>(
  //   "/app/button_registration", 10, EstimatePoseCallback); // Use the callback defined in the anonymous namespace

  // Add callback function for FoundationPose Detection3D array to broadcast to ROS TF tree
  config_map_[OUTPUT_POSE_COMPONENT_KEY].callback = std::bind(
    &FoundationPoseNode::FoundationPoseDetectionCallback, this, std::placeholders::_1,
    std::placeholders::_2);

  // Open configuration YAML file
  const std::string package_directory = ament_index_cpp::get_package_share_directory(
    "isaac_ros_foundationpose");
  std::filesystem::path yaml_path =
    package_directory / std::filesystem::path("config") /
    std::filesystem::path(configuration_file_);
  if (!std::filesystem::exists(yaml_path)) {
    RCLCPP_ERROR(this->get_logger(), "%s could not be found. Exiting.", yaml_path.string().c_str());
    throw std::runtime_error("Parameter parsing failure.");
  }

  // Parse parameters
  rcl_params_t * foundationpose_params = rcl_yaml_node_struct_init(rcutils_get_default_allocator());
  rcl_parse_yaml_file(yaml_path.c_str(), foundationpose_params);

  rcl_variant_t * max_hypothesis = rcl_yaml_node_struct_get(
    "foundationpose", "max_hypothesis", foundationpose_params);
  if (!max_hypothesis->integer_value) {
    RCLCPP_ERROR(this->get_logger(), "No max_hypothesis parameter found");
    throw std::runtime_error("Parameter parsing failure.");
  }
  max_hypothesis_ = static_cast<uint32_t>(*max_hypothesis->integer_value);

  rcl_variant_t * resized_image_width = rcl_yaml_node_struct_get(
    "foundationpose", "resized_image_width", foundationpose_params);
  if (!resized_image_width->integer_value) {
    RCLCPP_ERROR(this->get_logger(), "No resized_image_width parameter found");
    throw std::runtime_error("Parameter parsing failure.");
  }
  resized_image_width_ = static_cast<uint32_t>(*resized_image_width->integer_value);

  rcl_variant_t * resized_image_height = rcl_yaml_node_struct_get(
    "foundationpose", "resized_image_height", foundationpose_params);
  if (!resized_image_width->integer_value) {
    RCLCPP_ERROR(this->get_logger(), "No resized_image_height parameter found");
    throw std::runtime_error("Parameter parsing failure.");
  }
  resized_image_height_ = static_cast<uint32_t>(*resized_image_height->integer_value);

  rcl_variant_t * refine_crop_ratio = rcl_yaml_node_struct_get(
    "foundationpose", "refine_crop_ratio", foundationpose_params);
  if (!refine_crop_ratio->double_value) {
    RCLCPP_ERROR(this->get_logger(), "No refine_crop_ratio parameter found");
    throw std::runtime_error("Parameter parsing failure.");
  }
  refine_crop_ratio_ = static_cast<float>(*refine_crop_ratio->double_value);

  rcl_variant_t * score_crop_ratio = rcl_yaml_node_struct_get(
    "foundationpose", "score_crop_ratio", foundationpose_params);
  if (!score_crop_ratio->double_value) {
    RCLCPP_ERROR(this->get_logger(), "No score_crop_ratio parameter found");
    throw std::runtime_error("Parameter parsing failure.");
  }
  score_crop_ratio_ = static_cast<float>(*score_crop_ratio->double_value);

  rcl_variant_t * rot_normalizer = rcl_yaml_node_struct_get(
    "foundationpose", "rot_normalizer", foundationpose_params);
  if (!rot_normalizer->double_value) {
    RCLCPP_ERROR(this->get_logger(), "No rot_normalizer parameter found");
    throw std::runtime_error("Parameter parsing failure.");
  }
  rot_normalizer_ = static_cast<float>(*rot_normalizer->double_value);

  rcl_yaml_node_struct_fini(foundationpose_params);

  registerSupportedType<nvidia::isaac_ros::nitros::NitrosCameraInfo>();
  registerSupportedType<nvidia::isaac_ros::nitros::NitrosImage>();
  registerSupportedType<nvidia::isaac_ros::nitros::NitrosDetection3DArray>();
  registerSupportedType<nvidia::isaac_ros::nitros::NitrosTensorList>();

  startNitrosNode();
}

FoundationPoseNode::~FoundationPoseNode() = default;

void FoundationPoseNode::postLoadGraphCallback()
{
  RCLCPP_DEBUG(get_logger(), "[FoundationPoseNode] postLoadGraphCallback().");

  // Set the model configuration file path from parameter
  getNitrosContext().setParameterUInt32(
    "sampling", "nvidia::isaac_ros::FoundationposeSampling", "max_hypothesis",
    max_hypothesis_);

  getNitrosContext().setParameterUInt32(
    "render", "nvidia::isaac_ros::FoundationposeRender", "resized_image_width",
    resized_image_width_);

  getNitrosContext().setParameterUInt32(
    "render_score", "nvidia::isaac_ros::FoundationposeRender", "resized_image_width",
    resized_image_width_);

  getNitrosContext().setParameterUInt32(
    "render", "nvidia::isaac_ros::FoundationposeRender", "resized_image_height",
    resized_image_height_);

  getNitrosContext().setParameterUInt32(
    "render_score", "nvidia::isaac_ros::FoundationposeRender", "resized_image_height",
    resized_image_height_);

  getNitrosContext().setParameterFloat32(
    "render", "nvidia::isaac_ros::FoundationposeRender", "crop_ratio", refine_crop_ratio_);

  getNitrosContext().setParameterFloat32(
    "render_score", "nvidia::isaac_ros::FoundationposeRender", "crop_ratio", score_crop_ratio_);

  getNitrosContext().setParameterFloat32(
    "transform", "nvidia::isaac_ros::FoundationposeTransformation", "rot_normalizer",
    rot_normalizer_);

  // Set the mesh path from parameter
  getNitrosContext().setParameterStr(
    "render", "nvidia::isaac_ros::FoundationposeRender", "mesh_file_path", mesh_file_path_);

  getNitrosContext().setParameterStr(
    "render_score", "nvidia::isaac_ros::FoundationposeRender", "mesh_file_path", mesh_file_path_);

  getNitrosContext().setParameterStr(
    "render", "nvidia::isaac_ros::FoundationposeRender", "texture_path", texture_path_);

  getNitrosContext().setParameterStr(
    "render_score", "nvidia::isaac_ros::FoundationposeRender", "texture_path", texture_path_);

  getNitrosContext().setParameterStr(
    "decoder", "nvidia::isaac_ros::FoundationposeDecoder", "mesh_file_path", mesh_file_path_);

  getNitrosContext().setParameterStr(
    "decoder", "nvidia::isaac_ros::FoundationposeDecoder", "mesh_file_path", mesh_file_path_);

  getNitrosContext().setParameterStr(
    "transform", "nvidia::isaac_ros::FoundationposeTransformation", "mesh_file_path",
    mesh_file_path_);

  // Set the depth threshold
  getNitrosContext().setParameterFloat32(
    "render", "nvidia::isaac_ros::FoundationposeRender", "min_depth", min_depth_);

  getNitrosContext().setParameterFloat32(
    "render_score", "nvidia::isaac_ros::FoundationposeRender", "min_depth", min_depth_);

  getNitrosContext().setParameterFloat32(
    "render", "nvidia::isaac_ros::FoundationposeRender", "max_depth", max_depth_);

  getNitrosContext().setParameterFloat32(
    "render_score", "nvidia::isaac_ros::FoundationposeRender", "max_depth", max_depth_);

  getNitrosContext().setParameterFloat32(
    "sampling", "nvidia::isaac_ros::FoundationposeSampling", "min_depth", min_depth_);

  // Set the refine iterations
  getNitrosContext().setParameterInt32(
    "render", "nvidia::isaac_ros::FoundationposeRender", "refine_iterations",
    refine_iterations_);
  getNitrosContext().setParameterInt32(
    "transform", "nvidia::isaac_ros::FoundationposeTransformation", "refine_iterations",
    refine_iterations_);

  // Set the refine network TensorRT configs from parameter
  getNitrosContext().setParameterStr(
    "refine_inference", "nvidia::gxf::TensorRtInference", "model_file_path",
    refine_model_file_path_);

  getNitrosContext().setParameterStr(
    "refine_inference", "nvidia::gxf::TensorRtInference", "engine_file_path",
    refine_engine_file_path_);

  getNitrosContext().setParameter1DStrVector(
    "refine_inference", "nvidia::gxf::TensorRtInference", "input_tensor_names",
    refine_input_tensor_names_);

  getNitrosContext().setParameter1DStrVector(
    "refine_inference", "nvidia::gxf::TensorRtInference", "input_binding_names",
    refine_input_binding_names_);

  getNitrosContext().setParameter1DStrVector(
    "refine_inference", "nvidia::gxf::TensorRtInference", "output_tensor_names",
    refine_output_tensor_names_);

  getNitrosContext().setParameter1DStrVector(
    "refine_inference", "nvidia::gxf::TensorRtInference", "output_binding_names",
    refine_output_binding_names_);

  // Set the score network TensorRT configs from parameter
  getNitrosContext().setParameterStr(
    "score_inference", "nvidia::gxf::TensorRtInference", "model_file_path",
    score_model_file_path_);

  getNitrosContext().setParameterStr(
    "score_inference", "nvidia::gxf::TensorRtInference", "engine_file_path",
    score_engine_file_path_);

  getNitrosContext().setParameter1DStrVector(
    "score_inference", "nvidia::gxf::TensorRtInference", "input_tensor_names",
    score_input_tensor_names_);

  getNitrosContext().setParameter1DStrVector(
    "score_inference", "nvidia::gxf::TensorRtInference", "input_binding_names",
    score_input_binding_names_);

  getNitrosContext().setParameter1DStrVector(
    "score_inference", "nvidia::gxf::TensorRtInference", "output_tensor_names",
    score_output_tensor_names_);

  getNitrosContext().setParameter1DStrVector(
    "score_inference", "nvidia::gxf::TensorRtInference", "output_binding_names",
    score_output_binding_names_);
}

void FoundationPoseNode::FoundationPoseDetectionCallback(
  const gxf_context_t context, nitros::NitrosTypeBase & msg)
{
  // // Check if estimate_pose_or_not is true
  // if (!estimate_pose_or_not) {
  //   RCLCPP_WARN(get_logger(), "[FoundationPoseNode] Estimate pose condition not met. Skipping callback.");
  //   return;  // Exit the function if the condition is not met
  // }

  // Reset the flag to false after checking
  estimate_pose_or_not = false;

  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_ =
    std::make_unique<tf2_ros::TransformBroadcaster>(*this);
  geometry_msgs::msg::TransformStamped transform_stamped;

  auto msg_entity = nvidia::gxf::Entity::Shared(context, msg.handle);

  // Populate timestamp information back into the header
  auto maybe_input_timestamp = msg_entity->get<nvidia::gxf::Timestamp>();
  if (!maybe_input_timestamp) {    // Fallback to label 'timestamp'
    maybe_input_timestamp = msg_entity->get<nvidia::gxf::Timestamp>("timestamp");
  }
  if (maybe_input_timestamp) {
    transform_stamped.header.stamp.sec = static_cast<int32_t>(
      maybe_input_timestamp.value()->acqtime / static_cast<uint64_t>(1e9));
    transform_stamped.header.stamp.nanosec = static_cast<uint32_t>(
      maybe_input_timestamp.value()->acqtime % static_cast<uint64_t>(1e9));
  } else {
    RCLCPP_WARN(
      get_logger(),
      "[FoundationPoseNode] Failed to get timestamp");
  }

  //  Extract foundation pose list to a struct type defined in detection3_d_array_message.hpp
  auto foundation_pose_detections_array_expected = nvidia::isaac::GetDetection3DListMessage(
    msg_entity.value());
  if (!foundation_pose_detections_array_expected) {
    RCLCPP_ERROR(
      get_logger(), "[FoundationPoseNode] Failed to get detections data from message entity");
    return;
  }
  auto foundation_pose_detections_array = foundation_pose_detections_array_expected.value();

  // Extract number of tags detected
  size_t tags_count = foundation_pose_detections_array.count;

  std::cout << "[FoundationPoseNode] Detected " << tags_count << " object" << std::endl;

  if (tags_count > 0) {
    // struct is defined in fiducial_message.hpp
    auto pose = foundation_pose_detections_array.poses.at(0).value();

    transform_stamped.header.frame_id = msg.frame_id;
    transform_stamped.child_frame_id = tf_frame_name_;
    transform_stamped.transform.translation.x = pose->translation.x();
    transform_stamped.transform.translation.y = pose->translation.y();
    transform_stamped.transform.translation.z = pose->translation.z();
    transform_stamped.transform.rotation.x = pose->rotation.quaternion().x();
    transform_stamped.transform.rotation.y = pose->rotation.quaternion().y();
    transform_stamped.transform.rotation.z = pose->rotation.quaternion().z();
    transform_stamped.transform.rotation.w = pose->rotation.quaternion().w();

    tf_broadcaster_->sendTransform(transform_stamped);
  }
}

}  // namespace foundationpose
}  // namespace isaac_ros
}  // namespace nvidia

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::foundationpose::FoundationPoseNode)

























// Original code
// // SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
// // Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// //
// // Licensed under the Apache License, Version 2.0 (the "License");
// // you may not use this file except in compliance with the License.
// // You may obtain a copy of the License at
// //
// // http://www.apache.org/licenses/LICENSE-2.0
// //
// // Unless required by applicable law or agreed to in writing, software
// // distributed under the License is distributed on an "AS IS" BASIS,
// // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// // See the License for the specific language governing permissions and
// // limitations under the License.
// //
// // SPDX-License-Identifier: Apache-2.0

// #include <filesystem>

// #include "ament_index_cpp/get_package_share_directory.hpp"
// #include "detection3_d_array_message/detection3_d_array_message.hpp"
// #include "geometry_msgs/msg/transform_stamped.hpp"
// #include "rclcpp/rclcpp.hpp"
// #include "tf2_ros/transform_broadcaster.h"

// #include "isaac_ros_nitros_camera_info_type/nitros_camera_info.hpp"
// #include "isaac_ros_nitros_detection3_d_array_type/nitros_detection3_d_array.hpp"
// #include "isaac_ros_nitros_image_type/nitros_image.hpp"
// #include "isaac_ros_nitros_tensor_list_type/nitros_tensor_list.hpp"

// #include "isaac_ros_foundationpose/foundationpose_node.hpp"

// namespace nvidia
// {
// namespace isaac_ros
// {
// namespace foundationpose
// {

// using nvidia::gxf::optimizer::GraphIOGroupSupportedDataTypesInfoList;

// constexpr char INPUT_DEPTH_COMPONENT_KEY[] = "sync/depth_receiver";
// constexpr char INPUT_DEPTH_TENSOR_FORMAT[] = "nitros_image_32FC1";
// constexpr char INPUT_DEPTH_TOPIC_NAME[] = "pose_estimation/depth_image";

// constexpr char INPUT_RGB_IMAGE_COMPONENT_KEY[] = "sync/rgb_image_receiver";
// constexpr char INPUT_RGB_IMAGE_TENSOR_FORMAT[] = "nitros_image_rgb8";
// constexpr char INPUT_RGB_IMAGE_TOPIC_NAME[] = "pose_estimation/image";

// constexpr char INPUT_CAMERA_INFO_COMPONENT_KEY[] = "sync/camera_model_receiver";
// constexpr char INPUT_CAMERA_INFO_FORMAT[] = "nitros_camera_info";
// constexpr char INPUT_CAMERA_INFO_TOPIC_NAME[] = "pose_estimation/camera_info";

// constexpr char INPUT_SEGMENTATION_COMPONENT_KEY[] = "sync/mask_receiver";
// constexpr char INPUT_SEGMENTATION_FORMAT[] = "nitros_image_mono8";
// constexpr char INPUT_SEGMENTATION_TOPIC_NAME[] = "pose_estimation/segmentation";

// constexpr char OUTPUT_MATRIX_COMPONENT_KEY[] = "pose_matrix_sink/sink";
// constexpr char OUTPUT_MATRIX_FORMAT[] = "nitros_tensor_list_nchw";
// constexpr char OUTPUT_MATRIX_TOPIC_NAME[] = "pose_estimation/pose_matrix_output";

// constexpr char OUTPUT_POSE_COMPONENT_KEY[] = "pose_sink/sink";
// constexpr char OUTPUT_POSE_FORMAT[] = "nitros_detection3_d_array";
// constexpr char OUTPUT_POSE_TOPIC_NAME[] = "pose_estimation/output";

// constexpr char APP_YAML_FILENAME[] = "config/nitros_foundationpose_node.yaml";
// constexpr char PACKAGE_NAME[] = "isaac_ros_foundationpose";

// const std::vector<std::pair<std::string, std::string>> EXTENSIONS = {
//   {"isaac_ros_gxf", "gxf/lib/std/libgxf_std.so"},
//   {"isaac_ros_gxf", "gxf/lib/multimedia/libgxf_multimedia.so"},
//   {"isaac_ros_gxf", "gxf/lib/serialization/libgxf_serialization.so"},
//   {"isaac_ros_gxf", "gxf/lib/cuda/libgxf_cuda.so"},
//   {"gxf_isaac_depth_image_proc", "gxf/lib/libgxf_isaac_depth_image_proc.so"},
//   {"gxf_isaac_sgm", "gxf/lib/libgxf_isaac_sgm.so"},
//   {"gxf_isaac_messages", "gxf/lib/libgxf_isaac_messages.so"},
//   {"gxf_isaac_ros_messages", "gxf/lib/libgxf_isaac_ros_messages.so"},
//   {"gxf_isaac_tensor_rt", "gxf/lib/libgxf_isaac_tensor_rt.so"},
//   {"gxf_isaac_foundationpose", "gxf/lib/libgxf_isaac_foundationpose.so"},
// };

// const std::vector<std::string> PRESET_EXTENSION_SPEC_NAMES = {
//   "isaac_ros_foundationpose",
// };
// const std::vector<std::string> EXTENSION_SPEC_FILENAMES = {};
// const std::vector<std::string> GENERATOR_RULE_FILENAMES = {"config/namespace_injector_rule.yaml"};
// #pragma GCC diagnostic push
// #pragma GCC diagnostic ignored "-Wpedantic"
// const nitros::NitrosPublisherSubscriberConfigMap CONFIG_MAP = {
//   {INPUT_DEPTH_COMPONENT_KEY,
//     {
//       .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
//       .qos = rclcpp::QoS(10),
//       .compatible_data_format = INPUT_DEPTH_TENSOR_FORMAT,
//       .topic_name = INPUT_DEPTH_TOPIC_NAME,
//     }},
//   {INPUT_RGB_IMAGE_COMPONENT_KEY,
//     {
//       .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
//       .qos = rclcpp::QoS(10),
//       .compatible_data_format = INPUT_RGB_IMAGE_TENSOR_FORMAT,
//       .topic_name = INPUT_RGB_IMAGE_TOPIC_NAME,
//     }},
//   {INPUT_CAMERA_INFO_COMPONENT_KEY,
//     {
//       .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
//       .qos = rclcpp::QoS(10),
//       .compatible_data_format = INPUT_CAMERA_INFO_FORMAT,
//       .topic_name = INPUT_CAMERA_INFO_TOPIC_NAME,
//     }},
//   {INPUT_SEGMENTATION_COMPONENT_KEY,
//     {
//       .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
//       .qos = rclcpp::QoS(10),
//       .compatible_data_format = INPUT_SEGMENTATION_FORMAT,
//       .topic_name = INPUT_SEGMENTATION_TOPIC_NAME,
//     }},
//   {OUTPUT_POSE_COMPONENT_KEY,
//     {
//       .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
//       .qos = rclcpp::QoS(10),
//       .compatible_data_format = OUTPUT_POSE_FORMAT,
//       .topic_name = OUTPUT_POSE_TOPIC_NAME,
//       .frame_id_source_key = INPUT_RGB_IMAGE_COMPONENT_KEY,
//     }},
//   {OUTPUT_MATRIX_COMPONENT_KEY,
//     {
//       .type = nitros::NitrosPublisherSubscriberType::NEGOTIATED,
//       .qos = rclcpp::QoS(10),
//       .compatible_data_format = OUTPUT_MATRIX_FORMAT,
//       .topic_name = OUTPUT_MATRIX_TOPIC_NAME,
//       .frame_id_source_key = INPUT_RGB_IMAGE_COMPONENT_KEY,
//     }},
// };
// #pragma GCC diagnostic pop

// FoundationPoseNode::FoundationPoseNode(rclcpp::NodeOptions options)
// : nitros::NitrosNode(
//     options, APP_YAML_FILENAME, CONFIG_MAP, PRESET_EXTENSION_SPEC_NAMES, EXTENSION_SPEC_FILENAMES,
//     GENERATOR_RULE_FILENAMES, EXTENSIONS, PACKAGE_NAME),
//   configuration_file_(
//     declare_parameter<std::string>("configuration_file", "foundationpose_model_config.yaml")),
//   mesh_file_path_(declare_parameter<std::string>("mesh_file_path", "textured_simple.obj")),
//   texture_path_(declare_parameter<std::string>("texture_path", "texture_map.png")),
//   min_depth_(declare_parameter<float>("min_depth", 0.1)),
//   max_depth_(declare_parameter<float>("max_depth", 4.0)),
//   refine_iterations_(declare_parameter<int>("refine_iterations", 1)),
//   refine_model_file_path_(
//     declare_parameter<std::string>("refine_model_file_path", "/tmp/refine_model.onnx")),
//   refine_engine_file_path_(
//     declare_parameter<std::string>("refine_engine_file_path", "/tmp/refine_trt_engine.plan")),
//   score_model_file_path_(
//     declare_parameter<std::string>("score_model_file_path", "/tmp/score_model.onnx")),
//   score_engine_file_path_(
//     declare_parameter<std::string>("score_engine_file_path", "/tmp/score_trt_engine.plan")),

//   refine_input_tensor_names_(
//     declare_parameter<StringList>("refine_input_tensor_names", StringList())),
//   refine_input_binding_names_(
//     declare_parameter<StringList>("refine_input_binding_names", StringList())),
//   score_input_tensor_names_(
//     declare_parameter<StringList>("score_input_tensor_names", StringList())),
//   score_input_binding_names_(
//     declare_parameter<StringList>("score_input_binding_names", StringList())),

//   refine_output_tensor_names_(
//     declare_parameter<StringList>("refine_output_tensor_names", StringList())),
//   refine_output_binding_names_(
//     declare_parameter<StringList>("refine_output_binding_names", StringList())),
//   score_output_tensor_names_(
//     declare_parameter<StringList>("score_output_tensor_names", StringList())),
//   score_output_binding_names_(
//     declare_parameter<StringList>("score_output_binding_names", StringList())),

//   tf_frame_name_(declare_parameter<std::string>("tf_frame_name", "fp_object"))
// {
//   RCLCPP_DEBUG(get_logger(), "[FoundationPoseNode] Constructor");

//   // Add callback function for FoundationPose Detection3D array to broadcast to ROS TF tree
//   config_map_[OUTPUT_POSE_COMPONENT_KEY].callback = std::bind(
//     &FoundationPoseNode::FoundationPoseDetectionCallback, this, std::placeholders::_1,
//     std::placeholders::_2);

//   // Open configuration YAML file
//   const std::string package_directory = ament_index_cpp::get_package_share_directory(
//     "isaac_ros_foundationpose");
//   std::filesystem::path yaml_path =
//     package_directory / std::filesystem::path("config") /
//     std::filesystem::path(configuration_file_);
//   if (!std::filesystem::exists(yaml_path)) {
//     RCLCPP_ERROR(this->get_logger(), "%s could not be found. Exiting.", yaml_path.string().c_str());
//     throw std::runtime_error("Parameter parsing failure.");
//   }

//   // Parse parameters
//   rcl_params_t * foundationpose_params = rcl_yaml_node_struct_init(rcutils_get_default_allocator());
//   rcl_parse_yaml_file(yaml_path.c_str(), foundationpose_params);

//   rcl_variant_t * max_hypothesis = rcl_yaml_node_struct_get(
//     "foundationpose", "max_hypothesis", foundationpose_params);
//   if (!max_hypothesis->integer_value) {
//     RCLCPP_ERROR(this->get_logger(), "No max_hypothesis parameter found");
//     throw std::runtime_error("Parameter parsing failure.");
//   }
//   max_hypothesis_ = static_cast<uint32_t>(*max_hypothesis->integer_value);

//   rcl_variant_t * resized_image_width = rcl_yaml_node_struct_get(
//     "foundationpose", "resized_image_width", foundationpose_params);
//   if (!resized_image_width->integer_value) {
//     RCLCPP_ERROR(this->get_logger(), "No resized_image_width parameter found");
//     throw std::runtime_error("Parameter parsing failure.");
//   }
//   resized_image_width_ = static_cast<uint32_t>(*resized_image_width->integer_value);

//   rcl_variant_t * resized_image_height = rcl_yaml_node_struct_get(
//     "foundationpose", "resized_image_height", foundationpose_params);
//   if (!resized_image_width->integer_value) {
//     RCLCPP_ERROR(this->get_logger(), "No resized_image_height parameter found");
//     throw std::runtime_error("Parameter parsing failure.");
//   }
//   resized_image_height_ = static_cast<uint32_t>(*resized_image_height->integer_value);

//   rcl_variant_t * refine_crop_ratio = rcl_yaml_node_struct_get(
//     "foundationpose", "refine_crop_ratio", foundationpose_params);
//   if (!refine_crop_ratio->double_value) {
//     RCLCPP_ERROR(this->get_logger(), "No refine_crop_ratio parameter found");
//     throw std::runtime_error("Parameter parsing failure.");
//   }
//   refine_crop_ratio_ = static_cast<float>(*refine_crop_ratio->double_value);

//   rcl_variant_t * score_crop_ratio = rcl_yaml_node_struct_get(
//     "foundationpose", "score_crop_ratio", foundationpose_params);
//   if (!score_crop_ratio->double_value) {
//     RCLCPP_ERROR(this->get_logger(), "No score_crop_ratio parameter found");
//     throw std::runtime_error("Parameter parsing failure.");
//   }
//   score_crop_ratio_ = static_cast<float>(*score_crop_ratio->double_value);

//   rcl_variant_t * rot_normalizer = rcl_yaml_node_struct_get(
//     "foundationpose", "rot_normalizer", foundationpose_params);
//   if (!rot_normalizer->double_value) {
//     RCLCPP_ERROR(this->get_logger(), "No rot_normalizer parameter found");
//     throw std::runtime_error("Parameter parsing failure.");
//   }
//   rot_normalizer_ = static_cast<float>(*rot_normalizer->double_value);

//   rcl_yaml_node_struct_fini(foundationpose_params);

//   registerSupportedType<nvidia::isaac_ros::nitros::NitrosCameraInfo>();
//   registerSupportedType<nvidia::isaac_ros::nitros::NitrosImage>();
//   registerSupportedType<nvidia::isaac_ros::nitros::NitrosDetection3DArray>();
//   registerSupportedType<nvidia::isaac_ros::nitros::NitrosTensorList>();

//   startNitrosNode();
// }

// FoundationPoseNode::~FoundationPoseNode() = default;

// void FoundationPoseNode::postLoadGraphCallback()
// {
//   RCLCPP_DEBUG(get_logger(), "[FoundationPoseNode] postLoadGraphCallback().");

//   // Set the model configuration file path from parameter
//   getNitrosContext().setParameterUInt32(
//     "sampling", "nvidia::isaac_ros::FoundationposeSampling", "max_hypothesis",
//     max_hypothesis_);

//   getNitrosContext().setParameterUInt32(
//     "render", "nvidia::isaac_ros::FoundationposeRender", "resized_image_width",
//     resized_image_width_);

//   getNitrosContext().setParameterUInt32(
//     "render_score", "nvidia::isaac_ros::FoundationposeRender", "resized_image_width",
//     resized_image_width_);

//   getNitrosContext().setParameterUInt32(
//     "render", "nvidia::isaac_ros::FoundationposeRender", "resized_image_height",
//     resized_image_height_);

//   getNitrosContext().setParameterUInt32(
//     "render_score", "nvidia::isaac_ros::FoundationposeRender", "resized_image_height",
//     resized_image_height_);

//   getNitrosContext().setParameterFloat32(
//     "render", "nvidia::isaac_ros::FoundationposeRender", "crop_ratio", refine_crop_ratio_);

//   getNitrosContext().setParameterFloat32(
//     "render_score", "nvidia::isaac_ros::FoundationposeRender", "crop_ratio", score_crop_ratio_);

//   getNitrosContext().setParameterFloat32(
//     "transform", "nvidia::isaac_ros::FoundationposeTransformation", "rot_normalizer",
//     rot_normalizer_);

//   // Set the mesh path from parameter
//   getNitrosContext().setParameterStr(
//     "render", "nvidia::isaac_ros::FoundationposeRender", "mesh_file_path", mesh_file_path_);

//   getNitrosContext().setParameterStr(
//     "render_score", "nvidia::isaac_ros::FoundationposeRender", "mesh_file_path", mesh_file_path_);

//   getNitrosContext().setParameterStr(
//     "render", "nvidia::isaac_ros::FoundationposeRender", "texture_path", texture_path_);

//   getNitrosContext().setParameterStr(
//     "render_score", "nvidia::isaac_ros::FoundationposeRender", "texture_path", texture_path_);

//   getNitrosContext().setParameterStr(
//     "decoder", "nvidia::isaac_ros::FoundationposeDecoder", "mesh_file_path", mesh_file_path_);

//   getNitrosContext().setParameterStr(
//     "decoder", "nvidia::isaac_ros::FoundationposeDecoder", "mesh_file_path", mesh_file_path_);

//   getNitrosContext().setParameterStr(
//     "transform", "nvidia::isaac_ros::FoundationposeTransformation", "mesh_file_path",
//     mesh_file_path_);

//   // Set the depth threshold
//   getNitrosContext().setParameterFloat32(
//     "render", "nvidia::isaac_ros::FoundationposeRender", "min_depth", min_depth_);

//   getNitrosContext().setParameterFloat32(
//     "render_score", "nvidia::isaac_ros::FoundationposeRender", "min_depth", min_depth_);

//   getNitrosContext().setParameterFloat32(
//     "render", "nvidia::isaac_ros::FoundationposeRender", "max_depth", max_depth_);

//   getNitrosContext().setParameterFloat32(
//     "render_score", "nvidia::isaac_ros::FoundationposeRender", "max_depth", max_depth_);

//   getNitrosContext().setParameterFloat32(
//     "sampling", "nvidia::isaac_ros::FoundationposeSampling", "min_depth", min_depth_);

//   // Set the refine iterations
//   getNitrosContext().setParameterInt32(
//     "render", "nvidia::isaac_ros::FoundationposeRender", "refine_iterations",
//     refine_iterations_);
//   getNitrosContext().setParameterInt32(
//     "transform", "nvidia::isaac_ros::FoundationposeTransformation", "refine_iterations",
//     refine_iterations_);

//   // Set the refine network TensorRT configs from parameter
//   getNitrosContext().setParameterStr(
//     "refine_inference", "nvidia::gxf::TensorRtInference", "model_file_path",
//     refine_model_file_path_);

//   getNitrosContext().setParameterStr(
//     "refine_inference", "nvidia::gxf::TensorRtInference", "engine_file_path",
//     refine_engine_file_path_);

//   getNitrosContext().setParameter1DStrVector(
//     "refine_inference", "nvidia::gxf::TensorRtInference", "input_tensor_names",
//     refine_input_tensor_names_);

//   getNitrosContext().setParameter1DStrVector(
//     "refine_inference", "nvidia::gxf::TensorRtInference", "input_binding_names",
//     refine_input_binding_names_);

//   getNitrosContext().setParameter1DStrVector(
//     "refine_inference", "nvidia::gxf::TensorRtInference", "output_tensor_names",
//     refine_output_tensor_names_);

//   getNitrosContext().setParameter1DStrVector(
//     "refine_inference", "nvidia::gxf::TensorRtInference", "output_binding_names",
//     refine_output_binding_names_);

//   // Set the score network TensorRT configs from parameter
//   getNitrosContext().setParameterStr(
//     "score_inference", "nvidia::gxf::TensorRtInference", "model_file_path",
//     score_model_file_path_);

//   getNitrosContext().setParameterStr(
//     "score_inference", "nvidia::gxf::TensorRtInference", "engine_file_path",
//     score_engine_file_path_);

//   getNitrosContext().setParameter1DStrVector(
//     "score_inference", "nvidia::gxf::TensorRtInference", "input_tensor_names",
//     score_input_tensor_names_);

//   getNitrosContext().setParameter1DStrVector(
//     "score_inference", "nvidia::gxf::TensorRtInference", "input_binding_names",
//     score_input_binding_names_);

//   getNitrosContext().setParameter1DStrVector(
//     "score_inference", "nvidia::gxf::TensorRtInference", "output_tensor_names",
//     score_output_tensor_names_);

//   getNitrosContext().setParameter1DStrVector(
//     "score_inference", "nvidia::gxf::TensorRtInference", "output_binding_names",
//     score_output_binding_names_);
// }

// void FoundationPoseNode::FoundationPoseDetectionCallback(
//   const gxf_context_t context, nitros::NitrosTypeBase & msg)
// {
//   std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_ =
//     std::make_unique<tf2_ros::TransformBroadcaster>(*this);
//   geometry_msgs::msg::TransformStamped transform_stamped;

//   auto msg_entity = nvidia::gxf::Entity::Shared(context, msg.handle);

//   // Populate timestamp information back into the header
//   auto maybe_input_timestamp = msg_entity->get<nvidia::gxf::Timestamp>();
//   if (!maybe_input_timestamp) {    // Fallback to label 'timestamp'
//     maybe_input_timestamp = msg_entity->get<nvidia::gxf::Timestamp>("timestamp");
//   }
//   if (maybe_input_timestamp) {
//     transform_stamped.header.stamp.sec = static_cast<int32_t>(
//       maybe_input_timestamp.value()->acqtime / static_cast<uint64_t>(1e9));
//     transform_stamped.header.stamp.nanosec = static_cast<uint32_t>(
//       maybe_input_timestamp.value()->acqtime % static_cast<uint64_t>(1e9));
//   } else {
//     RCLCPP_WARN(
//       get_logger(),
//       "[FoundationPoseNode] Failed to get timestamp");
//   }

//   //  Extract foundation pose list to a struct type defined in detection3_d_array_message.hpp
//   auto foundation_pose_detections_array_expected = nvidia::isaac::GetDetection3DListMessage(
//     msg_entity.value());
//   if (!foundation_pose_detections_array_expected) {
//     RCLCPP_ERROR(
//       get_logger(), "[FoundationPoseNode] Failed to get detections data from message entity");
//     return;
//   }
//   auto foundation_pose_detections_array = foundation_pose_detections_array_expected.value();

//   // Extract number of tags detected
//   size_t tags_count = foundation_pose_detections_array.count;

//   std::cout << "tags_count: " << tags_count << std::endl;

//   if (tags_count > 0) {
//     // struct is defined in fiducial_message.hpp
//     auto pose = foundation_pose_detections_array.poses.at(0).value();

//     transform_stamped.header.frame_id = msg.frame_id;
//     transform_stamped.child_frame_id = tf_frame_name_;
//     transform_stamped.transform.translation.x = pose->translation.x();
//     transform_stamped.transform.translation.y = pose->translation.y();
//     transform_stamped.transform.translation.z = pose->translation.z();
//     transform_stamped.transform.rotation.x = pose->rotation.quaternion().x();
//     transform_stamped.transform.rotation.y = pose->rotation.quaternion().y();
//     transform_stamped.transform.rotation.z = pose->rotation.quaternion().z();
//     transform_stamped.transform.rotation.w = pose->rotation.quaternion().w();

//     tf_broadcaster_->sendTransform(transform_stamped);
//   }
// }

// }  // namespace foundationpose
// }  // namespace isaac_ros
// }  // namespace nvidia

// #include "rclcpp_components/register_node_macro.hpp"
// RCLCPP_COMPONENTS_REGISTER_NODE(nvidia::isaac_ros::foundationpose::FoundationPoseNode)































































