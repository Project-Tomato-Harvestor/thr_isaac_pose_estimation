# #################### YOLOV8; without drop_node
# # SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# # Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# # http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# #
# # SPDX-License-Identifier: Apache-2.0
# import os
# from typing import Any, Dict
# from ament_index_python.packages import get_package_share_directory
# # from isaac_ros_examples import IsaacROSLaunchFragment
# import launch
# from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
# from launch.launch_description_sources import PythonLaunchDescriptionSource
# from launch.substitutions import LaunchConfiguration
# from launch_ros.actions import ComposableNodeContainer
# from launch_ros.descriptions import ComposableNode

# # Expected number of input messages in 1 second
# INPUT_IMAGES_EXPECT_FREQ = 30
# # Number of input messages to be dropped in 1 second
# # INPUT_IMAGES_DROP_FREQ = 20
# INPUT_IMAGES_DROP_FREQ = 0

# # Camera resolution
# WIDTH, HEIGHT = 640, 480
# # WIDTH, HEIGHT = 848, 480

# # YOLOV8 models expetc 640x640 encoded image size
# YOLOV8_MODEL_INPUT_SIZE = 640

# VISUALIZATION_DOWNSCALING_FACTOR = 10

# MESH_FILE_PATH = '/isaac_ros_assets/isaac_ros_foundationpose/rpb/rpb.obj'
# TEXTURE_PATH = '/isaac_ros_assets/isaac_ros_foundationpose/rpb/baked_mesh_tex0.png'
# REFINE_MODEL_PATH = '/tmp/refine_model.onnx'
# REFINE_ENGINE_PATH = '/isaac_ros_assets/models/foundationpose/refine_trt_engine.plan'
# SCORE_MODEL_PATH = '/tmp/score_model.onnx'
# SCORE_ENGINE_PATH = '/isaac_ros_assets/models/foundationpose/score_trt_engine.plan'
# MODEL_PATH = '/isaac_ros_assets/models/yolov8/best.onnx'
# ENGINE_PATH = '/isaac_ros_assets/models/yolov8/best.plan'

# # REFINE_MODEL_PATH = '/tmp/refine_model.onnx'
# # REFINE_ENGINE_PATH = '/tmp/refine_trt_engine.plan'
# # SCORE_MODEL_PATH = '/tmp/score_model.onnx'
# # SCORE_ENGINE_PATH = '/tmp/score_trt_engine.plan'

# # class IsaacROSFoundationPoseLaunchFragment(IsaacROSLaunchFragment):
# class IsaacROSFoundationPoseLaunchFragment():

#     @staticmethod
#     def get_composable_nodes():

#         # Drop node parameters
#         input_images_expect_freq = LaunchConfiguration('input_images_expect_freq')
#         input_images_drop_freq = LaunchConfiguration('input_images_drop_freq')
        
#         # FoundationPose parameters
#         mesh_file_path = LaunchConfiguration('mesh_file_path')
#         texture_path = LaunchConfiguration('texture_path')
#         refine_model_file_path = LaunchConfiguration('refine_model_file_path')
#         refine_engine_file_path = LaunchConfiguration('refine_engine_file_path')
#         score_model_file_path = LaunchConfiguration('score_model_file_path')
#         score_engine_file_path = LaunchConfiguration('score_engine_file_path')
        
#         # YOLOV8 parameters
#         input_width = WIDTH
#         input_height = HEIGHT
#         input_to_YOLOV8_ratio = input_width / YOLOV8_MODEL_INPUT_SIZE
#         model_file_path = LaunchConfiguration('model_file_path')
#         engine_file_path = LaunchConfiguration('engine_file_path')
#         input_tensor_names = LaunchConfiguration('input_tensor_names')
#         input_binding_names = LaunchConfiguration('input_binding_names')
#         output_tensor_names = LaunchConfiguration('output_tensor_names')
#         output_binding_names = LaunchConfiguration('output_binding_names')
#         verbose = LaunchConfiguration('verbose')
#         force_engine_update = LaunchConfiguration('force_engine_update')
 
#         # YOLOV8 Decoder parameters
#         confidence_threshold = LaunchConfiguration('confidence_threshold')
#         nms_threshold = LaunchConfiguration('nms_threshold')
        
#         return {

#             # Yolo objection detection pipeline
#             'tensor_rt_node': ComposableNode(
#                 name='tensor_rt',
#                 package='isaac_ros_tensor_rt',
#                 plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
#                 parameters=[{
#                     'model_file_path': model_file_path,
#                     'engine_file_path': engine_file_path,
#                     # 'output_binding_names': output_binding_names,
#                     # 'output_tensor_names': output_tensor_names,
#                     # 'input_tensor_names': input_tensor_names,
#                     # 'input_binding_names': input_binding_names,
#                     # 'verbose': verbose,
#                     # 'force_engine_update': force_engine_update
#                 }]
#             ),
            
#             'yolov8_decoder_node': ComposableNode(
#                 name='yolov8_decoder_node',
#                 package='isaac_ros_yolov8',
#                 plugin='nvidia::isaac_ros::yolov8::YoloV8DecoderNode',
#                 parameters=[{
#                     # 'confidence_threshold': confidence_threshold,
#                     # 'nms_threshold': nms_threshold,
#                     'confidence_threshold': 0.6,
#                     'nms_threshold': 0.45,
#                 }],
                
#                 # remappings=[
#                 #     ('detections_output', '/d405/detections_output'),
#                 # ],
                
#                 remappings=[
#                     ('detections_output', '/d435/detections_output'),
#                 ],
#             ),
            
            

#             # Create a binary segmentation mask from a Detection2DArray published by RT-DETR.
#             # The segmentation mask is of size
#             # int(IMAGE_WIDTH/input_to_YOLOV8_ratio) x int(IMAGE_HEIGHT/input_to_YOLOV8_ratio)
#             'detection2_d_to_mask_node': ComposableNode(
#                 name='detection2_d_to_mask',
#                 package='isaac_ros_foundationpose',
#                 plugin='nvidia::isaac_ros::foundationpose::Detection2DToMask',
#                 parameters=[{
#                     'mask_width': int(input_width/input_to_YOLOV8_ratio),
#                     'mask_height': int(input_height/input_to_YOLOV8_ratio)}],
                
#                 # remappings=[('detection2_d_array', '/d405/detections_output'),
#                 #             ('segmentation', '/d405/yolo_segmentation'),
#                 #             ],
                
#                 remappings=[('detection2_d_array', '/d435/detections_output'),
#                             ('segmentation', '/d435/yolo_segmentation'),
#                             ],
#             ),

#             # Resize segmentation mask to ESS model image size so it can be used by FoundationPose
#             # FoundationPose requires depth, rgb image and segmentation mask to be of the same size
#             # Resize from int(IMAGE_WIDTH/input_to_YOLOV8_ratio) x
#             # int(IMAGE_HEIGHT/input_to_YOLOV8_ratio)
#             # to ESS_MODEL_IMAGE_WIDTH x ESS_MODEL_IMAGE_HEIGHT
#             # output height constraint is used since keep_aspect_ratio is False
#             # and the image is padded
#             'resize_mask_node': ComposableNode(
#                 name='resize_mask_node',
#                 package='isaac_ros_image_proc',
#                 plugin='nvidia::isaac_ros::image_proc::ResizeNode',
#                 parameters=[{
#                     'input_width': int(input_width/input_to_YOLOV8_ratio),
#                     'input_height': int(input_height/input_to_YOLOV8_ratio),
#                     'output_width': input_width,
#                     'output_height': input_height,
#                     'keep_aspect_ratio': False,
#                     'disable_padding': False
#                 }],
                
#                 # remappings=[
#                 #     ('image', '/d405/yolo_segmentation'),
#                 #     ('camera_info', '/d405/color/camera_info'),
#                 #     ('resize/image', '/d405/segmentation'),
#                 #     ('resize/camera_info', '/camera_info_segmentation'),
#                 # ],
                
#                 remappings=[
#                     ('image', '/d435/yolo_segmentation'),
#                     ('camera_info', '/d435/color/camera_info'),
#                     ('resize/image', '/d435/segmentation'),
#                     ('resize/camera_info', '/camera_info_segmentation'),
#                 ],
                
#             ),

#             # 'resize_left_viz': ComposableNode(
#             #     name='resize_left_viz',
#             #     package='isaac_ros_image_proc',
#             #     plugin='nvidia::isaac_ros::image_proc::ResizeNode',
#             #     parameters=[{
#             #         'input_width': input_width,
#             #         'input_height': input_height,
#             #         'output_width': int(input_width/VISUALIZATION_DOWNSCALING_FACTOR) * 2,
#             #         'output_height': int(input_height/VISUALIZATION_DOWNSCALING_FACTOR) * 2,
#             #         'keep_aspect_ratio': False,
#             #         'encoding_desired': 'rgb8',
#             #         'disable_padding': False
#             #     }],
                
#             #     # remappings=[
#             #     #     ('image', '/d405/color/image_rect_raw'),
#             #     #     ('camera_info', '/d405/color/camera_info'),
#             #     #     ('resize/image', '/rgb/image_rect_color_viz'),
#             #     #     ('resize/camera_info', '/rgb/camera_info_viz')
#             #     # ],
                
#             #     remappings=[
#             #         ('image', '/rgb/image_rect_color'),
#             #         ('camera_info', '/rgb/camera_info'),
#             #         ('resize/image', '/rgb/image_rect_color_viz'),
#             #         ('resize/camera_info', '/rgb/camera_info_viz')
#             #     ],
#             # ),

#             'foundationpose_node': ComposableNode(
#                 name='foundationpose_node',
#                 package='isaac_ros_foundationpose',
#                 plugin='nvidia::isaac_ros::foundationpose::FoundationPoseNode',
#                 parameters=[{
#                     'mesh_file_path': mesh_file_path,
#                     'texture_path': texture_path,

#                     # 'refine_model_file_path': refine_model_file_path,
#                     'refine_engine_file_path': refine_engine_file_path,
#                     'refine_input_tensor_names': ['input_tensor1', 'input_tensor2'],
#                     'refine_input_binding_names': ['input1', 'input2'],
#                     'refine_output_tensor_names': ['output_tensor1', 'output_tensor2'],
#                     'refine_output_binding_names': ['output1', 'output2'],

#                     # 'score_model_file_path': score_model_file_path,
#                     'score_engine_file_path': score_engine_file_path,
#                     'score_input_tensor_names': ['input_tensor1', 'input_tensor2'],
#                     'score_input_binding_names': ['input1', 'input2'],
#                     'score_output_tensor_names': ['output_tensor'],
#                     'score_output_binding_names': ['output1'],
#                 }],
                
#                 # remappings=[
#                 #     ('pose_estimation/depth_image', '/depth_image'),
#                 #     ('pose_estimation/image', '/rgb/image_rect_color'),
#                 #     ('pose_estimation/camera_info', '/rgb/camera_info'),
#                 #     ('pose_estimation/segmentation', '/d405/segmentation'),
#                 #     ('pose_estimation/output', '/output')
#                 # ],
                
#                 # remappings=[
#                 #     ('pose_estimation/depth_image', '/depth_image'),
#                 #     ('pose_estimation/image', '/rgb/image_rect_color'),
#                 #     ('pose_estimation/camera_info', '/rgb/camera_info'),
#                 #     ('pose_estimation/segmentation', '/d435/segmentation'),
#                 #     ('pose_estimation/output', '/output')
#                 # ],
                
                
                
#                 # remappings=[
#                 #     ('pose_estimation/depth_image', '/d405/depth_image_fpe'),
#                 #     ('pose_estimation/image', '/d405/color_image_fpe'),
#                 #     ('pose_estimation/camera_info', '/d405/camera_info_fpe'),
#                 #     ('pose_estimation/segmentation', '/d405/segmentation'),
#                 #     ('pose_estimation/output', '/output')
#                 # ],
                
#                 remappings=[
#                     ('pose_estimation/depth_image', '/d435/depth_image_fpe'),
#                     ('pose_estimation/image', '/d435/color_image_fpe'),
#                     ('pose_estimation/camera_info', '/d435/camera_info_fpe'),
#                     ('pose_estimation/segmentation', '/d435/segmentation'),
#                     ('pose_estimation/output', '/output')
#                 ],
                
                
#             ),

            
#         }

#     @staticmethod
#     def get_launch_actions():
        
#         network_image_width = LaunchConfiguration('network_image_width')
#         network_image_height = LaunchConfiguration('network_image_height')
#         image_mean = LaunchConfiguration('image_mean')
#         image_stddev = LaunchConfiguration('image_stddev')

#         encoder_dir = get_package_share_directory('isaac_ros_dnn_image_encoder')

#         return {
#             'input_images_expect_freq': DeclareLaunchArgument(
#                 'input_images_expect_freq',
#                 default_value=str(INPUT_IMAGES_EXPECT_FREQ),
#                 description='Expected number of input messages in 1 second'),

#             'input_images_drop_freq': DeclareLaunchArgument(
#                 'input_images_drop_freq',
#                 default_value=str(INPUT_IMAGES_DROP_FREQ),
#                 description='Number of input messages to be dropped in 1 second'),

#             'mesh_file_path': DeclareLaunchArgument(
#                 'mesh_file_path',
#                 default_value=MESH_FILE_PATH,
#                 description='The absolute file path to the mesh file'),

#             'texture_path': DeclareLaunchArgument(
#                 'texture_path',
#                 default_value=TEXTURE_PATH,
#                 description='The absolute file path to the texture map'),

#             'refine_model_file_path': DeclareLaunchArgument(
#                 'refine_model_file_path',
#                 default_value=REFINE_MODEL_PATH,
#                 description='The absolute file path to the refine model'),

#             'refine_engine_file_path': DeclareLaunchArgument(
#                 'refine_engine_file_path',
#                 default_value=REFINE_ENGINE_PATH,
#                 description='The absolute file path to the refine trt engine'),

#             'score_model_file_path': DeclareLaunchArgument(
#                 'score_model_file_path',
#                 default_value=SCORE_MODEL_PATH,
#                 description='The absolute file path to the score model'),

#             'score_engine_file_path': DeclareLaunchArgument(
#                 'score_engine_file_path',
#                 default_value=SCORE_ENGINE_PATH,
#                 description='The absolute file path to the score trt engine'),

#             'network_image_width': DeclareLaunchArgument(
#                 'network_image_width',
#                 default_value='640',
#                 description='The input image width that the network expects'
#             ),
#             'network_image_height': DeclareLaunchArgument(
#                 'network_image_height',
#                 default_value='640',
#                 description='The input image height that the network expects'
#             ),
#             'image_mean': DeclareLaunchArgument(
#                 'image_mean',
#                 default_value='[0.0, 0.0, 0.0]',
#                 description='The mean for image normalization'
#             ),
#             'image_stddev': DeclareLaunchArgument(
#                 'image_stddev',
#                 default_value='[1.0, 1.0, 1.0]',
#                 description='The standard deviation for image normalization'
#             ),

#             'model_file_path': DeclareLaunchArgument(
#                 'model_file_path',
#                  default_value=MODEL_PATH,
#                 description='The absolute file path to the ONNX file'
#             ),
#             'engine_file_path': DeclareLaunchArgument(
#                 'engine_file_path',
#                  default_value=ENGINE_PATH,
#                 description='The absolute file path to the TensorRT engine file'
#             ),
#             'input_tensor_names': DeclareLaunchArgument(
#                 'input_tensor_names',
#                 default_value='["input_tensor"]',
#                 description='A list of tensor names to bound to the specified input binding names'
#             ),
#             'input_binding_names': DeclareLaunchArgument(
#                 'input_binding_names',
#                 default_value='["images"]',
#                 description='A list of input tensor binding names (specified by model)'
#             ),
#             'output_tensor_names': DeclareLaunchArgument(
#                 'output_tensor_names',
#                 default_value='["output_tensor"]',
#                 description='A list of tensor names to bound to the specified output binding names'
#             ),
#             'output_binding_names': DeclareLaunchArgument(
#                 'output_binding_names',
#                 default_value='["output0"]',
#                 description='A list of output tensor binding names (specified by model)'
#             ),
#             'verbose': DeclareLaunchArgument(
#                 'verbose',
#                 default_value='False',
#                 description='Whether TensorRT should verbosely log or not'
#             ),
#             'force_engine_update': DeclareLaunchArgument(
#                 'force_engine_update',
#                 default_value='False',
#                 description='Whether TensorRT should update the TensorRT engine file or not'
#             ),
#             'confidence_threshold': DeclareLaunchArgument(
#                 'confidence_threshold',
#                 default_value='0.6',
#                 description='Confidence threshold to filter candidate detections during NMS'
#             ),
#             'nms_threshold': DeclareLaunchArgument(
#                 'nms_threshold',
#                 default_value='0.45', # 0.45, 0.8
#                 description='NMS IOU threshold'
#             ),

#             'yolov8_encoder_launch': IncludeLaunchDescription(
#                 PythonLaunchDescriptionSource(
#                     [os.path.join(encoder_dir, 'launch', 'dnn_image_encoder.launch.py')]
#                 ),
#                 launch_arguments={
#                     # 'input_image_width': str(interface_specs['camera_resolution']['width']),
#                     # 'input_image_height': str(interface_specs['camera_resolution']['height']),
#                     'input_image_width': str(WIDTH),
#                     'input_image_height': str(HEIGHT),
#                     'network_image_width': network_image_width,
#                     'network_image_height': network_image_height,
#                     'image_mean': image_mean,
#                     'image_stddev': image_stddev,
#                     'attach_to_shared_component_container': 'True',
#                     'component_container_name': '/isaac_ros_examples/container',
#                     'dnn_image_encoder_namespace': 'yolov8_encoder',
#                     # 'image_input_topic': '/d405/color/image_rect_raw',
#                     # 'camera_info_input_topic': '/d405/color/camera_info',
#                     'image_input_topic': '/d435/color/image_raw',
#                     'camera_info_input_topic': '/d435/color/camera_info',
#                     'tensor_output_topic': '/tensor_pub',
#                 }.items(),
#             ),

            
#         }
    
# # def generate_launch_description():
# #     foundationpose_container = ComposableNodeContainer(
# #         package='rclcpp_components',
# #         name='foundationpose_container',
# #         namespace='',
# #         executable='component_container_mt',
# #         composable_node_descriptions=[
# #             *IsaacROSFoundationPoseLaunchFragment.get_composable_nodes().values(),
# #         ],
# #         output='screen'
# #     )

# #     return launch.LaunchDescription(
# #         [foundationpose_container] +
# #         IsaacROSFoundationPoseLaunchFragment.get_launch_actions().values()
# #     )

# def generate_launch_description():
#     foundationpose_container = ComposableNodeContainer(
#         package='rclcpp_components',
#         name='foundationpose_container',
#         namespace='',
#         executable='component_container_mt',
#         composable_node_descriptions=[
#             *IsaacROSFoundationPoseLaunchFragment.get_composable_nodes().values(),
#         ],
#         output='screen'
#     )

#     return launch.LaunchDescription(
#         [foundationpose_container] +
#         list(IsaacROSFoundationPoseLaunchFragment.get_launch_actions().values())  # Convert dict_values to list
#     )

























# ################### YOLOV8 (still problematic)
# # SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# # Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# # http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# #
# # SPDX-License-Identifier: Apache-2.0

# import os

# from ament_index_python.packages import get_package_share_directory
# import launch
# from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
# from launch.conditions import IfCondition, UnlessCondition
# from launch.launch_description_sources import PythonLaunchDescriptionSource
# from launch.substitutions import LaunchConfiguration
# from launch_ros.actions import ComposableNodeContainer, Node
# from launch_ros.descriptions import ComposableNode

# MESH_FILE_NAME = '/tmp/textured_simple.obj'
# TEXTURE_MAP_NAME = '/tmp/texture_map.png'
# REFINE_MODEL_NAME = '/tmp/refine_model.onnx'
# REFINE_ENGINE_NAME = '/tmp/refine_trt_engine.plan'
# SCORE_MODEL_NAME = '/tmp/score_model.onnx'
# SCORE_ENGINE_NAME = '/tmp/score_trt_engine.plan'
# MODEL_PATH = '/isaac_ros_assets/models/yolov8/best.onnx'
# ENGINE_PATH = '/isaac_ros_assets/models/yolov8/best.plan'

# # Camera resolution
# WIDTH, HEIGHT = 640, 480
# # WIDTH, HEIGHT = 848, 480

# # YOLOV8 models expetc 640x640 encoded image size
# YOLOV8_MODEL_INPUT_SIZE = 640

# def generate_launch_description():
#     """Generate launch description for testing relevant nodes."""
#     rviz_config_path = os.path.join(
#         get_package_share_directory('isaac_ros_foundationpose'),
#         'rviz', 'foundationpose.rviz')
    
#     # FoundationPose parameters
#     mesh_file_path = LaunchConfiguration('mesh_file_path')
#     texture_path = LaunchConfiguration('texture_path')
#     refine_model_file_path = LaunchConfiguration('refine_model_file_path')
#     refine_engine_file_path = LaunchConfiguration('refine_engine_file_path')
#     score_model_file_path = LaunchConfiguration('score_model_file_path')
#     score_engine_file_path = LaunchConfiguration('score_engine_file_path')
#     mask_height = LaunchConfiguration('mask_height')
#     mask_width = LaunchConfiguration('mask_width')
#     launch_rviz = LaunchConfiguration('launch_rviz')
#     launch_bbox_to_mask = LaunchConfiguration('launch_bbox_to_mask')
    
#     # YOLOV8 parameters
#     input_width = WIDTH
#     input_height = HEIGHT
#     input_to_YOLOV8_ratio = input_width / YOLOV8_MODEL_INPUT_SIZE
    
#     network_image_width = LaunchConfiguration('network_image_width')
#     network_image_height = LaunchConfiguration('network_image_height')
#     image_mean = LaunchConfiguration('image_mean')
#     image_stddev = LaunchConfiguration('image_stddev')

#     encoder_dir = get_package_share_directory('isaac_ros_dnn_image_encoder')
    

#     launch_args = [
#         DeclareLaunchArgument(
#             'mesh_file_path',
#             default_value=MESH_FILE_NAME,
#             description='The absolute file path to the mesh file'),

#         DeclareLaunchArgument(
#             'texture_path',
#             default_value=TEXTURE_MAP_NAME,
#             description='The absolute file path to the texture map'),

#         DeclareLaunchArgument(
#             'refine_model_file_path',
#             default_value=REFINE_MODEL_NAME,
#             description='The absolute file path to the refine model'),

#         DeclareLaunchArgument(
#             'refine_engine_file_path',
#             default_value=REFINE_ENGINE_NAME,
#             description='The absolute file path to the refine trt engine'),

#         DeclareLaunchArgument(
#             'score_model_file_path',
#             default_value=SCORE_MODEL_NAME,
#             description='The absolute file path to the score model'),

#         DeclareLaunchArgument(
#             'score_engine_file_path',
#             default_value=SCORE_ENGINE_NAME,
#             description='The absolute file path to the score trt engine'),

#         DeclareLaunchArgument(
#             'mask_height',
#             default_value='480',
#             description='The height of the mask generated from the bounding box'),

#         DeclareLaunchArgument(
#             'mask_width',
#             default_value='640',
#             description='The width of the mask generated from the bounding box'),

#         DeclareLaunchArgument(
#             'launch_bbox_to_mask',
#             default_value='True',
#             description='Flag to enable bounding box to mask converter'),
        
#         DeclareLaunchArgument(
#             'network_image_width',
#             default_value='640',
#             description='The input image width that the network expects'
#         ),
        
#         DeclareLaunchArgument(
#             'network_image_height',
#             default_value='640',
#             description='The input image height that the network expects'
#         ),
        
#         DeclareLaunchArgument(
#             'image_mean',
#             default_value='[0.0, 0.0, 0.0]',
#             description='The mean for image normalization'
#         ),
        
#         DeclareLaunchArgument(
#             'image_stddev',
#             default_value='[1.0, 1.0, 1.0]',
#             description='The standard deviation for image normalization'
#         ),
        
#         DeclareLaunchArgument(
#             'model_file_path',
#                 default_value=MODEL_PATH,
#             description='The absolute file path to the ONNX file'
#         ),
        
#         DeclareLaunchArgument(
#             'engine_file_path',
#                 default_value=ENGINE_PATH,
#             description='The absolute file path to the TensorRT engine file'
#         ),
#         DeclareLaunchArgument(
#             'input_tensor_names',
#             default_value='["input_tensor"]',
#             description='A list of tensor names to bound to the specified input binding names'
#         ),
        
#         DeclareLaunchArgument(
#             'input_binding_names',
#             default_value='["images"]',
#             description='A list of input tensor binding names (specified by model)'
#         ),
        
#         DeclareLaunchArgument(
#             'output_tensor_names',
#             default_value='["output_tensor"]',
#             description='A list of tensor names to bound to the specified output binding names'
#         ),

#         DeclareLaunchArgument(
#             'launch_rviz',
#             default_value='True',
#             description='Flag to enable Rviz2 launch'),
        
#         DeclareLaunchArgument(
#             'output_binding_names',
#             default_value='["output0"]',
#             description='A list of output tensor binding names (specified by model)'
#         ),
        
#         DeclareLaunchArgument(
#             'verbose',
#             default_value='False',
#             description='Whether TensorRT should verbosely log or not'
#         ),
        
#         DeclareLaunchArgument(
#             'force_engine_update',
#             default_value='False',
#             description='Whether TensorRT should update the TensorRT engine file or not'
#         ),
        
#         DeclareLaunchArgument(
#             'confidence_threshold',
#             default_value='0.6',
#             description='Confidence threshold to filter candidate detections during NMS'
#         ),
        
#         DeclareLaunchArgument(
#             'nms_threshold',
#             default_value='0.45', # 0.45, 0.8
#             description='NMS IOU threshold'
#         ),
        
#         IncludeLaunchDescription(
#         PythonLaunchDescriptionSource(
#             [os.path.join(encoder_dir, 'launch', 'dnn_image_encoder.launch.py')]
#         ),
#         launch_arguments={
#             'input_image_width': str(WIDTH),
#             'input_image_height': str(HEIGHT),
#             'network_image_width': network_image_width,
#             'network_image_height': network_image_height,
#             'image_mean': image_mean,
#             'image_stddev': image_stddev,
#             'attach_to_shared_component_container': 'True',
#             'component_container_name': '/isaac_ros_examples/container',
#             'dnn_image_encoder_namespace': 'yolov8_encoder',
#             # 'image_input_topic': '/d405/color/image_rect_raw',
#             # 'camera_info_input_topic': '/d405/color/camera_info',
#             'image_input_topic': '/d435/color/image_raw',
#             'camera_info_input_topic': '/d435/color/camera_info',
#             'tensor_output_topic': '/tensor_pub',
#         }.items(),
#     ),

#     ]

    
    

#     detection2_d_to_mask_node = ComposableNode(
#         name='detection2_d_to_mask',
#         package='isaac_ros_foundationpose',
#         plugin='nvidia::isaac_ros::foundationpose::Detection2DToMask',
#         parameters=[{
#             'mask_width': mask_width,
#             'mask_height': mask_height
#         }],
#         remappings=[('detection2_d_array', '/d435/detections_output'),
#                     ('segmentation', '/d435/yolo_segmentation'),
#         ],
#     )
    
#     resize_mask_node = ComposableNode(
#         name='resize_mask_node',
#         package='isaac_ros_image_proc',
#         plugin='nvidia::isaac_ros::image_proc::ResizeNode',
#         parameters=[{
#             'input_width': int(input_width/input_to_YOLOV8_ratio),
#             'input_height': int(input_height/input_to_YOLOV8_ratio),
#             'output_width': input_width,
#             'output_height': input_height,
#             'keep_aspect_ratio': False,
#             'disable_padding': False
#         }],
        
#         # remappings=[
#         #     ('image', '/d405/yolo_segmentation'),
#         #     ('camera_info', '/d405/color/camera_info'),
#         #     ('resize/image', '/d405/segmentation'),
#         #     ('resize/camera_info', '/camera_info_segmentation'),
#         # ],
        
#         remappings=[
#             ('image', '/d435/yolo_segmentation'),
#             ('camera_info', '/d435/color/camera_info'),
#             ('resize/image', '/d435/segmentation'),
#             ('resize/camera_info', '/camera_info_segmentation'),
#         ],
        
#     )
    
    
#     foundationpose_node = ComposableNode(
#         name='foundationpose',
#         package='isaac_ros_foundationpose',
#         plugin='nvidia::isaac_ros::foundationpose::FoundationPoseNode',
#         parameters=[{
#             'mesh_file_path': mesh_file_path,
#             'texture_path': texture_path,

#             'refine_model_file_path': refine_model_file_path,
#             'refine_engine_file_path': refine_engine_file_path,
#             'refine_input_tensor_names': ['input_tensor1', 'input_tensor2'],
#             'refine_input_binding_names': ['input1', 'input2'],
#             'refine_output_tensor_names': ['output_tensor1', 'output_tensor2'],
#             'refine_output_binding_names': ['output1', 'output2'],

#             'score_model_file_path': score_model_file_path,
#             'score_engine_file_path': score_engine_file_path,
#             'score_input_tensor_names': ['input_tensor1', 'input_tensor2'],
#             'score_input_binding_names': ['input1', 'input2'],
#             'score_output_tensor_names': ['output_tensor'],
#             'score_output_binding_names': ['output1'],
#         }],
        
#         # remappings=[
#         #     ('pose_estimation/depth_image', 'depth_registered/image_rect'),
#         #     ('pose_estimation/image', 'rgb/image_rect_color'),
#         #     ('pose_estimation/camera_info', 'rgb/camera_info'),
#         #     ('pose_estimation/segmentation', 'segmentation'),
#         #     ('pose_estimation/output', 'output')
#         # ],
        
#         remappings=[
#             ('pose_estimation/depth_image', '/d435/depth_image_fpe'),
#             ('pose_estimation/image', '/d435/color_image_fpe'),
#             ('pose_estimation/camera_info', '/d435/camera_info_fpe'),
#             ('pose_estimation/segmentation', '/d435/segmentation'),
#             ('pose_estimation/output', '/output')
#         ],
    
#     )
    
    
#     rviz_node = Node(
#         package='rviz2',
#         executable='rviz2',
#         name='rviz2',
#         arguments=['-d', rviz_config_path],
#         condition=IfCondition(launch_rviz)
#     )

#     foundationpose_bbox_container = ComposableNodeContainer(
#         name='foundationpose_container',
#         namespace='foundationpose_container',
#         package='rclcpp_components',
#         executable='component_container_mt',
#         composable_node_descriptions=[
#             detection2_d_to_mask_node,
#             resize_mask_node,
#             foundationpose_node,
#             ],
#         output='screen',
#         condition=IfCondition(launch_bbox_to_mask)
#     )

#     foundationpose_container = ComposableNodeContainer(
#         name='foundationpose_container',
#         namespace='foundationpose_container',
#         package='rclcpp_components',
#         executable='component_container_mt',
#         composable_node_descriptions=[foundationpose_node],
#         output='screen',
#         condition=UnlessCondition(launch_bbox_to_mask)
#     )

#     return launch.LaunchDescription(
#         launch_args + 
#         [foundationpose_container,
#         foundationpose_bbox_container,
#         rviz_node]
#     )

























































################## No YOLOV8 (runs in other launch file)
# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import os

from ament_index_python.packages import get_package_share_directory
import launch
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode

MESH_FILE_NAME = '/tmp/textured_simple.obj'
TEXTURE_MAP_NAME = '/tmp/texture_map.png'
REFINE_MODEL_NAME = '/tmp/refine_model.onnx'
REFINE_ENGINE_NAME = '/tmp/refine_trt_engine.plan'
SCORE_MODEL_NAME = '/tmp/score_model.onnx'
SCORE_ENGINE_NAME = '/tmp/score_trt_engine.plan'

# Camera resolution
WIDTH, HEIGHT = 640, 480
# WIDTH, HEIGHT = 848, 480

# YOLOV8 models expetc 640x640 encoded image size
YOLOV8_MODEL_INPUT_SIZE = 640

def generate_launch_description():
    """Generate launch description for testing relevant nodes."""
    rviz_config_path = os.path.join(
        get_package_share_directory('isaac_ros_foundationpose'),
        'rviz', 'foundationpose.rviz')

    launch_args = [
        DeclareLaunchArgument(
            'mesh_file_path',
            default_value=MESH_FILE_NAME,
            description='The absolute file path to the mesh file'),

        DeclareLaunchArgument(
            'texture_path',
            default_value=TEXTURE_MAP_NAME,
            description='The absolute file path to the texture map'),

        DeclareLaunchArgument(
            'refine_model_file_path',
            default_value=REFINE_MODEL_NAME,
            description='The absolute file path to the refine model'),

        DeclareLaunchArgument(
            'refine_engine_file_path',
            default_value=REFINE_ENGINE_NAME,
            description='The absolute file path to the refine trt engine'),

        DeclareLaunchArgument(
            'score_model_file_path',
            default_value=SCORE_MODEL_NAME,
            description='The absolute file path to the score model'),

        DeclareLaunchArgument(
            'score_engine_file_path',
            default_value=SCORE_ENGINE_NAME,
            description='The absolute file path to the score trt engine'),

        DeclareLaunchArgument(
            'mask_height',
            default_value='480',
            description='The height of the mask generated from the bounding box'),

        DeclareLaunchArgument(
            'mask_width',
            default_value='640',
            description='The width of the mask generated from the bounding box'),

        DeclareLaunchArgument(
            'launch_bbox_to_mask',
            default_value='True',
            description='Flag to enable bounding box to mask converter'),

        DeclareLaunchArgument(
            'launch_rviz',
            default_value='True',
            description='Flag to enable Rviz2 launch'),

    ]

    # FoundationPose parameters
    mesh_file_path = LaunchConfiguration('mesh_file_path')
    texture_path = LaunchConfiguration('texture_path')
    refine_model_file_path = LaunchConfiguration('refine_model_file_path')
    refine_engine_file_path = LaunchConfiguration('refine_engine_file_path')
    score_model_file_path = LaunchConfiguration('score_model_file_path')
    score_engine_file_path = LaunchConfiguration('score_engine_file_path')
    mask_height = LaunchConfiguration('mask_height')
    mask_width = LaunchConfiguration('mask_width')
    launch_rviz = LaunchConfiguration('launch_rviz')
    launch_bbox_to_mask = LaunchConfiguration('launch_bbox_to_mask')
    
    # YOLOV8 parameters
    input_width = WIDTH
    input_height = HEIGHT
    input_to_YOLOV8_ratio = input_width / YOLOV8_MODEL_INPUT_SIZE
    

    detection2_d_to_mask_node = ComposableNode(
        name='detection2_d_to_mask',
        package='isaac_ros_foundationpose',
        plugin='nvidia::isaac_ros::foundationpose::Detection2DToMask',
        parameters=[{
            'mask_width': mask_width,
            'mask_height': mask_height
        }],
        remappings=[('detection2_d_array', '/d435/detections_output'),
                    ('segmentation', '/d435/yolo_segmentation'),
        ],
    )
    
    resize_mask_node = ComposableNode(
        name='resize_mask_node',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        parameters=[{
            'input_width': int(input_width/input_to_YOLOV8_ratio),
            'input_height': int(input_height/input_to_YOLOV8_ratio),
            'output_width': input_width,
            'output_height': input_height,
            'keep_aspect_ratio': False,
            'disable_padding': False
        }],
        
        # remappings=[
        #     ('image', '/d405/yolo_segmentation'),
        #     ('camera_info', '/d405/color/camera_info'),
        #     ('resize/image', '/d405/segmentation'),
        #     ('resize/camera_info', '/camera_info_segmentation'),
        # ],
        
        remappings=[
            ('image', '/d435/yolo_segmentation'),
            ('camera_info', '/d435/color/camera_info'),
            ('resize/image', '/d435/segmentation'),
            ('resize/camera_info', '/camera_info_segmentation'),
        ],
        
    )
    
    
    foundationpose_node = ComposableNode(
        name='foundationpose',
        package='isaac_ros_foundationpose',
        plugin='nvidia::isaac_ros::foundationpose::FoundationPoseNode',
        parameters=[{
            'mesh_file_path': mesh_file_path,
            'texture_path': texture_path,

            'refine_model_file_path': refine_model_file_path,
            'refine_engine_file_path': refine_engine_file_path,
            'refine_input_tensor_names': ['input_tensor1', 'input_tensor2'],
            'refine_input_binding_names': ['input1', 'input2'],
            'refine_output_tensor_names': ['output_tensor1', 'output_tensor2'],
            'refine_output_binding_names': ['output1', 'output2'],

            'score_model_file_path': score_model_file_path,
            'score_engine_file_path': score_engine_file_path,
            'score_input_tensor_names': ['input_tensor1', 'input_tensor2'],
            'score_input_binding_names': ['input1', 'input2'],
            'score_output_tensor_names': ['output_tensor'],
            'score_output_binding_names': ['output1'],
        }],
        
        # remappings=[
        #     ('pose_estimation/depth_image', 'depth_registered/image_rect'),
        #     ('pose_estimation/image', 'rgb/image_rect_color'),
        #     ('pose_estimation/camera_info', 'rgb/camera_info'),
        #     ('pose_estimation/segmentation', 'segmentation'),
        #     ('pose_estimation/output', 'output')
        # ],
        
        remappings=[
            ('pose_estimation/depth_image', '/d435/depth_image_fpe'),
            ('pose_estimation/image', '/d435/color_image_fpe'),
            ('pose_estimation/camera_info', '/d435/camera_info_fpe'),
            ('pose_estimation/segmentation', '/d435/segmentation'),
            ('pose_estimation/output', '/output')
        ],
    
    )
    
    
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_path],
        condition=IfCondition(launch_rviz)
    )

    foundationpose_bbox_container = ComposableNodeContainer(
        name='foundationpose_container',
        namespace='foundationpose_container',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[
            detection2_d_to_mask_node,
            resize_mask_node,
            foundationpose_node,
            ],
        output='screen',
        condition=IfCondition(launch_bbox_to_mask)
    )

    foundationpose_container = ComposableNodeContainer(
        name='foundationpose_container',
        namespace='foundationpose_container',
        package='rclcpp_components',
        executable='component_container_mt',
        composable_node_descriptions=[foundationpose_node],
        output='screen',
        condition=UnlessCondition(launch_bbox_to_mask)
    )

    return launch.LaunchDescription(launch_args + [foundationpose_container,
                                                   foundationpose_bbox_container,
                                                   rviz_node])














































#################### Original one
# # SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# # Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# # http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# #
# # SPDX-License-Identifier: Apache-2.0

# import os

# from ament_index_python.packages import get_package_share_directory
# import launch
# from launch.actions import DeclareLaunchArgument
# from launch.conditions import IfCondition, UnlessCondition
# from launch.substitutions import LaunchConfiguration
# from launch_ros.actions import ComposableNodeContainer, Node
# from launch_ros.descriptions import ComposableNode

# MESH_FILE_NAME = '/tmp/textured_simple.obj'
# TEXTURE_MAP_NAME = '/tmp/texture_map.png'
# REFINE_MODEL_NAME = '/tmp/refine_model.onnx'
# REFINE_ENGINE_NAME = '/tmp/refine_trt_engine.plan'
# SCORE_MODEL_NAME = '/tmp/score_model.onnx'
# SCORE_ENGINE_NAME = '/tmp/score_trt_engine.plan'


# def generate_launch_description():
#     """Generate launch description for testing relevant nodes."""
#     rviz_config_path = os.path.join(
#         get_package_share_directory('isaac_ros_foundationpose'),
#         'rviz', 'foundationpose.rviz')

#     launch_args = [
#         DeclareLaunchArgument(
#             'mesh_file_path',
#             default_value=MESH_FILE_NAME,
#             description='The absolute file path to the mesh file'),

#         DeclareLaunchArgument(
#             'texture_path',
#             default_value=TEXTURE_MAP_NAME,
#             description='The absolute file path to the texture map'),

#         DeclareLaunchArgument(
#             'refine_model_file_path',
#             default_value=REFINE_MODEL_NAME,
#             description='The absolute file path to the refine model'),

#         DeclareLaunchArgument(
#             'refine_engine_file_path',
#             default_value=REFINE_ENGINE_NAME,
#             description='The absolute file path to the refine trt engine'),

#         DeclareLaunchArgument(
#             'score_model_file_path',
#             default_value=SCORE_MODEL_NAME,
#             description='The absolute file path to the score model'),

#         DeclareLaunchArgument(
#             'score_engine_file_path',
#             default_value=SCORE_ENGINE_NAME,
#             description='The absolute file path to the score trt engine'),

#         DeclareLaunchArgument(
#             'mask_height',
#             default_value='480',
#             description='The height of the mask generated from the bounding box'),

#         DeclareLaunchArgument(
#             'mask_width',
#             default_value='640',
#             description='The width of the mask generated from the bounding box'),

#         DeclareLaunchArgument(
#             'launch_bbox_to_mask',
#             default_value='False',
#             description='Flag to enable bounding box to mask converter'),

#         DeclareLaunchArgument(
#             'launch_rviz',
#             default_value='False',
#             description='Flag to enable Rviz2 launch'),

#     ]

#     mesh_file_path = LaunchConfiguration('mesh_file_path')
#     texture_path = LaunchConfiguration('texture_path')
#     refine_model_file_path = LaunchConfiguration('refine_model_file_path')
#     refine_engine_file_path = LaunchConfiguration('refine_engine_file_path')
#     score_model_file_path = LaunchConfiguration('score_model_file_path')
#     score_engine_file_path = LaunchConfiguration('score_engine_file_path')
#     mask_height = LaunchConfiguration('mask_height')
#     mask_width = LaunchConfiguration('mask_width')
#     launch_rviz = LaunchConfiguration('launch_rviz')
#     launch_bbox_to_mask = LaunchConfiguration('launch_bbox_to_mask')

#     foundationpose_node = ComposableNode(
#         name='foundationpose',
#         package='isaac_ros_foundationpose',
#         plugin='nvidia::isaac_ros::foundationpose::FoundationPoseNode',
#         parameters=[{
#             'mesh_file_path': mesh_file_path,
#             'texture_path': texture_path,

#             'refine_model_file_path': refine_model_file_path,
#             'refine_engine_file_path': refine_engine_file_path,
#             'refine_input_tensor_names': ['input_tensor1', 'input_tensor2'],
#             'refine_input_binding_names': ['input1', 'input2'],
#             'refine_output_tensor_names': ['output_tensor1', 'output_tensor2'],
#             'refine_output_binding_names': ['output1', 'output2'],

#             'score_model_file_path': score_model_file_path,
#             'score_engine_file_path': score_engine_file_path,
#             'score_input_tensor_names': ['input_tensor1', 'input_tensor2'],
#             'score_input_binding_names': ['input1', 'input2'],
#             'score_output_tensor_names': ['output_tensor'],
#             'score_output_binding_names': ['output1'],
#         }],
#         remappings=[
#             ('pose_estimation/depth_image', 'depth_registered/image_rect'),
#             ('pose_estimation/image', 'rgb/image_rect_color'),
#             ('pose_estimation/camera_info', 'rgb/camera_info'),
#             ('pose_estimation/segmentation', 'segmentation'),
#             ('pose_estimation/output', 'output')])

#     rviz_node = Node(
#         package='rviz2',
#         executable='rviz2',
#         name='rviz2',
#         arguments=['-d', rviz_config_path],
#         condition=IfCondition(launch_rviz))

#     detection2_d_to_mask_node = ComposableNode(
#         name='detection2_d_to_mask',
#         package='isaac_ros_foundationpose',
#         plugin='nvidia::isaac_ros::foundationpose::Detection2DToMask',
#         parameters=[{
#             'mask_width': mask_width,
#             'mask_height': mask_height
#         }])

#     foundationpose_bbox_container = ComposableNodeContainer(
#         name='foundationpose_container',
#         namespace='foundationpose_container',
#         package='rclcpp_components',
#         executable='component_container_mt',
#         composable_node_descriptions=[
#             foundationpose_node,
#             detection2_d_to_mask_node],
#         output='screen',
#         condition=IfCondition(launch_bbox_to_mask)
#     )

#     foundationpose_container = ComposableNodeContainer(
#         name='foundationpose_container',
#         namespace='foundationpose_container',
#         package='rclcpp_components',
#         executable='component_container_mt',
#         composable_node_descriptions=[foundationpose_node],
#         output='screen',
#         condition=UnlessCondition(launch_bbox_to_mask)
#     )

#     return launch.LaunchDescription(launch_args + [foundationpose_container,
#                                                    foundationpose_bbox_container,
#                                                    rviz_node])
