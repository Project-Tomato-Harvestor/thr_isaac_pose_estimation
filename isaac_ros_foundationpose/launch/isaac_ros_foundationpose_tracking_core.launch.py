#################### YOLOV8
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
from typing import Any, Dict
from ament_index_python.packages import get_package_share_directory
from isaac_ros_examples import IsaacROSLaunchFragment
import launch
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode

# Camera resolution
WIDTH, HEIGHT = 640, 480
# WIDTH, HEIGHT = 848, 480

# YOLOV8 models expetc 640x640 encoded image size
YOLOV8_MODEL_INPUT_SIZE = 640

# REFINE_MODEL_PATH = '/tmp/refine_model.onnx'
# REFINE_ENGINE_PATH = '/tmp/refine_trt_engine.plan'
# SCORE_MODEL_PATH = '/tmp/score_model.onnx'
# SCORE_ENGINE_PATH = '/tmp/score_trt_engine.plan'

MESH_FILE_PATH = '/workspaces/isaac_ros-dev/isaac_ros_assets/isaac_ros_foundationpose/rpb/rpb.obj'
TEXTURE_PATH = '/workspaces/isaac_ros-dev/isaac_ros_assets/isaac_ros_foundationpose/rpb/baked_mesh_tex0.png'
REFINE_MODEL_PATH = '/tmp/refine_model.onnx'
REFINE_ENGINE_PATH = '/isaac_ros_assets/models/foundationpose/refine_trt_engine.plan'
SCORE_MODEL_PATH = '/tmp/score_model.onnx'
SCORE_ENGINE_PATH = '/workspaces/isaac_ros-dev/isaac_ros_assets/models/foundationpose/score_trt_engine.plan'
MODEL_PATH = '/workspaces/isaac_ros-dev/isaac_ros_assets/models/yolov8/best.onnx'
ENGINE_PATH = '/workspaces/isaac_ros-dev/isaac_ros_assets/models/yolov8/best.plan'


class IsaacROSFoundationPoseTrackingLaunchFragment(IsaacROSLaunchFragment):

    @staticmethod
    def get_composable_nodes(interface_specs: Dict[str, Any]) -> Dict[str, ComposableNode]:
        
        # FoundationPose parameters
        mesh_file_path = LaunchConfiguration('mesh_file_path')
        texture_path = LaunchConfiguration('texture_path')
        refine_model_file_path = LaunchConfiguration('refine_model_file_path')
        refine_engine_file_path = LaunchConfiguration('refine_engine_file_path')
        score_model_file_path = LaunchConfiguration('score_model_file_path')
        score_engine_file_path = LaunchConfiguration('score_engine_file_path')
        
        # YOLOV8 parameters
        input_width = interface_specs['camera_resolution']['width']
        input_height = interface_specs['camera_resolution']['height']
        # input_width = WIDTH
        # input_height = HEIGHT
        input_to_YOLOV8_ratio = input_width / YOLOV8_MODEL_INPUT_SIZE
        model_file_path = LaunchConfiguration('model_file_path')
        engine_file_path = LaunchConfiguration('engine_file_path')
        input_tensor_names = LaunchConfiguration('input_tensor_names')
        input_binding_names = LaunchConfiguration('input_binding_names')
        output_tensor_names = LaunchConfiguration('output_tensor_names')
        output_binding_names = LaunchConfiguration('output_binding_names')
        verbose = LaunchConfiguration('verbose')
        force_engine_update = LaunchConfiguration('force_engine_update')
 
        # YOLOV8 Decoder parameters
        confidence_threshold = LaunchConfiguration('confidence_threshold')
        nms_threshold = LaunchConfiguration('nms_threshold')
        
        return {
            # Yolo objection detection pipeline
            'tensor_rt_node': ComposableNode(
                name='tensor_rt',
                package='isaac_ros_tensor_rt',
                plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
                parameters=[{
                    'model_file_path': model_file_path,
                    'engine_file_path': engine_file_path,
                    'output_binding_names': output_binding_names,
                    'output_tensor_names': output_tensor_names,
                    'input_tensor_names': input_tensor_names,
                    'input_binding_names': input_binding_names,
                    'verbose': verbose,
                    'force_engine_update': force_engine_update
                }]
            ),
            'yolov8_decoder_node': ComposableNode(
                 name='yolov8_decoder_node',
                 package='isaac_ros_yolov8',
                 plugin='nvidia::isaac_ros::yolov8::YoloV8DecoderNode',
                 parameters=[{
                     'confidence_threshold': confidence_threshold,
                     'nms_threshold': nms_threshold,
                 }],
                
                # remappings=[
                #     ('detections_output', '/d435/detections_output'),
                #     ('selected_target_output', '/d435/selected_target_output'),
                # ],
                
                remappings=[
                    ('detections_output', '/d435/detections_output'),
                    ('selected_target_output', '/d435/selected_target_output'),
                    ('selected_target_kf_output', '/d435/selected_target_kf_output'),
                ],   
            ),

            # Create a binary segmentation mask from a Detection2DArray published by YOLOV8.
            # The segmentation mask is of size
            # int(IMAGE_WIDTH/input_to_YOLOV8_ratio) x int(IMAGE_HEIGHT/input_to_YOLOV8_ratio)
            'detection2_d_to_mask_node': ComposableNode(
                name='detection2_d_to_mask',
                package='isaac_ros_foundationpose',
                plugin='nvidia::isaac_ros::foundationpose::Detection2DToMask',
                parameters=[{
                    'mask_width': int(input_width/input_to_YOLOV8_ratio),
                    'mask_height': int(input_height/input_to_YOLOV8_ratio)}],
                
                # remappings=[
                #             ('detection2_d_array', '/d435/selected_target_output'),
                #             ('segmentation', '/d435/yolo_segmentation'),
                # ],
                
                remappings=[
                            ('detection2_d_array', '/d435/selected_target_kf_output'),
                            ('segmentation', '/d435/yolo_segmentation'),
                ],
            ),

            
            'selector_node': ComposableNode(
                name='selector_node',
                package='isaac_ros_foundationpose',
                plugin='nvidia::isaac_ros::foundationpose::Selector',
                parameters=[{
                }],
                
                remappings=[ # from d435 directly
                    ('image', '/d435/color/image_raw'),
                    ('camera_info', '/d435/color/camera_info'),
                    ('depth_image', '/d435/depth_converted'),
                    ('segmentation', '/d435/yolo_segmentation'),
                ],
            ),

            'foundationpose_node': ComposableNode(
                name='foundationpose_node',
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
                    
                    # 'tf_frame_name' : ['fp_mesh'],
                    
                    
                }],
                
                # remappings=[
                #     ('pose_estimation/depth_image', '/depth_image'),
                #     ('pose_estimation/image', '/rgb/image_rect_color'),
                #     ('pose_estimation/camera_info', '/rgb/camera_info'),
                #     ('pose_estimation/segmentation', '/d435/segmentation'),
                #     ('pose_estimation/output', '/d435/fpe_output')
                # ],
            ),

            'foundationpose_tracking_node': ComposableNode(
                name='foundationpose_tracking_node',
                package='isaac_ros_foundationpose',
                plugin='nvidia::isaac_ros::foundationpose::FoundationPoseTrackingNode',
                parameters=[{
                    'mesh_file_path': mesh_file_path,
                    'texture_path': texture_path,

                    'refine_model_file_path': refine_model_file_path,
                    'refine_engine_file_path': refine_engine_file_path,
                    'refine_input_tensor_names': ['input_tensor1', 'input_tensor2'],
                    'refine_input_binding_names': ['input1', 'input2'],
                    'refine_output_tensor_names': ['output_tensor1', 'output_tensor2'],
                    'refine_output_binding_names': ['output1', 'output2'],
                    
                    # 'tf_frame_name' : ['fp_mesh'],
                    
                }],
                
                # remappings=[
                #     ('tracking/depth_image', '/d435/depth_converted'),
                #     ('tracking/image', '/d435/color/image_raw'),
                #     ('tracking/camera_info', '/d435/color/camera_info'),
                #     ('tracking/segmentation', '/d435/segmentation'),
                #     ('tracking/output', '/d435/fpt_output')
                # ],
                
            ),
            

        }

    @staticmethod
    def get_launch_actions(interface_specs: Dict[str, Any]) -> \
            Dict[str, launch.actions.OpaqueFunction]:
        
        network_image_width = LaunchConfiguration('network_image_width')
        network_image_height = LaunchConfiguration('network_image_height')
        image_mean = LaunchConfiguration('image_mean')
        image_stddev = LaunchConfiguration('image_stddev')

        encoder_dir = get_package_share_directory('isaac_ros_dnn_image_encoder')

        return {
            'mesh_file_path': DeclareLaunchArgument(
                'mesh_file_path',
                default_value=MESH_FILE_PATH,
                description='The absolute file path to the mesh file'),

            'texture_path': DeclareLaunchArgument(
                'texture_path',
                default_value=TEXTURE_PATH,
                description='The absolute file path to the texture map'),

            'refine_model_file_path': DeclareLaunchArgument(
                'refine_model_file_path',
                default_value=REFINE_MODEL_PATH,
                description='The absolute file path to the refine model'),

            'refine_engine_file_path': DeclareLaunchArgument(
                'refine_engine_file_path',
                default_value=REFINE_ENGINE_PATH,
                description='The absolute file path to the refine trt engine'),

            'score_model_file_path': DeclareLaunchArgument(
                'score_model_file_path',
                default_value=SCORE_MODEL_PATH,
                description='The absolute file path to the score model'),

            'score_engine_file_path': DeclareLaunchArgument(
                'score_engine_file_path',
                default_value=SCORE_ENGINE_PATH,
                description='The absolute file path to the score trt engine'),

            'network_image_width': DeclareLaunchArgument(
                'network_image_width',
                default_value='640',
                description='The input image width that the network expects'
            ),
            'network_image_height': DeclareLaunchArgument(
                'network_image_height',
                default_value='640',
                description='The input image height that the network expects'
            ),
            'image_mean': DeclareLaunchArgument(
                'image_mean',
                default_value='[0.0, 0.0, 0.0]',
                description='The mean for image normalization'
            ),
            'image_stddev': DeclareLaunchArgument(
                'image_stddev',
                default_value='[1.0, 1.0, 1.0]',
                description='The standard deviation for image normalization'
            ),

            'model_file_path': DeclareLaunchArgument(
                'model_file_path',
                 default_value=MODEL_PATH,
                description='The absolute file path to the ONNX file'
            ),
            'engine_file_path': DeclareLaunchArgument(
                'engine_file_path',
                 default_value=ENGINE_PATH,
                description='The absolute file path to the TensorRT engine file'
            ),
            'input_tensor_names': DeclareLaunchArgument(
                'input_tensor_names',
                default_value='["input_tensor"]',
                description='A list of tensor names to bound to the specified input binding names'
            ),
            'input_binding_names': DeclareLaunchArgument(
                'input_binding_names',
                default_value='["images"]',
                description='A list of input tensor binding names (specified by model)'
            ),
            'output_tensor_names': DeclareLaunchArgument(
                'output_tensor_names',
                default_value='["output_tensor"]',
                description='A list of tensor names to bound to the specified output binding names'
            ),
            'output_binding_names': DeclareLaunchArgument(
                'output_binding_names',
                default_value='["output0"]',
                description='A list of output tensor binding names (specified by model)'
            ),
            'verbose': DeclareLaunchArgument(
                'verbose',
                default_value='False',
                description='Whether TensorRT should verbosely log or not'
            ),
            'force_engine_update': DeclareLaunchArgument(
                'force_engine_update',
                default_value='False',
                description='Whether TensorRT should update the TensorRT engine file or not'
            ),
            'confidence_threshold': DeclareLaunchArgument(
                'confidence_threshold',
                default_value='0.5,
                description='Confidence threshold to filter candidate detections during NMS'
            ),
            'nms_threshold': DeclareLaunchArgument(
                'nms_threshold',
                default_value='0.45', # 0.45, 0.8
                description='NMS IOU threshold'
            ),

            'yolov8_encoder_launch': IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    [os.path.join(encoder_dir, 'launch', 'dnn_image_encoder.launch.py')]
                ),
                launch_arguments={
                    'input_image_width': str(interface_specs['camera_resolution']['width']),
                    'input_image_height': str(interface_specs['camera_resolution']['height']),
                    # 'input_image_width': str(640),
                    # 'input_image_height': str(480),
                    'network_image_width': network_image_width,
                    'network_image_height': network_image_height,
                    'image_mean': image_mean,
                    'image_stddev': image_stddev,
                    'attach_to_shared_component_container': 'True',
                    'component_container_name': '/isaac_ros_examples/container',
                    'dnn_image_encoder_namespace': 'yolov8_encoder',
                    'image_input_topic': '/d435/color/image_raw',
                    'camera_info_input_topic': '/d435/color/camera_info',
                    'tensor_output_topic': '/tensor_pub',
                }.items(),
            ),
            
        }


def generate_launch_description():
    foundationpose_tracking_container = ComposableNodeContainer(
        package='rclcpp_components',
        name='foundationpose_tracking_container',
        namespace='',
        executable='component_container_mt',
        composable_node_descriptions=[
            *IsaacROSFoundationPoseTrackingLaunchFragment.get_composable_nodes().values(),
        ],
        output='screen'
    )

    return launch.LaunchDescription(
        [foundationpose_tracking_container] +
        IsaacROSFoundationPoseTrackingLaunchFragment.get_launch_actions().values()
    )











# #################### YOLOV8 (old)
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
# from isaac_ros_examples import IsaacROSLaunchFragment
# import launch
# from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
# from launch.launch_description_sources import PythonLaunchDescriptionSource
# from launch.substitutions import LaunchConfiguration
# from launch_ros.actions import ComposableNodeContainer
# from launch_ros.descriptions import ComposableNode

# # Expected number of input messages in 1 second
# INPUT_IMAGES_EXPECT_FREQ = 30
# # Number of input messages to be dropped in 1 second
# INPUT_IMAGES_DROP_FREQ = 20

# # Camera resolution
# WIDTH, HEIGHT = 640, 480
# # WIDTH, HEIGHT = 848, 480

# # YOLOV8 models expetc 640x640 encoded image size
# YOLOV8_MODEL_INPUT_SIZE = 640

# VISUALIZATION_DOWNSCALING_FACTOR = 10

# # REFINE_MODEL_PATH = '/tmp/refine_model.onnx'
# # REFINE_ENGINE_PATH = '/tmp/refine_trt_engine.plan'
# # SCORE_MODEL_PATH = '/tmp/score_model.onnx'
# # SCORE_ENGINE_PATH = '/tmp/score_trt_engine.plan'

# MESH_FILE_PATH = '/workspaces/isaac_ros-dev/isaac_ros_assets/isaac_ros_foundationpose/rpb/rpb.obj'
# TEXTURE_PATH = '/workspaces/isaac_ros-dev/isaac_ros_assets/isaac_ros_foundationpose/rpb/baked_mesh_tex0.png'
# REFINE_MODEL_PATH = '/tmp/refine_model.onnx'
# REFINE_ENGINE_PATH = '/isaac_ros_assets/models/foundationpose/refine_trt_engine.plan'
# SCORE_MODEL_PATH = '/tmp/score_model.onnx'
# SCORE_ENGINE_PATH = '/workspaces/isaac_ros-dev/isaac_ros_assets/models/foundationpose/score_trt_engine.plan'
# MODEL_PATH = '/workspaces/isaac_ros-dev/isaac_ros_assets/models/yolov8/best.onnx'
# ENGINE_PATH = '/workspaces/isaac_ros-dev/isaac_ros_assets/models/yolov8/best.plan'


# class IsaacROSFoundationPoseTrackingLaunchFragment(IsaacROSLaunchFragment):

#     @staticmethod
#     def get_composable_nodes(interface_specs: Dict[str, Any]) -> Dict[str, ComposableNode]:
        
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
#         input_width = interface_specs['camera_resolution']['width']
#         input_height = interface_specs['camera_resolution']['height']
#         # input_width = WIDTH
#         # input_height = HEIGHT
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
#             # Drops input_images_expect_freq out of input_images_drop_freq input messages
#             'drop_node':  ComposableNode(
#                 name='drop_node',
#                 package='isaac_ros_nitros_topic_tools',
#                 plugin='nvidia::isaac_ros::nitros::NitrosCameraDropNode',
#                 parameters=[{
#                     'X': input_images_drop_freq,
#                     'Y': input_images_expect_freq,
#                     'mode': 'mono+depth',
#                     'depth_format_string': 'nitros_image_mono16'
#                 }],
#                 # remappings=[
#                 #     ('image_1', 'image_rect'),
#                 #     ('camera_info_1', 'camera_info_rect'),
#                 #     ('depth_1', 'depth'),
#                 #     ('image_1_drop', 'rgb/image_rect_color'),
#                 #     ('camera_info_1_drop', 'rgb/camera_info'),
#                 #     ('depth_1_drop', 'depth_image'),
#                 # ]
                
#                 remappings=[
#                     ('image_1', '/d435/color/image_raw'),
#                     ('camera_info_1', '/d435/color/camera_info'),
#                     ('depth_1', '/d435/depth_converted'),
#                     ('image_1_drop', '/rgb/image_rect_color'),
#                     ('camera_info_1_drop', '/rgb/camera_info'),
#                     ('depth_1_drop', '/depth_image'),
#                 ],
                
                
#                 # remappings=[
#                 #     ('image_1', '/d435/color_image_fpe'),
#                 #     ('camera_info_1', '/d435/camera_info_fpe'),
#                 #     ('depth_1', '/d435/depth_image_fpe'),
#                 #     ('image_1_drop', '/rgb/image_rect_color'),
#                 #     ('camera_info_1_drop', '/rgb/camera_info'),
#                 #     ('depth_1_drop', '/depth_image'),
#                 # ],
#             ),
            
#             # Yolo objection detection pipeline
#             'tensor_rt_node': ComposableNode(
#                 name='tensor_rt',
#                 package='isaac_ros_tensor_rt',
#                 plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
#                 parameters=[{
#                     'model_file_path': model_file_path,
#                     'engine_file_path': engine_file_path,
#                     'output_binding_names': output_binding_names,
#                     'output_tensor_names': output_tensor_names,
#                     'input_tensor_names': input_tensor_names,
#                     'input_binding_names': input_binding_names,
#                     'verbose': verbose,
#                     'force_engine_update': force_engine_update
#                 }]
#             ),
#             'yolov8_decoder_node': ComposableNode(
#                  name='yolov8_decoder_node',
#                  package='isaac_ros_yolov8',
#                  plugin='nvidia::isaac_ros::yolov8::YoloV8DecoderNode',
#                  parameters=[{
#                      'confidence_threshold': confidence_threshold,
#                      'nms_threshold': nms_threshold,
#                  }],
#                 #  remappings=[
#                 #     ('detections_output', '/d435/detections_output'),
#                 # ],
                
#                 remappings=[
#                     ('detections_output', '/d435/detections_output'),
#                     ('selected_target_output', '/d435/selected_target_output'),
#                     ('selected_target_kf_output', '/d435/selected_target_kf_output'),
#                 ],
                 
#             ),

#             # Create a binary segmentation mask from a Detection2DArray published by YOLOV8.
#             # The segmentation mask is of size
#             # int(IMAGE_WIDTH/input_to_YOLOV8_ratio) x int(IMAGE_HEIGHT/input_to_YOLOV8_ratio)
#             'detection2_d_to_mask_node': ComposableNode(
#                 name='detection2_d_to_mask',
#                 package='isaac_ros_foundationpose',
#                 plugin='nvidia::isaac_ros::foundationpose::Detection2DToMask',
#                 parameters=[{
#                     'mask_width': int(input_width/input_to_YOLOV8_ratio),
#                     'mask_height': int(input_height/input_to_YOLOV8_ratio)}],
#                 # remappings=[('detection2_d_array', 'detections_output'),
#                 #             ('segmentation', 'yolo_segmentation')
#                 #             ],
                
#                 # remappings=[('detection2_d_array', '/d435/detections_output'),
#                 #             ('segmentation', '/d435/yolo_segmentation'),
#                 #             ],
                
#                 # remappings=[
#                 #             ('detection2_d_array', '/d435/selected_target_output'),
#                 #             ('segmentation', '/d435/yolo_segmentation'),
#                 # ],
                
#                 remappings=[
#                             ('detection2_d_array', '/d435/selected_target_kf_output'),
#                             ('segmentation', '/d435/yolo_segmentation'),
#                 ],
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
#                 #     ('image', 'yolo_segmentation'),
#                 #     ('camera_info', 'rgb/camera_info'),
#                 #     ('resize/image', 'segmentation'),
#                 #     ('resize/camera_info', 'camera_info_segmentation')
#                 # ]
#                 # remappings=[
#                 #     ('image', '/d435/yolo_segmentation'),
#                 #     ('camera_info', '/rgb/camera_info'),
#                 #     ('resize/image', '/d435/segmentation'),
#                 #     ('resize/camera_info', '/camera_info_segmentation'),
#                 # ],
#                 remappings=[
#                     ('image', '/d435/yolo_segmentation'),
#                     ('camera_info', '/d435/color/camera_info'),
#                     ('resize/image', '/d435/segmentation'),
#                     ('resize/camera_info', '/camera_info_segmentation'),
#                 ],
#             ),

            
#             'selector_node': ComposableNode(
#                 name='selector_node',
#                 package='isaac_ros_foundationpose',
#                 plugin='nvidia::isaac_ros::foundationpose::Selector',
#                 parameters=[{
#                     # 'reset_period': 20000, # [ms]
#                     'reset_period': 5000, # [ms]
#                     # 'reset_period': 60000, # [ms]
#                 }],
#                 # remappings=[
#                 #     ('image', '/rgb/image_rect_color'),
#                 #     ('camera_info', '/rgb/camera_info'),
#                 #     ('depth_image', '/depth_image'),
#                 # ],
                
#                 # remappings=[ # from drop_node
#                 #     ('image', '/rgb/image_rect_color'),
#                 #     ('camera_info', '/rgb/camera_info'),
#                 #     ('depth_image', '/depth_image'),
#                 #     ('segmentation', '/d435/segmentation'),
#                 # ],
                
#                 # remappings=[ # from d435 directly
#                 #     ('image', '/d435/color/image_raw'),
#                 #     ('camera_info', '/d435/color/camera_info'),
#                 #     ('depth_image', '/d435/depth_converted'),
#                 #     ('segmentation', '/d435/segmentation'),
#                 # ],
                
#                 remappings=[ # from d435 directly
#                     ('image', '/d435/color/image_raw'),
#                     ('camera_info', '/d435/color/camera_info'),
#                     ('depth_image', '/d435/depth_converted'),
#                     ('segmentation', '/d435/yolo_segmentation'),
#                 ],
#             ),

#             'foundationpose_node': ComposableNode(
#                 name='foundationpose_node',
#                 package='isaac_ros_foundationpose',
#                 plugin='nvidia::isaac_ros::foundationpose::FoundationPoseNode',
#                 parameters=[{
#                     'mesh_file_path': mesh_file_path,
#                     'texture_path': texture_path,
                    
#                     'refine_model_file_path': refine_model_file_path,
#                     'refine_engine_file_path': refine_engine_file_path,
#                     'refine_input_tensor_names': ['input_tensor1', 'input_tensor2'],
#                     'refine_input_binding_names': ['input1', 'input2'],
#                     'refine_output_tensor_names': ['output_tensor1', 'output_tensor2'],
#                     'refine_output_binding_names': ['output1', 'output2'],

#                     'score_model_file_path': score_model_file_path,
#                     'score_engine_file_path': score_engine_file_path,
#                     'score_input_tensor_names': ['input_tensor1', 'input_tensor2'],
#                     'score_input_binding_names': ['input1', 'input2'],
#                     'score_output_tensor_names': ['output_tensor'],
#                     'score_output_binding_names': ['output1'],
                    
#                     # 'tf_frame_name' : [''],
                    
#                     # 'mesh_file_path': MESH_FILE_PATH,
#                     # 'texture_path': TEXTURE_PATH,
#                     # 'refine_model_file_path': REFINE_MODEL_PATH,
#                     # 'refine_engine_file_path': REFINE_ENGINE_PATH,
#                     # 'refine_input_tensor_names': ['input_tensor1', 'input_tensor2'],
#                     # 'refine_input_binding_names': ['input1', 'input2'],
#                     # 'refine_output_tensor_names': ['output_tensor1', 'output_tensor2'],
#                     # 'refine_output_binding_names': ['output1', 'output2'],

#                     # 'score_model_file_path': SCORE_MODEL_PATH,
#                     # 'score_engine_file_path': SCORE_ENGINE_PATH,
#                     # 'score_input_tensor_names': ['input_tensor1', 'input_tensor2'],
#                     # 'score_input_binding_names': ['input1', 'input2'],
#                     # 'score_output_tensor_names': ['output_tensor'],
#                     # 'score_output_binding_names': ['output1'],
#                 }],
#                 # remappings=[
#                 #     ('pose_estimation/depth_image', 'depth_image'),
#                 #     ('pose_estimation/image', 'rgb/image_rect_color'),
#                 #     ('pose_estimation/camera_info', 'rgb/camera_info'),
#                 #     ('pose_estimation/segmentation', 'segmentation'),
#                 #     # ('pose_estimation/output', 'output')
#                 #     ],
                
#                 # remappings=[
#                 #     ('pose_estimation/depth_image', '/depth_image'),
#                 #     ('pose_estimation/image', '/rgb/image_rect_color'),
#                 #     ('pose_estimation/camera_info', '/rgb/camera_info'),
#                 #     ('pose_estimation/segmentation', '/d435/segmentation'),
#                 #     ('pose_estimation/output', '/d435/fpe_output')
#                 # ],
#             ),

#             'foundationpose_tracking_node': ComposableNode(
#                 name='foundationpose_tracking_node',
#                 package='isaac_ros_foundationpose',
#                 plugin='nvidia::isaac_ros::foundationpose::FoundationPoseTrackingNode',
#                 parameters=[{
#                     'mesh_file_path': mesh_file_path,
#                     'texture_path': texture_path,

#                     'refine_model_file_path': refine_model_file_path,
#                     'refine_engine_file_path': refine_engine_file_path,
#                     'refine_input_tensor_names': ['input_tensor1', 'input_tensor2'],
#                     'refine_input_binding_names': ['input1', 'input2'],
#                     'refine_output_tensor_names': ['output_tensor1', 'output_tensor2'],
#                     'refine_output_binding_names': ['output1', 'output2'],
                    
#                     # 'mesh_file_path': MESH_FILE_PATH,
#                     # 'texture_path': TEXTURE_PATH,

#                     # 'refine_model_file_path': REFINE_MODEL_PATH,
#                     # 'refine_engine_file_path': REFINE_ENGINE_PATH,
#                     # 'refine_input_tensor_names': ['input_tensor1', 'input_tensor2'],
#                     # 'refine_input_binding_names': ['input1', 'input2'],
#                     # 'refine_output_tensor_names': ['output_tensor1', 'output_tensor2'],
#                     # 'refine_output_binding_names': ['output1', 'output2'],
#                 }],
#                 # remappings=[
#                 #     ('tracking/depth_image', '/depth_image'),
#                 #     ('tracking/image', '/rgb/image_rect_color'),
#                 #     ('tracking/camera_info', '/rgb/camera_info'),
#                 #     ('tracking/segmentation', '/d435/segmentation'),
#                 #     # ('tracking/output', 'output')
#                 # ],
                
#                 # remappings=[
#                 #     ('tracking/depth_image', '/d435/depth_converted'),
#                 #     ('tracking/image', '/d435/color/image_raw'),
#                 #     ('tracking/camera_info', '/d435/color/camera_info'),
#                 #     ('tracking/segmentation', '/d435/segmentation'),
#                 #     ('tracking/output', '/d435/fpt_output')
#                 # ],
                
#             ),
            

#         }

#     @staticmethod
#     def get_launch_actions(interface_specs: Dict[str, Any]) -> \
#             Dict[str, launch.actions.OpaqueFunction]:
        
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
#                 default_value='0.4',
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
#                     'input_image_width': str(interface_specs['camera_resolution']['width']),
#                     'input_image_height': str(interface_specs['camera_resolution']['height']),
#                     # 'input_image_width': str(640),
#                     # 'input_image_height': str(480),
#                     'network_image_width': network_image_width,
#                     'network_image_height': network_image_height,
#                     'image_mean': image_mean,
#                     'image_stddev': image_stddev,
#                     'attach_to_shared_component_container': 'True',
#                     'component_container_name': '/isaac_ros_examples/container',
#                     'dnn_image_encoder_namespace': 'yolov8_encoder',
#                     'image_input_topic': '/d435/color/image_raw',
#                     'camera_info_input_topic': '/d435/color/camera_info',
#                     # 'image_input_topic': 'image_rect',
#                     # 'camera_info_input_topic': 'camera_info_rect',
#                     'tensor_output_topic': '/tensor_pub',
#                 }.items(),
#             ),
            
#         }


# def generate_launch_description():
#     foundationpose_tracking_container = ComposableNodeContainer(
#         package='rclcpp_components',
#         name='foundationpose_tracking_container',
#         namespace='',
#         executable='component_container_mt',
#         composable_node_descriptions=[
#             *IsaacROSFoundationPoseTrackingLaunchFragment.get_composable_nodes().values(),
#             # IsaacROSFoundationPoseTrackingLaunchFragment.get_composable_nodes({})['flange_to_camera_tf_broadcaster'],
#         ],
#         output='screen'
#     )

#     return launch.LaunchDescription(
#         [foundationpose_tracking_container] +
#         IsaacROSFoundationPoseTrackingLaunchFragment.get_launch_actions().values()
#     )








































#################### RT-DETR
# # # SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
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

# from typing import Any, Dict

# from isaac_ros_examples import IsaacROSLaunchFragment
# import launch
# from launch.actions import DeclareLaunchArgument
# from launch.substitutions import LaunchConfiguration
# from launch_ros.actions import ComposableNodeContainer
# from launch_ros.descriptions import ComposableNode


# # RT-DETR models expect 640x640 encoded image size
# RT_DETR_MODEL_INPUT_SIZE = 640
# # RT-DETR models expect 3 image channels
# RT_DETR_MODEL_NUM_CHANNELS = 3

# REFINE_MODEL_PATH = '/tmp/refine_model.onnx'
# REFINE_ENGINE_PATH = '/tmp/refine_trt_engine.plan'
# SCORE_MODEL_PATH = '/tmp/score_model.onnx'
# SCORE_ENGINE_PATH = '/tmp/score_trt_engine.plan'


# class IsaacROSFoundationPoseTrackingLaunchFragment(IsaacROSLaunchFragment):

#     @staticmethod
#     def get_composable_nodes(interface_specs: Dict[str, Any]) -> Dict[str, ComposableNode]:

#         # FoundationPose parameters
#         mesh_file_path = LaunchConfiguration('mesh_file_path')
#         texture_path = LaunchConfiguration('texture_path')
#         refine_model_file_path = LaunchConfiguration('refine_model_file_path')
#         refine_engine_file_path = LaunchConfiguration('refine_engine_file_path')
#         score_model_file_path = LaunchConfiguration('score_model_file_path')
#         score_engine_file_path = LaunchConfiguration('score_engine_file_path')
#         # RT-DETR parameters
#         rt_detr_model_file_path = LaunchConfiguration('rt_detr_model_file_path')
#         rt_detr_engine_file_path = LaunchConfiguration('rt_detr_engine_file_path')
#         input_width = interface_specs['camera_resolution']['width']
#         input_height = interface_specs['camera_resolution']['height']
#         input_to_RT_DETR_ratio = input_width / RT_DETR_MODEL_INPUT_SIZE
#         return {
#             # Resize and pad input images to RT-DETR model input image size
#             # Resize from IMAGE_WIDTH x IMAGE_HEIGHT to
#             # IMAGE_WIDTH/input_TO_RT_DETR_RATIO x IMAGE_HEIGHT/input_TO_RT_DETR_RATIO
#             # output height constraint is not used since keep_aspect_ratio is True
#             'resize_left_rt_detr_node': ComposableNode(
#                 name='resize_left_rt_detr_node',
#                 package='isaac_ros_image_proc',
#                 plugin='nvidia::isaac_ros::image_proc::ResizeNode',
#                 parameters=[{
#                     'input_width': input_width,
#                     'input_height': input_height,
#                     'output_width': RT_DETR_MODEL_INPUT_SIZE,
#                     'output_height': RT_DETR_MODEL_INPUT_SIZE,
#                     'keep_aspect_ratio': True,
#                     'encoding_desired': 'rgb8',
#                     'disable_padding': True
#                 }],
#                 remappings=[
#                     ('image', 'image_rect'),
#                     ('camera_info', 'camera_info_rect'),
#                     ('resize/image', 'color_image_resized'),
#                     ('resize/camera_info', 'camera_info_resized')
#                 ]
#             ),
#             # Pad the image from IMAGE_WIDTH/input_TO_RT_DETR_RATIO x
#             # IMAGE_HEIGHT/input_TO_RT_DETR_RATIO
#             # to RT_DETR_MODEL_INPUT_WIDTH x RT_DETR_MODEL_INPUT_HEIGHT
#             'pad_node': ComposableNode(
#                 name='pad_node',
#                 package='isaac_ros_image_proc',
#                 plugin='nvidia::isaac_ros::image_proc::PadNode',
#                 parameters=[{
#                     'output_image_width': RT_DETR_MODEL_INPUT_SIZE,
#                     'output_image_height': RT_DETR_MODEL_INPUT_SIZE,
#                     'padding_type': 'BOTTOM_RIGHT'
#                 }],
#                 remappings=[(
#                     'image', 'color_image_resized'
#                 )]
#             ),

#             # Convert image to tensor and reshape
#             'image_to_tensor_node': ComposableNode(
#                 name='image_to_tensor_node',
#                 package='isaac_ros_tensor_proc',
#                 plugin='nvidia::isaac_ros::dnn_inference::ImageToTensorNode',
#                 parameters=[{
#                     'scale': False,
#                     'tensor_name': 'image',
#                 }],
#                 remappings=[
#                     ('image', 'padded_image'),
#                     ('tensor', 'normalized_tensor'),
#                 ]
#             ),

#             'interleave_to_planar_node': ComposableNode(
#                 name='interleaved_to_planar_node',
#                 package='isaac_ros_tensor_proc',
#                 plugin='nvidia::isaac_ros::dnn_inference::InterleavedToPlanarNode',
#                 parameters=[{
#                     'input_tensor_shape': [RT_DETR_MODEL_INPUT_SIZE,
#                                            RT_DETR_MODEL_INPUT_SIZE,
#                                            RT_DETR_MODEL_NUM_CHANNELS]
#                 }],
#                 remappings=[
#                     ('interleaved_tensor', 'normalized_tensor')
#                 ]
#             ),

#             'reshape_node': ComposableNode(
#                 name='reshape_node',
#                 package='isaac_ros_tensor_proc',
#                 plugin='nvidia::isaac_ros::dnn_inference::ReshapeNode',
#                 parameters=[{
#                     'output_tensor_name': 'input_tensor',
#                     'input_tensor_shape': [RT_DETR_MODEL_NUM_CHANNELS,
#                                            RT_DETR_MODEL_INPUT_SIZE,
#                                            RT_DETR_MODEL_INPUT_SIZE],
#                     'output_tensor_shape': [1, RT_DETR_MODEL_NUM_CHANNELS,
#                                             RT_DETR_MODEL_INPUT_SIZE,
#                                             RT_DETR_MODEL_INPUT_SIZE]
#                 }],
#                 remappings=[
#                     ('tensor', 'planar_tensor')
#                 ],
#             ),

#             'rtdetr_preprocessor_node': ComposableNode(
#                 name='rtdetr_preprocessor',
#                 package='isaac_ros_rtdetr',
#                 plugin='nvidia::isaac_ros::rtdetr::RtDetrPreprocessorNode',
#                 remappings=[
#                     ('encoded_tensor', 'reshaped_tensor')
#                 ]
#             ),

#             # RT-DETR objection detection pipeline
#             'tensor_rt_node': ComposableNode(
#                 name='tensor_rt',
#                 package='isaac_ros_tensor_rt',
#                 plugin='nvidia::isaac_ros::dnn_inference::TensorRTNode',
#                 parameters=[{
#                     'model_file_path': rt_detr_model_file_path,
#                     'engine_file_path': rt_detr_engine_file_path,
#                     'output_binding_names': ['labels', 'boxes', 'scores'],
#                     'output_tensor_names': ['labels', 'boxes', 'scores'],
#                     'input_tensor_names': ['images', 'orig_target_sizes'],
#                     'input_binding_names': ['images', 'orig_target_sizes'],
#                     'force_engine_update': False
#                 }]
#             ),
#             'rtdetr_decoder_node': ComposableNode(
#                 name='rtdetr_decoder',
#                 package='isaac_ros_rtdetr',
#                 plugin='nvidia::isaac_ros::rtdetr::RtDetrDecoderNode',
#                 parameters=[{
#                     'confidence_threshold': 0.5,
#                 }],
#             ),

#             # Create a binary segmentation mask from a Detection2DArray published by RT-DETR.
#             # The segmentation mask is of size
#             # int(IMAGE_WIDTH/input_to_RT_DETR_ratio) x int(IMAGE_HEIGHT/input_to_RT_DETR_ratio)
#             'detection2_d_to_mask_node': ComposableNode(
#                 name='detection2_d_to_mask',
#                 package='isaac_ros_foundationpose',
#                 plugin='nvidia::isaac_ros::foundationpose::Detection2DToMask',
#                 parameters=[{
#                     'mask_width': int(input_width/input_to_RT_DETR_ratio),
#                     'mask_height': int(input_height/input_to_RT_DETR_ratio)}],
#                 remappings=[('detection2_d_array', 'detections_output'),
#                             ('segmentation', 'rt_detr_segmentation')]),

#             # Resize segmentation mask to ESS model image size so it can be used by FoundationPose
#             # FoundationPose requires depth, rgb image and segmentation mask to be of the same size
#             # Resize from int(IMAGE_WIDTH/input_to_RT_DETR_ratio) x
#             # int(IMAGE_HEIGHT/input_to_RT_DETR_ratio)
#             # to ESS_MODEL_IMAGE_WIDTH x ESS_MODEL_IMAGE_HEIGHT
#             # output height constraint is used since keep_aspect_ratio is False
#             # and the image is padded
#             'resize_mask_node': ComposableNode(
#                 name='resize_mask_node',
#                 package='isaac_ros_image_proc',
#                 plugin='nvidia::isaac_ros::image_proc::ResizeNode',
#                 parameters=[{
#                     'input_width': int(input_width/input_to_RT_DETR_ratio),
#                     'input_height': int(input_height/input_to_RT_DETR_ratio),
#                     'output_width': input_width,
#                     'output_height': input_height,
#                     'keep_aspect_ratio': False,
#                     'disable_padding': False
#                 }],
#                 remappings=[
#                     ('image', 'rt_detr_segmentation'),
#                     ('camera_info', 'camera_info_resized'),
#                     ('resize/image', 'segmentation'),
#                     ('resize/camera_info', 'camera_info_segmentation')
#                 ]
#             ),

#             'selector_node': ComposableNode(
#                 name='selector_node',
#                 package='isaac_ros_foundationpose',
#                 plugin='nvidia::isaac_ros::foundationpose::Selector',
#                 parameters=[{
#                     # Expect to reset after the rosbag play complete
#                     'reset_period': 65000
#                 }],
#                 remappings=[
#                     ('image', 'image_rect'),
#                     ('camera_info', 'camera_info_rect'),
#                     ('depth_image', 'depth'),
#                 ]
#             ),

#             'foundationpose_node': ComposableNode(
#                 name='foundationpose_node',
#                 package='isaac_ros_foundationpose',
#                 plugin='nvidia::isaac_ros::foundationpose::FoundationPoseNode',
#                 parameters=[{
#                     'mesh_file_path': mesh_file_path,
#                     'texture_path': texture_path,

#                     'refine_model_file_path': refine_model_file_path,
#                     'refine_engine_file_path': refine_engine_file_path,
#                     'refine_input_tensor_names': ['input_tensor1', 'input_tensor2'],
#                     'refine_input_binding_names': ['input1', 'input2'],
#                     'refine_output_tensor_names': ['output_tensor1', 'output_tensor2'],
#                     'refine_output_binding_names': ['output1', 'output2'],

#                     'score_model_file_path': score_model_file_path,
#                     'score_engine_file_path': score_engine_file_path,
#                     'score_input_tensor_names': ['input_tensor1', 'input_tensor2'],
#                     'score_input_binding_names': ['input1', 'input2'],
#                     'score_output_tensor_names': ['output_tensor'],
#                     'score_output_binding_names': ['output1'],
#                 }]
#             ),

#             'foundationpose_tracking_node': ComposableNode(
#                 name='foundationpose_tracking_node',
#                 package='isaac_ros_foundationpose',
#                 plugin='nvidia::isaac_ros::foundationpose::FoundationPoseTrackingNode',
#                 parameters=[{
#                     'mesh_file_path': mesh_file_path,
#                     'texture_path': texture_path,

#                     'refine_model_file_path': refine_model_file_path,
#                     'refine_engine_file_path': refine_engine_file_path,
#                     'refine_input_tensor_names': ['input_tensor1', 'input_tensor2'],
#                     'refine_input_binding_names': ['input1', 'input2'],
#                     'refine_output_tensor_names': ['output_tensor1', 'output_tensor2'],
#                     'refine_output_binding_names': ['output1', 'output2'],
#                 }])

#         }

#     @staticmethod
#     def get_launch_actions(interface_specs: Dict[str, Any]) -> \
#             Dict[str, launch.actions.OpaqueFunction]:

#         return {
#             'mesh_file_path': DeclareLaunchArgument(
#                 'mesh_file_path',
#                 default_value='',
#                 description='The absolute file path to the mesh file'),

#             'texture_path': DeclareLaunchArgument(
#                 'texture_path',
#                 default_value='',
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

#             'rt_detr_model_file_path': DeclareLaunchArgument(
#                 'rt_detr_model_file_path',
#                 default_value='',
#                 description='The absolute file path to the RT-DETR ONNX file'),

#             'rt_detr_engine_file_path': DeclareLaunchArgument(
#                 'rt_detr_engine_file_path',
#                 default_value='',
#                 description='The absolute file path to the RT-DETR TensorRT engine file'),
#         }


# def generate_launch_description():
#     foundationpose_tracking_container = ComposableNodeContainer(
#         package='rclcpp_components',
#         name='foundationpose_tracking_container',
#         namespace='',
#         executable='component_container_mt',
#         composable_node_descriptions=IsaacROSFoundationPoseTrackingLaunchFragment
#         .get_composable_nodes().values(),
#         output='screen'
#     )

#     return launch.LaunchDescription(
#         [foundationpose_tracking_container] +
#         IsaacROSFoundationPoseTrackingLaunchFragment.get_launch_actions().values())












