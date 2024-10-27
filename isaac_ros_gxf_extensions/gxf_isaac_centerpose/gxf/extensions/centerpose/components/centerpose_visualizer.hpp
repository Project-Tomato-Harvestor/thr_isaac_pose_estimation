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
#pragma once

#include "Eigen/Dense"
#include "gxf/core/parameter_parser_std.hpp"
#include "gxf/std/allocator.hpp"
#include "gxf/std/codelet.hpp"
#include "gxf/std/receiver.hpp"
#include "gxf/std/timestamp.hpp"
#include "gxf/std/transmitter.hpp"

namespace nvidia {
namespace isaac {
namespace centerpose {

class CenterPoseVisualizer : public gxf::Codelet {
 public:
  gxf_result_t registerInterface(gxf::Registrar* registrar) override;
  gxf_result_t start() override;
  gxf_result_t tick() override;
  gxf_result_t stop() override;

 private:
  gxf::Parameter<gxf::Handle<gxf::Receiver>> video_buffer_input_;
  gxf::Parameter<gxf::Handle<gxf::Receiver>> detections_input_;
  gxf::Parameter<gxf::Handle<gxf::Receiver>> camera_model_input_;
  gxf::Parameter<gxf::Handle<gxf::Transmitter>> output_;
  gxf::Parameter<gxf::Handle<gxf::Allocator>> allocator_;

  gxf::Parameter<bool> show_axes_;
  gxf::Parameter<int32_t> bounding_box_color_;
};

}  // namespace centerpose
}  // namespace isaac
}  // namespace nvidia
