/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <torch/extension.h>
#include "rasterize_points.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rasterize_t", &RasterizeTCUDA);
  m.def("rasterize_t_backward", &RasterizeTBackwardCUDA);
  m.def("mark_visible", &markVisible);
  m.def("compute_relocation", &ComputeRelocationCUDA);
  m.def("compute_relocation_student_t", &ComputeRelocationStudentTCUDA);
}