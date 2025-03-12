#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene_nt
import os
from tqdm import tqdm
from os import makedirs
from t_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene.nt_model import NTModel

import time


def render_set(model_path, name, iteration, views, primitives, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    render_avg_time = 0.0
    render_start_event = torch.cuda.Event(enable_timing=True)
    render_end_event = torch.cuda.Event(enable_timing=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # if idx == 0: time1 = time.time()
        # rendering = render(view, primitives, pipeline, background)["render"]
        
        render_start_event.record()
        rendering = render(view, primitives, pipeline, background)["render"]
        render_end_event.record()
        torch.cuda.synchronize()
        render_avg_time += render_start_event.elapsed_time(render_end_event)

        # NOTE: comment out image saving code to get real FPS
        gt = view.original_image[0:3, :, :]
        rendering = torch.clamp(rendering, 0.0, 1.0)
        gt = torch.clamp(gt, 0.0, 1.0)
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
    
    # NOTE: comment out image saving code to get real FPS
    # print("average render time: ", render_avg_time/1000/len(views))
    # print("FPS:", len(views)/(render_avg_time/1000.0))
    

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        primitives = NTModel(dataset.sh_degree, dataset.nu_degree)
        scene = Scene_nt(dataset, primitives, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), primitives, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), primitives, pipeline, background)
        # NOTE: print nu_degree
        nu_degree = scene.primitives.get_nu_degree
        print("\nDegree of Freedom: max {} min {} mean {} var {}".format(nu_degree.max(), nu_degree.min(), nu_degree.mean(), nu_degree.var()))

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)