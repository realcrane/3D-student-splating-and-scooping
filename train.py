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

import os
import json
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from t_renderer import render, network_gui
import sys
from scene import Scene_nt, NTModel
from utils.general_utils import safe_state, op_sigmoid
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.nt_model import build_scaling_rotation
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from os import makedirs
import torchvision
from PIL import Image
import torchvision.transforms.functional as tf

from utils.general_utils import get_expon_lr_func


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    if dataset.cap_max == -1:
        print("Please specify the maximum number of Gaussians using --cap_max.")
        exit()
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    primitives = NTModel(dataset.sh_degree, dataset.nu_degree)
    scene = Scene_nt(dataset, primitives)
    """
    NOTE: 
    This part of code is tricky,
    the setting for learning rate is different with first and second order optimizer.
    For second order optimizer, the real learning rate is approximately lr^2.
    We are adjusting suitable learning rate for our second order optimizer here.
    This is made based on real learning rate (base_lr * spatial_lr_scale) of Train scene from Tanks&Temples dataset.
    The '0.001226442339330813' is the base_lr we want for start,
    and the '1.226630619638022e-05' is the base_lr we at the end.
    """
    train_scene_scale = 7.45176315307617
    opt.position_lr_init = pow(0.001226442339330813*primitives.spatial_lr_scale/train_scene_scale, 0.5)/primitives.spatial_lr_scale
    opt.position_lr_final = pow(1.226630619638022e-05*primitives.spatial_lr_scale/train_scene_scale, 0.5)/primitives.spatial_lr_scale

    xyz_lr_sqrt_args = get_expon_lr_func(lr_init=opt.position_lr_init*primitives.spatial_lr_scale,
                                                    lr_final=opt.position_lr_final*primitives.spatial_lr_scale,
                                                    lr_delay_mult=opt.position_lr_delay_mult,
                                                    max_steps=opt.position_lr_max_steps)

    print("spatial lr scale: {}".format(primitives.spatial_lr_scale))
    # print(opt.position_lr_init)
    # print(opt.position_lr_final)
    print("training lr range: {} - {}".format(pow(xyz_lr_sqrt_args(1),2), pow(xyz_lr_sqrt_args(opt.position_lr_max_steps),2)))

    C_burnin = dataset.C_burnin
    C = dataset.C
    burnin_iterations = dataset.burnin_iterations


    primitives.training_setup(opt, C_burnin, C, burnin_iterations)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        primitives.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    for iteration in range(first_iter, opt.iterations + 1):        
        iter_start.record()

        xyz_lr = primitives.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            primitives.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        else:
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, primitives, pipe, bg)
        image = render_pkg["render"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        loss = loss + args.opacity_reg * torch.abs(primitives.get_opacity).mean()
        loss = loss + args.scale_reg * torch.abs(primitives.get_scaling).mean()

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()
            
            # Optimizer step
            if iteration < opt.iterations:
                # NOTE: SGHMC optimization
                sig = (op_sigmoid(1 - torch.abs(primitives.get_opacity)))

                L = build_scaling_rotation(primitives.get_scaling, primitives.get_rotation)
                actual_covariance = L @ L.transpose(1, 2)

                _, total_max, total_mean, total_min = primitives.optimizer.step(sig=sig.detach(), cov=actual_covariance.detach())
                primitives.optimizer.zero_grad(set_to_none = True)
                if tb_writer:
                    tb_writer.add_scalar('sghmc_total_max', total_max.item(), iteration)
                    tb_writer.add_scalar('sghmc_total_mean', total_mean.item(), iteration)
                    tb_writer.add_scalar('sghmc_total_min', total_min.item(), iteration)

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), dataset)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if iteration < opt.densify_until_iter and iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                dead_mask = (primitives.get_opacity < dataset.opacity_threshold).squeeze(-1)
                dead_mask2 = (primitives.get_opacity > -dataset.opacity_threshold).squeeze(-1)
                dead_mask =  torch.logical_and(dead_mask, dead_mask2)
                
                primitives.recycle_components(dead_mask=dead_mask)
                primitives.add_components(cap_max=args.cap_max)

                torch.cuda.empty_cache()

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((primitives.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene_nt, renderFunc, renderArgs, dataset):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.primitives, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()

                    """
                    NOTE: save internal results first, then load saved images to calculate PSNR.
                    This may seem silly, but it is the only way to get the same PSNR as using 'metric.py'.
                    The reason is that the 'metric.py' scipt loads saved image, so the calculation is done with integer type.
                    Without saving, it is calculated with float type.
                    My experience is that you can get a higher PSNR without saving first (some work actually used this trick...).
                    """
                    render_path = os.path.join(scene.model_path, "ours_{}".format(iteration), config['name'], "renders")
                    gts_path = os.path.join(scene.model_path, "ours_{}".format(iteration), config['name'], "gt")
                    makedirs(render_path, exist_ok=True)
                    makedirs(gts_path, exist_ok=True)
                    torchvision.utils.save_image(image, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
                    torchvision.utils.save_image(gt_image, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

                    render = Image.open(os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
                    gt = Image.open(os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
                    render = tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda()
                    gt = tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda()
                    psnr_test += psnr(render, gt).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))

                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
        
        positive_pts_mask = torch.where(scene.primitives.get_opacity >= 0, True, False).squeeze(-1)
        print("positive componets number: ", (positive_pts_mask==True).sum().item())
        print("negative componets number: ", (positive_pts_mask==False).sum().item())
        
        nu_degree = scene.primitives.get_nu_degree
        print("degree of freedom: max: {} min: {} mean: {} std: {}".format(nu_degree.max(), nu_degree.min(), nu_degree.mean(), nu_degree.std()))
        
        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.primitives.get_opacity, iteration)
            tb_writer.add_histogram("scene/nu_degree_histogram", scene.primitives.get_nu_degree, iteration)
            tb_writer.add_scalar('total_points', scene.primitives.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[40_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    
    if args.config is not None:
        # Load the configuration file
        config = load_config(args.config)
        # Set the configuration parameters on args, if they are not already set by command line arguments
        for key, value in config.items():
            setattr(args, key, value)

    args.test_iterations.append(args.iterations)
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
