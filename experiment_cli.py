"""Dependency-free argument validation for the experiment runner."""
import argparse
import json
import math
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(description="Neural Plagiarism experiment runner")
    parser.add_argument('--target_folder', required=True, help='Folder containing target images')
    parser.add_argument('--start', type=int, default=0, help='Starting index for processing images')
    parser.add_argument('--end', type=int, default=10, help='Ending index for processing images')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device index')
    parser.add_argument('--image_length', type=int, default=512, help='Length of the image (square assumed)')
    parser.add_argument('--model_id', default='Manojb/stable-diffusion-2-1-base', help='Model ID for the diffusion pipeline')
    parser.add_argument('--num_images', type=int, default=1, help='Number of images to generate per prompt')
    parser.add_argument('--guidance_scale', type=float, default=7.5, help='Guidance scale for stable diffusion')
    parser.add_argument('--num_inference_steps', type=int, default=50, help='Legacy option; this runner uses --attack_num_inference_steps')
    parser.add_argument('--attack_num_inference_steps', type=int, default=50, help='Number of steps for reverse diffusion')
    parser.add_argument('--output_folder', default='./outputs/', help='Folder for saving output images and logs')
    parser.add_argument('--start_step', type=int, default=0, help='Starting step for optimization.')
    parser.add_argument('--shortcut_step', type=int, default=-1, help='Build a shortcut from this step to step 0.')
    parser.add_argument('--iters', type=int, default=10, help='Number of optimization iterations')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for optimization')
    parser.add_argument('--gamma1', type=float, default=0.1, help='Weight for latent difference')
    parser.add_argument('--gamma2', type=float, default=1e5, help='Weight for semantic difference')
    parser.add_argument('--gamma3', type=float, default=1e-3, help='Weight for image difference')
    parser.add_argument('--eps', nargs='+', type=float, default=[10, 15], help='One bound for all selected steps, or one bound per --k value')
    parser.add_argument('--k', nargs='+', type=int, default=[25, 45], help='Selected timesteps')
    parser.add_argument('--watermark_text', type=str, default='test', help='Legacy option; watermark embedding/detection is disabled in this runner')
    parser.add_argument('--watermark_method', default='dwtDctSvd', help='Watermarking method to use (e.g., "dwtDctSvd", "rivaGan")')
    parser.add_argument('--gen_seed', type=int, default=0, help='Seed for random generation of images')
    parser.add_argument('--decode_inv', action='store_true', help='Learn the VAE encoding by regression')
    parser.add_argument('--dry-run', action='store_true', help='Validate arguments and list inputs without loading models or writing outputs')
    return parser


def parse_args(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    folder = Path(args.target_folder)
    if not folder.is_dir():
        parser.error('--target_folder must be an existing directory')
    if args.start < 0 or args.end <= args.start:
        parser.error('require 0 <= --start < --end (end is exclusive)')
    if args.iters < 1 or args.num_images < 1:
        parser.error('--iters and --num_images must be positive')
    if args.gpu < 0:
        parser.error('--gpu must be nonnegative')
    if args.gen_seed < 0 or args.gen_seed >= 2**32 - 5:
        parser.error('--gen_seed must be between 0 and 2**32 - 6')
    if args.image_length < 8 or args.image_length % 8:
        parser.error('--image_length must be a positive multiple of 8')
    steps = args.attack_num_inference_steps
    if steps < 2 or steps > 1000 or not 0 <= args.start_step < steps:
        parser.error('require 2 <= --attack_num_inference_steps <= 1000 and 0 <= --start_step < steps')
    if any(k < max(1, args.start_step) or k >= steps for k in args.k):
        parser.error('--k must contain step indices >= max(1, start_step) and < attack_num_inference_steps')
    if len(set(args.k)) != len(args.k):
        parser.error('--k must contain distinct step indices')
    if args.shortcut_step != -1 and not max(args.k) <= args.shortcut_step < steps:
        parser.error('--shortcut_step must be -1 or >= every --k value and < attack_num_inference_steps')
    if len(args.eps) == 1:
        args.eps = args.eps * len(args.k)
    if len(args.eps) != len(args.k):
        parser.error('--eps requires one value or exactly one value per --k step')
    if any(not math.isfinite(e) or e < 0 for e in args.eps):
        parser.error('--eps values must be finite and nonnegative')
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error('--lr must be finite and positive')
    if not math.isfinite(args.guidance_scale) or args.guidance_scale <= 1:
        parser.error("--guidance_scale must be finite and > 1 for this runner's paired text embeddings")
    if any(not math.isfinite(g) or g < 0 for g in (args.gamma1, args.gamma2, args.gamma3)):
        parser.error('--gamma1, --gamma2 and --gamma3 must be finite and nonnegative')
    suffixes = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    images = sorted(str(p) for p in folder.iterdir() if p.is_file() and p.suffix.lower() in suffixes)
    images = images[args.start:args.end]
    if not images:
        parser.error('no supported images in the selected range')
    if args.gen_seed + 10 * (args.num_images - 1) + len(images) - 1 + 5 >= 2**32:
        parser.error('generated seeds exceed the NumPy seed range; reduce --gen_seed or --num_images')
    if args.dry_run:
        print(json.dumps({'arguments': vars(args), 'inputs': images,
                          'note': 'Configuration check only; no model loaded, images decoded or results generated.'}, indent=2))
        parser.exit()
    output = Path(args.output_folder)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error('--output_folder must be empty or new to avoid mixing or overwriting experiment results')
    return args, images
