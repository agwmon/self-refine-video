import os

import torch
from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video

from functions.pipeline_wan_pnp import WanPipeline


MODEL_ID = "Wan2.2-T2V-A14B-Diffusers"
DTYPE = torch.bfloat16
VAE_DTYPE = torch.float32
DEVICE = "cuda"

PROMPT = (
    "A gymnast on a pommel horse swings their legs in wide circles (flares), "
    "supporting their entire weight on alternating hands."
)
# Base negative prompt in Wan2.2 (https://github.com/Wan-Video/Wan2.2)
NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
    "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
    "杂乱的背景，三条腿，背景人很多，倒着走"
)

SEED = 50
SAVE_DIR = "output/test"

HEIGHT, WIDTH = 480, 832
NUM_FRAMES = 81
GUIDANCE_SCALE = 4
GUIDANCE_SCALE_2 = 3
NUM_INFERENCE_STEPS = 40
FPS = 16

# Hyperparameters for P&P.
# List format: [(2, 5, 3), (6, 14, 1)] is also allowed.
STOCHASTIC_PLAN = [
    {"start": 2, "end": 5, "steps": 3},
    {"start": 6, "end": 14, "steps": 1},
]

# Threshold for uncertainty to determine certain and uncertain regions.
# Lower values increase refinement strength, but may introduce artifacts.
THS_UNCERTAINTY = 0.25
P_NORM = 1
CERTAIN_PERCENTAGE = 0.999  # If certain area percentage is larger, skip P&P iterations.


def build_pipeline() -> WanPipeline:
    vae = AutoencoderKLWan.from_pretrained(
        MODEL_ID,
        subfolder="vae",
        torch_dtype=VAE_DTYPE,
    )
    pipe = WanPipeline.from_pretrained(MODEL_ID, vae=vae, torch_dtype=DTYPE)
    return pipe.to(DEVICE)


def main() -> None:
    os.makedirs(SAVE_DIR, exist_ok=True)

    pipe = build_pipeline()
    output = pipe(
        stochastic_plan=STOCHASTIC_PLAN,
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
        guidance_scale=GUIDANCE_SCALE,
        guidance_scale_2=GUIDANCE_SCALE_2,
        num_inference_steps=NUM_INFERENCE_STEPS,
        generator=torch.Generator(device=DEVICE).manual_seed(SEED),
        ths_uncertainty=THS_UNCERTAINTY,
        p_norm=P_NORM,
        certain_percentage=CERTAIN_PERCENTAGE,
    ).frames[0]

    export_to_video(output, f"{SAVE_DIR}/output.mp4", fps=FPS)


if __name__ == "__main__":
    main()