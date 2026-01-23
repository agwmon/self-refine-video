from importlib import metadata
import os

import torch
from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video, load_image
import json
from functions.pipeline_wan_i2v_pnp import WanImageToVideoPnPPipeline


MODEL_ID = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
DTYPE = torch.bfloat16
VAE_DTYPE = torch.float32
DEVICE = "cuda"

IMAGE = load_image("assets/robot_050.jpg")
PROMPT = (
    "The video opens with a close-up shot of two robotic arms positioned on either side of a metallic refrigerator in a clean, modern indoor setting, likely a laboratory or testing facility. Both arms are visible, showcasing their intricate black, silver, and white components, with cylindrical attachments. The refrigerator has a sleek, silver finish with a slightly reflective surface, situated against a corner of a white wall. A small section of a stainless steel sink is visible on the left side of the frame, indicating proximity to a kitchen or utility area. The floor beneath the refrigerator is covered with light-colored tiles, and a faint yellow line marking is noticeable near the bottom right corner of the image. Several people's legs are visible on the floor towards the right side while walking.  The left robotic arm opens the refrigerator's door, revealing its contents. It then retrieves a cold drink bottle and places it on the counter beside the refrigerator. Simultaneously, the right robotic arm gently pushes the refrigerator door closed. The scene remains consistent, focusing on the interaction between the robotic arms and the refrigerator without introducing any new objects. The video captures the precision and efficiency of the robotic arms in a controlled, modern environment."
)

NEGATIVE_PROMPT = (
    "low quality"
)

SEED = 42
SAVE_DIR = "output/test2"

HEIGHT, WIDTH = 480, 832
NUM_FRAMES = 81
GUIDANCE_SCALE = 3.5
NUM_INFERENCE_STEPS = 40
FPS = 16

# Hyperparameters for P&P.
# List format: [(2, 5, 3), (6, 13, 1)] is also allowed.
STOCHASTIC_PLAN = [
    {"start": 1, "end": 5, "steps": 3},
    {"start": 6, "end": 13, "steps": 1},
]

# Threshold for uncertainty to determine certain and uncertain regions.
# Lower values increase refinement strength, but may introduce artifacts (difference in color tone).
THS_UNCERTAINTY = 0.0 # I2V does not cause over-saturation issue, so set to 0.
P_NORM = 1 # Fix
CERTAIN_PERCENTAGE = 0.999  # If certain area percentage is larger, skip P&P iterations.

def _build_stochastic_step_map(plan):
    step_map = {}
    if not plan:
        return step_map
    for entry in plan:
        if isinstance(entry, dict):
            start = entry.get("start", entry.get("begin"))
            end = entry.get("end", entry.get("stop"))
            steps = entry.get("steps", entry.get("anneal", entry.get("num_anneal_steps", 1)))
            if start is None or end is None:
                raise ValueError("stochastic_plan dict entries must contain 'start' and 'end' keys.")
        else:
            if len(entry) != 3:
                raise ValueError("Tuple entries in stochastic_plan must be of the form (start, end, num_anneal_steps).")
            start, end, steps = entry

        start_i = int(start)
        end_i = int(end)
        steps_i = int(steps)
        if start_i < 0 or end_i < 0:
            raise ValueError("stochastic_plan indices must be non-negative.")
        if end_i < start_i:
            raise ValueError(f"stochastic_plan end ({end_i}) must be >= start ({start_i}).")
        if steps_i < 1:
            continue

        for idx in range(start_i, end_i + 1):
            step_map[idx] = steps_i

    return step_map


def _compute_total_nfe(num_inference_steps, stochastic_plan):
    step_map = _build_stochastic_step_map(stochastic_plan)
    extra = sum(step_map.get(i, 0) for i in range(num_inference_steps))
    return num_inference_steps + extra


def build_pipeline() -> WanImageToVideoPnPPipeline:
    vae = AutoencoderKLWan.from_pretrained(
        MODEL_ID,
        subfolder="vae",
        torch_dtype=VAE_DTYPE,
    )
    pipe = WanImageToVideoPnPPipeline.from_pretrained(MODEL_ID, vae=vae, torch_dtype=DTYPE)
    return pipe.to(DEVICE)

def main() -> None:
    os.makedirs(SAVE_DIR, exist_ok=True)

    total_nfe = _compute_total_nfe(NUM_INFERENCE_STEPS, STOCHASTIC_PLAN)
    print(f"Total NFE: {total_nfe} (base {NUM_INFERENCE_STEPS} + extra {total_nfe - NUM_INFERENCE_STEPS})")

    pipe = build_pipeline()

    metadata = {
        "prompt": PROMPT,
        "negative_prompt": NEGATIVE_PROMPT,
        "height": HEIGHT,
        "width": WIDTH,
        "num_frames": NUM_FRAMES,
        "guidance_scale": GUIDANCE_SCALE,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "stochastic_plan": STOCHASTIC_PLAN,
        "ths_uncertainty": THS_UNCERTAINTY,
        "p_norm": P_NORM,
        "certain_percentage": CERTAIN_PERCENTAGE,
    }

    with open(os.path.join(SAVE_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    output = pipe(
        stochastic_plan=STOCHASTIC_PLAN,
        image=IMAGE,
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        height=HEIGHT,
        width=WIDTH,
        num_frames=NUM_FRAMES,
        guidance_scale=GUIDANCE_SCALE,
        num_inference_steps=NUM_INFERENCE_STEPS,
        generator=torch.Generator(device=DEVICE).manual_seed(SEED),
        ths_uncertainty=THS_UNCERTAINTY,
        p_norm=P_NORM,
        certain_percentage=CERTAIN_PERCENTAGE,
    ).frames[0]

    export_to_video(output, f"{SAVE_DIR}/output_pnp.mp4", fps=FPS)

    ########## Comparison with Base Wan ##########
    # from diffusers import WanImageToVideoPipeline
    # pipe_base = WanImageToVideoPipeline.from_pretrained(MODEL_ID, 
    #                                         vae=pipe.vae,
    #                                         text_encoder=pipe.text_encoder,
    #                                         tokenizer=pipe.tokenizer,
    #                                         image_encoder=pipe.image_encoder,
    #                                         transformer=pipe.transformer,
    #                                         scheduler=pipe.scheduler,
    #                                         image_processor=pipe.image_processor,  
    #                                         transformer_2=pipe.transformer_2)
                                        
    # output = pipe_base(
    #     image=IMAGE,
    #     prompt=PROMPT,
    #     negative_prompt=NEGATIVE_PROMPT,
    #     height=HEIGHT,
    #     width=WIDTH,
    #     num_frames=NUM_FRAMES,
    #     guidance_scale=GUIDANCE_SCALE,
    #     num_inference_steps=NUM_INFERENCE_STEPS,
    #     generator=torch.Generator(device=DEVICE).manual_seed(SEED),
    # ).frames[0]

    # export_to_video(output, f"{SAVE_DIR}/output_base.mp4", fps=FPS)
    #######################################
                                            




if __name__ == "__main__":
    main()