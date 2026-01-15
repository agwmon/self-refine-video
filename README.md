# self-refine-video

Video generation with P&P (Perturb-and-Refine) using Wan2.2 on Diffusers.

## Overview

This repo contains a minimal script to run Wan2.2 T2V with a P&P refinement schedule and export the result to MP4.

## Dependencies

Follow the model card for environment details:
- https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B-Diffusers

Minimum dependencies (install in your environment):
- diffusers
- transformers
- torch

## Usage

Edit prompts and hyperparameters in [inference_pnp.py](inference_pnp.py), then run the script to generate a video:

- Output video: `output/test/output.mp4`
