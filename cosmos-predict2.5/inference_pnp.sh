####################################
# Base Video-to-World Inference
####################################
python examples/inference.py \
    -i assets/base/robot_013_pnp.json \
    -o outputs/base_video2world_robot_013 \
    --disable-guardrails

####################################
# P&P Inference
####################################
python examples/inference_pnp.py \
    -i assets/base/robot_013_pnp.json \
    -o outputs/base_video2world_robot_013_pnp \
    --disable-guardrails