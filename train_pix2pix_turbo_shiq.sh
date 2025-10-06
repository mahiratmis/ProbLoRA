#!/bin/bash

clear

# Define the output directory, appending a timestamp to ensure uniqueness
# This helps prevent overwriting previous outputs and aids in tracking runs
WANDB_PROJECT_NAME="pix2pix_turbo_shiq_lr4e4_modifiedlora_msl1_perceptual_conv2d_rank16_8"
# OUTPUT_DIR="outputs_pix2pix/$WANDB_PROJECT_NAME$(date +'%Y-%m-%d_%H-%M-%S')"
OUTPUT_DIR="outputs_pix2pix/$WANDB_PROJECT_NAME"
mkdir -p "$OUTPUT_DIR"  # Create the output directory if it does not already exist

# Copy necessary scripts and source files to the output directory for reference
# This ensures that the current version of the scripts is preserved with the output
cp -r my_config_pix2pix.sh inference_paired_shiq.sh train_pix2pix_turbo_shiq.sh src "$OUTPUT_DIR"

# Export an environment variable required for certain configurations of distributed training
# `NCCL_P2P_DISABLE=1` disables peer-to-peer communication, which might be necessary on certain hardware setups
export NCCL_P2P_DISABLE=1
export TORCH_DISTRIBUTED_DEBUG="DETAIL"

accelerate launch --main_process_port 28502 --config_file my_config_pix2pix.sh \
    src/train_pix2pix_turbo_shiq.py \
    --pretrained_model_name_or_path="stabilityai/sd-turbo" \
    --output_dir="$OUTPUT_DIR" \
    --gradient_accumulation_steps=1 \
    --train_batch_size=4 \
    --num_training_epochs=25000 \
    --max_train_steps=500000 \
    --enable_xformers_memory_efficient_attention --viz_freq 25 \
    --report_to "wandb" --tracker_project_name "$WANDB_PROJECT_NAME" \
    --patch_size=512 \
    --learning_rate=4e-4