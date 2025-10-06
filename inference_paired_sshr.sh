# clear screen
clear

checkpoint_number="84001"
runname="pix2pix_turbo_sshr_defaultlora_msl1_perceptual"
run_dir="outputs_pix2pix/$runname"
out_dir="$run_dir/$checkpoint_number/preds"
model_path="$run_dir/checkpoints/model_$checkpoint_number.pkl"
prompt="remove specular highlight from image"

mkdir -p $out_dir
export CUDA_VISIBLE_DEVICES=1
python src/inference_paired_sshr.py --model_path "$model_path" \
    --output_dir "$out_dir" \
    --prompt="$prompt" > $out_dir/0log.txt

echo "All done $runname"