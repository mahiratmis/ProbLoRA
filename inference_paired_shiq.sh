# clear screen
clear

checkpoint_number="84001"
runname="shiq_lr4e4_problora_msl1_perceptual"
run_dir="outputs_pix2pix/$runname"
out_dir="$run_dir/$checkpoint_number/preds_fid"
model_path="$run_dir/checkpoints/model_$checkpoint_number.pkl"
prompt="remove specular highlight from image"

mkdir -p $out_dir
export CUDA_VISIBLE_DEVICES=0
python src/inference_paired_shiq.py --model_path "$model_path" \
    --output_dir "$out_dir" \
    --prompt="$prompt" > $out_dir/0log.txt

echo "All done $runname" 
