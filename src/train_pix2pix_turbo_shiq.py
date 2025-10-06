import os
import gc
import numpy as np
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm.auto import tqdm
import wandb

import diffusers
from diffusers.utils.import_utils import is_xformers_available
from diffusers.optimization import get_scheduler


from pix2pix_turbo import Pix2Pix_Turbo
from my_utils.training_utils import parse_args_paired_training, SHIQData
from data_transform import ToTensor, Compose, RandomHorizontalFlip

from losses import CombinedLoss

def main(args):
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
    )

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.output_dir, "eval"), exist_ok=True)
    # print("args.lora_rank_unet, args.lora_rank_vae", args.lora_rank_unet, args.lora_rank_vae)
    if args.pretrained_model_name_or_path == "stabilityai/sd-turbo":
        net_pix2pix = Pix2Pix_Turbo(lora_rank_unet=args.lora_rank_unet, lora_rank_vae=args.lora_rank_vae)
        net_pix2pix.set_train()
    else:
        net_pix2pix = Pix2Pix_Turbo(pretrained_path=args.pretrained_model_name_or_path, lora_rank_unet=args.lora_rank_unet, lora_rank_vae=args.lora_rank_vae)
        net_pix2pix.set_train()        

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            net_pix2pix.unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available, please install it by running `pip install xformers`")

    if args.gradient_checkpointing:
        net_pix2pix.unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # make the optimizer
    layers_to_opt = []
    for n, _p in net_pix2pix.unet.named_parameters():
        if "lora" in n:
            assert _p.requires_grad
            layers_to_opt.append(_p)
    layers_to_opt += list(net_pix2pix.unet.conv_in.parameters())
    for n, _p in net_pix2pix.vae.named_parameters():
        if "lora" in n and "vae_skip" in n:
            assert _p.requires_grad
            layers_to_opt.append(_p)
    layers_to_opt = layers_to_opt + list(net_pix2pix.vae.decoder.skip_conv_1.parameters()) + \
        list(net_pix2pix.vae.decoder.skip_conv_2.parameters()) + \
        list(net_pix2pix.vae.decoder.skip_conv_3.parameters()) + \
        list(net_pix2pix.vae.decoder.skip_conv_4.parameters())
    
    # print("Learning rate: ", args.learning_rate)

    optimizer = torch.optim.AdamW(layers_to_opt, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,)
    lr_scheduler = get_scheduler(args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power,)

    transforms_train = Compose([
        RandomHorizontalFlip(),
        ToTensor()
    ])

    transforms_test = Compose([
        ToTensor()
    ])

    dataset_train = SHIQData(root="../datasets/SHIQ_data_10825/train/", transform=transforms_train, tokenizer=net_pix2pix.tokenizer)
    dl_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers)
    dataset_val = SHIQData(root="../datasets/SHIQ_data_10825/test/", transform=transforms_test, tokenizer=net_pix2pix.tokenizer)
    dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=args.dataloader_num_workers)

    # Prepare everything with our `accelerator`.
    net_pix2pix, optimizer, dl_train, lr_scheduler = accelerator.prepare(
        net_pix2pix, optimizer, dl_train, lr_scheduler
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move al networksr to device and cast to weight_dtype
    net_pix2pix.to(accelerator.device, dtype=weight_dtype)

    criterion = CombinedLoss(
        max_val=1.0,
        psnr_min=10.0,
        psnr_max=50.0,
        lambda_pix=1.0,
        lambda_ssim=0.0,
        lambda_perceptual=1.0,
        lambda_lpips=0.0,
        lambda_psnr=0.0,
        lambda_l2=0.0        
    )     

    # Prepare criterion
    criterion = accelerator.prepare(criterion)    

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    progress_bar = tqdm(range(0, args.max_train_steps), initial=0, desc="Steps",
        disable=not accelerator.is_local_main_process,)

    # start the training loop
    global_step = 0
    for epoch in range(0, args.num_training_epochs):
        for step, batch in enumerate(dl_train):
            l_acc = [net_pix2pix]
            with accelerator.accumulate(*l_acc):
                x_src = batch["conditioning_pixel_values"]
                x_tgt = batch["output_pixel_values"]
                B, C, H, W = x_src.shape
                # print("x_src.shape , x_tgt.shape", x_src.shape, x_tgt.shape)
                # print('batch["input_ids"].shape', batch["input_ids"].shape)


                # forward pass
                x_tgt_pred = net_pix2pix(x_src, prompt_tokens=batch["input_ids"], deterministic=True)
                total_loss, loss_dict = criterion(x_tgt_pred, x_tgt)
                accelerator.backward(total_loss, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(layers_to_opt, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    logs = {}
                    # log all the losses
                    logs["loss_pixel_loss"] = loss_dict["pixel_loss"]
                    logs["loss_lpips"] = loss_dict["lpips_loss"]
                    logs["loss_ssim"] = loss_dict["ssim_loss"]
                    logs["loss_perceptual"] = loss_dict["perceptual_loss"]
                    logs["loss_psnr"] = loss_dict["psnr_loss"]  
                    logs["loss_l2"] = loss_dict["l2_loss"]                  

                    progress_bar.set_postfix(**logs)

                    # viz some images
                    if global_step % args.viz_freq == 1:
                        log_dict = {
                            "train/source": [wandb.Image(x_src[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(B)],
                            "train/target": [wandb.Image(x_tgt[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(B)],
                            "train/model_output": [wandb.Image(x_tgt_pred[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(B)],
                        }
                        for k in log_dict:
                            logs[k] = log_dict[k]

                    # checkpoint the model
                    if global_step % args.checkpointing_steps == 1:
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        accelerator.unwrap_model(net_pix2pix).save_model(outf)

                    # compute validation set scores
                    if global_step % args.eval_freq == 1:

                        print("Evaluating on validation set...")

                        l_pixel, l_lpips, l_perceptual, l_ssim, lpsnr, l_l2 = [], [], [], [], [], []
                        if args.track_val_fid:
                            os.makedirs(os.path.join(args.output_dir, "eval", f"fid_{global_step}"), exist_ok=True)
                        for step, batch_val in enumerate(dl_val):
                            if step >= args.num_samples_eval:
                                break
                            x_src = batch_val["conditioning_pixel_values"].cuda()
                            x_tgt = batch_val["output_pixel_values"].cuda()
                            B, C, H, W = x_src.shape
                            assert B == 1, "Use batch size 1 for eval."
                            with torch.no_grad():
                                # forward pass
                                x_tgt_pred = accelerator.unwrap_model(net_pix2pix)(x_src, prompt_tokens=batch_val["input_ids"].cuda(), deterministic=True)
                                total_loss, loss_dict = criterion(x_tgt_pred, x_tgt)                                

                                l_pixel.append(loss_dict["pixel_loss"])
                                l_lpips.append(loss_dict["lpips_loss"])
                                l_ssim.append(loss_dict["ssim_loss"])
                                l_perceptual.append(loss_dict["perceptual_loss"])
                                lpsnr.append(loss_dict["psnr_loss"])
                                l_l2.append(loss_dict["l2_loss"])

                        logs["val/lpixel"] = np.mean(l_pixel)
                        logs["val/lpips"] = np.mean(l_lpips)                     
                        logs["val/perceptual"] = np.mean(l_perceptual)
                        logs["val/ssim"] = np.mean(l_ssim)
                        logs["val/psnr"] = np.mean(lpsnr)   
                        logs["val/l2"] = np.mean(l_l2)                       
                        gc.collect()
                        torch.cuda.empty_cache()
                    accelerator.log(logs, step=global_step)


if __name__ == "__main__":
    args = parse_args_paired_training()
    main(args)
