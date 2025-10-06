import os
import requests
from tqdm import tqdm
from diffusers import DDPMScheduler
import torch


def make_1step_sched():
    noise_scheduler_1step = DDPMScheduler.from_pretrained("stabilityai/sd-turbo", subfolder="scheduler")
    noise_scheduler_1step.set_timesteps(1, device="cuda")
    noise_scheduler_1step.alphas_cumprod = noise_scheduler_1step.alphas_cumprod.cuda()
    return noise_scheduler_1step


def my_vae_encoder_fwd(self, sample):
    # print("Sample before conv_in", sample.shape)
    sample = self.conv_in(sample)
    # print("Sample after conv_in", sample.shape)
    l_blocks = []
    # down
    for down_block in self.down_blocks:
        l_blocks.append(sample)
        sample = down_block(sample)
        # print("Sample after downblock", sample.shape)
    # middle
    sample = self.mid_block(sample)
    # print("Sample after mid", sample.shape)
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)
    # print("Sample after conv out", sample.shape)
    self.current_down_blocks = l_blocks
    return sample


def my_vae_decoder_fwd_pvt(self, sample, latent_embeds=None):
    sample = self.conv_in(sample)
    upscale_dtype = next(iter(self.up_blocks.parameters())).dtype

    # **Middle Block Processing**
    sample = self.mid_block(sample, latent_embeds)
    sample = sample.to(upscale_dtype)
    print("sample.shape", sample.shape)

    if not self.ignore_skip:
        skip_convs = [self.skip_conv_1, self.skip_conv_2, self.skip_conv_3, self.skip_conv_4]

        # **Upsample factors to match decoder**
        upscale_sizes = [64, 128, 256, 512]  # Target sizes (assuming 512 final resolution)

        for scale in self.incoming_skip_acts[::-1]:
            print("scale.shape", scale.shape)

        # **Iterate Over Decoder Blocks**
        for idx, up_block in enumerate(self.up_blocks):
            skip_in = skip_convs[idx](self.incoming_skip_acts[::-1][idx] * self.gamma)  # Process skip feature
            
            # **Ensure skip_in matches sample's spatial size**
            target_size = (upscale_sizes[idx], upscale_sizes[idx])
            skip_in = torch.nn.functional.interpolate(skip_in, size=target_size, mode="bilinear", align_corners=True)
            
            print(f"Upsampled skip {idx}: {skip_in.shape}, Target: {target_size}")

            # **Merge Skip Connection & Decode**
            sample = sample + skip_in
            sample = up_block(sample, latent_embeds)
    else:
        for idx, up_block in enumerate(self.up_blocks):
            sample = up_block(sample, latent_embeds)

    # **Post-processing**
    if latent_embeds is None:
        sample = self.conv_norm_out(sample)
    else:
        sample = self.conv_norm_out(sample, latent_embeds)

    sample = self.conv_act(sample)
    sample = self.conv_out(sample)

    return sample

def my_vae_decoder_fwd(self, sample, latent_embeds=None):
    # print("DECODER Sample before conv_in", sample.shape)
    
    sample = self.conv_in(sample)
    # print("DECODER Sample after conv_in", sample.shape)
    upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
    # middle
    sample = self.mid_block(sample, latent_embeds)
    # print("DECODER Sample after mid_block", sample.shape)
    sample = sample.to(upscale_dtype)
    if not self.ignore_skip:
        skip_convs = [self.skip_conv_1, self.skip_conv_2, self.skip_conv_3, self.skip_conv_4]
        # up
        # print("len incoming skip acts", len(self.incoming_skip_acts))
        for idx, up_block in enumerate(self.up_blocks):
            # print(f"self.incoming_skip_acts[::-1][{idx}].shape {self.incoming_skip_acts[::-1][idx].shape}")
            skip_in = skip_convs[idx](self.incoming_skip_acts[::-1][idx] * self.gamma)
            # add skip
            sample = sample + skip_in
            sample = up_block(sample, latent_embeds)
            # print("DECODER Sample after up_block skip", sample.shape)
    else:
        for idx, up_block in enumerate(self.up_blocks):
            sample = up_block(sample, latent_embeds)
    # post-process
    if latent_embeds is None:
        sample = self.conv_norm_out(sample)
    else:
        sample = self.conv_norm_out(sample, latent_embeds)
    sample = self.conv_act(sample)
    # print("DECODER Sample before conv_out skip", sample.shape)
    sample = self.conv_out(sample)
    # before upsampling layer
    # sample = self.conv_act(sample)
    # print("DECODER Sample after conv_out skip", sample.shape)
    # _, _, h, w = sample.shape
    # sample = torch.nn.functional.interpolate(sample, size=(h*4, w*4), mode="bilinear", align_corners=True)
    return sample

def download_url(url, outf):
    if not os.path.exists(outf):
        print(f"Downloading checkpoint to {outf}")
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
        with open(outf, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR, something went wrong")
        print(f"Downloaded successfully to {outf}")
    else:
        print(f"Skipping download, {outf} already exists")
