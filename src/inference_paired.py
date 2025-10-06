import os
import argparse
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torchvision.transforms.functional as F
from pix2pix_turbo_burstormer import Pix2Pix_Turbo
from image_prep import canny_from_pil

import h5py

from glob import glob

def numpy_to_torch(image):
    #print("image.shape : ", image.shape)
    image = image.transpose((0, 3, 1, 2))
    torch_tensor = torch.from_numpy(image.copy())
    return torch_tensor

def reduce_channels(tensor):
    """
    Reduce a tensor of shape (C, H, W) where C=4 to (3, H, W) by averaging 
    the second and fourth channels.
    
    Args:
        tensor (torch.Tensor): Input tensor of shape (4, H, W).
        
    Returns:
        torch.Tensor: Output tensor of shape (3, H, W).
    """
    if tensor.size(0) != 4:
        raise ValueError("Input tensor must have 4 channels in the first dimension.")
    
    # Compute the averaged second and fourth channels
    tensor[1] = (tensor[1] + tensor[3]) / 2
    
    # Select the desired channels (0, 1, 2) directly without extra concatenation
    return tensor[:3]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image', type=str, required=True, help='path to the input image')
    parser.add_argument('--prompt', type=str, required=True, default="from dark image to daytime image" ,help='the prompt to be used')
    parser.add_argument('--model_name', type=str, default='', help='name of the pretrained model to be used')
    parser.add_argument('--model_path', type=str, default='', help='path to a model state dict to be used')
    parser.add_argument('--output_dir', type=str, default='output', help='the directory to save the output')
    parser.add_argument('--low_threshold', type=int, default=100, help='Canny low threshold')
    parser.add_argument('--high_threshold', type=int, default=200, help='Canny high threshold')
    parser.add_argument('--gamma', type=float, default=0.4, help='The sketch interpolation guidance amount')
    parser.add_argument('--seed', type=int, default=42, help='Random seed to be used')
    parser.add_argument('--use_fp16', action='store_true', help='Use Float16 precision for faster inference')
    # dark data
    parser.add_argument('--gt_dir', type=str, required=True, default="../datasets/Sony/long/", help='path to the ground truth images')    
    parser.add_argument('--input_dir', type=str, required=True, default="../datasets/Sony/short/", help='path to the input images') 
    parser.add_argument('--img_id', type=int, required=True, default=10003, help='id of the image to be processed')
    parser.add_argument('--patch_size', type=int, default=0, required=True, help='size of patches')     
    args = parser.parse_args()

    # only one of model_name and model_path should be provided
    if args.model_name == '' != args.model_path == '':
        raise ValueError('Either model_name or model_path should be provided')

    os.makedirs(args.output_dir, exist_ok=True)

    # initialize the model
    model = Pix2Pix_Turbo(pretrained_name=args.model_name, pretrained_path=args.model_path)
    # model.to('cpu')
    # model.unet.to('cpu')
    # model.vae.to('cpu')
    # model.text_encoder.to('cpu')
    # model.timesteps.to('cpu')
    model.set_eval()
    if args.use_fp16:
        model.half()

    n_burst = 4
    hdf5_path = '../datasets/Sony/1_dataset.h5'
    hdf5_file = h5py.File(hdf5_path, 'r')

    # Access the HDF5 groups
    # gt_group = hdf5_file['gt']
    input_group = hdf5_file['input']   
    img_id = str(args.img_id)
    # gt_image = gt_group[img_id][...] 
    # Get input images
    input_subgroup = input_group[img_id]
    input_keys = list(input_subgroup.keys())
    input_indices = np.random.choice(
        range(len(input_keys)), n_burst, replace=len(input_keys) < n_burst)
    input_images = np.array([input_subgroup[str(i)][...] for i in input_indices])

    hdf5_file.close()
    del input_subgroup

    if args.patch_size == 0:
        input_patches = input_images
        # gt_patch = gt_image
    else:
        raw_ratio = 2
        ps=args.patch_size 
        _, _, H, W, _ = input_images.shape
        xx = np.random.randint(0, W - ps)
        yy = np.random.randint(0, H - ps)
        input_patches = input_images[:, :, yy:yy + ps, xx:xx + ps, :]
        # gt_patch = gt_image[:, yy*raw_ratio:yy*raw_ratio +ps*raw_ratio, xx*raw_ratio:xx*raw_ratio +ps*raw_ratio, :] 
        # gt_patch = torch.nn.functional.interpolate(
        #             torch.from_numpy(gt_patch).permute(0, 3, 1, 2),
        #             size=(args.patch_size, args.patch_size),
        #             mode='bilinear',
        #             align_corners=True
        #         ).permute(0, 2, 3, 1).numpy()  

   
    input_patches = np.squeeze(input_patches, axis=1)                   
    
    input_patches = numpy_to_torch(input_patches)        
    # gt_patch = numpy_to_torch(gt_patch) 
    # Apply max pooling to simulate burst aggregation
    input_patches, _ = torch.max(input_patches, dim=0, keepdim=True) # 1,C,H,W where C=4
    # convert input to rgb
    input_patches = reduce_channels(input_patches.squeeze(0))
    input_image = input_patches.unsqueeze(0).cuda()  

    # make sure that the input image is a multiple of 8
    # input_image = Image.open(args.input_image).convert('RGB')
    # new_width = input_image.width - input_image.width % 8
    # new_height = input_image.height - input_image.height % 8
    # input_image = input_image.resize((new_width, new_height), Image.LANCZOS)
    # bname = os.path.basename(args.input_image)

    # translate the image
    with torch.no_grad():
        # if args.model_name == 'edge_to_image':
        #     canny = canny_from_pil(input_image, args.low_threshold, args.high_threshold)
        #     canny_viz_inv = Image.fromarray(255 - np.array(canny))
        #     canny_viz_inv.save(os.path.join(args.output_dir, bname.replace('.png', '_canny.png')))
        #     c_t = F.to_tensor(canny).unsqueeze(0).cuda()
        #     if args.use_fp16:
        #         c_t = c_t.half()
        #     output_image = model(c_t, args.prompt)

        # elif args.model_name == 'sketch_to_image_stochastic':
        #     image_t = F.to_tensor(input_image) < 0.5
        #     c_t = image_t.unsqueeze(0).cuda().float()
        #     torch.manual_seed(args.seed)
        #     B, C, H, W = c_t.shape
        #     noise = torch.randn((1, 4, H // 8, W // 8), device=c_t.device)
        #     if args.use_fp16:
        #         c_t = c_t.half()
        #         noise = noise.half()
        #     output_image = model(c_t, args.prompt, deterministic=False, r=args.gamma, noise_map=noise)

        # else:
        c_t = input_image
        if args.use_fp16:
            c_t = c_t.half()
        output_image = model(c_t, args.prompt)

        output_pil = transforms.ToPILImage()(output_image[0].cpu() * 0.5 + 0.5)

    # save the output image
    os.makedirs(args.output_dir, exist_ok=True)
    output_pil.save(os.path.join(args.output_dir, f"{args.img_id}.png"))
