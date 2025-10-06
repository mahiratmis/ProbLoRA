import os
import argparse
import numpy as np
import torch
from pix2pix_turbo import Pix2Pix_Turbo
from my_utils.training_utils import SSHRDataset, generate_sshr_testing_data_list, ImageTransform
from skimage.metrics import structural_similarity as compare_ssim

from torchvision.utils import save_image
from tqdm import tqdm

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

def psnr(pred, target, pixel_max_cnt = 1):
    mse = torch.mul(target - pred, target - pred)
    rmse_avg = (torch.mean(mse).item()) ** 0.5
    p = 20 * np.log10(pixel_max_cnt / rmse_avg)
    return p

def grey_psnr(pred, target, pixel_max_cnt = 255):
    pred = torch.sum(pred, dim = 0)
    target = torch.sum(target, dim = 0)
    mse = torch.mul(target - pred, target - pred)
    rmse_avg = (torch.mean(mse).item()) ** 0.5
    p = 20 * np.log10(pixel_max_cnt * 3 / rmse_avg)
    return p

def ssim(pred, target):
    pred = pred.clone().data.permute(0, 2, 3, 1).cpu().numpy()
    target = target.clone().data.permute(0, 2, 3, 1).cpu().numpy()
    target = target[0]
    pred = pred[0]
    # print("target.shape, pred.shape", target.shape, pred.shape) 
    ssim = compare_ssim(target, pred, channel_axis = 2, data_range = 1.0)
    return ssim

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, required=True, default="from dark image to daytime image" ,help='the prompt to be used')
    parser.add_argument('--model_name', type=str, default='', help='name of the pretrained model to be used')
    parser.add_argument('--model_path', type=str, default='', help='path to a model state dict to be used')
    parser.add_argument('--output_dir', type=str, default='output', help='the directory to save the output')
    parser.add_argument('--use_fp16', action='store_true', help='Use Float16 precision for faster inference')  
    args = parser.parse_args()

    # only one of model_name and model_path should be provided
    if args.model_name == '' != args.model_path == '':
        raise ValueError('Either model_name or model_path should be provided')

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # initialize the model
    model = Pix2Pix_Turbo(pretrained_name=args.model_name, pretrained_path=args.model_path)
    model.to(device)
    model.set_eval()
    if args.use_fp16:
        model.half()
   
   
    val_img_list = generate_sshr_testing_data_list(data_dir="../datasets", data_list_file="../datasets/SSHR/test_7_tuples.lst")
    dataset_val = SSHRDataset(img_list=val_img_list, img_transform=ImageTransform(), phase="test", tokenizer=model.tokenizer)
    dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=2)


    metrics_list = []
    #ssim = SSIM()
    #psnr = PSNR()
    it = 0
    # save the output image
    os.makedirs(args.output_dir, exist_ok=True)
    num_test_images = len(dl_val)

    with tqdm(total=len(dl_val)) as pbar:
        for data in dl_val:
            A = data['conditioning_pixel_values'].half() if args.use_fp16 else data['conditioning_pixel_values']
            D = data['output_pixel_values'].half() if args.use_fp16 else data['output_pixel_values']
            image_name = data['image_name'][0]
            
            A = A.to(device)
            D = D.to(device)
            D = D*0.5 + 0.5  # range 0 to 1
            with torch.no_grad():  
                Dpred = model(A, args.prompt)*0.5 + 0.5  # range 0 to 1
                # print(Dpred.shape, D.shape)
                ssim_val = ssim(Dpred, D)
                psnr_val = psnr(Dpred, D)
                mse_val = torch.mean((Dpred - D) ** 2)*100
                metrics_list.append([ssim_val, psnr_val, mse_val.detach().cpu().numpy()])
                it+=1
                # print(np.sum(np.asarray(metrics_list), 0)/it)
                save_image(Dpred, os.path.join(args.output_dir, image_name + "_D.png"))
            avg = np.sum(np.asarray(metrics_list), 0)/it
            pbar.set_description(f"Processed {image_name} SSIM: {avg[0]:.4f} PSNR: {avg[1]:.4f} MSE: {avg[2]:.4f}")
            pbar.update(1)
    avg = np.mean(np.asarray(metrics_list), 0)
    print(f"SSIM: {avg[0]:.4f} PSNR: {avg[1]:.4f} MSE: {avg[2]:.4f}")
