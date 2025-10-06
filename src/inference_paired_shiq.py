import os
import argparse
import numpy as np
import torch
from pix2pix_turbo import Pix2Pix_Turbo
from my_utils.training_utils import SHIQData
from data_transform import ToTensor, Compose
from skimage.metrics import structural_similarity as compare_ssim
from tqdm import tqdm
import lpips                             
from torchmetrics.image.fid import FrechetInceptionDistance

from torchvision.utils import save_image


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

def psnr(pred, target, pixel_max_cnt=1):
    """
    Compute PSNR between pred and target images.
    Assumes input tensors are in the [0,1] range.
    """
    mse_val = torch.mean((target - pred) ** 2).item()
    rmse_avg = mse_val ** 0.5
    p = 20 * np.log10(pixel_max_cnt / rmse_avg)
    return p

def ssim(pred, target):
    """
    Compute SSIM between pred and target images.
    Expects tensors with shape [batch, channels, height, width].
    """
    # Rearrange tensors to [batch, height, width, channels]
    pred_np = pred.clone().data.permute(0, 2, 3, 1).cpu().numpy()
    target_np = target.clone().data.permute(0, 2, 3, 1).cpu().numpy()
    pred_np = pred_np[0]
    target_np = target_np[0]
    ssim_val = compare_ssim(target_np, pred_np, channel_axis=2, data_range=1.0)
    return ssim_val

def lpips_score(pred: torch.Tensor, target: torch.Tensor, lpips_fn) -> float:
    """LPIPS between two batches, expects inputs in [-1,1]."""
    return lpips_fn(pred, target).item()

def local_mse(pred: torch.Tensor,
              gt: torch.Tensor,
              mask: torch.Tensor,
              reduction: str = 'mean',
              eps: float = 1e-8) -> torch.Tensor:
    """
    Compute local MSE over masked regions.

    Args:
        pred (Tensor): Predicted tensor, shape (B, C, H, W).
        gt   (Tensor): Ground-truth tensor, same shape as pred.
        mask (Tensor): Binary mask, shape (B, 1, H, W) or broadcastable to pred.
        reduction (str): 'mean', 'sum', or 'none'.
        eps (float): Small value to avoid division by zero.

    Returns:
        Tensor: Scalar loss if reduction is 'mean' or 'sum'; 
                else per-element squared error tensor if 'none'.
    """
    # 1) Broadcast mask to match pred/gt channels
    if mask.dim() == pred.dim() - 1:
        # mask is (B, H, W) → add channel dim
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(pred)  # now (B, C, H, W)

    # 2) Apply mask
    masked_pred = pred * mask
    masked_gt   = gt   * mask

    # 3) Compute squared error
    sq_err = (masked_pred - masked_gt).pow(2)

    if reduction == 'none':
        return sq_err

    # 4) Sum over all dimensions
    total_error = sq_err.sum()

    if reduction == 'sum':
        return total_error

    # 5) 'mean': divide by number of masked elements
    num_pixels = mask.sum()
    return total_error / (num_pixels + eps)

def print_overall_metrics(metrics):
    # overall
    arr = np.array(metrics)
    p_avg, s_avg, l_avg, m_avg, lm_avg = arr.mean(axis=0)
    print("\nAverages".ljust(20), 
          f"PSNR: {p_avg:.4f}   SSIM: {s_avg:.4f}   LPIPS: {l_avg:.4f} MSE: {m_avg:.4f} LMSE: {lm_avg:.4f}")
    print("-"*60)  


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt', type=str, required=True, default="from dark image to daytime image" ,help='the prompt to be used')
    parser.add_argument('--model_name', type=str, default='', help='name of the pretrained model to be used')
    parser.add_argument('--model_path', type=str, default='', help='path to a model state dict to be used')
    parser.add_argument('--output_dir', type=str, default='output', help='the directory to save the output')
    parser.add_argument('--low_threshold', type=int, default=100, help='Canny low threshold')
    parser.add_argument('--high_threshold', type=int, default=200, help='Canny high threshold')
    parser.add_argument('--gamma', type=float, default=0.4, help='The sketch interpolation guidance amount')
    parser.add_argument('--seed', type=int, default=42, help='Random seed to be used')
    parser.add_argument('--use_fp16', action='store_true', help='Use Float16 precision for faster inference')
   
    args = parser.parse_args()

    # only one of model_name and model_path should be provided
    if args.model_name == '' != args.model_path == '':
        raise ValueError('Either model_name or model_path should be provided')

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # initialize the model
    model = Pix2Pix_Turbo(pretrained_name=args.model_name, pretrained_path=args.model_path)
    # model.to('cpu')
    # model.unet.to('cpu')
    # model.vae.to('cpu')
    # model.text_encoder.to('cpu')
    # model.timesteps.to('cpu')
    model.to(device)
    model.set_eval()
    # if args.use_fp16:
    #     model.half()

    transforms_test = Compose([
        ToTensor()
    ])        

    dataset_val = SHIQData(root="../datasets/SHIQ_data_10825/test/", transform=transforms_test, tokenizer=model.tokenizer)
    dl_val = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=2)


    metrics_list = []
    #ssim = SSIM()
    #psnr = PSNR()
    it = 0
    # save the output image
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize LPIPS
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    # note: LPIPS expects inputs in range [-1,1]    

     # --- FID için metrik objesi ---
    fid = FrechetInceptionDistance(feature=2048, reset_real_features=False).to(device)
    # reset_real_features=False diyoruz ki aligned ve unaligned bir arada hesaba girsin    

    with tqdm(total=len(dl_val)) as pbar:
        for data in dl_val:
            A = data['conditioning_pixel_values'].to(device)
            D = data['output_pixel_values'].to(device)
            # ground truth ve pred aralığı: [0,1] yap
            D = D * 0.5 + 0.5
            delta = (A-D)[0]
            # print(delta.shape, A.shape, D.shape)
            mask = 0.3*delta[0]+0.59*delta[1]+0.11*delta[2]
            mask = mask>0.707*mask.max()              

            with torch.no_grad():
                Dpred = model(A, prompt_tokens=data["input_ids"].to(device)) * 0.5 + 0.5

                # PSNR, SSIM, LPIPS, MSE, LMSE
                psnr_val = psnr(Dpred, D)
                ssim_val = ssim(Dpred, D)
                lpips_val = lpips_score(2*Dpred-1, 2*D-1, lpips_fn)
                mse_val   = torch.mean((Dpred - D) ** 2) * 100
                lmse_val  = torch.mean((mask * (Dpred - D)) ** 2) * 100
                metrics_list.append([
                    psnr_val, ssim_val, lpips_val,
                    mse_val.item(), lmse_val.item()
                ])

                # --- FID update ---
                # torchmetrics FID, hem fake hem real'i aynı anda update alır
                fid.update((D * 255).clamp(0,255).to(torch.uint8), real=True)
                fid.update((Dpred * 255).clamp(0,255).to(torch.uint8), real=False)

                save_image(Dpred, os.path.join(args.output_dir, data['image_name'][0] + "_D.png"))

            # progress bar
            avg = np.mean(np.array(metrics_list), axis=0)
            pbar.set_description(
                f"PSNR: {avg[0]:.4f} SSIM: {avg[1]:.4f} LPIPS: {avg[2]:.4f}"
                f" MSE: {avg[3]:.4f} LMSE: {avg[4]:.4f}"
            )
            pbar.update(1)

    # Birincil metrikleri yazdır
    print_overall_metrics(metrics_list)

    # --- FID'i hesapla ve yazdır ---
    fid_value = fid.compute().item()
    print(f"FID: {fid_value:.4f}")