import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import vgg19, VGG19_Weights
from pytorch_msssim import ssim  # Install pytorch-msssim library
from lpips import LPIPS  # Install lpips library
from accelerate import Accelerator

class MultiScaleLoss(nn.Module):
    def __init__(self):
        super(MultiScaleLoss, self).__init__()
        self.l1_loss = nn.L1Loss()

    def forward(self, pred, target):
        device = pred.device  # Ensure the loss operates on the same device as inputs
        loss = self.l1_loss(pred, target)
        # return loss.to(device)
        for scale in [0.5, 0.25]:
            pred_resized = F.interpolate(pred, scale_factor=scale, mode='bilinear', align_corners=False)
            target_resized = F.interpolate(target, scale_factor=scale, mode='bilinear', align_corners=False)
            loss += self.l1_loss(pred_resized, target_resized)
        return loss

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.eval()
        self.vgg_layers = nn.Sequential(*list(vgg[:16]))
        for p in self.vgg_layers.parameters():
            p.requires_grad = False

        # ImageNet normalization parameters
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer('std',  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, pred, target):
        # If your model outputs are in [-1,1], first bring them to [0,1]:

        # Now normalize to ImageNet stats
        pred   = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std

        # Extract features and compute MSE
        pred_feats   = self.vgg_layers(pred)
        target_feats = self.vgg_layers(target)
        return F.mse_loss(pred_feats, target_feats)


class PerceptualLossGamma(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features.eval()
        self.vgg_layers = nn.Sequential(*list(vgg[:16]))
        for p in self.vgg_layers.parameters():
            p.requires_grad = False
        self.gamma = 0.5

    def forward(self, pred, target):
        pred_gamma = torch.pow(pred.clamp(min=1e-8), self.gamma)
        target_gamma = torch.pow(target.clamp(min=1e-8), self.gamma)
        
        # 2. Dynamic normalization (her görüntünün kendi stats'i)
        pred_norm = (pred_gamma - pred_gamma.mean()) / (pred_gamma.std() + 1e-8)
        target_norm = (target_gamma - target_gamma.mean()) / (target_gamma.std() + 1e-8)
        
        # 3. 0-1 aralığına getir
        pred_norm = (pred_norm - pred_norm.min()) / (pred_norm.max() - pred_norm.min() + 1e-8)
        target_norm = (target_norm - target_norm.min()) / (target_norm.max() - target_norm.min() + 1e-8)
        
        pred_feats = self.vgg_layers(pred_norm)
        target_feats = self.vgg_layers(target_norm)
        return F.mse_loss(pred_feats, target_feats)


class CombinedLoss(nn.Module):
    def __init__(self, max_val=1.0, psnr_min=10.0, psnr_max=50.0,
                 lambda_pix=1.0, lambda_ssim=1.0, lambda_perceptual=1.0, lambda_lpips=1.0,
                 lambda_psnr=1.0, lambda_l2=1.0, lambda_l1=1.0,
                 lambda_color=0.0, lambda_grad=0.0):
        super(CombinedLoss, self).__init__()
        self.max_val = max_val
        self.psnr_min = psnr_min
        self.psnr_max = psnr_max
        self.lambda_pix = lambda_pix
        self.lambda_ssim = lambda_ssim
        self.lambda_perceptual = lambda_perceptual
        self.lambda_lpips = lambda_lpips
        self.lambda_psnr = lambda_psnr
        self.lambda_l2 = lambda_l2
        self.lambda_l1 = lambda_l1
        self.lambda_color = lambda_color
        self.lambda_grad = lambda_grad

        self.multi_scale_loss = MultiScaleLoss()
        self.perceptual_loss = PerceptualLoss()
        self.lpips_loss = LPIPS(net='vgg')

    def psnr_loss(self, pred, target):
        mse = torch.mean((pred - target) ** 2)
        psnr = 20 * torch.log10(self.max_val / torch.sqrt(mse + 1e-8))
        psnr_normalized = (psnr - self.psnr_min) / (self.psnr_max - self.psnr_min)
        psnr_normalized = torch.clamp(psnr_normalized, 0, 1)
        return 1 - psnr_normalized

    def color_loss(self, pred, target):
        pred_mean = pred.mean(dim=(2, 3))  # shape: (B, C)
        target_mean = target.mean(dim=(2, 3))
        return F.mse_loss(pred_mean, target_mean)

    def grad_loss(self, pred, target):
        def gradient(x):
            dx = x[:, :, :, :-1] - x[:, :, :, 1:]
            dy = x[:, :, :-1, :] - x[:, :, 1:, :]
            return dx, dy

        dx_pred, dy_pred = gradient(pred)
        dx_target, dy_target = gradient(target)
        return F.l1_loss(dx_pred, dx_target) + F.l1_loss(dy_pred, dy_target)

    def forward(self, pred, target):
        # Denormalize from [-1, 1]
        pred_scaled = pred * 0.5 + 0.5
        target_scaled = target * 0.5 + 0.5

        # Individual losses
        pix_loss = self.multi_scale_loss(pred_scaled, target_scaled) if self.lambda_pix > 0 else torch.tensor(0, device=pred.device)
        l2_loss = F.mse_loss(pred_scaled, target_scaled) if self.lambda_l2 > 0 else torch.tensor(0, device=pred.device)
        l1_loss = F.l1_loss(pred_scaled, target_scaled) if self.lambda_l1 > 0 else torch.tensor(0, device=pred.device)
        perceptual_loss = self.perceptual_loss(pred_scaled, target_scaled) if self.lambda_perceptual > 0 else torch.tensor(0, device=pred.device)
        lpips_loss = self.lpips_loss(pred, target).mean() if self.lambda_lpips > 0 else torch.tensor(0, device=pred.device)
        ssim_loss = 1 - ssim(pred_scaled, target_scaled, data_range=self.max_val, size_average=True) if self.lambda_ssim > 0 else torch.tensor(0, device=pred.device)
        psnr_loss = self.psnr_loss(pred_scaled, target_scaled) if self.lambda_psnr > 0 else torch.tensor(0, device=pred.device)
        color_loss = self.color_loss(pred_scaled, target_scaled) if self.lambda_color > 0 else torch.tensor(0, device=pred.device)
        grad_loss = self.grad_loss(pred_scaled, target_scaled) if self.lambda_grad > 0 else torch.tensor(0, device=pred.device)

        # Combine
        total_loss = (
            self.lambda_pix * pix_loss +
            self.lambda_ssim * ssim_loss +
            self.lambda_perceptual * perceptual_loss +
            self.lambda_lpips * lpips_loss +
            self.lambda_psnr * psnr_loss +
            self.lambda_l2 * l2_loss +
            self.lambda_l1 * l1_loss +
            self.lambda_color * color_loss +
            self.lambda_grad * grad_loss
        )

        loss_dict = {
            "pixel_loss": self.lambda_pix * pix_loss.item(),
            "ssim_loss": self.lambda_ssim * ssim_loss.item(),
            "perceptual_loss": self.lambda_perceptual * perceptual_loss.item(),
            "lpips_loss": self.lambda_lpips * lpips_loss.item(),
            "psnr_loss": self.lambda_psnr * psnr_loss.item(),
            "l2_loss": self.lambda_l2 * l2_loss.item(),
            "l1_loss": self.lambda_l1 * l1_loss.item(),
            "color_loss": self.lambda_color * color_loss.item(),
            "grad_loss": self.lambda_grad * grad_loss.item(),
            "total_loss": total_loss.item()
        }

        return total_loss, loss_dict

