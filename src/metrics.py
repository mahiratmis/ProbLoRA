import torch
import numpy as np
from skimage.metrics import structural_similarity as compare_ssim
import lpips
from torchmetrics.image.fid import FrechetInceptionDistance
from typing import List, Optional


class Metrics:
    """
    Compute and track image quality metrics:
      - PSNR
      - SSIM
      - LPIPS
      - Global MSE
      - Local MSE (masked)
      - FID
    """

    # DEFAULT_METRICS = ["psnr", "ssim", "lpips", "mse", "fid"]
    DEFAULT_METRICS = ["psnr", "ssim", "lpips", "mse"]

    def __init__(
        self,
        device: torch.device,
        metric_names: Optional[List[str]] = None
    ):
        self.device = device
        self.metric_names = metric_names or self.DEFAULT_METRICS

        # LPIPS (expects inputs in [-1, 1])
        self.lpips_fn = lpips.LPIPS(net="alex").to(device)

        # FID: accumulate real & fake images, then compute once
        self.fid = FrechetInceptionDistance(
            feature=2048,
            reset_real_features=False
        ).to(device)

        # Storage for per-sample scores
        self.storage = {name: [] for name in self.metric_names}

    def psnr(self, pred: torch.Tensor, target: torch.Tensor, pixel_max: float = 1.0) -> float:
        """Compute PSNR assuming inputs in [0, 1]."""
        mse_val = torch.mean((target - pred) ** 2).item()
        rmse = np.sqrt(mse_val)
        return 20 * np.log10(pixel_max / (rmse + 1e-12))

    def ssim(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """
        Compute SSIM.
        Expects inputs shape [B, C, H, W] in [0, 1].
        Returns SSIM for the first sample in the batch.
        """
        # move to CPU, permute to HWC
        p = pred[0].permute(1, 2, 0).cpu().numpy()
        t = target[0].permute(1, 2, 0).cpu().numpy()
        return compare_ssim(t, p, channel_axis=2, data_range=1.0)

    def lpips_score(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Compute LPIPS, inputs expected in [-1, 1]."""
        with torch.no_grad():
            return self.lpips_fn(pred, target).item()

    def mse(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Global MSE over all pixels and channels."""
        return torch.mean((target - pred).detach() ** 2).item()

    def local_mse(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        reduction: str = "mean",
        eps: float = 1e-8
    ) -> torch.Tensor:
        """
        Compute masked MSE.
        mask: shape (B, 1, H, W) or broadcastable to (B, C, H, W)
        reduction: 'none', 'sum', or 'mean'
        """
        # ensure mask has same dims
        if mask.dim() == pred.dim() - 1:
            mask = mask.unsqueeze(1)
        mask = mask.expand_as(pred)

        sq_err = (pred - target).pow(2) * mask

        if reduction == "none":
            return sq_err

        total_err = sq_err.sum()
        if reduction == "sum":
            return total_err

        # mean reduction
        num_pix = mask.sum()
        return total_err / (num_pix + eps)

    def update_metrics(self, name: str, value: float):
        """Store a single metric value under `name`."""
        if name not in self.storage:
            raise KeyError(f"Metric '{name}' is not tracked.")
        self.storage[name].append(value)

    def update_fid(self, real: torch.Tensor, fake: torch.Tensor):
        """
        Update FID with one batch:
        real/fake should be in [0,1], uint8 conversion done here.
        """
        # scale to [0,255] uint8
        real_u8 = (real * 255).clamp(0, 255).to(torch.uint8)
        fake_u8 = (fake * 255).clamp(0, 255).to(torch.uint8)
        self.fid.update(real_u8, real=True)
        self.fid.update(fake_u8, real=False)

    def get_metric_means_dict(self) -> dict:
        """Return average of each stored metric, computing FID last."""
        means = {
            name: float(np.mean(vals)) if vals else float("nan")
            for name, vals in self.storage.items()
            if name != "fid"
        }

        if "fid" in self.storage:
            means["fid"] = float(self.fid.compute().item())

        return means

    def print_overall_metrics(self):
        """Print all metric means in one line, 4 decimals each."""
        means = self.get_metric_means()
        line = " ".join(f"{k}: {v:.4f}" for k, v in means.items())
        print(line)

    def reset(self):
        """
        Reset stored metric values and FID internal state.
        """
        # Clear stored lists
        for vals in self.storage.values():
            vals.clear()
        # Reset FID state
        self.fid.reset()        
