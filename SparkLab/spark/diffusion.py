"""
Module for sky images diffusion to detector images.
"""

from dataclasses import dataclass
from typing import Callable

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.types import Tensor


__all__ = [
    'NoiseScheduler',
    'DiffusionRegistry',
    'get_diff_registry',
    'extract',
    'Sampler',
    'sample',
]


class NoiseScheduler:
    """
    Noise scheduler for the diffusion process.
    """
    def __init__(self, timesteps: int) -> None:
        self.timesteps = timesteps
    
    def linear(self, start: float, stop: float) -> Tensor:
        """
        Linear noise schedule for the forward diffusion process.
        From DDPM (Ho et al, 2020): https://arxiv.org/pdf/2006.11239.
        """
        return torch.linspace(start, stop, self.timesteps)
    
    def quadratic(self, start: float, stop: float) -> Tensor:
        """
        Quadratic noise schedule for the forward diffusion process.
        From DDIM (Song et al, 2021): https://arxiv.org/pdf/2010.02502.
        """
        return torch.linspace(start ** 0.5, stop ** 0.5, self.timesteps) ** 2

    def cosine(self, s: float = 0.008) -> Tensor:
        """
        Cosine noise schedule for the forward diffusion process.
        From Improved DDPM (Nichol, Dhariwal, 2021): https://arxiv.org/abs/2102.09672.
        """
        x = torch.linspace(0, self.timesteps, self.timesteps + 1) / self.timesteps
        x = torch.cos(0.5 * torch.pi * (x + s) / (1 + s)).pow(2)
        x = x / x[0]
        betas = 1 - (x[1:] / x[:-1])
        return torch.clip(betas, 1e-4, 0.9999)
    
    def sigmoid(self, start: float, stop: float) -> Tensor:
        """
        Sigmoid noise schedule for the forward diffusion process.
        From (Jabri et al, 2022): https://arxiv.org/pdf/2212.11972.
        """
        _betas = torch.linspace(-6.0, 6.0, self.timesteps)
        return torch.sigmoid(_betas) * (stop - start) + start


@dataclass
class DiffusionRegistry:
    """Container with pre-computed quantities for the diffusion process."""
    # forward diffusion
    sqrt_alphas_cumprod: Tensor
    sqrt_one_minus_alphas_cumprod: Tensor
    # model sampling
    alphas_cumprod_prev: Tensor
    posterior_sigma: Tensor


def get_diff_registry(betas: Tensor) -> DiffusionRegistry:
    """Computes the quantities needed for the diffusion process."""
    # double precision to avoid alpha-drift
    betas = betas.to(torch.float64)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    # forward diffusion q(x_t | x_{t-1})
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    # sampling (sigmas from DDIM https://arxiv.org/pdf/2010.02502)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    posterior_sigma = torch.sqrt(
        (1.0 - alphas_cumprod_prev) * (1.0 - alphas_cumprod / alphas_cumprod_prev),
    ) / sqrt_one_minus_alphas_cumprod
    # define table with pre-computed vals
    registry = DiffusionRegistry(
        sqrt_alphas_cumprod.float(),
        sqrt_one_minus_alphas_cumprod.float(),
        alphas_cumprod_prev.float(),
        posterior_sigma.float(),
    )
    return registry


def extract(vals: Tensor, t: Tensor, x_dims: int) -> Tensor:
    """
    Extracts the appropriate vals_t value for a batch of indices, for
    a generic signal (e.g., images, latent vectors, video, etc.).
    """
    b, *_ = t.shape
    out = vals.gather(-1, t)
    return out.reshape(b, *((1,) * (x_dims - 1)))


class Sampler(nn.Module):
    """
    Defines the sampling process from the model for inference.
    """
    def __init__(self, betas: Tensor) -> None:
        super().__init__()
        registry = get_diff_registry(betas)
        # register buffers for pre-comp quantities
        for key, value in registry.__dict__.items():
            self.register_buffer(key, value)

    def q_sample(self, x: Tensor, t: Tensor, noise: Tensor | None = None) -> Tensor:
        """Defines forward diffusion for input tensor by adding noise."""
        if noise is None:
            noise = torch.randn_like(x)
            
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, len(x.shape))
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, len(x.shape))
        
        return sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise
    
    @torch.no_grad()
    def p_sample(self, model: nn.Module, x: Tensor, t: Tensor, eta: float) -> Tensor:
        """
        Samples the signal from the model at step `t - 1`, from
        DDIM (Song et al, 2021): https://arxiv.org/pdf/2010.02502.
        """
        if len(t) != x.shape[0]:
            raise ValueError(
                f'Invalid timestep shape {t.shape}: must be equal to input batch {x.shape[0]}.'
            )

        x_dims = len(x.shape)
        pred_noise = model(x, t)

        # see https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
        a_t_prev = extract(self.alphas_cumprod_prev, t, x_dims)
        sqrt_a_t = extract(self.sqrt_alphas_cumprod, t, x_dims)
        sqrt_one_m_a_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_dims)
        sigma = eta * extract(self.posterior_sigma, t, x_dims)

        x0_pred = (x - sqrt_one_m_a_t * pred_noise) / sqrt_a_t
        dir_to_x = torch.sqrt(1.0 - a_t_prev - sigma ** 2) * pred_noise
        diff_noise = sigma * torch.randn_like(x)

        return torch.sqrt(a_t_prev) * x0_pred + dir_to_x + diff_noise


@torch.no_grad()
def _sample(
    sample_fn: Callable[[Tensor, Tensor], Tensor], 
    x_start: Tensor,
    timesteps: list[int],
    batch_size: int,
    full_process: bool,
) -> Tensor | list[Tensor]:
    """Sampling algorithm for signal denoising through time steps."""
    img = x_start
    diff_process: list[Tensor] = []
    for idx in tqdm(timesteps, desc=f'Sampling', total=len(timesteps)):
        t = torch.full((batch_size,), idx, device=x_start.device, dtype=torch.long)
        img = sample_fn(img, t)
        if full_process:
            diff_process.append(img.cpu())
    
    out = img if not full_process else diff_process
    return out


@torch.no_grad()
def sample(
    model: nn.Module,
    sampler: Sampler,
    timesteps: int | list[int],
    x_shape: torch.Size,
    eta: float = 0.0,
    x_t: Tensor | None = None,
    full_process: bool = False,
) -> Tensor | list[Tensor]:
    """Samples images from the model through denoising diffusion process."""
    device = next(model.parameters()).device
    batch_size = x_shape[0]
    x_start = (
        x_t if x_t is not None
        else torch.randn(x_shape, device=device)
    )
    timesteps_ = (
        list(range(0, timesteps))[::-1] if isinstance(timesteps, int) else timesteps[::-1]
    )
    sampler = sampler.to(device)
    sample_fn = lambda x, t: sampler.p_sample(model, x, t, eta)
    return _sample(sample_fn, x_start, timesteps_, batch_size, full_process)


# end